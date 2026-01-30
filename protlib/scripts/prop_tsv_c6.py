import argparse
import os
import sys

import tqdm

from functools import partial

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-g', '--graph', type=str)
parser.add_argument('-o', '--output', type=str)

parser.add_argument('-d', '--device', type=str, default="1")
parser.add_argument('-n', '--n-props', type=int)
parser.add_argument('-dr', '--direction', type=str, default='backward')
parser.add_argument('-f', '--func', type=str, default='max')


parser.add_argument('-b', '--batch-size', type=int, default=30000)
parser.add_argument('-bi', '--batch-inner', type=int, default=5000)


def get_kernel(direction):
    kernel = cp.ElementwiseKernel(
        """
        int64 i_,
        raw int64 j_,
        int64 col,
        int64 l,
        int64 lx
        """,
        'raw float32 arr',
        """
        int k;

        for (k = 0; k < l; k++) {{
            arr[i_ * lx + col] = {0}(arr[i_ * lx + col], arr[i_ * lx + j_[k]]);
        }}
        """.format(direction),
        'propagate_{0}_kernel'.format(direction)
    )

    return kernel


def propagate_bwd(mat, G, kernel):
    indexer = cp.arange(mat.shape[0])

    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        adj = cp.asarray(adj, dtype=cp.int64)
        kernel(indexer, adj, f, adj.shape[0], mat.shape[1], mat.ravel())

    return


def propagate_fwd(mat, G, kernel):
    indexer = cp.arange(mat.shape[0])

    D = get_depths(G, True)
    for i in range(len(D)):
        for f in D[i]:

            adj = G.terms_list[f]['adj']

            if len(adj) == 0:
                continue

            adj = cp.asarray(adj, dtype=cp.int64)
            kernel(indexer, adj, f, adj.shape[0], mat.shape[1], mat.ravel())

    return


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cupy as cp
    import cudf

    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph, get_depths = [None] * 5

    prop_max_kernel = get_kernel('max')
    prop_min_kernel = get_kernel('min')

    propagate_d = propagate_fwd if args.direction == 'forward' else propagate_bwd
    propagate = partial(propagate_d, kernel=prop_max_kernel if args.func == 'max' else prop_min_kernel)

    trainTerms = cudf.read_csv(args.path, sep='\t', usecols=['EntryID', 'term'])
    ontologies = []
    for ns, terms_dict in obo_parser(args.graph).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    back_prot_id = index = trainTerms['EntryID'].drop_duplicates().reset_index(drop=True)
    length = len(back_prot_id)
    prot_id = cudf.Series(cp.arange(length), back_prot_id)
    trainTerms['id'] = trainTerms['EntryID'].map(prot_id)

    flg = True

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for i in tqdm.tqdm(range(0, length, args.batch_size)):
        print('Checkpoint0')
        sample = trainTerms.query(f'(id >= {i}) & (id < {i + args.batch_size})')
        batch_len = min(args.batch_size, length - i)
        print('Checkpoint1')

        for G in ontologies:
            mapper = cudf.Series(get_funcs_mapper(G))
            sample['term_id'] = sample['term'].map(mapper)
            sample_ont = sample.dropna().astype({'term_id': cp.int32})

            sample_ont['id'] = sample_ont['id'] - i

            mat = cp.zeros((batch_len, G.idxs), dtype=cp.float32)
            cp.add.at(
                mat, (sample_ont['id'].values, sample_ont['term_id'].values), 1
            )
            # mat.scatter_add((sample_ont['id'].values, sample_ont['term_id'].values), 1)
            mat = cp.clip(mat, 0, 1)

            for _ in range(args.n_props):
                propagate(mat, G)

            for j in range(0, mat.shape[0], args.batch_inner):
                row, col = cp.nonzero(mat[j: j + args.batch_inner])

                sample_batch = cudf.DataFrame({

                    'EntryID': back_prot_id[i + j + row].reset_index(drop=True),
                    'term': cudf.Series(get_funcs_mapper(G, False))[col].reset_index(drop=True),
                }).sort_values(['EntryID', 'term'], ascending=True)

                ns_id, ns_str = get_ns_id(G)
                asp = ns_str.upper() + 'O'

                sample_batch['aspect'] = asp

                with open(args.output, 'w' if flg else 'a') as f:
                    sample_batch.to_csv(f, index=False, sep='\t', header=flg)
                    flg = False

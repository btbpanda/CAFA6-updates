import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tqdm
import yaml
from numba import njit, prange
from pandas import Series, DataFrame

print(os.path.abspath(os.path.join(__file__, '../../../')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

try:
    from protlib.metric import obo_parser, Graph, ia_parser, get_funcs_mapper
except Exception:
    obo_parser, Graph, ia_parser, get_funcs_mapper = None, None, None, None

parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--og    -path', type=str)
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-t', '--terms', type=str)
parser.add_argument('-g', '--graph', type=str)
parser.add_argument('-i', '--ia', type=str)
parser.add_argument('-s', '--seq', type=str)

parser.add_argument('-b', '--batch-size', type=int)
parser.add_argument('-p', '--propagate', type=bool, default=False)
parser.add_argument('-cf', '--cafa', type=int, default=5)

ASPECT_CAFA5 = {'BPO': 'biological_process', 'MFO': 'molecular_function', 'CCO': 'cellular_component'}
ASPECT_CAFA6 = {'P': 'biological_process', 'F': 'molecular_function', 'C': 'cellular_component'}



@njit
def prop_max_cpu(mat, k, adj):
    for i in prange(mat.shape[0]):
        if mat[i, k] == 1:
            continue

        for j in adj:
            if mat[i, j] == 1:
                mat[i, k] = 1
                continue

    return


def propagate_target(mat, G):
    for f in G.order:

        adj = G.terms_list[f]['children']

        if len(adj) == 0:
            continue

        prop_max_cpu(mat, f, np.asarray(adj))

    return


if __name__ == '__main__':
    args = parser.parse_args()

    path = args.output
    os.makedirs(path, exist_ok=True)

    trainTerms = pd.read_csv(args.terms, sep='\t')

    terms = trainTerms.set_index('EntryID')
    unique_terms = set(trainTerms['EntryID'])

    terms['namespace'] = terms['aspect'].map(
        ASPECT_CAFA5 if args.cafa == 5 else ASPECT_CAFA6
    )

    vec_train_protein_ids = pd.read_feather(
        args.seq, columns=['EntryID'],
    )['EntryID'].values

    ia_dict = ia_parser(args.ia)
    ontologies = []

    for ns, terms_dict in obo_parser(args.graph).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, True))

    priors_D = {
        'biological_process': {
            'psum': 0, 'pcount': 0, 'nsum': 0, 'ncount': 0
        },
        'molecular_function': {
            'psum': 0, 'pcount': 0, 'nsum': 0, 'ncount': 0
        },
        'cellular_component': {
            'psum': 0, 'pcount': 0, 'nsum': 0, 'ncount': 0
        },
    }


    for n, i in tqdm.tqdm(enumerate(range(0, vec_train_protein_ids.shape[0], args.batch_size))):

        idx = vec_train_protein_ids[i: i + args.batch_size]
        num = Series(np.arange(idx.shape[0]), index=idx)
        trm = terms.loc[[x for x in idx if x in unique_terms]]

        # reformat targets
        for ont in ontologies:

            ns, terms_names = ont.namespace, [x['id'] for x in ont.terms_list]
            os.makedirs(os.path.join(path, ns), exist_ok=True)

            trm_ont = trm.query(f"namespace == '{ns}'").copy()

            trm_ont['id'] = trm_ont['term'].map(get_funcs_mapper(ont)).values
            trm_ont['n'] = num.loc[trm_ont.index].values

            trg = np.zeros((num.shape[0], ont.idxs), dtype=np.float32)
            np.add.at(trg, (trm_ont['n'].values, trm_ont['id'].values), 1)

            if args.propagate:
                propagate_target(trg, ont)

            # create NaNs from graph
            for k, node in enumerate(ont.terms_list):
                adj = node['adj']
                if len(adj) > 0:
                    na = np.nonzero(np.nansum(trg[:, adj], axis=1) == 0)[0]
                    assert np.nansum(trg[na, k]) == 0, 'Should be empty'
                    trg[na, k] = np.nan

            trg = DataFrame(trg, columns=terms_names)
            # update priors data
            bs, nulls = trg.shape[0], trg.isnull().sum().values

            priors_D[ns]['psum'] = priors_D[ns]['psum'] + trg.sum().fillna(0).values
            priors_D[ns]['pcount'] = priors_D[ns]['pcount'] + bs - nulls

            priors_D[ns]['nsum'] = priors_D[ns]['nsum'] + nulls
            priors_D[ns]['ncount'] = priors_D[ns]['ncount'] + bs

            # save partitioned
            trg['EntryID'] = idx
            trg.to_parquet(os.path.join(path, ns, f'part_{str(n).zfill(2)}.parquet'))

    for ns in priors_D:
        priors = priors_D[ns]

        mean = np.where(priors['pcount'] == 0, 0, priors['psum'] / priors['pcount'])
        nulls = priors['nsum'] / priors['ncount']

        np.save(
            os.path.join(path, ns, f'prior.npy'), mean
        )

        np.save(
            os.path.join(path, ns, f'nulls.npy'), nulls
        )

        print(f'Total {ns} parsed: {priors['psum'].sum()}')


    
    # count priors
    # for ont in ontologies:
    #     trg = pd.read_parquet(glob.glob(os.path.join(path, ont.namespace, f'part_*')),
    #                           columns=[x['id'] for x in ont.terms_list])
    #     mean = trg.mean().fillna(0).values
    #     nulls = trg.isnull().mean().values
    #
    #     joblib.dump(mean, os.path.join(path, ont.namespace, f'prior.pkl'))
    #     joblib.dump(nulls, os.path.join(path, ont.namespace, f'nulls.pkl'))

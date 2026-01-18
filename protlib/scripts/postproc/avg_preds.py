import argparse
import os
import sys

import tqdm
import polars as pl

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()

parser.add_argument('-imax', '--input-max', type=str)
parser.add_argument('-imin', '--input-min', type=str)
parser.add_argument('-o', '--output-path', type=str)

parser.add_argument('-d', '--device', type=str, default="1")


def write_csv_batch(df, file, mode='w', batch_size=10000000, **kwargs):
    for i in tqdm.tqdm(range(0, len(df), batch_size)):
        batch = df[i: i + batch_size]

        with open(file, mode) as f:
            batch.to_csv(f, **kwargs)
            mode = 'a'

    return


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cupy as cp
    import cudf

    try:
        from protlib.metric import get_funcs_mapper, get_ns_id, obo_parser, Graph
    except Exception:
        get_funcs_mapper, get_ns_id, obo_parser, Graph = [None] * 4

    pred_max = cudf.from_pandas(
        pl.read_csv(
            args.input_max, has_header=False, separator='\t', new_columns=['EntryID', 'term', 'p'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'p': pl.Float32},
        ).to_pandas()
    )

    pred_min = cudf.from_pandas(
        pl.read_csv(
            args.input_min, has_header=False, separator='\t', new_columns=['EntryID', 'term', 'p'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'p': pl.Float32},
        ).to_pandas()
    )

    pred = cudf.concat([pred_max, pred_min], ignore_index=True)

    pred = pred.groupby(['EntryID', 'term'])['p'].mean().reset_index().sort_values(['EntryID', 'p'], ascending=False)

    write_csv_batch(
        pred, args.output_path,
        mode='w', batch_size=10000000,
        sep='\t', header=False, index=False
    )
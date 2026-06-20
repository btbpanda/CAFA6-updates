"""This is the final script which post process stacker results and make submission file
"""
import os
import sys
import subprocess
import yaml
import tqdm
import polars as pl
from pathlib import Path

def write_csv_batch(df, file, mode='w', batch_size=10000000, **kwargs):
    for i in tqdm.tqdm(range(0, len(df), batch_size)):
        batch = df[i: i + batch_size]

        with open(file, mode) as f:
            batch.to_csv(f, **kwargs)
            mode = 'a'

    return

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['devices'][0]) # args.device

    import cudf

    data_path = Path(config['data_path']).resolve()

    uniprot_prop_test_path = data_path / 'uniprot/prop/test'
    cafa6_path = data_path / 'cafa-6-protein-function-prediction'
    graph_path = cafa6_path / 'Train/go-basic.obo'

    models_path = Path(config['models_path']).resolve()
    output_path = models_path / 'submission.tsv'
    postproc_path = models_path / 'pred_{direction}.tsv'

    PYTHON = sys.executable
    DEVICE = config['devices'][0]

    stacker_0 = models_path / 'prediction_nn_cross_pbtfidf_f_42' / 'pred_tta_0.tsv'
    stacker_1 = models_path / 'prediction_nn_cross_pbtfidf_f_53' / 'pred_tta_0.tsv'

    # -----------------------------------
    # AVERAGE STACKER SEEDS
    # -----------------------------------

    command = sys.executable + f""" ./protlib/scripts/postproc/avg_preds.py \
            --input-max {stacker_0} \
            --input-min {stacker_1} \
            --output-path {output_path} \
            --device 0"""
    subprocess.run(command, shell=True)

    # -----------------------------------
    # DO POSTPROCESSING
    # -----------------------------------

    command = sys.executable + f""" ./protlib/scripts/postproc/step_c6_pl.py \
            --graph-path {graph_path} \
            --input-path {output_path} \
            --output-path {postproc_path} \
            --device {DEVICE} \
            --batch_size 15000 \
            --batch_inner 3000 \
            --lr 0.7 \
            --direction {{direction}}"""

    reduction = sys.executable + f""" ./protlib/scripts/postproc/avg_preds.py \
            --input-max {str(postproc_path).format(direction='max')} \
            --input-min {str(postproc_path).format(direction='min')} \
            --output-path {output_path} \
            --device {DEVICE}"""

    for _ in range(3):
        subprocess.run(command.format(direction='max'), shell=True)
        subprocess.run(command.format(direction='min'), shell=True)

        subprocess.run(reduction, shell=True)

    # -----------------------------------
    # ADD KNOWN TERMS
    # -----------------------------------

    no_know_terms = cudf.from_pandas(
        pl.read_csv(
            uniprot_prop_test_path / 'no-know.tsv',
            separator='\t', columns=['EntryID', 'term'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'aspect': pl.Categorical, }
        ).with_columns(
            p=pl.lit(0.99, dtype=pl.Float32)
        ).to_pandas()
    )

    lim_know_terms = cudf.from_pandas(
        pl.read_csv(
            uniprot_prop_test_path / 'lim-know.tsv',
            separator='\t', columns=['EntryID', 'term'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'aspect': pl.Categorical, }
        ).with_columns(
            p=pl.lit(0.99, dtype=pl.Float32)
        ).to_pandas()
    )

    part_know_terms = cudf.from_pandas(
        pl.read_csv(
            uniprot_prop_test_path / 'part-know.tsv',
            separator='\t', columns=['EntryID', 'term'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'aspect': pl.Categorical, }
        ).with_columns(
            p=pl.lit(0.99, dtype=pl.Float32)
        ).to_pandas()
    )

    pred = cudf.from_pandas(
        pl.read_csv(
            output_path, has_header=False, separator='\t', new_columns=['EntryID', 'term', 'p'],
            schema={'EntryID': pl.Categorical, 'term': pl.Categorical, 'p': pl.Float32}
        ).to_pandas()
    )

    pred = cudf.concat([
        no_know_terms, lim_know_terms, part_know_terms, pred
    ], ignore_index=True)

    pred = pred.groupby(['EntryID', 'term'])['p'].max().reset_index() \
        .sort_values(['EntryID', 'p'], ascending=False)

    # -----------------------------------
    # THIS IS THE END ...
    # -----------------------------------

    write_csv_batch(
        pred, output_path,
        mode='w', batch_size=10000000,
        sep='\t', header=False, index=False
    )


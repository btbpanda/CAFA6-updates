"""Save evidence / types features in format that stacker accepts
"""

import subprocess
import yaml
import polars as pl
from pathlib import Path

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    # TODO: Check paths later ...
    uni_codes_path = data_path / 'cafa6-uniprot-codes'

    raw_codes_path = uni_codes_path / 'raw'
    prop_codes_path = uni_codes_path / 'prop'
    evidence_path = prop_codes_path / 'evidence'
    type_path = prop_codes_path / 'type'

    # features_path = data_path / 'uniprot/raw/features' # raw uniprot auto labels

    stacker_feats_path = data_path / 'features'
    stacker_feats_path.mkdir(parents=True, exist_ok=True)

    terms = pl.read_csv(
        data_path / 'train_terms.tsv', separator='\t' # propagated CAFA 6 terms
    )

    # -----------------------------------
    # Total evidence count: 1 evidence eq 1 term
    # -----------------------------------

    evidence = pl.read_csv(
        uni_codes_path / 'train_auto.tsv',
        separator='\t'
    )

    for i in range(2, 6):
        evidence.filter(
            pl.col('cnt') >= i
        ).select(
            'EntryID', 'term'
        ).write_csv(
            stacker_feats_path / f'train_ev_cnt_{i}.tsv', separator='\t'
        )

    evidence = pl.read_csv(
        uni_codes_path / 'old_train_auto.tsv',
        separator='\t'
    )

    for i in range(2, 6):
        evidence.filter(
            pl.col('cnt') >= i
        ).select(
            'EntryID', 'term'
        ).write_csv(
            stacker_feats_path / f'old_train_ev_cnt_{i}.tsv', separator='\t'
        )

    evidence = pl.read_csv(
        uni_codes_path / 'test_auto228.tsv',
        separator='\t'
    )

    for i in range(2, 6):
        evidence.filter(
            pl.col('cnt') >= i
        ).select(
            'EntryID', 'term'
        ).write_csv(
            stacker_feats_path / f'test_ev_cnt_{i}.tsv', separator='\t'
        )

    # -----------------------------------
    # Total evidence count: 1 evidence eq 1 term at 1 source
    # -----------------------------------

    for dataset in ['train', 'old_train', 'test']:

        evidence = pl.concat([
            pl.read_csv(x, separator='\t') for x in evidence_path.glob(f'{dataset}_*.tsv')
        ]).group_by(
            'EntryID', 'term'
        ).agg(
            pl.col('cnt').sum()
        )

        for i in range(2, 6):
            evidence.filter(
                pl.col('cnt') >= i
            ).select(
                'EntryID', 'term'
            ).write_csv(
                stacker_feats_path / f'{dataset}_ev_codes_cnt_{i}.tsv', separator='\t'
            )

    # -----------------------------------
    # Copy all propagated data
    # -----------------------------------

    for path in prop_codes_path.glob('*/*.tsv'):

        pl.read_csv(
            path, separator='\t'
        ).select(
            'EntryID', 'term'
        ).write_csv(
            stacker_feats_path / f'{path.stem}_prop.tsv', separator='\t'
        )

    # -----------------------------------
    # Just copy all auto labels (pure 1/0 labels as old version + non propagated data)
    # -----------------------------------

    task = f"""
    cp {data_path / 'uniprot/prop/features/train_auto.tsv'} {stacker_feats_path / 'train_auto.tsv'}
    cp {data_path / 'uniprot/prop/features/old_train_auto.tsv'} {stacker_feats_path / 'old_train_auto.tsv'}
    cp {data_path / 'uniprot/prop/features/test_auto228.tsv'} {stacker_feats_path / 'test_auto.tsv'}
    cp {data_path / 'uniprot/raw/features/train_auto.tsv'} {stacker_feats_path / 'train_raw_auto.tsv'}
    cp {data_path / 'uniprot/raw/features/old_train_auto.tsv'} {stacker_feats_path / 'old_train_raw_auto.tsv'}
    cp {data_path / 'uniprot/raw/features/test_auto228.tsv'} {stacker_feats_path / 'test_raw_auto.tsv'}
    
    cp {evidence_path / '*'} {stacker_feats_path}
    cp {type_path / '*'} {stacker_feats_path}
    """
    subprocess.run(task, shell=True)
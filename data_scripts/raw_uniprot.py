"""This is parser of uniprot to parquet. Select only proteins from any of CAFA 5 / CAFA 6

"""
import argparse
import os

import pandas as pd
import tqdm
import yaml

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str)
parser.add_argument('-o', '--output', type=str, )

if __name__ == '__main__':
    args = parser.parse_args()

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    fasta_path = data_path / 'helpers/fasta'
    # output path
    uniprot_path = data_path / 'uniprot'
    uniprot_path.mkdir(parents=True, exist_ok=True)


    train = set(pd.read_feather(
        fasta_path / 'train_seq.feather', columns=['EntryID']
    )['EntryID'])

    old_train = set(pd.read_feather(
        fasta_path / 'old_train_seq.feather', columns=['EntryID']
    )['EntryID'])

    test = set(pd.read_feather(
        fasta_path / 'test_seq.feather', columns=['EntryID']
    )['EntryID'])

    idxs = train.union(old_train).union(test)

    reader = pd.read_csv(
        data_path / args.file,
        sep='\t',
        header=None,
        names=['x', 'EntryID', 'xx', 'type', 'term', 'y', 'source', 'yyy', 'z', 'zz', 'zzz', 'a', 'aa', 'date', 'b',
               'bb', 'bbb'],
        usecols=['EntryID', 'type', 'term', 'source', 'date'],
        chunksize=1_000_000,
        na_filter=True
    )

    store = []

    for n, batch in tqdm.tqdm(enumerate(reader)):

        if n == 0:
            batch = batch.dropna()

        filtred = batch[batch['EntryID'].isin(idxs)]
        filtred = filtred.drop_duplicates()

        store.append(filtred)

    store = pd.concat(store, ignore_index=True).drop_duplicates()
    store.to_parquet(uniprot_path / args.output)


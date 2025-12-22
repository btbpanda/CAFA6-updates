import argparse
import os

import pandas as pd
import tqdm
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str)
parser.add_argument('-o', '--output', type=str, )
parser.add_argument('-ft', '--fasta', type=str, )

if __name__ == '__main__':
    args = parser.parse_args()

    train = set(pd.read_feather(
        os.path.join(args.fasta, 'train_seq.feather'), columns=['EntryID']
    )['EntryID'])

    old_train = set(pd.read_feather(
        os.path.join(args.fasta, 'old_train_seq.feather'), columns=['EntryID']
    )['EntryID'])

    test = set(pd.read_feather(
        os.path.join(args.fasta, 'test_seq.feather'), columns=['EntryID']
    )['EntryID'])

    idxs = train.union(old_train).union(test)

    reader = pd.read_csv(
        args.file,
        sep='\t',
        header=None,
        names=['x', 'EntryID', 'xx', 'type', 'term', 'y', 'source', 'yyy', 'z', 'zz', 'zzz', 'a', 'aa', 'date', 'b',
               'bb', 'bbb'],
        usecols=['EntryID', 'type',  'term', 'source', ],
        chunksize=1_000_000,
        na_filter=True
    )

    store = []

    for n, batch in tqdm.tqdm(enumerate(reader)):

        if n == 0:
            batch = batch.dropna()

        filtred = batch[(batch['EntryID'].isin(idxs)) & (~batch['type'].fillna('').str.startswith('NOT'))]
        filtred = filtred[['EntryID', 'term', 'source', ]].drop_duplicates()

        if len(store) > 0 and len(filtred) > 0 and \
                (store[-1].iloc[-1].values == filtred.iloc[0].values).all():
            filtred = filtred[1:]

        store.append(filtred)

    store = pd.concat(store, ignore_index=True).drop_duplicates()

    os.makedirs(os.path.join(args.output, 'raw'), exist_ok=True)
    store.to_parquet(os.path.join(args.output, 'raw', 'uniprot.parquet'), )

    kaggle_codes = set('IDA IMP TAS IPI IEP IGI IC EXP HTP HDA HMP HGI HEP'.split())

    # save labeling by categories
    is_kaggle = store['source'].isin(kaggle_codes)

    # save ground truth labeling
    store[store['EntryID'].isin(train) & is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'train_terms.tsv'), sep='\t', index=False
    )

    store[store['EntryID'].isin(old_train) & is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'old_train_terms.tsv'), sep='\t', index=False
    )

    store[store['EntryID'].isin(test - train) & is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'test_leak_terms.tsv'), sep='\t', index=False
    )

    # save features labeling
    store[store['EntryID'].isin(train) & ~is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'train_auto.tsv'), sep='\t', index=False
    )

    store[store['EntryID'].isin(old_train) & ~is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'old_train_auto.tsv'), sep='\t', index=False
    )

    store[store['EntryID'].isin(test) & ~is_kaggle].to_csv(
        os.path.join(args.output, 'raw', 'test_auto.tsv'), sep='\t', index=False
    )
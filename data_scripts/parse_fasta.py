"""This script parse annoying fasta files to common format ...
Here we build train / old_train based on CAFA 5 / CAFA 6 data
train = CAFA 6
old_train = CAFA 5 - CAFA 6

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import yaml
from pathlib import Path

import os
from Bio import SeqIO, Seq

def parse_line_c6_train(line):

    D = {}
    D['seq'] = str(line.seq)
    _, D['EntryID'], D['tax_name'] = line.name.split('|')
    D['tax_name'] = D['tax_name'].split('_')[1]

    return pd.Series(D)

def parse_line_c5_train(line):

    D = {}
    D['seq'] = str(line.seq)
    D['EntryID'] = line.name
    D['tax_name'] = line.description.split(' ')[1].split('|')[2]
    D['tax_name'] = D['tax_name'].split('_')[1]

    return pd.Series(D)

def parse_line_c6_test(line):

    D = {}
    D['seq'] = str(line.seq)
    D['EntryID'], tax = line.description.split(' ')
    D['taxonomyID'] = int(tax)

    return pd.Series(D)



if __name__ == '__main__':
    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()

    # -----------------------------------
    # CAFA 6 Train
    # -----------------------------------

    # Load sequences
    sequences = SeqIO.parse(
        data_path / 'cafa-6-protein-function-prediction/Train/train_sequences.fasta', 'fasta'
    )

    train_df = pd.concat([pd.DataFrame(parse_line_c6_train(x)).T for x in sequences])

    # fetch taxonomy
    train_tax = pd.read_csv(
        data_path / 'cafa-6-protein-function-prediction/Train/train_taxonomy.tsv',
        sep='\t', header=None, names=['EntryID', 'taxonomyID']
    )

    train_df = pd.merge(train_df, train_tax, on='EntryID', how='left')

    # -----------------------------------
    # CAFA 5 Train
    # -----------------------------------

    # Load sequences
    sequences = SeqIO.parse(
        data_path / 'cafa-5-protein-function-prediction/Train/train_sequences.fasta', 'fasta'
    )

    old_train_df = pd.concat([pd.DataFrame(parse_line_c5_train(x)).T for x in sequences])

    # fetch taxonomy
    train_tax = pd.read_csv(
        data_path / 'cafa-5-protein-function-prediction/Train/train_taxonomy.tsv',
        sep='\t', header=None, names=['EntryID', 'taxonomyID']
    )

    old_train_df = pd.merge(old_train_df, train_tax, on='EntryID', how='left')
    old_train_df = old_train_df[~old_train_df['EntryID'].isin(train_df['EntryID'])].reset_index(drop=True)

    # -----------------------------------
    # CAFA 6 Test
    # -----------------------------------

    # Load sequences
    sequences = SeqIO.parse(
        data_path / 'cafa-6-protein-function-prediction/Test/testsuperset.fasta', 'fasta'
    )

    test_df = pd.concat([pd.DataFrame(parse_line_c6_test(x)).T for x in sequences]).astype({'taxonomyID': np.int64})

    # -----------------------------------
    # SAVE
    # -----------------------------------

    dump_path = data_path / 'helpers/fasta'
    dump_path.mkdir(parents=True, exist_ok=True)

    train_df[:15000].to_feather(dump_path / 'train_seq.feather', )
    old_train_df[:12000].to_feather(dump_path / 'old_train_seq.feather', )
    test_df[:25000].to_feather(dump_path / 'test_seq.feather', )


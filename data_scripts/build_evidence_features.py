"""Build features based on the evidence codes of term and type of relation with term

"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

TYPES = [
    'involved_in', 'enables', 'located_in', 'part_of', 'is_active_in', 'acts_upstream_of_or_within'
]

IEA_CODES = [
    'IEA', 'IBA', 'ISO', 'ISS', 'NAS', 'ND', 'ISM',
]

KAGGLE_CODES = 'IDA IMP TAS IPI IEP IGI IC EXP HTP HDA HMP HGI HEP'.split()

def process_uniprot(uniprot, ):
    uniprot['is_train'] = uniprot['EntryID'].isin(train['EntryID'])
    uniprot['is_old_train'] = uniprot['EntryID'].isin(old_train['EntryID'])
    uniprot['is_test'] = uniprot['EntryID'].isin(test['EntryID'])

    uniprot['is_kaggle'] = uniprot['source'].isin(set('IDA IMP TAS IPI IEP IGI IC EXP HTP HDA HMP HGI HEP'.split()))
    uniprot['is_pos'] = ~uniprot['type'].str.startswith('NOT')
    uniprot['valid_terms'] = uniprot['term'].isin(aspect)

    uniprot['type'] = uniprot['type'].where(uniprot['type'].isin(TYPES), 'other')
    uniprot['evidence'] = uniprot['source'].where(uniprot['source'].isin(IEA_CODES), 'other')
    uniprot['aspect'] = uniprot['term'].map(aspect)

    return uniprot

if __name__ == '__main__':

    from protlib.cafa_utils import obo_parser, Graph

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    uniprot_path = data_path / 'uniprot'
    graph_path = data_path / 'cafa-6-protein-function-prediction/Train/go-basic.obo'

    evidence_path = data_path / 'cafa6-uniprot-codes/raw/evidence'
    evidence_path.mkdir(parents=True, exist_ok=True)
    type_path = data_path / 'cafa6-uniprot-codes/raw/type'
    type_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # GRAPH
    # -----------------------------------

    asp_D = {
        'biological_process': 'BPO',
        'molecular_function': 'MFO',
        'cellular_component': 'CCO'
    }

    ontologies, roots, aspect = [], [], {}
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))
        roots.append([x['id'] for x in ontologies[-1].terms_list if len(x['adj']) == 0][0])

        aspect = {**aspect, **{
            x['id']: asp_D[ontologies[-1].namespace] for x in ontologies[-1].terms_list
        }}

    # -----------------------------------
    # Competition data
    # -----------------------------------

    terms = pd.read_csv(
        data_path / 'cafa-6-protein-function-prediction/Train/train_terms.tsv', sep='\t'
    )

    train = pd.read_feather(data_path / 'helpers/fasta/train_seq.feather')
    old_train = pd.read_feather(data_path / 'helpers/fasta/old_train_seq.feather')
    test = pd.read_feather(data_path / 'helpers/fasta/test_seq.feather')

    # -----------------------------------
    # GOA data
    # -----------------------------------

    uniprot228 = pd.read_parquet(uniprot_path / 'raw_uniprot228.parquet')
    uniprot228 = process_uniprot(uniprot228, )

    # -----------------------------------
    # EVIDENCE FEATURES
    # -----------------------------------

    for code in IEA_CODES + ['other']:
        efeats228 = uniprot228.query(f'~is_kaggle & is_pos & valid_terms & (evidence == "{code}")') \
            [['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
            .drop_duplicates(ignore_index=True)

        for dataset in ['train', 'old_train', 'test']:

            path = evidence_path / f'{dataset}_{code}.tsv'
            efeats228.query(f'is_{dataset}')[['EntryID', 'term', ]] \
                .to_csv(path, sep='\t', index=False)

            print(f'Code: {code}. Saved shape: ', pd.read_csv(path, sep='\t')['EntryID'].drop_duplicates().shape)

    # -----------------------------------
    # TYPE FEATURES
    # -----------------------------------

    for code in TYPES + ['other']:
        efeats228 = uniprot228.query(f'~is_kaggle & is_pos & valid_terms & (type == "{code}")') \
            [['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
            .drop_duplicates(ignore_index=True)

        for dataset in ['train', 'old_train', 'test']:
            path = type_path / f'{dataset}_{code}.tsv'

            efeats228.query(f'is_{dataset}')[['EntryID', 'term', ]] \
                .to_csv(path, sep='\t', index=False)

            print(f'Code: {code}. Saved shape: ', pd.read_csv(path, sep='\t')['EntryID'].drop_duplicates().shape)
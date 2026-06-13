import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml


def process_uniprot(uniprot, ):
    uniprot['is_train'] = uniprot['EntryID'].isin(train['EntryID'])
    uniprot['is_old_train'] = uniprot['EntryID'].isin(old_train['EntryID'])
    uniprot['is_test'] = uniprot['EntryID'].isin(test['EntryID'])

    uniprot['is_kaggle'] = uniprot['source'].isin(set('IDA IMP TAS IPI IEP IGI IC EXP HTP HDA HMP HGI HEP'.split()))
    uniprot['is_pos'] = ~uniprot['type'].str.startswith('NOT')
    uniprot['valid_terms'] = uniprot['term'].isin(aspect)

    return uniprot

if __name__ == '__main__':

    from protlib.cafa_utils import obo_parser, Graph

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    uniprot_path = data_path / 'uniprot'
    graph_path = data_path / 'cafa-6-protein-function-prediction/Train/go-basic.obo'

    labels_path = data_path / 'uniprot/raw/labels'
    features_path = data_path / 'uniprot/raw/features'

    test_path = data_path / 'uniprot/raw/test'
    test_ext_path = data_path / 'uniprot/raw/test_ext'

    for path in [labels_path, features_path, test_path, test_ext_path]:
        path.mkdir(exist_ok=True, parents=True)

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

    uniprot226 = pd.read_parquet(uniprot_path / 'raw_uniprot226.parquet')
    uniprot226 = process_uniprot(uniprot226, )

    uniprot228 = pd.read_parquet(uniprot_path / 'raw_uniprot228.parquet')
    uniprot228 = process_uniprot(uniprot228, )

    # -----------------------------------
    # LABELS
    # -----------------------------------

    # 226 sample is used for:
    # 1) reconcile train. Checked: it exactly matches train
    # 2) make labeling for old_train
    # 3) form stacking time based validation
    labels226 = uniprot226.query('is_kaggle & is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)
    labels226['aspect'] = labels226['term'].map(aspect)

    # 228 sample is used for:
    # 1) form stacking time based validation
    # 2) submit directly to LB
    labels228 = uniprot228.query('is_kaggle & is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)
    labels228['aspect'] = labels228['term'].map(aspect)

    # check 226 occurance for 228
    labels228_unseen = pd.merge(
        labels228, labels226[['EntryID', 'term']].assign(is_seen_term=1),
        on=['EntryID', 'term'], how='left'
    ).query('is_seen_term.isnull()').drop('is_seen_term', axis=1)

    labels228_unseen['is_seen_prot'] = labels228_unseen['EntryID'].isin(labels226['EntryID'])
    labels228_unseen = pd.merge(
        labels228_unseen, labels226[['EntryID', 'aspect']].drop_duplicates().assign(is_seen_ont=1),
        on=['EntryID', 'aspect'], how='left'
    ).fillna({'is_seen_ont': 0}).astype({'is_seen_ont': bool})


    # -----------------------------------
    # VALIDATION: no knowledge
    # -----------------------------------

    labels228_unseen.query('~is_seen_prot')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_ext_path / 'no-know.tsv', sep='\t', index=False)

    labels228_unseen.query('~is_seen_prot & is_test')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_path / 'no-know.tsv', sep='\t', index=False)

    # -----------------------------------
    # VALIDATION: limited knowledge
    # -----------------------------------

    labels228_unseen.query('is_seen_prot & ~is_seen_ont')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_ext_path / 'lim-know.tsv', sep='\t', index=False)

    labels228_unseen.query('is_seen_prot & ~is_seen_ont & is_test')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_path / 'lim-know.tsv', sep='\t', index=False)

    # -----------------------------------
    # VALIDATION: partial knowledge
    # -----------------------------------

    labels228_unseen.query('is_seen_prot & is_seen_ont')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_ext_path / 'part-know.tsv', sep='\t', index=False)

    labels228_unseen.query('is_seen_prot & is_seen_ont & is_test')[['EntryID', 'term', 'aspect']] \
        .to_csv(test_path / 'part-know.tsv', sep='\t', index=False)

    # -----------------------------------
    # OLD TRAIN LABELS
    # -----------------------------------

    labels226.query('is_old_train')[['EntryID', 'term', 'aspect']] \
        .to_csv(labels_path / 'old_train_terms.tsv', sep='\t', index=False)

    # -----------------------------------
    # FEATURES: e-labels
    # -----------------------------------

    efeats226 = uniprot226.query('~is_kaggle & is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    efeats228 = uniprot228.query('~is_kaggle & is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    efeats226.query('is_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'train_auto.tsv', sep='\t', index=False)

    efeats226.query('is_old_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'old_train_auto.tsv', sep='\t', index=False)

    efeats226.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'test_auto226.tsv', sep='\t', index=False)

    efeats228.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'test_auto228.tsv', sep='\t', index=False)

    efeats_union = pd.concat([
        efeats226.query('is_test'),
        efeats228.query('is_test')
    ]).groupby(['EntryID', 'term', ]).size().rename('cnt').reset_index()

    efeats_union.drop('cnt', axis=1).to_csv(features_path / 'test_auto_union.tsv', sep='\t', index=False)

    # -----------------------------------
    # FEATURES: NOT e-labels
    # -----------------------------------

    not_efeats226 = uniprot226.query('~is_kaggle & ~is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    not_efeats228 = uniprot228.query('~is_kaggle & ~is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    not_efeats226.query('is_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_train_auto.tsv', sep='\t', index=False)

    not_efeats226.query('is_old_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_old_train_auto.tsv', sep='\t', index=False)

    not_efeats226.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_test_auto226.tsv', sep='\t', index=False)

    not_efeats228.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_test_auto228.tsv', sep='\t', index=False)

    not_efeats_union = pd.concat([
        not_efeats226.query('is_test'),
        not_efeats228.query('is_test')
    ]).groupby(['EntryID', 'term', ]).size().rename('cnt').reset_index()

    not_efeats_union.drop('cnt', axis=1).to_csv(features_path / 'not_test_auto_union.tsv', sep='\t', index=False)

    # -----------------------------------
    # FEATURES: NOT correct labels
    # -----------------------------------

    not_labels226 = uniprot226.query('is_kaggle & ~is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    not_labels228 = uniprot228.query('is_kaggle & ~is_pos & valid_terms')[
        ['EntryID', 'term', 'is_train', 'is_old_train', 'is_test']] \
        .drop_duplicates(ignore_index=True)

    not_labels226.query('is_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_train_terms.tsv', sep='\t', index=False)

    not_labels226.query('is_old_train')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_old_train_terms.tsv', sep='\t', index=False)

    not_labels226.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_test_terms226.tsv', sep='\t', index=False)

    not_labels228.query('is_test')[['EntryID', 'term', ]] \
        .to_csv(features_path / 'not_test_terms228.tsv', sep='\t', index=False)

    not_labels_union = pd.concat([
        not_labels226.query('is_test'),
        not_labels228.query('is_test')
    ]).groupby(['EntryID', 'term', ]).size().rename('cnt').reset_index()

    not_labels_union.drop('cnt', axis=1).to_csv(features_path / 'not_test_terms_union.tsv', sep='\t', index=False)




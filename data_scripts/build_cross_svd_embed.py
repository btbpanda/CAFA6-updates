"""Build cross embedding: we build one hot -> SVD embedding on separate ontologies
The idea is to boost limited knowledge part: to predict BP we can use MF+CC info ...
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import polars as pl
import pandas as pd
import cupy as cp
import cuml
from protlib.metric import obo_parser, Graph

FREQ_CO = 5

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    cafa6_path = data_path / 'cafa-6-protein-function-prediction'
    graph_path = cafa6_path / 'Train/go-basic.obo'
    helpers_path = data_path / 'helpers'

    embed_path = Path(config['embed_path']).resolve()

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    train_terms = pd.read_csv(cafa6_path / 'train_terms.tsv', sep='\t')
    vc = train_terms['term'].value_counts()

    for G in ontologies:

        alias = ''.join(map(lambda x: x[0], G.namespace.split('_')))
        ont_embed_path = embed_path / f'svd512{alias}'
        ont_embed_path.mkdir(parents=True, exist_ok=True)

        # -----------------------------------
        # Fit ont embedding
        # -----------------------------------

        root = [x['id'] for x in G.terms_list if len(x['adj']) == 0]
        assert len(root) == 1
        root = root[0]

        all_terms = [x['id'] for x in G.terms_list]
        freq = vc[vc.index.isin(all_terms) & (vc >= FREQ_CO)]
        columns = freq.index.tolist()

        target = pl.read_parquet(
            list(
                helpers_path.glob(f'cafa6-sparse-labels/{G.namespace}/part_*.parquet')
            ),
            columns=columns
        ).filter(
            pl.col(root) == 1
        ).with_columns(
            pl.all().fill_null(0)
        ).to_numpy()

        svd = cuml.TruncatedSVD(n_components=512, algorithm='full')
        svd.fit(cp.asarray(target))

        # -----------------------------------
        # Predict on train
        # -----------------------------------

        target = pl.read_parquet(
            list(
                helpers_path.glob(f'cafa6-sparse-labels/{G.namespace}/part_*.parquet')
            ),
            columns=['EntryID'] + columns
        ).with_columns(
            pl.all().fill_null(0)
        )

        ids = target['EntryID'].to_numpy()
        np.save(ont_embed_path / 'train_ids.npy', ids)

        embed = svd.transform(
            target.drop('EntryID').to_numpy()
        )
        np.save(ont_embed_path / 'train_embeds.npy', embed)

        # -----------------------------------
        # Predict on old train
        # -----------------------------------

        target = pl.read_parquet(
            list(
                helpers_path.glob(f'uniprot-old-sparse-labels/{G.namespace}/part_*.parquet')
            ),
            columns=['EntryID'] + columns
        ).with_columns(
            pl.all().fill_null(0)
        )

        old_ids = target['EntryID'].to_numpy()
        np.save(ont_embed_path / 'old_train_ids.npy', old_ids)

        old_embed = svd.transform(
            target.drop('EntryID').to_numpy()
        )

        np.save(ont_embed_path / 'old_train_embeds.npy', old_embed)

        # -----------------------------------
        # Predict on test - just collect train + old train and fill with 0 the rest
        # -----------------------------------

        test = pl.from_pandas(
            pd.read_feather(
                helpers_path / 'fasta/test_seq.feather',
                columns=['EntryID']
            )
        )
        embed_names = [f'embed_{x}' for x in range(512)]

        embed_placeholder = test.with_columns(
            pl.lit(0, dtype=pl.Float32).alias(x) for x in embed_names
        ).filter(
            ~pl.col('EntryID').is_in(ids),
            ~pl.col('EntryID').is_in(old_ids),

        )
        embed_test = pl.concat([
            pl.from_numpy(
                np.concatenate([ids, old_ids]), schema=['EntryID']
            ),

            pl.from_numpy(
                np.concatenate([embed, old_embed]), schema=embed_names
            ),
        ], how='horizontal')

        embed_test = pl.concat([embed_placeholder, embed_test], )
        embed_test = test.join(embed_test, on='EntryID', how='left', validate='1:1', maintain_order='left')

        np.save(ont_embed_path / 'test_ids.npy', embed_test['EntryID'].to_numpy())
        np.save(ont_embed_path / 'test_embeds.npy', embed_test.select(embed_names).to_numpy())




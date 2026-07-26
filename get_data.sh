#!/bin/bash

# download contents
python ./data_scripts/download.py
# create train / old_train / test
python ./data_scripts/parse_fasta.py
# create goa 226/228 parquets
python ./data_scripts/uniprot_to_parquet.py
# create labels and features from goa
python ./data_scripts/parse_goa.py
# create extended features list from goa
python ./data_scripts/build_evidence_features.py

# Here we start GPU part - originally executed in Kaggle Kernels
# TODO: test this part on local Sergey's env before submit to kaggle!!!!

# propagate GOA annotations and build sparse cond multilabel targets (0/1/null labels)
python ./data_scripts/uniprot_prop_and_sparse.py
# propagate evidence and types features
python ./data_scripts/uniprot_codes_prop.py
# store features in stacker format (CPU)
python ./data_scripts/stacker_format_feats.py
# build SVD cross embedding (GPU, but possible to switch to CPU)
python ./data_scripts/build_cross_svd_embed.py

# articles
python ./articles/article_download.py \
    --seq-file "./helpers/fasta/old_train_seq.feather" \
    --ids-file "./old_train_ids.txt" \
    --output-mapping "./old_train_protein_to_pmid.csv" \
    --output-abstracts "./old_train_final_abstracts.csv"

python ./articles/article_download.py \
    --seq-file "./helpers/fasta/train_seq.feather" \
    --ids-file "./train_ids.txt" \
    --output-mapping "./train_protein_to_pmid.csv" \
    --output-abstracts "./train_final_abstracts.csv"

python ./articles/article_download.py \
    --seq-file "./helpers/fasta/test_seq.feather" \
    --ids-file "./test_ids.txt" \
    --output-mapping "./test_protein_to_pmid.csv" \
    --output-abstracts "./test_final_abstracts.csv"

python ./articles/article_prep.py \
    --train-abstracts train_final_abstracts.csv \
    --old-train-abstracts old_train_final_abstracts.csv \
    --test-abstracts test_final_abstracts.csv \
    --test-protein-ids test_ids.txt \
    --train-seq ./helpers/fasta/train_seq.feather \
    --old-train-seq ./helpers/fasta/old_train_seq.feather \
    --test-seq ./helpers/fasta/test_seq.feather \
    --output-dir ./processed_abstracts \
    --embeds-dir ./articlestfidf

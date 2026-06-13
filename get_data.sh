
# download contents
python ./data_scipts/download.py
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
# store features in stacker format
python ./data_scripts/stacker_format_feats.py
# build SVD cross embedding


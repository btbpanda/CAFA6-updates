
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
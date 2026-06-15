python article_download.py \
    --seq-file "../helpers/fasta/old_train_seq.feather" \
    --ids-file "./old_train_ids.txt" \
    --output-mapping "./old_train_protein_to_pmid.csv" \
    --output-abstracts "./old_train_final_abstracts.csv" \
    --email "admin@university.edu"

python article_download.py \
    --seq-file "../helpers/fasta/train_seq.feather" \
    --ids-file "./train_ids.txt" \
    --output-mapping "./train_protein_to_pmid.csv" \
    --output-abstracts "./train_final_abstracts.csv" \
    --email "admin@university.edu"

python article_download.py \
    --seq-file "../helpers/fasta/test_seq.feather" \
    --ids-file "./test_ids.txt" \
    --output-mapping "./test_protein_to_pmid.csv" \
    --output-abstracts "./test_final_abstracts.csv" \
    --email "admin@university.edu"

python article_prep.py \
    --train-abstracts train_final_abstracts.csv \
    --old-train-abstracts old_train_final_abstracts.csv \
    --test-abstracts test_final_abstracts.csv \
    --test-protein-ids test_uniprot_ids.txt \
    --train-seq ../helpers/fasta/train_seq.feather \
    --old-train-seq ../helpers/fasta/old_train_seq.feather \
    --test-seq ../helpers/fasta/test_seq.feather \
    --output-dir ../processed_abstracts \
    --embeds-dir ../embeds/articlestfidf

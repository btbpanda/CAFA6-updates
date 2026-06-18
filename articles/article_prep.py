import os
import argparse
import numpy as np
import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import yaml
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def parse_arguments():
    """Parses command line arguments for file paths and parameters."""
    parser = argparse.ArgumentParser(description="Process article abstracts and generate TF-IDF embeddings.")

    # Input file paths
    parser.add_argument("--train-abstracts", type=str, required=True, help="Path to train abstracts CSV")
    parser.add_argument("--old-train-abstracts", type=str, required=True, help="Path to old train abstracts CSV")
    parser.add_argument("--test-abstracts", type=str, required=True, help="Path to test abstracts CSV")
    parser.add_argument("--test-protein-ids", type=str, required=True, help="Path to test protein IDs text file")

    # Sequence file paths
    parser.add_argument("--train-seq", type=str, required=True, help="Path to train sequences Feather file")
    parser.add_argument("--old-train-seq", type=str, required=True, help="Path to old train sequences Feather file")
    parser.add_argument("--test-seq", type=str, required=True, help="Path to test sequences Feather file")

    # Output directories
    parser.add_argument("--output-dir", type=str, default="processed_abstracts", help="Directory for stats and models")
    parser.add_argument("--embeds-dir", type=str, default="./embeds/articlestfidf", help="Directory for embeddings")

    # TF-IDF Parameters
    parser.add_argument("--tfidf-max-features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--tfidf-min-df", type=float, default=3, help="Min document frequency for TF-IDF")
    parser.add_argument("--tfidf-max-df", type=float, default=0.9, help="Max document frequency for TF-IDF")
    parser.add_argument("--tfidf-ngram-range", type=int, nargs=2, default=[1, 2], help="N-gram range (min max)")

    return parser.parse_args()


def load_data(data_path, args, ):
    """Loads all input datasets using Polars."""
    print("\n[1/4] Loading data...")

    dtypes = {
        "ProteinID": pl.Utf8,
        "PMID": pl.Utf8,
        "Title": pl.Utf8,
        "Abstract": pl.Utf8
    }

    data_path = Path(data_path)

    print("  Loading train abstracts...")
    train_df = pl.read_csv(data_path / args.train_abstracts, dtypes=dtypes)

    print("  Loading old train abstracts...")
    old_train_df = pl.read_csv(data_path / args.old_train_abstracts, dtypes=dtypes)

    print("  Loading test abstracts...")
    test_df = pl.read_csv(data_path / args.test_abstracts, dtypes=dtypes)

    print("  Loading test protein IDs...")
    with open(data_path / args.test_protein_ids, 'r') as f:
        test_protein_ids = set(line.strip() for line in f)

    print(f"\nLoading Statistics:")
    print(f"  Train abstracts: {len(train_df):,} records")
    print(f"  Old train abstracts: {len(old_train_df):,} records")
    print(f"  Test abstracts: {len(test_df):,} records")
    print(f"  Test protein IDs: {len(test_protein_ids):,}")

    return train_df, old_train_df, test_df, test_protein_ids


def preprocess_data(train_df, old_train_df, test_df):
    """Handles missing values, combines text, and removes duplicates."""
    print("\n  Processing missing values...")
    for df in [train_df, old_train_df, test_df]:
        df = df.with_columns([
            pl.col("Title").fill_null(""),
            pl.col("Abstract").fill_null("")
        ])

    print("  Combining Title and Abstract...")
    for df in [train_df, old_train_df, test_df]:
        df = df.with_columns(
            (pl.col("Title") + " " + pl.col("Abstract")).alias("full_text")
        )

    print("  Removing duplicates based on PMID...")
    train_df = train_df.unique(subset=["PMID"])
    old_train_df = old_train_df.unique(subset=["PMID"])
    test_df = test_df.unique(subset=["PMID"])

    print(f"\nAfter processing:")
    print(f"  Train abstracts: {len(train_df):,}")
    print(f"  Old train abstracts: {len(old_train_df):,}")
    print(f"  Test abstracts: {len(test_df):,}")

    return train_df, old_train_df, test_df


def calculate_statistics(train_df, old_train_df, test_df, test_protein_ids, stats_path):
    """Calculates and saves statistics about abstract counts per protein."""
    print("\n[2/4] Calculating statistics...")

    def get_counts(df):
        return df.group_by("ProteinID").agg(pl.count().alias("abstract_count"))

    train_counts = get_counts(train_df)
    old_train_counts = get_counts(old_train_df)
    test_counts = get_counts(test_df)

    test_proteins_with_abstracts = set(test_counts["ProteinID"])
    test_proteins_without_abstracts = test_protein_ids - test_proteins_with_abstracts

    def get_distribution(counts_df):
        return Counter(counts_df["abstract_count"].to_list())

    train_dist = get_distribution(train_counts)
    old_train_dist = get_distribution(old_train_counts)
    test_dist = get_distribution(test_counts)
    test_dist[0] = len(test_proteins_without_abstracts)

    # Write stats to file
    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ABSTRACT COUNT STATISTICS PER PROTEIN\n")
        f.write("=" * 70 + "\n\n")

        def write_section(name, counts_df, dist, total_proteins):
            f.write(f"{name}:\n")
            f.write("-" * 70 + "\n")
            counts_list = counts_df["abstract_count"].to_list() if len(counts_df) > 0 else [0]
            
            f.write(f"Total proteins: {len(counts_df):,}\n")
            if counts_list:
                f.write(f"Mean abstracts: {np.mean(counts_list):.2f}\n")
                f.write(f"Median: {np.median(counts_list):.0f}\n")
                f.write(f"Min: {np.min(counts_list)}, Max: {np.max(counts_list)}\n\n")
            
            f.write("Distribution:\n")
            for count in sorted(dist.keys())[:20]:
                freq = dist[count]
                pct = (freq / total_proteins * 100) if total_proteins > 0 else 0
                f.write(f"  {count:3d} abstracts: {freq:6d} proteins ({pct:5.2f}%)\n")
            
            if len(dist) > 20:
                f.write(f"  ... (showing first 20 of {len(dist)} values)\n")
            f.write("\n")

        write_section("TRAIN SET", train_counts, train_dist, len(train_counts))
        write_section("OLD TRAIN SET", old_train_counts, old_train_dist, len(old_train_counts))
        
        f.write("=" * 70 + "\n\n")
        f.write("TEST SET:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total proteins: {len(test_protein_ids):,}\n")
        f.write(f"Proteins with abstracts: {len(test_proteins_with_abstracts):,}\n")
        f.write(f"Proteins without abstracts: {len(test_proteins_without_abstracts):,}\n")
        
        test_counts_list = test_counts["abstract_count"].to_list() if len(test_counts) > 0 else []
        if test_counts_list:
            f.write(f"Mean abstracts (with abstracts): {np.mean(test_counts_list):.2f}\n")
            f.write(f"Median: {np.median(test_counts_list):.0f}\n")
            f.write(f"Min: {np.min(test_counts_list)}, Max: {np.max(test_counts_list)}\n\n")
        
        f.write("Distribution:\n")
        for count in sorted(test_dist.keys())[:20]:
            freq = test_dist[count]
            pct = (freq / len(test_protein_ids) * 100) if len(test_protein_ids) > 0 else 0
            f.write(f"  {count:3d} abstracts: {freq:6d} proteins ({pct:5.2f}%)\n")
        
        if len(test_dist) > 20:
            f.write(f"  ... (showing first 20 of {len(test_dist)} values)\n")

    print(f"  Statistics saved to: {stats_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS:")
    print("=" * 70)
    
    train_list = train_counts["abstract_count"].to_list() if len(train_counts) > 0 else [0]
    old_train_list = old_train_counts["abstract_count"].to_list() if len(old_train_counts) > 0 else [0]
    
    print(f"\nTRAIN:")
    print(f"  Proteins: {len(train_counts):,}")
    print(f"  Mean abstracts: {np.mean(train_list):.2f}")
    
    print(f"\nOLD TRAIN:")
    print(f"  Proteins: {len(old_train_counts):,}")
    print(f"  Mean abstracts: {np.mean(old_train_list):.2f}")

    print(f"\nTEST:")
    total_test = len(test_protein_ids)
    with_abs = len(test_proteins_with_abstracts)
    print(f"  Total proteins: {total_test:,}")
    print(f"  With abstracts: {with_abs:,} ({with_abs/total_test*100:.1f}%)")
    print(f"  Without abstracts: {len(test_proteins_without_abstracts):,}")

    return train_counts, old_train_counts, test_counts, test_dist


def train_tfidf(train_df, old_train_df, test_df, args, model_path, vocab_path):
    """Trains the TF-IDF vectorizer and saves the model."""
    print("\n[3/4] Training TF-IDF...")

    all_texts = (
        train_df["full_text"].to_list() +
        old_train_df["full_text"].to_list() +
        test_df["full_text"].to_list()
    )

    print(f"  Total documents for training: {len(all_texts):,}")

    print("  Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
        max_df=args.tfidf_max_df,
        ngram_range=tuple(args.tfidf_ngram_range),
        stop_words='english',
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True
    )

    tfidf.fit(all_texts)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")

    print("  Saving TF-IDF model...")
    with open(model_path, 'wb') as f:
        pickle.dump(tfidf, f)

    with open(vocab_path, 'wb') as f:
        pickle.dump(tfidf.vocabulary_, f)

    print(f"  Model saved to: {model_path}")
    print(f"  Vocabulary saved to: {vocab_path}")

    # Show top words by IDF
    print("\n  Top 20 words by IDF (most informative):")
    feature_names = tfidf.get_feature_names_out()
    idf_scores = tfidf.idf_
    sorted_indices = np.argsort(idf_scores)[::-1]

    for i in range(min(20, len(feature_names))):
        idx = sorted_indices[i]
        print(f"    {i+1:2d}. {feature_names[idx]:30s} (IDF: {idf_scores[idx]:.3f})")

    return tfidf


def create_embeddings(seq_path, protein_to_texts, tfidf, max_features, entry_id_col="EntryID"):
    """Generates TF-IDF embeddings aligned with sequence file order."""
    seq_df = pd.read_feather(seq_path)
    embeddings = []

    print(f"  Processing {os.path.basename(seq_path)}...")
    
    for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="    Generating embeddings"):
        entry_id = row[entry_id_col]

        if entry_id in protein_to_texts and protein_to_texts[entry_id]:
            texts = protein_to_texts[entry_id]
            tfidf_matrix = tfidf.transform(texts)
            # Summing vectors for all articles related to the protein
            combined_embedding = np.array(tfidf_matrix.sum(axis=0)).flatten()
            embeddings.append(combined_embedding)
        else:
            zero_embedding = np.zeros(max_features, dtype=np.float32)
            embeddings.append(zero_embedding)

    return np.vstack(embeddings), len(seq_df)


def generate_final_embeddings(data_path, args, tfidf, train_df, old_train_df, test_df):
    """Generates and saves final embedding files."""
    print("\n[4/4] Generating TF-IDF embeddings aligned with sequences...")

    def map_protein_to_texts(df):
        mapping = {}
        for row in df.iter_rows(named=True):
            pid = row["ProteinID"]
            text = row["full_text"]
            if pid not in mapping:
                mapping[pid] = []
            mapping[pid].append(text)
        return mapping

    data_path = Path(data_path)

    print("  Creating protein-to-text mappings...")
    train_map = map_protein_to_texts(train_df)
    old_train_map = map_protein_to_texts(old_train_df)
    test_map = map_protein_to_texts(test_df)

    print(f"  Unique proteins in train map: {len(train_map)}")
    print(f"  Unique proteins in old_train map: {len(old_train_map)}")
    print(f"  Unique proteins in test map: {len(test_map)}")

    # Train
    train_embeds, train_count = create_embeddings(
        data_path / args.train_seq, train_map, tfidf, args.tfidf_max_features
    )
    train_path = data_path / args.embeds_dir / "train_embeds.npy"
    np.save(train_path, train_embeds)
    print(f"  Train embeddings shape: {train_embeds.shape} -> Saved to {train_path}")

    # Old Train
    old_train_embeds, old_train_count = create_embeddings(
        data_path / args.old_train_seq, old_train_map, tfidf, args.tfidf_max_features
    )
    old_train_path = data_path / args.embeds_dir / "old_train_embeds.npy"
    np.save(old_train_path, old_train_embeds)
    print(f"  Old train embeddings shape: {old_train_embeds.shape} -> Saved to {old_train_path}")

    # Test
    test_embeds, test_count = create_embeddings(
        data_path / args.test_seq, test_map, tfidf, args.tfidf_max_features
    )
    test_path = data_path / args.embeds_dir / "test_embeds.npy"
    np.save(test_path, test_embeds)
    print(f"  Test embeddings shape: {test_embeds.shape} -> Saved to {test_path}")

    return train_path, old_train_path, test_path


def get_file_size(path):
    """Returns human-readable file size."""
    size = Path(path).stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.2f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.2f} MB"
    else:
        return f"{size/1024**3:.2f} GB"


def main():
    args = parse_arguments()

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    output_dir = data_path / args.output_dir
    embeds_dir = data_path / args.embeds_dir

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    stats_path = output_dir / "abstract_statistics.txt"
    tfidf_model_path = output_dir / "tfidf_model.pkl"
    tfidf_vocab_path = output_dir / "tfidf_vocab.pkl"

    print("=" * 70)
    print("Processing Article Abstracts")
    print("=" * 70)

    # 1. Load Data
    train_df, old_train_df, test_df, test_protein_ids = load_data(data_path, args)

    # 2. Preprocess
    train_df, old_train_df, test_df = preprocess_data(train_df, old_train_df, test_df)

    # 3. Statistics
    calculate_statistics(train_df, old_train_df, test_df, test_protein_ids, stats_path)

    # 4. Train TF-IDF
    tfidf = train_tfidf(train_df, old_train_df, test_df, args, tfidf_model_path, tfidf_vocab_path)

    # 5. Generate Embeddings
    train_path, old_train_path, test_path = generate_final_embeddings(data_path, args, tfidf, train_df, old_train_df, test_df)

    # Final Report
    print("\n" + "=" * 70)
    print("FILE SIZES:")
    print("=" * 70)
    print(f"  TF-IDF model: {get_file_size(tfidf_model_path)}")
    print(f"  Train embeddings: {get_file_size(train_path)}")
    print(f"  Old Train embeddings: {get_file_size(old_train_path)}")
    print(f"  Test embeddings: {get_file_size(test_path)}")

    print("\n" + "=" * 70)
    print("SUCCESS! ALL TASKS COMPLETED.")
    print("=" * 70)

    print("\nGENERATED FILES:")
    print(f"  1. TF-IDF Model: {tfidf_model_path}")
    print(f"  2. TF-IDF Vocabulary: {tfidf_vocab_path}")
    print(f"  3. Statistics: {stats_path}")
    print(f"  4. Train Embeddings: {train_path}")
    print(f"  5. Old Train Embeddings: {old_train_path}")
    print(f"  6. Test Embeddings: {test_path}")

    print("\nUSAGE IN MODEL:")
    print("""
# Load embeddings
train_tfidf = np.load('path/to/train_embeds.npy')
old_train_tfidf = np.load('path/to/old_train_embeds.npy')
test_tfidf = np.load('path/to/test_embeds.npy')

# Alignment:
# train_tfidf[i] corresponds to train_seq.feather.iloc[i]['EntryID']
# Dimensions: (num_proteins, TFIDF_MAX_FEATURES)
""")

    print("\nRECOMMENDATIONS:")
    print("  - Each row contains summed TF-IDF vectors of all articles for a protein.")
    print("  - Rows with no articles contain zero vectors.")
    print("  - Order strictly matches the input Feather sequence files.")


if __name__ == "__main__":
    main()


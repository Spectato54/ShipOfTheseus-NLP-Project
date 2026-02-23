"""
Builds the T0->T3 paraphrase chain per document and saves the split datasets to disk.
"""
import pandas as pd
from src.utils.config import DATA_PROCESSED, STAGES


def build_chains(df):
    """
    Build paraphrase chains for each document in the dataset; preprocess_dataset() should be called before this function.
    Returns a new DataFrame with columns: source, key, paraphraser, family, T0, T1, T2, T3.
    """

    df = df.copy()

    # Separate T0 from paraphrased versions since T0 is the original text and shared across all chains
    t0 = (
        df[df["stage"] == "T0"][["key", "source", "text"]]
        .drop_duplicates(subset=["key", "source"])
        .rename(columns={"text": "T0"})
    )

    chains = []
    paraphrasers = [p for p in df["paraphraser"].unique() if p != "original"]

    for paraphraser in paraphrasers:
        subset = df[df["paraphraser"] == paraphraser]

        # Grab family for this paraphraser — same value for all rows in subset
        family = subset["family"].iloc[0]

        for stage in ["T1", "T2", "T3"]:
            stage_df = (
                subset[subset["stage"] == stage][["key", "source", "text"]]
                .drop_duplicates(subset=["key", "source"])
                .rename(columns={"text": stage})
            )
            if stage == "T1":
                merged = stage_df
            else:
                merged = merged.merge(stage_df, on=["key", "source"], how="outer")

        merged["paraphraser"] = paraphraser
        merged["family"] = family
        chains.append(merged)

    paraphrased = pd.concat(chains, ignore_index=True)

    result = paraphrased.merge(t0, on=["key", "source"], how="left")
    result = result[["key", "source", "paraphraser", "family", "T0", "T1", "T2", "T3"]]
    return result.reset_index(drop=True)


def save_chain(chains_df, dataset_name):
    """
    Save the chains DataFrame to a CSV file in the processed data directory.
    """
    output_path = DATA_PROCESSED / f"{dataset_name}_chains.csv"
    chains_df.to_csv(output_path, index=False)
    print(f"Saved chains for dataset '{dataset_name}' to: {output_path}")


def build_and_save_chains(datasets):
    """
    Build and save paraphrase chains for all datasets in the corpus.
    """
    chains = {}
    for name, df in datasets.items():
        print(f"Building chains for dataset '{name}'...")
        chains_df = build_chains(df)
        save_chain(chains_df, name)
        chains[name] = chains_df
        print(
            f"{len(chains_df):,} chains built "
            f"({chains_df['paraphraser'].nunique()} paraphrasers, "
            f"{chains_df['key'].nunique()} documents)\n"
        )
    return chains


def display_paraphrase_chain(df, dataset, key, source, paraphraser):
    """
    Display the paraphrase chain for a specific document (identified by key) and paraphraser.
	"""
    print(f"Document ID  : {key}")
    print(f"Data Set     : {dataset}")
    print(f"Source Set   : {source}")
    print(f"Paraphraser  : {paraphraser}")
    print("=" * 80)
    row = df[
		(df["key"] == key)
		& (df["source"] == source)
		& (df["paraphraser"] == paraphraser)
	]
    for t in STAGES:
        if row.empty:
            print(f"\n[{t}] — NOT FOUND")
            continue

        text = row.iloc[0][f"{t}"]
        word_count = len(text.split())
        print(f"\n[{t}] ({word_count} words):")
        print(f" {text}")

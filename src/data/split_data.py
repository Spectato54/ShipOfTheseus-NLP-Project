"""
Builds the T0->T3 paraphrase chain per document and saves the split datasets to disk.
"""

from src.utils.config import DATA_PROCESSED, STAGES


def build_chains(df):
    """
    Build paraphrase chains for each document in the dataset; preprocess_dataset() should be called before this function.
    Returns a new DataFrame with columns: source, key, paraphraser, family, T0, T1, T2, T3.
    """
    # Verify preprocess has been run
    for col in ["cleaned_text", "no_stop_text"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Run preprocess_dataset() before build_chains()."
            )

    # Separate T0 from paraphrased versions since T0 is the original text and shared across all chains
    t0_df = df[df["stage"] == "T0"][
        ["source", "key", "text", "cleaned_text", "no_stop_text"]
    ].rename(
        columns={
            "text": "T0_text",
            "cleaned_text": "T0_cleaned",
            "no_stop_text": "T0_no_stop",
        }
    )

    # Pivot the paraphrased versions (T1-T3) to wide format
    paraphrase_df = (
        df[df["stage"] != "T0"]
        .pivot_table(
            index=["source", "key", "paraphraser", "family"],
            columns="stage",
            values=["text", "cleaned_text", "no_stop_text"],
            aggfunc="first"
        )
        .reset_index()
    )

    # Flatten MultiIndex columns: ('text', 'T1') -> 'T1_text'
    paraphrase_df.columns = [
		f"{stage}_{col}" if stage else col
		for col, stage in paraphrase_df.columns
	]

    paraphrase_df.columns.name = None  # Remove the columns name from pivot

    rename_map = {}
    for stage in ["T1", "T2", "T3"]:
        rename_map[f"{stage}_cleaned_text"] = f"{stage}_cleaned"
        rename_map[f"{stage}_no_stop_text"] = f"{stage}_no_stop"
    paraphrase_df = paraphrase_df.rename(columns=rename_map)
    
    # Merge T0 with the paraphrased versions to build the full chains
    merged_df = paraphrase_df.merge(t0_df, on=["source", "key"], how="left")

    # Reorder columns for clarity
    order_cols = (["source", "key", "paraphraser", "family"] +
				  [f"{stage}_{suffix}" for stage in ["T0", "T1", "T2", "T3"] for suffix in ["text", "cleaned", "no_stop"]])
    
    chain_df = merged_df[[c for c in order_cols if c in merged_df.columns]]
    return chain_df


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

        text = row.iloc[0][f"{t}_text"]
        word_count = len(text.split())
        print(f"\n[{t}_text] ({word_count} words):")
        print(f" {text}")
        

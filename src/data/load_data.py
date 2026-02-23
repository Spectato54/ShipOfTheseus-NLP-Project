"""
Loads the Ship of Theseus paraphrased CSVs from the
data/raw/ directory and returns them as a dict of DataFrames.

Expected CSV columns: source, key, text, version_name
"""

import pandas as pd
from src.utils.config import (
    DATASETS,
    EXPECTED_COLUMNS,
    DATA_RAW,
    WORKING_DATASETS,
    find_family,
    ver_map,
    ver_parse,
)

def load_dataset(name):
    """
    Load a single dataset CSV by name ('sci_gen', 'wp', or 'xsum').
    Returns a DataFrame with an added 'stage' column (T0-T3).
    """

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")

    path = DATA_RAW / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)

    # Validate columns
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset '{name}' is missing columns: {missing}")

    # Add stage column
    df["stage"] = df["version_name"].apply(ver_map)
    df["paraphraser"] = df["version_name"].apply(ver_parse)
    df["family"] = df["version_name"].apply(find_family)
    print(f"Data Loaded for dataset '{name}'!")
    print(f"Dataset Shape of {name}: {df.shape}\n")
    return df

def load_all():
    """
    Load entire corpus.
    Returns a dict of DataFrames, keyed by dataset name.
    """
    datasets = {}
    for name in DATASETS.keys():
        df = load_dataset(name)
        datasets[name] = df

    return datasets

def load_working():
    """
    Load only the working datasets (Default: sci_gen, wp, xsum; can configure in src/utils/config.py).
    Returns a dict of DataFrames, keyed by dataset name.
    """
    datasets = {}
    for name in WORKING_DATASETS:
        df = load_dataset(name)
        datasets[name] = df

    return datasets

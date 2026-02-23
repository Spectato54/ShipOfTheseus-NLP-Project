"""
Configureation for Ship of Theseus Project
Paths, Constants, Paraphrase Helper Methods
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"

# All 7 datasets from Ship of Theseus Corpus
DATASETS = {
    "sci_gen": "sci_gen_paraphrased.csv",
    "wp": "wp_paraphrased.csv",
    "xsum": "xsum_paraphrased.csv",
    "eli5": "eli5_paraphrased.csv",
    "cmv": "cmv_paraphrased.csv",
    "tldr": "tldr_paraphrased.csv",
    "yelp": "yelp_paraphrased.csv",
}

# Subset of datasets to work with for now (we can add more later)
WORKING_DATASETS = ["sci_gen", "wp", "xsum"]

# Expected CSV columns
EXPECTED_COLUMNS = {"source", "key", "text", "version_name"}

# Sources
SOURCES = ["Eleuther-AI", "LLAMA", "Tsinghua", "BigScience", "Human", "PaLM", "OpenAI"]

# Paraphrasers
PARAPHRASERS = [
    "pegasus(full)",
    "pegasus(slight)",
    "dipper(high)",
    "palm",
    "dipper",
    "dipper(low)",
    "chatgpt",
]

# Stages
STAGES = ["T0", "T1", "T2", "T3"]

# Paraphraser Families (for analysis)
FAMILIES = {
    "pegasus": ["pegasus(full)", "pegasus(slight)"],
    "dipper": ["dipper(high)", "dipper", "dipper(low)"],
    "palm": ["palm"],
    "chatgpt": ["chatgpt"],
}


# Version Helper Methods
def ver_parse(ver):
    """Return the base paraphraser name (first segment before any underscore)."""
    if ver == "original":
        return ver
    v = ver.split("_")
    return v[0]


def ver_map(ver):
    """
    Map a version_name string to a stage label T0-T3.

    Logic: underscore count = number of paraphrase iterations applied.
        'original'               -> T0
        'chatgpt'                -> T1  (0 underscores)
        'chatgpt_chatgpt'        -> T2  (1 underscore)
        'chatgpt_chatgpt_chatgpt'-> T3  (2 underscores)
    """
    if ver == "original":
        return "T0"
        # Count underscores
    under_count = ver.count("_")

    if under_count == 0:
        return "T1"
    elif under_count == 1:
        return "T2"
    else:
        return "T3"


def find_family(ver):
    """Find the paraphraser family for a given version_name."""
    ver = ver_parse(ver)
    if ver == "original":
        return "none"
    for family, members in FAMILIES.items():
        if ver in members:
            return family
    return "unknown"

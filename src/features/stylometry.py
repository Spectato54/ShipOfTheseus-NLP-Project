"""
Stylometric Feature Extraction for Ship of Theseus Project.

Extracts 5+ linguistic features per text using spaCy:
  - Sentence length (mean, variance)
  - Type-Token Ratio (lexical diversity)
  - Punctuation ratio
  - POS tag distribution
  - Dependency tree depth (mean, max)
"""

import string
import numpy as np
import pandas as pd
import spacy
from collections import Counter
from scipy.spatial.distance import jensenshannon

# Load spaCy model once at import time
_NLP = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


# --- Core POS tags we track (Universal Dependencies) ---
TRACKED_POS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "CONJ", "CCONJ", "SCONJ", "NUM", "PART",
    "INTJ", "PUNCT", "SYM", "X",
]


def _tree_depth(token):
    """Compute depth of a token in the dependency tree (iterative)."""
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
    return depth


def extract_stylometric_features(text):
    """
    Extract stylometric features from a single text.

    Returns a dict with:
        sent_len_mean, sent_len_var  : sentence length stats (in words)
        ttr                          : Type-Token Ratio
        punct_ratio                  : fraction of chars that are punctuation
        dep_depth_mean, dep_depth_max: dependency tree depth stats
        pos_<TAG>                    : normalized POS tag frequencies
    """
    nlp = _get_nlp()

    if not isinstance(text, str) or len(text.strip()) == 0:
        features = {
            "sent_len_mean": 0.0, "sent_len_var": 0.0,
            "ttr": 0.0, "punct_ratio": 0.0,
            "dep_depth_mean": 0.0, "dep_depth_max": 0.0,
        }
        for pos in TRACKED_POS:
            features[f"pos_{pos}"] = 0.0
        return features

    # Truncate very long texts to first 5000 chars for speed
    doc = nlp(text[:5000])

    # --- Sentence length ---
    sent_lengths = [len([t for t in s if not t.is_punct]) for s in doc.sents]
    if len(sent_lengths) == 0:
        sent_lengths = [0]
    sent_len_mean = np.mean(sent_lengths)
    sent_len_var = np.var(sent_lengths)

    # --- Type-Token Ratio ---
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if len(tokens) > 0:
        ttr = len(set(tokens)) / len(tokens)
    else:
        ttr = 0.0

    # --- Punctuation ratio ---
    total_chars = len(text[:5000])
    punct_chars = sum(1 for c in text[:5000] if c in string.punctuation)
    punct_ratio = punct_chars / total_chars if total_chars > 0 else 0.0

    # --- POS tag distribution ---
    pos_counts = Counter(t.pos_ for t in doc)
    total_tokens = sum(pos_counts.values())
    pos_features = {}
    for pos in TRACKED_POS:
        pos_features[f"pos_{pos}"] = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0.0

    # --- Dependency depth ---
    depths = [_tree_depth(t) for t in doc]
    dep_depth_mean = np.mean(depths) if depths else 0.0
    dep_depth_max = max(depths) if depths else 0.0

    features = {
        "sent_len_mean": round(sent_len_mean, 4),
        "sent_len_var": round(sent_len_var, 4),
        "ttr": round(ttr, 4),
        "punct_ratio": round(punct_ratio, 4),
        "dep_depth_mean": round(dep_depth_mean, 4),
        "dep_depth_max": round(dep_depth_max, 4),
    }
    features.update(pos_features)
    return features


def extract_features_batch(texts, show_progress=True):
    """
    Extract stylometric features for a list of texts.

    Args:
        texts: list of strings
        show_progress: print progress every 100 texts

    Returns:
        pd.DataFrame with one row per text, columns = feature names
    """
    results = []
    n = len(texts)
    for i, text in enumerate(texts):
        results.append(extract_stylometric_features(text))
        if show_progress and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n} texts")
    if show_progress:
        print(f"  Done: {n}/{n} texts")
    return pd.DataFrame(results)


def compute_pos_divergence(pos_dist_a, pos_dist_b):
    """
    Compute Jensen-Shannon divergence between two POS distributions.

    Args:
        pos_dist_a, pos_dist_b: dicts mapping POS tag -> frequency (normalized)

    Returns:
        float: JS divergence (0 = identical, 1 = maximally different)
    """
    vec_a = np.array([pos_dist_a.get(f"pos_{p}", 0.0) for p in TRACKED_POS])
    vec_b = np.array([pos_dist_b.get(f"pos_{p}", 0.0) for p in TRACKED_POS])

    # Ensure they sum to 1 (add epsilon to avoid division by zero)
    eps = 1e-10
    vec_a = vec_a + eps
    vec_b = vec_b + eps
    vec_a = vec_a / vec_a.sum()
    vec_b = vec_b / vec_b.sum()

    return float(jensenshannon(vec_a, vec_b))


def aggregate_features_by_stage(chains_df, stages=("T0", "T1", "T2", "T3"),
                                sample_n=500, seed=42):
    """
    Extract and aggregate stylometric features across stages.

    Args:
        chains_df: DataFrame with columns T0, T1, T2, T3 (text)
        stages: which stages to process
        sample_n: rows to sample (None = all)
        seed: random seed

    Returns:
        dict: {stage: pd.DataFrame of features}
        Also returns the sampled chains_df for reference.
    """
    if sample_n and len(chains_df) > sample_n:
        chains_df = chains_df.sample(sample_n, random_state=seed)

    stage_features = {}
    for stage in stages:
        print(f"\nExtracting features for {stage}...")
        texts = chains_df[stage].dropna().tolist()
        stage_features[stage] = extract_features_batch(texts)

    return stage_features, chains_df

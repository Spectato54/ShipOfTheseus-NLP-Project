"""
Named Entity Recognition (NER) Analysis for Ship of Theseus Project.

Extracts named entities from text and computes entity retention metrics
(Jaccard, recall, precision) to measure content erasure across paraphrase
iterations.
"""

import numpy as np
import pandas as pd
import spacy

# Lazy-load spaCy model with only NER enabled (fast)
_NLP_NER = None


def _get_nlp():
    """Lazy-load spaCy NER-only pipeline."""
    global _NLP_NER
    if _NLP_NER is None:
        _NLP_NER = spacy.load(
            "en_core_web_sm",
            disable=["tagger", "parser", "lemmatizer"],
        )
    return _NLP_NER


def extract_entities(text):
    """
    Extract a set of named entities from text.

    Returns:
        set of lowercased, stripped entity strings
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    nlp = _get_nlp()
    doc = nlp(text[:5000])  # truncate for speed, matching stylometry.py
    return {ent.text.lower().strip() for ent in doc.ents}


def extract_entities_typed(text):
    """
    Extract named entities with their labels.

    Returns:
        set of (entity_text, entity_label) tuples
    """
    if not isinstance(text, str) or not text.strip():
        return set()
    nlp = _get_nlp()
    doc = nlp(text[:5000])
    return {(ent.text.lower().strip(), ent.label_) for ent in doc.ents}


def entity_retention_metrics(ref_entities, hyp_entities):
    """
    Compute entity retention metrics between reference and hypothesis entity sets.

    Args:
        ref_entities: set of entities from T0 (reference)
        hyp_entities: set of entities from Tn (hypothesis)

    Returns:
        dict with:
            jaccard   : |intersection| / |union|
            recall    : |intersection| / |ref| (entity preservation)
            precision : |intersection| / |hyp| (1 - hallucination rate)
            n_ref     : number of reference entities
            n_hyp     : number of hypothesis entities
            n_shared  : number of shared entities
    """
    if len(ref_entities) == 0 and len(hyp_entities) == 0:
        return {
            "jaccard": 1.0, "recall": 1.0, "precision": 1.0,
            "n_ref": 0, "n_hyp": 0, "n_shared": 0,
        }
    if len(ref_entities) == 0:
        return {
            "jaccard": 0.0, "recall": np.nan, "precision": 0.0,
            "n_ref": 0, "n_hyp": len(hyp_entities), "n_shared": 0,
        }
    if len(hyp_entities) == 0:
        return {
            "jaccard": 0.0, "recall": 0.0, "precision": np.nan,
            "n_ref": len(ref_entities), "n_hyp": 0, "n_shared": 0,
        }

    shared = ref_entities & hyp_entities
    union = ref_entities | hyp_entities

    return {
        "jaccard": len(shared) / len(union),
        "recall": len(shared) / len(ref_entities),
        "precision": len(shared) / len(hyp_entities),
        "n_ref": len(ref_entities),
        "n_hyp": len(hyp_entities),
        "n_shared": len(shared),
    }


def extract_ner_for_chains(chains_df, stages=("T0", "T1", "T2", "T3"),
                           show_progress=True, batch_size=500):
    """
    Extract entity sets for each stage in a chains DataFrame.
    Uses nlp.pipe() for batched processing (much faster than one-by-one).

    Args:
        chains_df: DataFrame with columns T0, T1, T2, T3 (text)
        stages: stage columns to process
        show_progress: print progress updates
        batch_size: spaCy pipe batch size

    Returns:
        chains_df with new columns: ner_T0, ner_T1, ner_T2, ner_T3
    """
    nlp = _get_nlp()
    df = chains_df.copy()
    n = len(df)
    for stage in stages:
        if show_progress:
            print(f"  Extracting NER for {stage} ({n} texts)...")
        # Prepare texts: replace NaN/non-string with empty, truncate
        texts = [
            str(t)[:5000] if isinstance(t, str) and t.strip() else ""
            for t in df[stage]
        ]
        entities = []
        for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
            entities.append({ent.text.lower().strip() for ent in doc.ents})
            if show_progress and (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{n}")
        df[f"ner_{stage}"] = entities
        if show_progress:
            print(f"    Done: {n}/{n}")
    return df


def compute_retention_for_chains(chains_df, stages=("T1", "T2", "T3"),
                                 show_progress=True):
    """
    Compute entity retention metrics for each row, comparing each stage to T0.

    Args:
        chains_df: DataFrame with ner_T0, ner_T1, ner_T2, ner_T3 columns
        stages: stages to compare against T0
        show_progress: print progress updates

    Returns:
        chains_df with new columns: jaccard_T1, recall_T1, precision_T1, etc.
    """
    df = chains_df.copy()
    ref_list = df["ner_T0"].tolist()
    for stage in stages:
        if show_progress:
            print(f"  Computing retention metrics for {stage}...")
        hyp_list = df[f"ner_{stage}"].tolist()
        metrics_list = [
            entity_retention_metrics(ref, hyp)
            for ref, hyp in zip(ref_list, hyp_list)
        ]
        metrics_df = pd.DataFrame(metrics_list)
        for col in metrics_df.columns:
            df[f"{col}_{stage}"] = metrics_df[col].values
    return df

"""
SBERT Cosine Similarity for Ship of Theseus Project.

Computes sentence-level cosine similarity using Sentence-BERT embeddings.
This is complementary to BERTScore (token-level matching):
- BERTScore: token-level precision/recall/F1
- SBERT cosine: whole-sentence embedding similarity

Used for RQ1 to measure semantic preservation across paraphrase iterations.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_sbert_similarity(references, hypotheses, model_name="all-MiniLM-L6-v2",
                              batch_size=64):
    """
    Compute cosine similarity between reference and hypothesis texts
    using SBERT embeddings.

    Args:
        references: list of reference texts (T0)
        hypotheses: list of hypothesis texts (T1/T2/T3)
        model_name: sentence-transformers model name
        batch_size: encoding batch size

    Returns:
        np.array of cosine similarity scores (one per pair)
    """
    model = SentenceTransformer(model_name)

    ref_embeddings = model.encode(references, batch_size=batch_size, show_progress_bar=True)
    hyp_embeddings = model.encode(hypotheses, batch_size=batch_size, show_progress_bar=True)

    # Pairwise cosine similarity (diagonal = matched pairs)
    similarities = np.array([
        cosine_similarity([r], [h])[0][0]
        for r, h in zip(ref_embeddings, hyp_embeddings)
    ])

    return similarities


def mean_sbert_by_stage(chains_df, stages=("T1", "T2", "T3"),
                         model_name="all-MiniLM-L6-v2", batch_size=64):
    """
    Compute mean SBERT cosine similarity for each stage vs T0.

    Args:
        chains_df: DataFrame with T0, T1, T2, T3 text columns
        stages: stages to evaluate against T0
        model_name: SBERT model name
        batch_size: encoding batch size

    Returns:
        dict of {stage: mean_cosine_similarity}
    """
    model = SentenceTransformer(model_name)

    # Get valid rows (T0 not null)
    valid = chains_df[chains_df["T0"].notna()].copy()
    t0_texts = valid["T0"].tolist()

    print(f"Encoding T0 ({len(t0_texts)} texts)...")
    t0_embeddings = model.encode(t0_texts, batch_size=batch_size, show_progress_bar=True)

    results = {"T0": 1.0}  # T0 vs T0 = 1.0

    for stage in stages:
        stage_texts = valid[stage].fillna("").tolist()
        print(f"Encoding {stage} ({len(stage_texts)} texts)...")
        stage_embeddings = model.encode(stage_texts, batch_size=batch_size,
                                         show_progress_bar=True)

        # Pairwise cosine similarity
        sims = np.array([
            cosine_similarity([r], [h])[0][0]
            for r, h in zip(t0_embeddings, stage_embeddings)
        ])

        results[stage] = float(np.mean(sims))
        print(f"  {stage} mean SBERT cosine: {results[stage]:.4f}")

    return results

"""
Computes BLEU and ROUGE scores between T0 and T1/T2/T3 for each chain row.

BLEU and ROUGE are metrics for evaluating the similarity of two sentences.
BLEU computes precision by comparing n-grams in the candidate sentence with those in the reference sentence.
ROUGE computes recall by comparing n-grams in the candidate sentence with those in the reference sentence.
Both metrics are widely used in tasks such as machine translation, text summarization, and natural language generation.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from src.utils.config import STAGES

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def compute_bleu_batch(reference, hypothesis):
    """Sentence-level BLEU-2 with smoothing (unigram + bigram)"""
    results = []
    for ref, hyp in zip(reference, hypothesis):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        smoother = SmoothingFunction().method1
        results.append(
            corpus_bleu(
                [[ref_tokens]],
                [hyp_tokens],
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoother,
            )
        )
    return results


rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge_batch(reference, hypothesis):
    """ROUGE F1 score for a given metric ('rouge1', 'rouge2', 'rougeL')"""
    r1, r2, rL = [], [], []
    for ref, hyp in zip(reference, hypothesis):
        scores = rouge.score(ref, hyp)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return {"rouge1": r1, "rouge2": r2, "rougeL": rL}


def compute_bleu_rouge(chains_df, sample_n=None, seed=64):
    """
    Compute BLEU-2, ROUGE-1, ROUGE-2, and ROUGE-L between T0_text and
    each of T1_text, T2_text, T3_text for every row in chains_df.

    Returns a new DataFrame with added columns:
            bleu_T1, bleu_T2, bleu_T3
            rouge1_T1, rouge1_T2, rouge1_T3
            rouge2_T1, rouge2_T2, rouge2_T3
            rougeL_T1, rougeL_T2, rougeL_T3
    """
    df = chains_df.copy()
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).copy()
        print(
            f"Sampled {sample_n} rows for BLEU/ROUGE computation from {len(chains_df)}."
        )

    for stage in STAGES:
        if stage == "T0":
            continue  # Skip T0 since it's the reference
        t0_col = "T0_text"
        tn_col = f"{stage}_text"

        # Skip rows where either text is missing
        valid = df[t0_col].notna() & df[tn_col].notna()
        refs = df.loc[valid, t0_col].tolist()
        hyps = df.loc[valid, tn_col].tolist()

        if not refs:
            print(f"  No valid rows for {stage}, skipping.")
            continue
        
        # ROUGE
        rouge_scores = compute_rouge_batch(refs, hyps)
        df.loc[valid, f"rouge1_{stage}"] = rouge_scores["rouge1"]
        df.loc[valid, f"rouge2_{stage}"] = rouge_scores["rouge2"]
        df.loc[valid, f"rougeL_{stage}"] = rouge_scores["rougeL"]

        # BLEU
        bleu_scores = compute_bleu_batch(refs, hyps)
        df.loc[valid, f"bleu_{stage}"] = bleu_scores

        print(
            f"  {stage}: BLEU={np.mean(bleu_scores):.4f}  "
            f"ROUGE-L={np.mean(rouge_scores['rougeL']):.4f}  "
            f"({valid.sum()} rows)"
        )

    return df


def mean_scores_by_stage(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize mean scores per stage into a DataFrame.
    """
    rows = []
    for stage in STAGES:
        if stage == "T0":
            continue  # Skip T0 since it's the reference
        
        for metric, label in [
            ("bleu",   "BLEU-2"),
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougeL", "ROUGE-L"),
        ]:
            col = f"{metric}_{stage}"
            if col in scored_df.columns:
                rows.append({
                    "stage":  stage,
                    "metric": label,
                    "score":  scored_df[col].mean(skipna=True),
                })
    return pd.DataFrame(rows)

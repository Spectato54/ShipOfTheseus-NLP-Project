"""
Computes BERTScore F1 between T0 and T1/T2/T3 for each chain row.

BERTScore captures SEMANTIC similarity; it should decay more slowly
than BLEU/ROUGE, showing that meaning is preserved longer than style.
"""

import pandas as pd
import torch
from src.utils.config import STAGES
from bert_score import score as bert_score_fn

def compute_bertscore(chains_df, batch_size=64, lang="en"):
    """
    Compute BERTScore F1 between T0_text and each of T1/T2/T3.
    """

    df = chains_df.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BERTScore device: {device}")
    for stage in STAGES:

        if stage == 'T0':
            continue  # Skip T0 since it's the reference
        t0_col = "T0"
        tn_col = f"{stage}"
        valid_mask = df[t0_col].notna() & df[tn_col].notna()
        refs = df.loc[valid_mask, t0_col].tolist()
        hyps = df.loc[valid_mask, tn_col].tolist()

        if not hyps:
            print(f"No valid rows for {stage}, skipping.")
            continue

        _, _, F1 = bert_score_fn(
			hyps, refs,
			lang=lang,
			batch_size=batch_size,
			device=device,
			verbose=False
		)

        df.loc[valid_mask, f"bertscore_{stage}"] = F1.numpy()

    return df

def mean_bertscore_by_stage(scored_df):
	"""
	Summarize mean BERTScore F1 per stage into a DataFrame.
	"""
	rows = []
	for stage in STAGES:
		if stage == 'T0':
			continue  # Skip T0 since it's the reference
		col = f"bertscore_{stage}"
		if col in scored_df.columns:
			rows.append(
				{
					"stage": stage,
					"metric": "BERTSCORE-F1",
					"score": scored_df[col].mean(skipna=True),
				}
			)
	return pd.DataFrame(rows)

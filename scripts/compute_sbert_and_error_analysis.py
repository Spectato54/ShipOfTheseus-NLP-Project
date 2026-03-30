"""
Script to compute:
1. SBERT cosine similarity across T0-T3 (for RQ1)
2. RQ2 error analysis (misclassification direction, per-domain breakdown)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import DATA_PROCESSED, DATASETS, STAGES

SEED = 42
SAMPLE_N = 500
EXP_DIR = ROOT / "experiments"

# ═══════════════════════════════════════════════════════════════════
# PART 1: SBERT Cosine Similarity
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: SBERT Cosine Similarity")
print("=" * 70)

from src.similarity.sbert import mean_sbert_by_stage

all_sbert_results = {}
sbert_per_dataset = {}

# Load and sample chains
chains = {}
for name in DATASETS.keys():
    path = DATA_PROCESSED / f"{name}_chains.csv"
    df = pd.read_csv(path)
    chains[name] = df.sample(min(SAMPLE_N, len(df)), random_state=SEED)

combined = pd.concat(chains.values(), ignore_index=True)

# Compute SBERT for combined dataset
print("\n--- Computing SBERT for ALL datasets combined ---")
sbert_combined = mean_sbert_by_stage(combined, model_name="all-MiniLM-L6-v2", batch_size=128)
print(f"\nCombined SBERT results: {sbert_combined}")

# Also compute per-dataset
for name, df in chains.items():
    print(f"\n--- Computing SBERT for {name} ---")
    sbert_per_dataset[name] = mean_sbert_by_stage(df, model_name="all-MiniLM-L6-v2", batch_size=128)

# Save results
sbert_df = pd.DataFrame([
    {"dataset": "all_combined", **sbert_combined}
] + [
    {"dataset": name, **scores} for name, scores in sbert_per_dataset.items()
])
sbert_out = EXP_DIR / "baseline_similarity" / "sbert_cosine_results.csv"
sbert_df.to_csv(sbert_out, index=False)
print(f"\nSaved SBERT results to: {sbert_out}")
print(sbert_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# PART 2: RQ2 Error Analysis
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: RQ2 Error Analysis")
print("=" * 70)

from src.models.attribution import train_attribution, evaluate_by_stage, balance_human_ai
from sklearn.metrics import confusion_matrix, classification_report

# Load all 7 datasets with dataset name preserved
chains_with_dataset = []
for name in DATASETS.keys():
    path = DATA_PROCESSED / f"{name}_chains.csv"
    df = pd.read_csv(path).sample(min(SAMPLE_N, len(df)), random_state=SEED)
    df["dataset"] = name
    chains_with_dataset.append(df)

combined_with_ds = pd.concat(chains_with_dataset, ignore_index=True)

# Train balanced attribution model
print("\nTraining balanced attribution model...")
model_attr, vectorizer, balanced_df = train_attribution(
    combined_with_ds, balanced=True, seed=SEED
)

# Add dataset column back after balancing
# (balance_human_ai preserves index, so dataset column should be there)
print(f"Balanced dataset columns: {balanced_df.columns.tolist()}")

# --- Error Analysis at T3 ---
print("\n--- Error Analysis at T3 (worst case) ---")
texts_t3 = balanced_df["T3"].fillna("").tolist()
X_t3 = vectorizer.transform(texts_t3)
y_true = np.array(["Human" if s == "Human" else "AI" for s in balanced_df["source"]])
y_pred = model_attr.predict(X_t3)

# Misclassification direction
correct = y_true == y_pred
wrong = ~correct

print(f"\nT3 total: {len(y_true)}, correct: {correct.sum()}, wrong: {wrong.sum()}")

# Direction of misclassification
human_as_ai = ((y_true == "Human") & (y_pred == "AI")).sum()
ai_as_human = ((y_true == "AI") & (y_pred == "Human")).sum()
n_human = (y_true == "Human").sum()
n_ai = (y_true == "AI").sum()

print(f"\nMisclassification Direction at T3:")
print(f"  Human → AI (false positive): {human_as_ai}/{n_human} = {human_as_ai/n_human*100:.1f}%")
print(f"  AI → Human (false negative): {ai_as_human}/{n_ai} = {ai_as_human/n_ai*100:.1f}%")

# Per-domain breakdown
if "dataset" in balanced_df.columns:
    print("\n--- Per-Domain Error Rate at T3 ---")
    domain_errors = []
    for ds_name in sorted(balanced_df["dataset"].unique()):
        mask = balanced_df["dataset"] == ds_name
        ds_true = y_true[mask]
        ds_pred = y_pred[mask]
        ds_acc = (ds_true == ds_pred).mean()
        ds_err = 1 - ds_acc
        ds_n = mask.sum()
        domain_errors.append({
            "dataset": ds_name,
            "n_samples": ds_n,
            "accuracy": round(ds_acc, 4),
            "error_rate": round(ds_err, 4),
        })
        print(f"  {ds_name}: acc={ds_acc:.4f}, err={ds_err:.4f} (n={ds_n})")

    domain_df = pd.DataFrame(domain_errors).sort_values("error_rate", ascending=False)

# Per-stage error direction
print("\n--- Misclassification Direction by Stage ---")
error_directions = []
for stage in STAGES:
    texts = balanced_df[stage].fillna("").tolist()
    X = vectorizer.transform(texts)
    yp = model_attr.predict(X)

    h_as_ai = ((y_true == "Human") & (yp == "AI")).sum()
    ai_as_h = ((y_true == "AI") & (yp == "Human")).sum()

    error_directions.append({
        "stage": stage,
        "human_misclassified_as_AI": h_as_ai,
        "human_misclass_rate": round(h_as_ai / n_human, 4) if n_human > 0 else 0,
        "AI_misclassified_as_human": ai_as_h,
        "AI_misclass_rate": round(ai_as_h / n_ai, 4) if n_ai > 0 else 0,
    })

error_dir_df = pd.DataFrame(error_directions)
print(error_dir_df.to_string(index=False))

# Per-family error at T3
print("\n--- Per-Family Error Rate at T3 ---")
if "family" in balanced_df.columns:
    for fam in sorted(balanced_df["family"].unique()):
        if fam == "none":
            continue
        mask = balanced_df["family"] == fam
        fam_true = y_true[mask]
        fam_pred = y_pred[mask]
        if len(fam_true) > 0:
            fam_acc = (fam_true == fam_pred).mean()
            print(f"  {fam}: acc={fam_acc:.4f} (n={mask.sum()})")

# Save error analysis
error_out = EXP_DIR / "identity_forensics" / "error_analysis_t3.csv"
error_dir_df.to_csv(error_out, index=False)

if "dataset" in balanced_df.columns:
    domain_out = EXP_DIR / "identity_forensics" / "error_analysis_by_domain.csv"
    domain_df.to_csv(domain_out, index=False)
    print(f"\nSaved domain error analysis to: {domain_out}")

print(f"Saved error direction analysis to: {error_out}")
print("\nDone!")

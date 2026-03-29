"""
Authorship Attribution Model for Ship of Theseus Project.

RQ2: "Point of No Return" — At what iteration does the text
lose its original authorial identity?

Approach: Train on T0 (Human vs AI sources), test on T1/T2/T3.
Track accuracy/F1 degradation across iterations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


def build_tfidf_vectorizer(texts, max_features=5000):
    """Fit a TF-IDF vectorizer on training texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(texts)
    return vectorizer


def balance_human_ai(chains_df, label_col="source", seed=42):
    """
    Balance Human vs AI samples by downsampling the majority class.

    Returns:
        balanced DataFrame with equal Human and AI rows
    """
    df = chains_df.copy()
    df["_binary"] = df[label_col].apply(lambda x: "Human" if x == "Human" else "AI")

    human_df = df[df["_binary"] == "Human"]
    ai_df = df[df["_binary"] == "AI"]

    n_human = len(human_df)
    n_ai = len(ai_df)
    n_min = min(n_human, n_ai)

    print(f"  Before balancing: Human={n_human}, AI={n_ai}")
    print(f"  After balancing:  Human={n_min}, AI={n_min}")

    human_sampled = human_df.sample(n_min, random_state=seed)
    ai_sampled = ai_df.sample(n_min, random_state=seed)

    balanced = pd.concat([human_sampled, ai_sampled]).drop(columns=["_binary"])
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def train_attribution(chains_df, vectorizer=None, label_col="source",
                      max_features=5000, balanced=True, seed=42):
    """
    Train authorship attribution on T0 texts.

    Args:
        chains_df: DataFrame with T0 text and source/label column
        vectorizer: pre-fitted TfidfVectorizer (or None to create one)
        label_col: column to use as label (default: 'source')
        max_features: TF-IDF vocab size
        balanced: if True, balance Human vs AI samples before training
        seed: random seed for balanced sampling

    Returns:
        model, vectorizer, balanced_df (the balanced dataset for evaluation)
    """
    if balanced:
        print("Balancing Human vs AI samples...")
        chains_df = balance_human_ai(chains_df, label_col, seed)

    texts = chains_df["T0"].dropna().tolist()
    labels = chains_df.loc[chains_df["T0"].notna(), label_col].tolist()

    # Binary: Human vs AI
    binary_labels = ["Human" if l == "Human" else "AI" for l in labels]

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(texts, max_features)

    X = vectorizer.transform(texts)
    y = np.array(binary_labels)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)

    # Report training accuracy
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy (T0): {train_acc:.4f}")
    print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    return model, vectorizer, chains_df


def evaluate_by_stage(model, vectorizer, chains_df, label_col="source",
                      stages=("T0", "T1", "T2", "T3")):
    """
    Evaluate attribution model on each paraphrase stage.

    Args:
        model: trained classifier
        vectorizer: fitted TfidfVectorizer
        chains_df: DataFrame with T0-T3 text columns
        label_col: ground truth label column
        stages: stages to evaluate

    Returns:
        pd.DataFrame with columns: stage, accuracy, f1_macro, f1_weighted
    """
    results = []
    labels_all = chains_df[label_col].tolist()
    binary_labels = ["Human" if l == "Human" else "AI" for l in labels_all]
    y_true = np.array(binary_labels)

    for stage in stages:
        texts = chains_df[stage].fillna("").tolist()
        X = vectorizer.transform(texts)
        y_pred = model.predict(X)

        # Only evaluate on non-empty texts
        mask = chains_df[stage].notna()
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1_m = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        f1_w = f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0)

        results.append({
            "stage": stage,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_m, 4),
            "f1_weighted": round(f1_w, 4),
        })
        print(f"  {stage}: Acc={acc:.4f}  F1(macro)={f1_m:.4f}")

    return pd.DataFrame(results)

"""
Paraphraser Fingerprint Classifier for Ship of Theseus Project.

RQ3: Do different paraphrasing models leave distinct traces?
Can we identify which model (ChatGPT, Dipper, Pegasus, PaLM) was used?

Approach: Multi-class classification using stylometric + TF-IDF features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def train_fingerprint(X_train, y_train, n_estimators=200, random_state=42):
    """
    Train a multi-class paraphraser fingerprint classifier.

    Args:
        X_train: feature matrix (numpy array or sparse)
        y_train: paraphraser family labels
        n_estimators: number of trees

    Returns:
        trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Fingerprint training accuracy: {train_acc:.4f}")
    return model


def evaluate_fingerprint(model, X_test, y_test):
    """
    Evaluate fingerprint classifier.

    Returns:
        dict with accuracy, f1, classification_report, confusion_matrix
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}")
    print(f"\n{report}")

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "labels": sorted(set(y_test)),
        "report": report,
    }


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get top feature importances from RandomForest.

    Returns:
        pd.DataFrame with columns: feature, importance (sorted desc)
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return df.head(top_n).reset_index(drop=True)

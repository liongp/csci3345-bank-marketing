# Helper functions

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_random_split(path, include_duration=False):
    """Load the pipeline's already-preprocessed random split."""
    suffix = "with" if include_duration else "no"
    with np.load(path, allow_pickle=False) as data:
        return {
            split: (data[f"X_{split}_{suffix}"], data[f"y_{split}"])
            for split in ("train", "val", "test")
        }


def validate_splits(splits):
    """Check that the prepared splits can be compared with the same model."""
    n_features = None
    for name in ("train", "val", "test"):
        X, y = splits[name]
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(
                f"{name}: expected a 2D feature matrix and one label per row"
            )
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{name}: both classes (0 and 1) are required for ROC-AUC")
        if n_features is None:
            n_features = X.shape[1]
        elif X.shape[1] != n_features:
            raise ValueError(
                f"{name}: feature columns do not match the training matrix"
            )


def evaluate(model, X, y, threshold=0.5):
    """Return ranking and threshold-based metrics for the positive class."""
    positive_index = np.flatnonzero(model.classes_ == 1).item()
    scores = model.predict_proba(X)[:, positive_index]
    predictions = (scores >= threshold).astype(int)
    return {
        "n": len(y),
        "positive_rate": float(np.mean(y)),
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }

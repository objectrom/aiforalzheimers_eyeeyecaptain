from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Patient-level metrics.

    Args:
        y_true: shape [N] in {0,1}
        y_prob: shape [N] in [0,1] (probability of class 1)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUROC: handle edge cases where only one class appears
    auroc = None
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    sens = float(tp / max(tp + fn, 1))  # recall for AD (1)
    spec = float(tn / max(tn + fp, 1))  # recall for CO (0)

    return {
        "auroc": auroc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm.tolist(),
    }

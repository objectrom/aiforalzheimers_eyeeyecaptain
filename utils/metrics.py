import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def compute_metrics(logits, labels):
    probs = 1 / (1 + np.exp(-np.array(logits)))
    preds = (probs > 0.5).astype(int)

    return {
        "AUC": roc_auc_score(labels, probs),
        "Accuracy": accuracy_score(labels, preds)
    }

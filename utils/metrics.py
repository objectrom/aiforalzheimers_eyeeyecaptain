import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5, sweep: bool = True):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    def metrics_at(t):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc  = float((tp + tn) / max(tp + tn + fp + fn, 1))
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        return acc, sens, spec, cm

    auroc = None
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, y_prob))

    # default threshold metrics
    acc, sens, spec, cm = metrics_at(threshold)

    out = {
        "auroc": auroc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
    }

    if sweep and len(y_true) > 0:
        ths = np.linspace(0.01, 0.99, 99)
        best = {"t": 0.5, "j": -1e9, "acc": None, "sens": None, "spec": None, "cm": None}
        for t in ths:
            acc_t, sens_t, spec_t, cm_t = metrics_at(float(t))
            j = sens_t + spec_t - 1.0   # Youden's J
            if j > best["j"]:
                best = {"t": float(t), "j": float(j), "acc": acc_t, "sens": sens_t, "spec": spec_t, "cm": cm_t.tolist()}

        out.update({
            "best_threshold": best["t"],
            "best_youden_j": best["j"],
            "best_accuracy": best["acc"],
            "best_sensitivity": best["sens"],
            "best_specificity": best["spec"],
            "best_confusion_matrix": best["cm"],
        })

    return out

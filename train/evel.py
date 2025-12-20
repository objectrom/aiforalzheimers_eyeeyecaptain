import torch
from collections import defaultdict
from utils.aggregation import aggregate_patient_predictions
from utils.metrics import compute_metrics

def evaluate(model, loader, device):
    model.eval()
    model.to(device)

    patient_logits = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for x, y, patient in loader:
            x = x.to(device)
            logits = model(x).cpu().squeeze().tolist()

            if not isinstance(logits, list):
                logits = [logits]

            for logit in logits:
                patient_logits[patient[0]].append(logit)
                patient_labels[patient[0]] = y[0].item()

    preds, labels = [], []
    for p in patient_logits:
        preds.append(aggregate_patient_predictions(patient_logits[p]))
        labels.append(patient_labels[p])

    metrics = compute_metrics(preds, labels)
    return metrics

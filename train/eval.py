import torch
from collections import defaultdict
from utils.aggregation import aggregate_patient_predictions
from utils.metrics import compute_metrics

def evaluate(model, loader, device):
    device = torch.device(device)
    model.eval()
    model.to(device)

    patient_logits = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for x, y, patients in loader:
            x = x.to(device)
            y = y.float()

            logits = model(x).view(-1).cpu()

            for logit, label, patient_id in zip(logits, y, patients):
                patient_logits[patient_id].append(logit.item())
                patient_labels[patient_id] = label.item()

    preds, labels = [], []
    for patient_id in patient_logits:
        preds.append(
            aggregate_patient_predictions(patient_logits[patient_id])
        )
        labels.append(patient_labels[patient_id])

    metrics = compute_metrics(preds, labels)
    return metrics

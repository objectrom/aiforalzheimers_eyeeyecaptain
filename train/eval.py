from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_binary_metrics

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    patient_ids: List[str] = []
    attn: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch)
        prob = out.prob_ad.detach().cpu().numpy()
        lab = batch["label"].detach().cpu().numpy()
        y_true.extend(lab.tolist())
        y_prob.extend(prob.tolist())
        patient_ids.extend(list(batch["patient_id"]))
        attn.extend(out.attention_weights.detach().cpu().numpy())

    metrics = compute_binary_metrics(np.array(y_true), np.array(y_prob))
    return {
        "metrics": metrics,
        "patients": [
            {
                "patient_id": pid,
                "prediction": float(p),
                "label": int(y),
                "attention_weights": a.tolist(),
            }
            for pid, p, y, a in zip(patient_ids, y_prob, y_true, attn)
        ],
    }

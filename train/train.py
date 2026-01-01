from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from dataocteye.oct_dataset import OCTPatientDataset
from train.eval import evaluate

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_splits(dataset: OCTPatientDataset, num_folds: int, stratified: bool = True, seed: int = 42):
    patient_ids = [s.patient_id for s in dataset.samples]
    labels = np.array([s.label for s in dataset.samples], dtype=int)

    idxs = np.arange(len(dataset))
    if stratified:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(idxs, labels))
    else:
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(idxs))
    return folds

def train_one_fold(cfg: Dict, fold_index: int) -> Dict:
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset (patient-level)
    ds = OCTPatientDataset(
        root=cfg["data"]["root"],
        labels_csv=cfg["data"]["labels_csv"],
        prefer_flat_normed=bool(cfg["data"]["prefer_flat_normed"]),
        image_size=int(cfg["data"]["image_size"]),
        num_pairs=int(cfg["data"]["num_pairs"]),
        channels=int(cfg["data"]["channels"]),
        patient_prefix_map=cfg["data"].get("patient_prefix_map", {"CO": 0, "AD": 1}),
    )

    folds = make_splits(ds, int(cfg["split"]["num_folds"]), bool(cfg["split"]["stratified"]), int(cfg["seed"]))
    tr_idx, va_idx = folds[fold_index]

    train_set = Subset(ds, tr_idx.tolist())
    val_set = Subset(ds, va_idx.tolist())

    def collate(batch):
        # Keep patient_id as list of strings
        out = {
            "light": torch.stack([b["light"] for b in batch], dim=0),
            "dark":  torch.stack([b["dark"] for b in batch],  dim=0),
            "label": torch.stack([b["label"] for b in batch], dim=0),
            "patient_id": [b["patient_id"] for b in batch],
        }
        return out

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size_patients"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size_patients"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    # model
    from models.oct_classifier import FunctionalOCTClassifier
    model = FunctionalOCTClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
        pair_mlp_hidden=int(cfg["model"]["pair_mlp_hidden"]),
        attention_hidden=int(cfg["model"]["attention_hidden"]),
        num_pairs=int(cfg["data"]["num_pairs"]),
        num_classes=2,
    ).to(device)

    
    # Calculate class weights from training set
    train_labels = [ds.samples[i].label for i in tr_idx]
    class_counts = np.bincount(train_labels, minlength=2)
    total = len(train_labels)
    class_weights = torch.FloatTensor([1.0, 2.5]).to(device)
    
    print(f"Fold {fold_index} - Class distribution: CO={class_counts[0]}, AD={class_counts[1]}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"fold_{fold_index}")
    os.makedirs(run_dir, exist_ok=True)

    best_auroc = -1.0
    best_path = os.path.join(run_dir, "best.pt")

    history: List[Dict] = []
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        pbar = tqdm(train_loader, desc=f"train fold{fold_index} ep{epoch+1}")
        losses = []

        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                out = model(batch)
                loss = criterion(out.logits, batch["label"])

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses))

        # validation
        val = evaluate(model, val_loader, device)
        metrics = val["metrics"]
        auroc = metrics.get("auroc")
        auroc_val = float(auroc) if auroc is not None else -1.0

        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_metrics": metrics,
        }
        history.append(record)

        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # save best
        if cfg["logging"].get("save_best", True) and auroc_val > best_auroc:
            best_auroc = auroc_val
            torch.save({"model": model.state_dict(), "cfg": cfg, "fold": fold_index}, best_path)
            with open(os.path.join(run_dir, "best_eval.json"), "w", encoding="utf-8") as f:
                json.dump(val, f, indent=2)

    # final eval with best
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    final = evaluate(model, val_loader, device)
    with open(os.path.join(run_dir, "final_eval.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    return {
        "fold": fold_index,
        "best_auroc": best_auroc,
        "final_metrics": final["metrics"],
        "run_dir": run_dir,
    }

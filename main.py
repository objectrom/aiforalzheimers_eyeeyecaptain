from __future__ import annotations

import argparse
import os
import yaml
import json

from train.train import train_one_fold

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    fold_index = int(cfg["split"].get("fold_index", 0))
    num_folds = int(cfg["split"]["num_folds"])
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    results = []
    if fold_index == -1:
        for k in range(num_folds):
            results.append(train_one_fold(cfg, k))
    else:
        results.append(train_one_fold(cfg, fold_index))

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done. Summary written to", os.path.join(out_dir, "summary.json"))

if __name__ == "__main__":
    main()

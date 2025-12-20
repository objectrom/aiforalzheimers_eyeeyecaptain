import yaml
import torch
from torch.utils.data import DataLoader

from datasets.oct_dataset import RetinalOCTDataset
from models.oct_classifier import OCTClassifier
from train.train import train
from train.eval import evaluate   

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = RetinalOCTDataset(cfg["data_root"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],   
    )

    model = OCTClassifier()

    train(
        model=model,
        loader=loader,
        device=cfg["device"],
        epochs=cfg["epochs"],
        lr=cfg["learning_rate"]
    )

    metrics = evaluate(
        model=model,
        loader=loader,
        device=cfg["device"]
    )

    print("Evaluation results:", metrics)

if __name__ == "__main__":
    main()

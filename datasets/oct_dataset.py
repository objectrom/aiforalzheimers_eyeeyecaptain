import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RetinalOCTDataset(Dataset):
    """
    Loads OCT / fOCT scans stored as .npy files.
    Directory structure:
    data/raw/{AD,Control}/patient_xxx/scan_x.npy
    """

    def __init__(self, root_dir):
        self.samples = []

        for label, cls in enumerate(["Control", "AD"]):
            cls_path = os.path.join(root_dir, cls)
            for patient in os.listdir(cls_path):
                patient_path = os.path.join(cls_path, patient)
                for file in os.listdir(patient_path):
                    if file.endswith(".npy"):
                        self.samples.append({
                            "path": os.path.join(patient_path, file),
                            "label": label,
                            "patient": patient
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = np.load(item["path"])

        # normalization
        x = (x - x.mean()) / (x.std() + 1e-6)

        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, H, W)

        y = torch.tensor(item["label"], dtype=torch.float32)
        return x, y, item["patient"]

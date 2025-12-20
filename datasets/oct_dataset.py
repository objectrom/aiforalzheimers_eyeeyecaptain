import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RetinalOCTDataset(Dataset):
    """
    Functional OCT dataset using light ON/OFF reflectivity changes.
    Each .npy file is assumed to be (T, H, W).
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
        vol = np.load(item["path"])  

        assert vol.ndim == 3, "Expected (T, H, W) OCT volume"

        T = vol.shape[0]
        off = vol[: T // 2].mean(axis=0)
        on  = vol[T // 2 :].mean(axis=0)

        delta = on - off

        delta = (delta - delta.mean()) / (delta.std() + 1e-6)

        x = torch.tensor(delta, dtype=torch.float32).unsqueeze(0) 
        y = torch.tensor(item["label"], dtype=torch.float32)

        return x, y, item["patient"]

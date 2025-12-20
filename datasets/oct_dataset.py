import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

class RetinalOCTDataset(Dataset):
    """
    Functional OCT Dataset using LIGHT / DARK reflectivity volumes.
    Each patient folder contains:
        *_LIGHT.hdr + *.img
        *_DARK.hdr  + *.img
    """

    def __init__(self, root_dir):
        self.samples = []

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            light = None
            dark = None

            for f in os.listdir(patient_path):
                if "flat-normed" in f and "LIGHT" in f and f.endswith(".hdr"):
                    light = os.path.join(patient_path, f)
                if "flat-normed" in f and "DARK" in f and f.endswith(".hdr"):
                    dark = os.path.join(patient_path, f)

            if light and dark:
                label = 1 if patient.startswith("AD") else 0
                self.samples.append({
                    "light": light,
                    "dark": dark,
                    "label": label,
                    "patient": patient
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        light_vol = sitk.GetArrayFromImage(sitk.ReadImage(item["light"]))
        dark_vol  = sitk.GetArrayFromImage(sitk.ReadImage(item["dark"]))

        assert light_vol.shape == dark_vol.shape

        delta = light_vol - dark_vol  

        delta = delta.mean(axis=0)   

        delta = (delta - delta.mean()) / (delta.std() + 1e-6)

        x = torch.tensor(delta, dtype=torch.float32).unsqueeze(0)  
        y = torch.tensor(item["label"], dtype=torch.float32)

        return x, y, item["patient"]

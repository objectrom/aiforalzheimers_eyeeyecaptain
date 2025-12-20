import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

class RetinalOCTDataset(Dataset):
    """
    Functional OCT Dataset using flat-normed LIGHT/DARK volumes.

    Assumes directory structure:
        data/
          ├── AD057/
          ├── CO839/
          ├── YA211/
          └── ...

    Label rule:
        AD*        -> 1 (Alzheimer)
        CO*, YA*   -> 0 (Control)
    """

    def __init__(self, root_dir):
        self.samples = []

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            if patient.startswith("AD"):
                label = 1
            elif patient.startswith("CO") or patient.startswith("YA") or patient.startswith("AQ"):
                label = 0
            else:
                continue

            light, dark = None, None

            for f in os.listdir(patient_path):
                fname = f.lower()

                if (
                    "flat-normed" in fname
                    and "light" in fname
                    and fname.endswith(".hdr")
                ):
                    light = os.path.join(patient_path, f)

                if (
                    "flat-normed" in fname
                    and "dark" in fname
                    and fname.endswith(".hdr")
                ):
                    dark = os.path.join(patient_path, f)

            if light is not None and dark is not None:
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

        delta = (light_vol - dark_vol).astype("float32")   

        delta = delta.mean(axis=0)      

        delta = (delta - delta.mean()) / (delta.std() + 1e-6)

        x = torch.tensor(delta, dtype=torch.float32).unsqueeze(0) 
        y = torch.tensor(item["label"], dtype=torch.float32)

        return x, y, item["patient"]

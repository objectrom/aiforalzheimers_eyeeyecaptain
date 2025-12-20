import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset


class RetinalOCTDataset(Dataset):
    """
    Functional OCT Dataset using flat-normed LIGHT/DARK volumes.

    Directory:
        data/
          ├── AD057/
          ├── CO443/
          ├── YA604/
          └── ...

    Label:
        AD*        -> 1
        CO*, YA*, AQ* -> 0
    """

    def __init__(self, root_dir):
        self.samples = []

        for patient in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            if patient.startswith("AD"):
                label = 1
            elif patient.startswith(("CO", "YA", "AQ")):
                label = 0
            else:
                continue

            files = os.listdir(patient_path)

            light_files = [
                f for f in files
                if f.endswith(".hdr")
                and "flat-normed" in f
                and "LIGHT" in f
            ]

            dark_files = [
                f for f in files
                if f.endswith(".hdr")
                and "flat-normed" in f
                and "DARK" in f
            ]

            for lf in light_files:
                pid = patient

                df = next(
                    (d for d in dark_files if pid in d),
                    None
                )

                if df is None:
                    continue

                self.samples.append({
                    "light": os.path.join(patient_path, lf),
                    "dark": os.path.join(patient_path, df),
                    "label": label,
                    "patient": patient
                })

        print(f"[Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        light_vol = sitk.GetArrayFromImage(
            sitk.ReadImage(item["light"])
        )
        dark_vol = sitk.GetArrayFromImage(
            sitk.ReadImage(item["dark"])
        )

        T = min(light_vol.shape[0], dark_vol.shape[0])
        light_vol = light_vol[:T]
        dark_vol  = dark_vol[:T]

        delta = light_vol - dark_vol      
        delta = delta.mean(axis=0)      

        delta = (delta - delta.mean()) / (delta.std() + 1e-6)

        x = torch.tensor(delta, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(item["label"], dtype=torch.float32)

        return x, y, item["patient"]
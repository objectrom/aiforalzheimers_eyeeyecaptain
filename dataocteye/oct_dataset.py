from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _infer_hw_from_img_bytes(img_path: str, dtype=np.float32):
    nbytes = os.path.getsize(img_path)
    n = nbytes // np.dtype(dtype).itemsize

    # try common OCT widths (fast path)
    for w in (3208, 2048, 1536, 1024, 800, 768, 640, 512, 400, 384, 256):
        if n % w == 0:
            h = n // w
            return (h, w)

    # fallback: factor search near sqrt
    r = int(np.sqrt(n))
    for h in range(r, 0, -1):
        if n % h == 0:
            w = n // h
            return (h, w)

    raise ValueError(f"Cannot infer (H,W) from {img_path}, n={n}")

# -----------------------------
# HDR/IMG lightweight reader
# -----------------------------
_HDR_KEYVAL = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*=\s*(.+?)\s*$")

def _parse_hdr(path: str) -> Dict[str, str]:
    """Parse a simple key=value header.

    Many OCT pipelines output .hdr files with key-value metadata.
    This parser is permissive; it will keep unknown keys as strings.
    """
    meta: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _HDR_KEYVAL.match(line)
            if m:
                meta[m.group(1).lower()] = m.group(2)
    return meta

def _infer_shape_and_dtype(meta: Dict[str, str]) -> Tuple[Tuple[int, ...], np.dtype]:
    """Infer shape and dtype from header meta. Supports common conventions.

    Tries keys: dim, dims, size, sizes, datatype, type, pixeltype.
    Falls back to (H, W) = (meta['height'], meta['width']) if present.
    """
    # dims
    for k in ["dims", "dim", "size", "sizes"]:
        if k in meta:
            parts = re.split(r"[\s,;]+", meta[k].strip())
            ints = [int(p) for p in parts if p.strip().isdigit()]
            if len(ints) >= 2:
                # Use last two as (H, W) if more dims are present
                if len(ints) > 2:
                    shape = tuple(ints[-2:])
                else:
                    shape = tuple(ints)
                break
    else:
        if "height" in meta and "width" in meta:
            shape = (int(meta["height"]), int(meta["width"]))
        else:
            raise ValueError(f"Could not infer dims from header keys: {list(meta.keys())}")

    # dtype
    dtype_str = (meta.get("datatype") or meta.get("type") or meta.get("pixeltype") or "float32").lower()
    # very common mappings
    if "float" in dtype_str and "64" in dtype_str:
        dtype = np.float64
    elif "float" in dtype_str or "single" in dtype_str:
        dtype = np.float32
    elif "uint16" in dtype_str or ("unsigned" in dtype_str and "16" in dtype_str):
        dtype = np.uint16
    elif "int16" in dtype_str:
        dtype = np.int16
    elif "uint8" in dtype_str:
        dtype = np.uint8
    else:
        # default safe
        dtype = np.float32

    return tuple(shape), np.dtype(dtype)

def load_hdr_img(hdr_path: str, img_path: str) -> np.ndarray:
    meta = _parse_hdr(hdr_path)
    try:
        shape, dtype = _infer_shape_and_dtype(meta)
    except Exception:
        dtype = np.float32
        shape = (2048, 3208) 

    arr = np.fromfile(img_path, dtype=dtype)

    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size == expected:
            return arr.reshape(shape)

    shape2 = _infer_hw_from_img_bytes(img_path, dtype=dtype)
    expected2 = int(np.prod(shape2))
    if arr.size != expected2:
        # if it contains stacked slices, try [S,H,W]
        if arr.size % expected2 == 0:
            s = arr.size // expected2
            arr = arr.reshape((s, shape2[0], shape2[1]))[0]
            return arr
        raise ValueError(f"Unexpected data size: got {arr.size}, cannot match inferred shape {shape2}")

    return arr.reshape(shape2)

# -----------------------------
# Dataset (patient-level)
# -----------------------------

def default_label_from_id(patient_id: str, prefix_map: Dict[str, int]) -> int:
    for prefix, lab in prefix_map.items():
        if patient_id.startswith(prefix):
            return int(lab)
    raise ValueError(f"Cannot infer label for patient_id={patient_id}. Provide labels_csv or prefix_map.")

def _choose_files(files: List[str], prefer_flat_normed: bool) -> List[str]:
    """Pick preferred set of OCT files.

    Preference order when `prefer_flat_normed` is True:
      1) contains 'flat-normed'
      2) contains 'flat'
      3) anything else
    """
    if not files:
        return files
    if not prefer_flat_normed:
        return files

    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        if "flat-normed" in name or "_flat-normed_" in name:
            return 0
        if "flat" in name or "_flat_" in name:
            return 1
        return 2

    best = min(score(p) for p in files)
    return [p for p in files if score(p) == best]

@dataclass
class PatientSample:
    patient_id: str
    light_pairs: List[Tuple[str, str]]  # list of (hdr, img)
    dark_pairs: List[Tuple[str, str]]   # list of (hdr, img)
    label: int

class OCTPatientDataset(Dataset):
    """Returns one patient at a time, per contract:
    {
      'light': [5, C, H, W],
      'dark' : [5, C, H, W],
      'label': int,
      'patient_id': str
    }
    """
    def __init__(
        self,
        root: str,
        labels_csv: Optional[str],
        prefer_flat_normed: bool = True,
        image_size: int = 224,
        num_pairs: int = 5,
        channels: int = 3,
        patient_prefix_map: Optional[Dict[str, int]] = None,
        patient_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.root = root
        self.prefer_flat_normed = prefer_flat_normed
        self.image_size = int(image_size)
        self.num_pairs = int(num_pairs)
        self.channels = int(channels)
        self.patient_prefix_map = patient_prefix_map or {"CO": 0, "AD": 1}

        # labels
        self.labels = None
        if labels_csv is not None:
            df = pd.read_csv(labels_csv)
            if "patient_id" not in df.columns or "label" not in df.columns:
                raise ValueError("labels_csv must have columns: patient_id,label")
            self.labels = {str(pid): int(lab) for pid, lab in zip(df["patient_id"], df["label"])}

        # transforms (keep simple + stable)
        self.to_tensor = T.ToTensor()  # HWC or HW -> CHW in [0,1] if uint8; for float keep scale
        self.resize = T.Resize((self.image_size, self.image_size), antialias=True)

        # index patients
        all_patients = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if patient_ids is not None:
            allow = set(patient_ids)
            all_patients = [p for p in all_patients if p in allow]

        self.samples: List[PatientSample] = []
        for pid in all_patients:
            sample = self._build_patient(pid)
            if sample is not None:
                self.samples.append(sample)

        if not self.samples:
            raise RuntimeError(f"No patients found under {root}.")

    def _build_patient(self, patient_id: str) -> Optional[PatientSample]:
        pdir = os.path.join(self.root, patient_id)
        files = os.listdir(pdir)

        # match hdr/img pairs
        hdrs = [f for f in files if f.lower().endswith(".hdr")]
        imgs = {os.path.splitext(f)[0]: f for f in files if f.lower().endswith(".img")}

        light_hdrs = [os.path.join(pdir, f) for f in hdrs if "light" in f.lower()]
        dark_hdrs  = [os.path.join(pdir, f) for f in hdrs if "dark" in f.lower()]

        light_hdrs = _choose_files(light_hdrs, self.prefer_flat_normed)
        dark_hdrs  = _choose_files(dark_hdrs,  self.prefer_flat_normed)

        def pair(hdr_list: List[str]) -> List[Tuple[str, str]]:
            out = []
            for hdr in hdr_list:
                stem = os.path.splitext(os.path.basename(hdr))[0]
                if stem in imgs:
                    out.append((hdr, os.path.join(pdir, imgs[stem])))
            # stable ordering: try to sort by trailing index, else lexicographic
            def key(x):
                name = os.path.basename(x[0])
                m = re.search(r"(\d+)(?!.*\d)", name)
                return (int(m.group(1)) if m else 10**9, name)
            out.sort(key=key)
            return out

        light_pairs = pair(light_hdrs)
        dark_pairs  = pair(dark_hdrs)

        # We need exactly 5 each; if more, take first 5 (consistent)
        if len(light_pairs) < self.num_pairs or len(dark_pairs) < self.num_pairs:
            # skip silently (or you can raise)
            return None
        light_pairs = light_pairs[: self.num_pairs]
        dark_pairs  = dark_pairs[: self.num_pairs]

        label = self.labels.get(patient_id) if self.labels is not None else default_label_from_id(patient_id, self.patient_prefix_map)
        return PatientSample(patient_id=patient_id, light_pairs=light_pairs, dark_pairs=dark_pairs, label=label)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_one(self, hdr: str, img: str) -> torch.Tensor:
        arr = load_hdr_img(hdr, img)  # [H, W] (likely)
        # normalize per-image robustly to [0,1] using percentiles (stable for OCT)
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
        if hi > lo:
            a = (a - lo) / (hi - lo)
        a = np.clip(a, 0.0, 1.0)

        # ToTensor expects HxW or HxWxC
        t = torch.from_numpy(a).unsqueeze(0)  # [1,H,W]
        t = self.resize(t)
        if self.channels == 3:
            t = t.repeat(3, 1, 1)
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        light = torch.stack([self._load_one(h, i) for (h, i) in s.light_pairs], dim=0)  # [5,C,H,W]
        dark  = torch.stack([self._load_one(h, i) for (h, i) in s.dark_pairs],  dim=0)  # [5,C,H,W]

        return {
            "light": light,
            "dark": dark,
            "label": torch.tensor(int(s.label), dtype=torch.long),
            "patient_id": s.patient_id,
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils.aggregation import AttentionMIL

@dataclass
class ModelOutput:
    logits: torch.Tensor              # [B, 2]
    prob_ad: torch.Tensor             # [B]
    attention_weights: torch.Tensor   # [B, 5]

class FunctionalOCTClassifier(nn.Module):
    """Patient-level functional OCT classifier.

    Contract:
      Input:
        light: [B, 5, C, H, W]
        dark : [B, 5, C, H, W]
      Computations:
        Δ_i = D_i - L_i
        f(L), f(D), f(Δ) through shared encoder
        g_i = MLP([f(L), f(D), |f(D)-f(L)|, f(Δ)])
        MIL attention pooling over i=1..5
        Patient classifier -> logits
      Output:
        patient-level P(AD) and attention weights α_i
    """
    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        dropout: float = 0.2,
        pair_mlp_hidden: int = 512,
        attention_hidden: int = 256,
        num_pairs: int = 5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_pairs = num_pairs

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,        # feature extractor
            global_pool="avg",
        )
        feat_dim = getattr(self.encoder, "num_features", None)
        if feat_dim is None:
            # timm models usually expose num_features
            raise RuntimeError("Backbone does not expose num_features; pick a standard timm backbone.")

        self.pair_mlp = nn.Sequential(
            nn.Linear(feat_dim * 4, pair_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_mlp_hidden, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mil = AttentionMIL(dim=feat_dim, hidden=attention_hidden, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*K, C, H, W]
        return self.encoder(x)  # [B*K, D]

    def forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        light = batch["light"]  # [B,5,C,H,W]
        dark  = batch["dark"]   # [B,5,C,H,W]
        B, K, C, H, W = light.shape
        assert K == self.num_pairs, f"Expected K={self.num_pairs}, got {K}"

        delta = dark - light

        # Flatten pairs
        L = light.reshape(B * K, C, H, W)
        D = dark.reshape(B * K, C, H, W)
        X = delta.reshape(B * K, C, H, W)

        fL = self.encode(L)  # [B*K,D]
        fD = self.encode(D)
        fX = self.encode(X)
        absdiff = torch.abs(fD - fL)

        concat = torch.cat([fL, fD, absdiff, fX], dim=-1)  # [B*K, 4D]
        g = self.pair_mlp(concat)  # [B*K, D]
        g = g.reshape(B, K, -1)    # [B, K, D]

        patient_emb, alpha = self.mil(g)          # [B,D], [B,K]
        logits = self.classifier(patient_emb)     # [B,2]
        prob_ad = F.softmax(logits, dim=-1)[:, 1] # [B]

        return ModelOutput(logits=logits, prob_ad=prob_ad, attention_weights=alpha)

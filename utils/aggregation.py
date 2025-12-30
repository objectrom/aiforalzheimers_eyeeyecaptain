from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    """Attention-based MIL pooling (Ilse et al., ICML 2018).

    Input: g [B, K, D] where K=5 pairs
    Output:
        patient embedding [B, D]
        attention weights [B, K]
    """
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: torch.Tensor):
        # g: [B, K, D]
        h = torch.tanh(self.fc1(g))
        h = self.dropout(h)
        a = self.fc2(h).squeeze(-1)  # [B, K]
        alpha = F.softmax(a, dim=1)  # [B, K]
        pooled = torch.sum(g * alpha.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, alpha

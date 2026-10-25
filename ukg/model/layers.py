from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean over `dim` with mask (0/1), safe for all-masked."""
    mask = mask.to(dtype=x.dtype)
    num = (x * mask.unsqueeze(-1)).sum(dim=dim)
    den = mask.sum(dim=dim, keepdim=False).clamp_min(1.0)
    return num / den.unsqueeze(-1)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over `dim` with 0/1 mask; returns 0 when all masked."""
    mask = mask.to(dtype=scores.dtype)
    # Use masked scores inside exp to avoid 0 * inf -> NaN.
    masked_scores = scores + (mask - 1.0) * 1e9
    max_score, _ = masked_scores.max(dim=dim, keepdim=True)
    exp = torch.exp(masked_scores - max_score) * mask
    den = exp.sum(dim=dim, keepdim=True)

    # IMPORTANT: Avoid computing exp/den when den==0.
    # Even if we later select zeros via torch.where, the 0/0 division can still
    # create NaNs in the autograd graph and lead to non-finite gradients.
    den_safe = den.clamp_min(1e-12)
    out = exp / den_safe
    out = out * (den > 0).to(dtype=out.dtype)
    return out

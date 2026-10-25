from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from ukg.model.layers import MLP, masked_mean


class UncertainPathEncoder(nn.Module):
    """Encode uncertain relational paths.

    Inputs:
      paths_rel: [B, P, L] relation ids with inverse
      paths_w:   [B, P, L] confidences
      paths_len: [B, P] lengths in [0, L]
      paths_mask:[B, P] 0/1 whether path exists

    Outputs:
      z: [B, P, d_z]
    """

    def __init__(
        self,
        n_rel_with_inv: int,
        rel_dim: int,
        hidden_dim: int,
        max_path_len: int,
        dropout: float,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rel_with_inv, rel_dim)
        self.gru = nn.GRU(input_size=rel_dim, hidden_size=hidden_dim, batch_first=True)
        self.max_path_len = int(max_path_len)

        self.use_uncertainty = bool(use_uncertainty)

        if self.use_uncertainty:
            # uncertainty feature dim: min/mean/gmean/var/len_ratio
            self.unc_mlp = MLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout)
            self.out_mlp = MLP(input_dim=hidden_dim + hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout)
        else:
            # Ablation (w/o Unc): remove path uncertainty statistics encoding.
            # Keep original modules above for reference; do NOT compute/use confidence-derived stats here.
            self.unc_mlp = None
            self.out_mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        paths_rel: torch.Tensor,
        paths_w: torch.Tensor,
        paths_len: torch.Tensor,
        paths_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = paths_rel.device
        B, P, L = paths_rel.shape
        BP = B * P

        rel_emb = self.rel_emb(paths_rel.view(BP, L))  # [BP,L,d]
        lengths = paths_len.view(BP)

        # semantic encoding with GRU (only for length>0)
        s = torch.zeros((BP, self.gru.hidden_size), device=device)
        nonzero = lengths > 0
        if nonzero.any():
            idx = torch.nonzero(nonzero, as_tuple=False).squeeze(-1)
            rel_nz = rel_emb[idx]
            len_nz = lengths[idx].cpu()
            packed = nn.utils.rnn.pack_padded_sequence(rel_nz, len_nz, batch_first=True, enforce_sorted=False)
            _, h_last = self.gru(packed)
            s[idx] = h_last.squeeze(0)

        if self.use_uncertainty:
            # uncertainty stats
            w = paths_w.view(BP, L)
            # mask per step
            step_mask = (torch.arange(L, device=device).unsqueeze(0) < lengths.unsqueeze(1)).to(w.dtype)  # [BP,L]

            eps = 1e-8
            w_masked = w * step_mask

            # min: set masked positions to +inf, then min
            w_inf = torch.where(step_mask > 0, w, torch.full_like(w, float("inf")))
            w_min = torch.where(lengths > 0, w_inf.min(dim=1).values, torch.zeros((BP,), device=device))

            w_mean = torch.where(lengths > 0, w_masked.sum(dim=1) / lengths.clamp_min(1).to(w.dtype), torch.zeros((BP,), device=device))

            # geometric mean
            logw = torch.log(torch.clamp(w, min=eps))
            logw_masked = logw * step_mask
            w_gmean = torch.where(
                lengths > 0,
                torch.exp(logw_masked.sum(dim=1) / lengths.clamp_min(1).to(w.dtype)),
                torch.zeros((BP,), device=device),
            )

            # variance
            diff = (w - w_mean.unsqueeze(1)) * step_mask
            w_var = torch.where(
                lengths > 0,
                (diff * diff).sum(dim=1) / lengths.clamp_min(1).to(w.dtype),
                torch.zeros((BP,), device=device),
            )

            m_ratio = lengths.to(w.dtype) / float(self.max_path_len)

            unc = torch.stack([w_min, w_mean, w_gmean, w_var, m_ratio], dim=-1)
            u = self.unc_mlp(unc)
            z = self.out_mlp(torch.cat([s, u], dim=-1))
        else:
            # Ablation (w/o Unc): path representation only uses relation-sequence semantics.
            z = self.out_mlp(s)
        z = self.dropout(torch.relu(z))
        z = z.view(B, P, -1)

        # zero-out padded paths
        z = z * paths_mask.unsqueeze(-1)
        return z


class ConditionalPathAttention(nn.Module):
    """Relation-conditioned path attention.

    Computes V_{ht,r} for each candidate relation r.

    Inputs:
      z: [B,P,d]
      paths_mask: [B,P]
      C: [B,d_c]
      rel_emb_base: [R,d_r]

    Outputs:
      V: [B,R,d]
      alpha: [B,R,P]
    """

    def __init__(self, z_dim: int, c_dim: int, rel_dim: int, att_dim: int):
        super().__init__()
        self.Wp = nn.Linear(z_dim, att_dim, bias=False)
        self.Wc = nn.Linear(c_dim, att_dim, bias=False)
        self.Wr = nn.Linear(rel_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(att_dim))
        self.v = nn.Parameter(torch.zeros(att_dim))
        nn.init.normal_(self.v, mean=0.0, std=0.02)

    def forward(
        self,
        z: torch.Tensor,
        paths_mask: torch.Tensor,
        C: torch.Tensor,
        rel_emb_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ukg.model.layers import masked_softmax

        B, P, dz = z.shape
        R, dr = rel_emb_base.shape

        proj_z = self.Wp(z)  # [B,P,da]
        proj_c = self.Wc(C).unsqueeze(1).unsqueeze(2)  # [B,1,1,da]
        proj_r = self.Wr(rel_emb_base).unsqueeze(0).unsqueeze(2)  # [1,R,1,da]

        h = torch.tanh(proj_z.unsqueeze(1) + proj_c + proj_r + self.b.view(1, 1, 1, -1))  # [B,R,P,da]
        scores = torch.einsum("brpa,a->brp", h, self.v)  # [B,R,P]

        alpha = masked_softmax(scores, paths_mask.unsqueeze(1), dim=-1)  # [B,R,P]
        V = torch.einsum("brp,bpd->brd", alpha, z)  # [B,R,dz]
        return V, alpha

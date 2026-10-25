from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ukg.model.layers import MLP, masked_mean


class ConfidenceEncoder(nn.Module):
    """Encode edge confidence using [w, w^2] -> d."""

    def __init__(self, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = MLP(2, hidden_dim, out_dim, dropout=dropout)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: [B,E]
        feats = torch.stack([w, w * w], dim=-1)  # [B,E,2]
        return self.mlp(feats)


class RelationContextEncoder(nn.Module):
    """Edge-centric message passing over a local edge-induced subgraph.

    Inputs:
      edge_rel: [B, E]
      edge_dir: [B, E] in {0,1}
      edge_w:   [B, E]
      edge_adj: [B, E, E] adjacency among edges (1 if share endpoint)
      edge_mask:[B, E] 0/1

    Output:
      c: [B, d]
    """

    def __init__(
        self,
        n_rel_with_inv: int,
        rel_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_confidence: bool = True,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rel_with_inv, rel_dim)
        self.dir_emb = nn.Embedding(2, rel_dim)
        self.use_confidence = bool(use_confidence)

        if self.use_confidence:
            self.conf_enc = ConfidenceEncoder(out_dim=rel_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            # Ablation (w/o Unc): remove confidence encoder from RC edge representation.
            # Keep original line below for reference.
            # self.conf_enc = ConfidenceEncoder(out_dim=rel_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.conf_enc = None

        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(dropout)

        self.update_mlps = nn.ModuleList(
            [
                MLP(input_dim=rel_dim * 2, hidden_dim=hidden_dim, output_dim=rel_dim, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.gate_mlp = MLP(input_dim=rel_dim * 2, hidden_dim=hidden_dim, output_dim=1, dropout=dropout)

    def forward(
        self,
        edge_rel: torch.Tensor,
        edge_dir: torch.Tensor,
        edge_w: torch.Tensor,
        edge_adj: torch.Tensor,
        edge_mask: torch.Tensor,
        use_neighbor_gate: bool = True,
        context_pooling: str = "gate",
    ) -> torch.Tensor:
        B, E = edge_rel.shape
        if E == 0:
            # handle rare empty batch edge tensor
            return torch.zeros((B, self.rel_emb.embedding_dim), device=edge_rel.device)

        h = self.rel_emb(edge_rel) + self.dir_emb(edge_dir)
        if self.use_confidence:
            # `conf_enc` is guaranteed non-None when `use_confidence` is True.
            h = h + self.conf_enc(edge_w)
        h = self.dropout(h)

        # edge_adj already masked for padding edges in collate; still ensure self not included
        adj = edge_adj * (1.0 - torch.eye(E, device=edge_rel.device).unsqueeze(0))

        for layer in self.update_mlps:
            deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,E,1]
            m = torch.bmm(adj, h) / deg  # [B,E,d]
            inp = torch.cat([h, m], dim=-1)
            h = layer(inp)
            h = torch.relu(h)
            h = self.dropout(h)

        if bool(use_neighbor_gate) and context_pooling != "mean":
            # Uncertainty-aware neighbor gating
            q_local = masked_mean(h, edge_mask, dim=1)  # [B,d]
            q_local_exp = q_local.unsqueeze(1).expand(-1, E, -1)
            gate = torch.sigmoid(self.gate_mlp(torch.cat([h, q_local_exp], dim=-1))).squeeze(-1)  # [B,E]
            gate = gate * edge_mask
            h = h * gate.unsqueeze(-1)

        c = masked_mean(h, edge_mask, dim=1)  # [B,d]
        return c


class EntityPairContextFusion(nn.Module):
    """Fuse head and tail context vectors into a pair representation C_ht."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.mlp = MLP(input_dim=in_dim * 4, hidden_dim=hidden_dim, output_dim=out_dim, dropout=dropout)

    def forward(self, c_h: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c_h, c_t, c_h * c_t, torch.abs(c_h - c_t)], dim=-1)
        return self.mlp(x)

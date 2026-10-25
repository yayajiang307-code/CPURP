from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ukg.model.context_encoder import EntityPairContextFusion, RelationContextEncoder
from ukg.model.layers import MLP
from ukg.model.path_encoder import ConditionalPathAttention, UncertainPathEncoder


@dataclass
class ModelOutput:
    scores: torch.Tensor  # [B,R]
    w_hat: Optional[torch.Tensor]  # [B]


class UKGRelationPredictor(nn.Module):
    """Uncertain KG relation prediction with context + uncertain paths + relation-conditioned attention."""

    def __init__(
        self,
        n_base_relations: int,
        n_relations_with_inv: int,
        hidden_dim: int,
        num_layers: int,
        max_path_len: int,
        dropout: float,
        use_confidence_head: bool,
        use_relation_context: bool = True,
        use_relation_paths: bool = True,
        use_neighbor_gate: bool = True,
        use_query_path_attention: bool = True,
        context_pooling: str = "gate",
        path_pooling: str = "att",
        ablation: str = "",
    ):
        super().__init__()
        self.n_base_relations = int(n_base_relations)
        self.n_relations_with_inv = int(n_relations_with_inv)
        self.hidden_dim = int(hidden_dim)
        self.use_confidence_head = bool(use_confidence_head)
        self.use_relation_context = bool(use_relation_context)
        self.use_relation_paths = bool(use_relation_paths)
        self.use_neighbor_gate = bool(use_neighbor_gate)
        self.use_query_path_attention = bool(use_query_path_attention)
        self.context_pooling = str(context_pooling)
        self.path_pooling = str(path_pooling)

        self.ablation = (ablation or "").strip().lower()
        wo_unc = self.ablation in {"wo_unc", "w/o_unc", "wounc"}

        # Context encoders for head and tail use shared parameters (same edge feature space)
        self.context_encoder = RelationContextEncoder(
            n_rel_with_inv=self.n_relations_with_inv,
            rel_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_confidence=(not wo_unc),
        )
        self.pair_fusion = EntityPairContextFusion(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)

        self.path_encoder = UncertainPathEncoder(
            n_rel_with_inv=self.n_relations_with_inv,
            rel_dim=hidden_dim,
            hidden_dim=hidden_dim,
            max_path_len=max_path_len,
            dropout=dropout,
            use_uncertainty=(not wo_unc),
        )

        # Candidate relation embeddings (base relations only)
        self.cand_rel_emb = nn.Embedding(self.n_base_relations, hidden_dim)

        self.path_att = ConditionalPathAttention(z_dim=hidden_dim, c_dim=hidden_dim, rel_dim=hidden_dim, att_dim=hidden_dim)

        q_in_dim = hidden_dim * (3 if self.use_relation_context else 2)
        self.query_mlp = MLP(input_dim=q_in_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.scorer = nn.Linear(hidden_dim, 1)

        if self.use_confidence_head:
            self.conf_head = nn.Linear(hidden_dim, 1)
        else:
            self.conf_head = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        device = batch["h"].device
        B = batch["h"].shape[0]
        R = self.n_base_relations

        if self.use_relation_context:
            c_h = self.context_encoder(
                batch["head_edge_rel"],
                batch["head_edge_dir"],
                batch["head_edge_w"],
                batch["head_edge_adj"],
                batch["head_edge_mask"],
                use_neighbor_gate=self.use_neighbor_gate,
                context_pooling=self.context_pooling,
            )
            c_t = self.context_encoder(
                batch["tail_edge_rel"],
                batch["tail_edge_dir"],
                batch["tail_edge_w"],
                batch["tail_edge_adj"],
                batch["tail_edge_mask"],
                use_neighbor_gate=self.use_neighbor_gate,
                context_pooling=self.context_pooling,
            )
            C = self.pair_fusion(c_h, c_t)  # [B,d]
        else:
            C = torch.zeros((B, self.hidden_dim), device=device, dtype=self.cand_rel_emb.weight.dtype)

        if self.use_relation_paths:
            z = self.path_encoder(batch["paths_rel"], batch["paths_w"], batch["paths_len"], batch["paths_mask"])  # [B,P,d]
        else:
            # Ablation (w/o RP): disable relation paths.
            # Keep the original encoding line above for reference.
            # z = self.path_encoder(batch["paths_rel"], batch["paths_w"], batch["paths_len"], batch["paths_mask"])  # [B,P,d]
            z = None

        cand_ids = torch.arange(R, device=device)
        e_r = self.cand_rel_emb(cand_ids)  # [R,d]

        if self.use_relation_paths and self.use_query_path_attention and self.path_pooling != "mean":
            V, _ = self.path_att(z=z, paths_mask=batch["paths_mask"], C=C, rel_emb_base=e_r)  # [B,R,d]
        else:
            if self.use_relation_paths and z is not None:
                pooled_z = torch.sum(z * batch["paths_mask"].unsqueeze(-1), dim=1)
                pooled_z = pooled_z / batch["paths_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                pooled_z = torch.zeros((B, self.hidden_dim), device=device, dtype=self.cand_rel_emb.weight.dtype)
            V = pooled_z.unsqueeze(1).expand(-1, R, -1)

        C_exp = C.unsqueeze(1).expand(-1, R, -1)
        e_r_exp = e_r.unsqueeze(0).expand(B, -1, -1)

        if self.use_relation_context:
            q_in = torch.cat([C_exp, V, e_r_exp], dim=-1)
        else:
            q_in = torch.cat([V, e_r_exp], dim=-1)

        q = self.query_mlp(q_in)  # [B,R,d]
        q = torch.relu(q)
        scores = self.scorer(q).squeeze(-1)  # [B,R]

        w_hat = None
        if self.use_confidence_head and ("r" in batch):
            # gather true relation representation
            true_r = batch["r"].view(B, 1, 1).expand(-1, 1, q.shape[-1])
            q_true = q.gather(dim=1, index=true_r).squeeze(1)  # [B,d]
            w_hat = torch.sigmoid(self.conf_head(q_true)).squeeze(-1)

        return ModelOutput(scores=scores, w_hat=w_hat)

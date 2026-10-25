from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ukg.data import (
    Quadruple,
    UKGGraph,
    build_edge_adjacency,
    extract_k_hop_edges,
    extract_paths_bounded_bfs,
)


MAX_CONTEXT_EDGES_HARD_CAP = 256


@dataclass
class Sample:
    h: int
    t: int
    r: int  # base relation id
    w: float

    head_edges_rel: np.ndarray  # [E_h]
    head_edges_dir: np.ndarray  # [E_h]
    head_edges_w: np.ndarray  # [E_h]
    head_edges_adj: np.ndarray  # [E_h, E_h]

    tail_edges_rel: np.ndarray  # [E_t]
    tail_edges_dir: np.ndarray  # [E_t]
    tail_edges_w: np.ndarray  # [E_t]
    tail_edges_adj: np.ndarray  # [E_t, E_t]

    paths_rel: np.ndarray  # [P, L]
    paths_w: np.ndarray  # [P, L]
    paths_len: np.ndarray  # [P]
    paths_mask: np.ndarray  # [P]


class UKGDataset(Dataset):
    def __init__(
        self,
        quadruples: Sequence[Quadruple],
        graph: UKGGraph,
        max_hop: int,
        max_path_len: int,
        max_paths: int,
        max_context_edges: int,
        use_relation_paths: bool = True,
        # Original (kept for reference):
        # split: str,
        split: str = "train",
    ):
        self.quadruples = list(quadruples)
        self.graph = graph
        self.max_hop = int(max_hop)
        self.max_path_len = int(max_path_len)
        self.max_paths = int(max_paths)
        self.max_context_edges = min(int(max_context_edges), MAX_CONTEXT_EDGES_HARD_CAP)
        self.use_relation_paths = bool(use_relation_paths)
        self.split = split

    def __len__(self) -> int:
        return len(self.quadruples)

    def __getitem__(self, idx: int) -> Sample:
        q = self.quadruples[idx]
        mask_triplet = (q.h, q.t, q.r)

        head_edges = extract_k_hop_edges(
            self.graph,
            center=q.h,
            max_hop=self.max_hop,
            max_edges=self.max_context_edges,
            mask_triplet=mask_triplet,
        )
        tail_edges = extract_k_hop_edges(
            self.graph,
            center=q.t,
            max_hop=self.max_hop,
            max_edges=self.max_context_edges,
            mask_triplet=mask_triplet,
        )

        def to_arrays(edges):
            if len(edges) == 0:
                return (
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.float32),
                )
            rel = np.asarray([e.rel for e in edges], dtype=np.int64)
            direction = (rel >= self.graph.n_base_relations).astype(np.int64)
            w = np.asarray([e.w for e in edges], dtype=np.float32)
            adj = build_edge_adjacency(edges)
            return rel, direction, w, adj

        head_rel, head_dir, head_w, head_adj = to_arrays(head_edges)
        tail_rel, tail_dir, tail_w, tail_adj = to_arrays(tail_edges)

        if self.use_relation_paths:
            paths = extract_paths_bounded_bfs(
                self.graph,
                h=q.h,
                t=q.t,
                max_path_len=self.max_path_len,
                max_paths=self.max_paths,
                mask_triplet=mask_triplet,
            )
        else:
            # Ablation (w/o RP): disable relation paths.
            # Keep the original extraction code above for reference.
            # paths = extract_paths_bounded_bfs(
            #     self.graph,
            #     h=q.h,
            #     t=q.t,
            #     max_path_len=self.max_path_len,
            #     max_paths=self.max_paths,
            #     mask_triplet=mask_triplet,
            # )
            paths = []

        # pad paths to [max_paths, max_path_len]
        P, L = self.max_paths, self.max_path_len
        paths_rel = np.zeros((P, L), dtype=np.int64)
        paths_w = np.zeros((P, L), dtype=np.float32)
        paths_len = np.zeros((P,), dtype=np.int64)
        paths_mask = np.zeros((P,), dtype=np.float32)
        for i, (rels, ws) in enumerate(paths[:P]):
            m = len(rels)
            if m == 0:
                continue
            paths_rel[i, :m] = np.asarray(rels, dtype=np.int64)
            paths_w[i, :m] = np.asarray(ws, dtype=np.float32)
            paths_len[i] = m
            paths_mask[i] = 1.0

        return Sample(
            h=q.h,
            t=q.t,
            r=q.r,
            w=float(q.w),
            head_edges_rel=head_rel,
            head_edges_dir=head_dir,
            head_edges_w=head_w,
            head_edges_adj=head_adj,
            tail_edges_rel=tail_rel,
            tail_edges_dir=tail_dir,
            tail_edges_w=tail_w,
            tail_edges_adj=tail_adj,
            paths_rel=paths_rel,
            paths_w=paths_w,
            paths_len=paths_len,
            paths_mask=paths_mask,
        )


def _pad_edges(batch_edges_rel, batch_edges_dir, batch_edges_w, batch_edges_adj):
    B = len(batch_edges_rel)
    max_e = max((x.shape[0] for x in batch_edges_rel), default=0)
    rel = torch.zeros((B, max_e), dtype=torch.long)
    direction = torch.zeros((B, max_e), dtype=torch.long)
    w = torch.zeros((B, max_e), dtype=torch.float32)
    mask = torch.zeros((B, max_e), dtype=torch.float32)
    adj = torch.zeros((B, max_e, max_e), dtype=torch.float32)

    for i in range(B):
        e = batch_edges_rel[i].shape[0]
        if e == 0:
            continue
        rel[i, :e] = torch.from_numpy(batch_edges_rel[i])
        direction[i, :e] = torch.from_numpy(batch_edges_dir[i])
        w[i, :e] = torch.from_numpy(batch_edges_w[i])
        mask[i, :e] = 1.0
        a = torch.from_numpy(batch_edges_adj[i])
        adj[i, :e, :e] = a

    # ensure padded edges have no adjacency
    adj = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
    return rel, direction, w, adj, mask


def collate_samples(samples: List[Sample], ablation: Optional[str] = None) -> Dict[str, Any]:
    h = torch.tensor([s.h for s in samples], dtype=torch.long)
    t = torch.tensor([s.t for s in samples], dtype=torch.long)
    r = torch.tensor([s.r for s in samples], dtype=torch.long)
    conf = torch.tensor([s.w for s in samples], dtype=torch.float32)

    head_rel, head_dir, head_w, head_adj, head_mask = _pad_edges(
        [s.head_edges_rel for s in samples],
        [s.head_edges_dir for s in samples],
        [s.head_edges_w for s in samples],
        [s.head_edges_adj for s in samples],
    )
    tail_rel, tail_dir, tail_w, tail_adj, tail_mask = _pad_edges(
        [s.tail_edges_rel for s in samples],
        [s.tail_edges_dir for s in samples],
        [s.tail_edges_w for s in samples],
        [s.tail_edges_adj for s in samples],
    )

    paths_rel = torch.stack([torch.from_numpy(s.paths_rel) for s in samples], dim=0).long()  # [B,P,L]
    paths_w = torch.stack([torch.from_numpy(s.paths_w) for s in samples], dim=0).float()  # [B,P,L]
    paths_len = torch.stack([torch.from_numpy(s.paths_len) for s in samples], dim=0).long()  # [B,P]
    paths_mask = torch.stack([torch.from_numpy(s.paths_mask) for s in samples], dim=0).float()  # [B,P]

    ab = (ablation or "").strip().lower()
    if ab in {"wo_unc", "w/o_unc", "wounc"}:
        # Ablation (w/o Unc): block all confidence evidence from entering the model.
        # IMPORTANT: do NOT touch `conf` labels; only replace model-input edge/path confidences.
        head_w = torch.ones_like(head_w)
        tail_w = torch.ones_like(tail_w)
        paths_w = torch.ones_like(paths_w)

    return {
        "h": h,
        "t": t,
        "r": r,
        "conf": conf,
        "head_edge_rel": head_rel,
        "head_edge_dir": head_dir,
        "head_edge_w": head_w,
        "head_edge_adj": head_adj,
        "head_edge_mask": head_mask,
        "tail_edge_rel": tail_rel,
        "tail_edge_dir": tail_dir,
        "tail_edge_w": tail_w,
        "tail_edge_adj": tail_adj,
        "tail_edge_mask": tail_mask,
        "paths_rel": paths_rel,
        "paths_w": paths_w,
        "paths_len": paths_len,
        "paths_mask": paths_mask,
    }

from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Quadruple:
    h: int
    r: int  # base relation id in [0, n_base_rel)
    t: int
    w: float


@dataclass(frozen=True)
class DirectedEdge:
    src: int
    dst: int
    rel: int  # relation id with inverse in [0, 2*n_base_rel)
    w: float


class UKGMapping:
    def __init__(self):
        self.entity2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2entity: List[str] = []
        self.id2relation: List[str] = []

    def get_entity_id(self, name: str) -> int:
        if name in self.entity2id:
            return self.entity2id[name]
        idx = len(self.id2entity)
        self.entity2id[name] = idx
        self.id2entity.append(name)
        return idx

    def get_relation_id(self, name: str) -> int:
        if name in self.relation2id:
            return self.relation2id[name]
        idx = len(self.id2relation)
        self.relation2id[name] = idx
        self.id2relation.append(name)
        return idx


class UKGGraph:
    """Graph built from training quadruples, with inverse relations for traversal."""

    def __init__(self, n_entities: int, n_base_relations: int):
        self.n_entities = n_entities
        self.n_base_relations = n_base_relations
        self.n_relations_with_inv = n_base_relations * 2
        self.adj: List[List[DirectedEdge]] = [[] for _ in range(n_entities)]  # outgoing edges

    def inv_rel(self, r_base: int) -> int:
        return r_base + self.n_base_relations

    def add_quadruple(self, h: int, r_base: int, t: int, w: float) -> None:
        self.adj[h].append(DirectedEdge(src=h, dst=t, rel=r_base, w=float(w)))
        self.adj[t].append(DirectedEdge(src=t, dst=h, rel=self.inv_rel(r_base), w=float(w)))


def read_quadruples_tsv(path: str, mapping: UKGMapping) -> List[Quadruple]:
    res: List[Quadruple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 columns (h\tr\tt\tw) in {path}, got: {line}")
            h, r, t, w = parts
            hid = mapping.get_entity_id(h)
            tid = mapping.get_entity_id(t)
            rid = mapping.get_relation_id(r)
            res.append(Quadruple(h=hid, r=rid, t=tid, w=float(w)))
    return res


def build_mapping_from_splits(train_path: str, valid_path: str, test_path: str) -> Tuple[UKGMapping, List[Quadruple], List[Quadruple], List[Quadruple]]:
    mapping = UKGMapping()
    train = read_quadruples_tsv(train_path, mapping)
    valid = read_quadruples_tsv(valid_path, mapping)
    test = read_quadruples_tsv(test_path, mapping)
    return mapping, train, valid, test


def build_train_graph(train_quads: Sequence[Quadruple], n_entities: int, n_base_relations: int) -> UKGGraph:
    g = UKGGraph(n_entities=n_entities, n_base_relations=n_base_relations)
    for q in train_quads:
        g.add_quadruple(q.h, q.r, q.t, q.w)
    return g


def _mask_direct_edge(edge: DirectedEdge, h: int, t: int, r_base: int, n_base_rel: int) -> bool:
    # mask the target fact edge (and its inverse) to reduce label leakage
    if edge.src == h and edge.dst == t and edge.rel == r_base:
        return True
    if edge.src == t and edge.dst == h and edge.rel == (r_base + n_base_rel):
        return True
    return False


def extract_k_hop_edges(
    g: UKGGraph,
    center: int,
    max_hop: int,
    max_edges: int,
    mask_triplet: Optional[Tuple[int, int, int]] = None,
) -> List[DirectedEdge]:
    """Collect directed edges within k hops around center using BFS over entities.

    Returns a list of directed edges encountered (deduplicated by (src,dst,rel)).
    """
    seen_nodes = {center}
    frontier = [center]
    seen_edges = set()
    edges: List[DirectedEdge] = []

    for _ in range(max_hop):
        next_frontier = []
        for u in frontier:
            for e in g.adj[u]:
                if mask_triplet is not None:
                    h, t, r_base = mask_triplet
                    if _mask_direct_edge(e, h, t, r_base, g.n_base_relations):
                        continue
                key = (e.src, e.dst, e.rel)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(e)
                    if len(edges) >= max_edges:
                        return edges
                if e.dst not in seen_nodes:
                    seen_nodes.add(e.dst)
                    next_frontier.append(e.dst)
        frontier = next_frontier
        if not frontier:
            break
    return edges


def extract_paths_bounded_bfs(
    g: UKGGraph,
    h: int,
    t: int,
    max_path_len: int,
    max_paths: int,
    mask_triplet: Optional[Tuple[int, int, int]] = None,
) -> List[Tuple[List[int], List[float]]]:
    """Extract simple paths from h to t.

    Returns: list of (rel_id_with_inv_seq, confidence_seq).
    """
    if h == t:
        return []

    paths: List[Tuple[List[int], List[float]]] = []

    # queue entries: (current_node, rel_seq, w_seq, visited_nodes)
    # NOTE: Using a Python `set` for visited and copying it for each queue entry
    # can cause large transient memory use when the frontier grows. Because
    # `max_path_len` is small, we store visited nodes as a small tuple instead.
    queue: deque[Tuple[int, List[int], List[float], Tuple[int, ...]]] = deque([(h, [], [], (h,))])

    # Safety caps: in sparse/no-path cases, the BFS frontier can still grow very fast
    # (especially for high-degree nodes). Keep the budget conservative to limit CPU
    # spikes and transient memory pressure.
    length_factor = max(1, int(max_path_len))
    max_expansions = max(5_000, int(max_paths) * 1_500 // length_factor)
    max_queue_size = max(2_000, int(max_paths) * 800 // length_factor)
    expansions = 0

    while queue:
        if len(queue) > max_queue_size:
            return paths
        node, rels, ws, visited_nodes = queue.popleft()
        if len(rels) >= max_path_len:
            continue
        for e in g.adj[node]:
            expansions += 1
            if expansions > max_expansions:
                return paths
            if mask_triplet is not None:
                mh, mt, mr = mask_triplet
                if _mask_direct_edge(e, mh, mt, mr, g.n_base_relations):
                    continue
            if e.dst in visited_nodes:
                continue
            new_rels = rels + [e.rel]
            new_ws = ws + [float(e.w)]
            if e.dst == t:
                paths.append((new_rels, new_ws))
                if len(paths) >= max_paths:
                    return paths
            else:
                queue.append((e.dst, new_rels, new_ws, visited_nodes + (e.dst,)))

    return paths


def build_edge_adjacency(edges: Sequence[DirectedEdge]) -> np.ndarray:
    """Edge-edge adjacency matrix A where A[i,j]=1 if edges i and j share an endpoint."""
    n = len(edges)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    entity2edges: Dict[int, List[int]] = {}
    for i, e in enumerate(edges):
        entity2edges.setdefault(e.src, []).append(i)
        entity2edges.setdefault(e.dst, []).append(i)

    adj = np.zeros((n, n), dtype=np.float32)
    for _, idxs in entity2edges.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    adj[i, j] = 1.0
    return adj


def pad_1d(seqs: List[List[int]], pad: int) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max((len(s) for s in seqs), default=0)
    arr = np.full((len(seqs), max_len), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        arr[i, : len(s)] = np.asarray(s, dtype=np.int64)
        mask[i, : len(s)] = 1.0
    return arr, mask

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass
class RelationMetrics:
    mrr: float
    hits1: float
    hits3: float
    hits10: float
    accuracy: float
    macro_f1: float


@dataclass
class ConfidenceMetrics:
    mse: float
    mae: float


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def compute_relation_metrics(scores: np.ndarray, labels: np.ndarray) -> RelationMetrics:
    """Compute relation prediction metrics.

    Args:
        scores: [N, R] array of scores (higher is better)
        labels: [N] true relation id in [0, R)
    """
    if scores.ndim != 2:
        raise ValueError("scores must be 2D [N, R]")

    n, r = scores.shape
    if labels.shape[0] != n:
        raise ValueError("labels length must match scores")

    # ranking: 1 is best
    # argsort descending
    order = np.argsort(-scores, axis=1)
    # position of true label
    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(r)[None, :]
    ranks = inv[np.arange(n), labels] + 1

    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))

    y_pred = np.argmax(scores, axis=1)
    acc = float(np.mean(y_pred == labels))
    macro_f1 = _macro_f1(labels, y_pred, r)

    return RelationMetrics(mrr=mrr, hits1=hits1, hits3=hits3, hits10=hits10, accuracy=acc, macro_f1=macro_f1)


def compute_confidence_metrics(w_hat: np.ndarray, w_true: np.ndarray) -> ConfidenceMetrics:
    w_hat = w_hat.astype(np.float32)
    w_true = w_true.astype(np.float32)
    mse = float(np.mean((w_hat - w_true) ** 2))
    mae = float(np.mean(np.abs(w_hat - w_true)))
    return ConfidenceMetrics(mse=mse, mae=mae)

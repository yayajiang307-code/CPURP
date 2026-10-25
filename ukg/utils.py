import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or CUDA not available.
        pass


def select_device(device: str = "auto", gpu: Optional[str] = None):
    import torch

    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # auto
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Reading --config requires PyYAML. Install via: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file root must be a mapping/dict")
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict `base` with `updates` (in-place) and return base."""
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ukg.data import build_mapping_from_splits, build_train_graph
from ukg.dataloader import UKGDataset, collate_samples
from ukg.model.model import UKGRelationPredictor
from ukg.utils import deep_update, load_yaml_config, select_device, set_seed
from train import _resolve_config, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate uncertain KG relation prediction checkpoint")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: checkpoints/<dataset>/best.pt)")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="Which split to evaluate (default: test)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_rc", action="store_true", help="Ablation: disable relation context (RC) module (w/o RC)")
    p.add_argument("--no_rp", action="store_true", help="Ablation: disable relation paths (RP) module (w/o RP)")
    p.add_argument("--ablation", type=str, default=None, help="Additional ablation variants (e.g., wo_unc or wo_gq)")
    p.add_argument("--save_json", type=str, default=None, help="Optional path to save metrics as JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # reuse train's config resolution behavior
    class _Args:
        pass

    a = _Args()
    a.dataset = args.dataset
    a.data_dir = args.data_dir
    a.config = args.config
    a.gpu = args.gpu
    a.seed = args.seed
    a.epochs = None
    a.batch_size = None
    a.lr = None
    a.log_every = None
    a.max_steps = None
    a.hidden_dim = None
    a.num_layers = None
    a.max_hop = None
    a.max_path_len = None
    a.max_paths = None
    a.lambda_conf = None
    a.use_confidence_head = False
    a.no_confidence_head = False
    a.no_rc = args.no_rc
    a.no_rp = args.no_rp
    a.ablation = args.ablation

    cfg = _resolve_config(a)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"), gpu=args.gpu)

    data_dir = args.data_dir or str(Path("data") / args.dataset)
    train_path = str(Path(data_dir) / cfg["paths"]["train_file"])
    valid_path = str(Path(data_dir) / cfg["paths"]["valid_file"])
    test_path = str(Path(data_dir) / cfg["paths"]["test_file"])

    mapping, train_quads, valid_quads, test_quads = build_mapping_from_splits(train_path, valid_path, test_path)
    n_entities = len(mapping.id2entity)
    n_base_rel = len(mapping.id2relation)
    graph = build_train_graph(train_quads, n_entities=n_entities, n_base_relations=n_base_rel)

    model_cfg = cfg.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))
    max_hop = int(model_cfg.get("max_hop", 2))
    max_path_len = int(model_cfg.get("max_path_len", 4))
    max_paths = int(model_cfg.get("max_paths", 32))
    max_context_edges = int(model_cfg.get("max_context_edges", 128))
    use_conf_head = bool(model_cfg.get("use_confidence_head", True))
    lambda_conf = float(model_cfg.get("lambda_conf", 0.0))
    use_rc = bool(model_cfg.get("use_relation_context", True))
    use_rp = bool(model_cfg.get("use_relation_paths", True))

    use_neighbor_gate = bool(model_cfg.get("use_neighbor_gate", True))
    use_query_path_attention = bool(model_cfg.get("use_query_path_attention", True))
    context_pooling = str(model_cfg.get("context_pooling", "gate"))
    path_pooling = str(model_cfg.get("path_pooling", "att"))

    ablation = str(cfg.get("ablation", "") or "").strip().lower()
    if ablation in {"none", "null"}:
        ablation = ""

    if ablation == "wo_gq":
        use_neighbor_gate = False
        use_query_path_attention = False
        context_pooling = "mean"
        path_pooling = "mean"

    if args.split == "valid":
        quads = valid_quads
        split_name = "valid"
    else:
        quads = test_quads
        split_name = "test"

    ds = UKGDataset(
        quadruples=quads,
        graph=graph,
        max_hop=max_hop,
        max_path_len=max_path_len,
        max_paths=max_paths,
        max_context_edges=max_context_edges,
        use_relation_paths=use_rp,
        split=split_name,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 128)),
        shuffle=False,
        num_workers=0,
        collate_fn=functools.partial(collate_samples, ablation=ablation),
    )

    model = UKGRelationPredictor(
        n_base_relations=n_base_rel,
        n_relations_with_inv=n_base_rel * 2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_path_len=max_path_len,
        dropout=dropout,
        use_confidence_head=use_conf_head,
        use_relation_context=use_rc,
        use_relation_paths=use_rp,
        use_neighbor_gate=use_neighbor_gate,
        use_query_path_attention=use_query_path_attention,
        context_pooling=context_pooling,
        path_pooling=path_pooling,
        ablation=ablation,
    ).to(device)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif ablation == "wo_gq":
        ckpt_path = Path("runs") / f"CPURP_woGQ_{args.dataset}_seed{seed}" / "best.pt"
    else:
        ckpt_path = Path("checkpoints") / args.dataset / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    metrics, _, _ = evaluate(model, loader, device=device, lambda_conf=lambda_conf)

    title = "Valid results" if args.split == "valid" else "Test results"
    print(title)
    print(
        f"mrr={metrics['mrr']:.4f} h1={metrics['hits@1']:.4f} h3={metrics['hits@3']:.4f} h10={metrics['hits@10']:.4f} "
        f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}"
    )
    # Original behavior (kept for reference):
    # if use_conf_head and lambda_conf > 0:
    #     print(f"mse={metrics.get('mse', 0.0):.4f} mae={metrics.get('mae', 0.0):.4f}")
    print(f"mse={metrics.get('mse', 0.0):.4f} mae={metrics.get('mae', 0.0):.4f}")

    if args.save_json:
        out = {
            "dataset": args.dataset,
            "split": args.split,
            "ablation": ablation,
            "ckpt": str(ckpt_path),
            "metrics": metrics,
        }
        Path(args.save_json).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

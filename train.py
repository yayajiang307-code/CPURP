from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ukg.data import build_mapping_from_splits, build_train_graph
from ukg.dataloader import UKGDataset, collate_samples
from ukg.metrics import compute_confidence_metrics, compute_relation_metrics
from ukg.model.model import UKGRelationPredictor
from ukg.utils import AverageMeter, deep_update, load_yaml_config, select_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertain KG relation prediction (PyTorch)")

    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., CN15K, NL27K, PPI5K)")
    p.add_argument("--data_dir", type=str, default=None, help="Path to dataset dir containing train/val/test .tsv")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--max_hop", type=int, default=None)
    p.add_argument("--max_path_len", type=int, default=None)
    p.add_argument("--max_paths", type=int, default=None)
    p.add_argument("--lambda_conf", type=float, default=None)

    p.add_argument(
        "--log_every",
        type=int,
        default=None,
        help="Print training progress every N steps (0 disables per-step logs). If omitted, uses config training.log_every.",
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop training after N optimizer steps (0 disables). If omitted, uses config training.max_steps.",
    )

    p.add_argument(
        "--debug_nan",
        action="store_true",
        help="If loss becomes NaN/Inf, print detailed tensor stats and abort.",
    )

    p.add_argument("--use_confidence_head", action="store_true", help="Enable confidence prediction head")
    p.add_argument("--no_confidence_head", action="store_true", help="Disable confidence prediction head")

    p.add_argument("--no_rc", action="store_true", help="Ablation: disable relation context (RC) module (w/o RC)")

    p.add_argument("--no_rp", action="store_true", help="Ablation: disable relation paths (RP) module (w/o RP)")

    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Additional ablation variants (e.g., wo_unc or wo_gq)",
    )

    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # always load default config if present
    default_cfg_path = Path("configs/default.yaml")
    if default_cfg_path.exists():
        cfg = deep_update(cfg, load_yaml_config(str(default_cfg_path)))

    # user config
    if args.config:
        cfg = deep_update(cfg, load_yaml_config(args.config))

    # CLI overrides
    cfg.setdefault("seed", 42)
    if args.seed is not None:
        cfg["seed"] = args.seed

    cfg.setdefault("training", {})
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.log_every is not None:
        cfg["training"]["log_every"] = args.log_every
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps

    cfg.setdefault("model", {})
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        cfg["model"]["num_layers"] = args.num_layers
    if args.max_hop is not None:
        cfg["model"]["max_hop"] = args.max_hop
    if args.max_path_len is not None:
        cfg["model"]["max_path_len"] = args.max_path_len
    if args.max_paths is not None:
        cfg["model"]["max_paths"] = args.max_paths
    if args.lambda_conf is not None:
        cfg["model"]["lambda_conf"] = args.lambda_conf

    if args.no_rc:
        cfg["model"]["use_relation_context"] = False

    if getattr(args, "no_rp", False):
        cfg["model"]["use_relation_paths"] = False

    if args.use_confidence_head and args.no_confidence_head:
        raise ValueError("Cannot set both --use_confidence_head and --no_confidence_head")
    if args.use_confidence_head:
        cfg["model"]["use_confidence_head"] = True
    if args.no_confidence_head:
        cfg["model"]["use_confidence_head"] = False

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("train_file", "train.tsv")
    cfg["paths"].setdefault("valid_file", "val.tsv")
    cfg["paths"].setdefault("test_file", "test.tsv")

    cfg.setdefault("ablation", "")
    if getattr(args, "ablation", None) is not None:
        cfg["ablation"] = args.ablation

    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("use_neighbor_gate", True)
    model_cfg.setdefault("use_query_path_attention", True)
    model_cfg.setdefault("context_pooling", "gate")
    model_cfg.setdefault("path_pooling", "att")

    ablation = str(cfg.get("ablation", "") or "").strip().lower()
    if ablation in {"none", "null"}:
        ablation = ""
    if ablation == "wo_gq":
        model_cfg["use_neighbor_gate"] = False
        model_cfg["use_query_path_attention"] = False
        model_cfg["context_pooling"] = "mean"
        model_cfg["path_pooling"] = "mean"

    return cfg


@torch.no_grad()
def evaluate(model: UKGRelationPredictor, loader: DataLoader, device: torch.device, lambda_conf: float) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    model.eval()
    all_scores = []
    all_labels = []
    all_w_true = []
    all_w_hat = []

    rel_loss_meter = AverageMeter()
    conf_loss_meter = AverageMeter()

    ce = nn.CrossEntropyLoss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        out = model(batch)
        rel_loss = ce(out.scores, batch["r"])
        rel_loss_meter.update(rel_loss.item(), n=batch["r"].shape[0])

        if out.w_hat is not None and lambda_conf > 0:
            conf_loss = mse(out.w_hat, batch["conf"])
            conf_loss_meter.update(conf_loss.item(), n=batch["r"].shape[0])
        else:
            conf_loss = None

        all_scores.append(out.scores.detach().cpu().numpy())
        all_labels.append(batch["r"].detach().cpu().numpy())
        all_w_true.append(batch["conf"].detach().cpu().numpy())
        if out.w_hat is not None:
            all_w_hat.append(out.w_hat.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    rel_metrics = compute_relation_metrics(scores=scores, labels=labels)

    metrics: Dict[str, float] = {
        "relation_loss": rel_loss_meter.avg,
        "mrr": rel_metrics.mrr,
        "hits@1": rel_metrics.hits1,
        "hits@3": rel_metrics.hits3,
        "hits@10": rel_metrics.hits10,
        "accuracy": rel_metrics.accuracy,
        "macro_f1": rel_metrics.macro_f1,
    }

    w_hat_arr = np.concatenate(all_w_hat, axis=0) if all_w_hat else None
    w_true_arr = np.concatenate(all_w_true, axis=0)

    if w_hat_arr is not None and lambda_conf > 0:
        conf_metrics = compute_confidence_metrics(w_hat=w_hat_arr, w_true=w_true_arr)
        metrics.update({"confidence_loss": conf_loss_meter.avg, "mse": conf_metrics.mse, "mae": conf_metrics.mae})
    else:
        # Original behavior (kept for reference):
        # metrics.update({"confidence_loss": 0.0})
        metrics.update({"confidence_loss": 0.0, "mse": 0.0, "mae": 0.0})

    return metrics, scores, w_hat_arr


def main() -> None:
    args = parse_args()
    cfg = _resolve_config(args)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = select_device(cfg.get("device", "auto"), gpu=args.gpu)

    data_dir = args.data_dir or str(Path("data") / args.dataset)
    data_dir = str(Path(data_dir))

    train_path = str(Path(data_dir) / cfg["paths"]["train_file"])
    valid_path = str(Path(data_dir) / cfg["paths"]["valid_file"])
    test_path = str(Path(data_dir) / cfg["paths"]["test_file"])

    mapping, train_quads, valid_quads, test_quads = build_mapping_from_splits(train_path, valid_path, test_path)
    n_entities = len(mapping.id2entity)
    n_base_rel = len(mapping.id2relation)

    graph = build_train_graph(train_quads, n_entities=n_entities, n_base_relations=n_base_rel)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    ablation = str(cfg.get("ablation", "") or "").strip().lower()
    if ablation in {"none", "null"}:
        ablation = ""
    valid_ablations = {"", "wo_unc", "w/o_unc", "wounc", "wo_gq", "w/o_gq", "wogq"}
    if ablation not in valid_ablations:
        raise ValueError(f"Unknown --ablation={ablation!r}. Supported: {sorted(valid_ablations)}")

    log_every = int(train_cfg.get("log_every", 0))
    max_steps = int(train_cfg.get("max_steps", 0))

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

    if ablation == "wo_gq":
        use_neighbor_gate = False
        use_query_path_attention = False
        context_pooling = "mean"
        path_pooling = "mean"

    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 128))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    select_best_by = str(train_cfg.get("select_best_by", "mrr")).lower()

    train_ds = UKGDataset(
        quadruples=train_quads,
        graph=graph,
        max_hop=max_hop,
        max_path_len=max_path_len,
        max_paths=max_paths,
        max_context_edges=max_context_edges,
        use_relation_paths=use_rp,
        split="train",
    )
    valid_ds = UKGDataset(
        quadruples=valid_quads,
        graph=graph,
        max_hop=max_hop,
        max_path_len=max_path_len,
        max_paths=max_paths,
        max_context_edges=max_context_edges,
        use_relation_paths=use_rp,
        split="valid",
    )
    test_ds = UKGDataset(
        quadruples=test_quads,
        graph=graph,
        max_hop=max_hop,
        max_path_len=max_path_len,
        max_paths=max_paths,
        max_context_edges=max_context_edges,
        use_relation_paths=use_rp,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=functools.partial(collate_samples, ablation=ablation),
    )
    # Original behavior (kept for reference):
    # valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_samples)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_samples)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=functools.partial(collate_samples, ablation=ablation),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
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

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    if ablation == "wo_gq":
        ckpt_dir = Path("runs") / f"CPURP_woGQ_{args.dataset}_seed{seed}"
    else:
        ckpt_dir = Path("checkpoints") / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    log_fp = open(ckpt_dir / "train.log", "a", encoding="utf-8") if ablation == "wo_gq" else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fp is not None:
            log_fp.write(msg + "\n")

    best_metric = -1.0

    log(f"Dataset={args.dataset} entities={n_entities} relations={n_base_rel} device={device} seed={seed}")
    log(f"Train={len(train_ds)} Valid={len(valid_ds)} Test={len(test_ds)}")
    log(
        f"hidden_dim={hidden_dim} layers={num_layers} hop={max_hop} max_path_len={max_path_len} max_paths={max_paths}",
    )
    if ablation:
        log(f"ablation={ablation}")
    # Original behavior (kept for reference):
    # print(f"use_relation_context={use_rc} use_confidence_head={use_conf_head} lambda_conf={lambda_conf}", flush=True)
    log(f"use_relation_context={use_rc} use_relation_paths={use_rp} use_confidence_head={use_conf_head} lambda_conf={lambda_conf}")
    log(f"use_neighbor_gate = {use_neighbor_gate}")
    log(f"use_query_path_attention = {use_query_path_attention}")
    log(f"context_pooling = {context_pooling}")
    log(f"path_pooling = {path_pooling}")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        rel_meter = AverageMeter()
        conf_meter = AverageMeter()

        epoch_start = time.time()
        last_log_time = epoch_start
        n_steps = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            for k, v in batch.items():
                batch[k] = v.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(batch)
            rel_loss = ce(out.scores, batch["r"])

            if out.w_hat is not None and lambda_conf > 0:
                conf_loss = mse(out.w_hat, batch["conf"])
                loss = rel_loss + lambda_conf * conf_loss
            else:
                conf_loss = None
                loss = rel_loss

            if args.debug_nan and (not torch.isfinite(loss)):
                print("\n[debug_nan] Non-finite loss detected", flush=True)
                print(f"epoch={epoch} step={step}", flush=True)
                print(f"rel_loss={rel_loss.item()} conf_loss={(conf_loss.item() if conf_loss is not None else None)}", flush=True)
                print(
                    f"scores finite={bool(torch.isfinite(out.scores).all())} min={out.scores.min().item()} max={out.scores.max().item()}",
                    flush=True,
                )
                if out.w_hat is not None:
                    print(
                        f"w_hat finite={bool(torch.isfinite(out.w_hat).all())} min={out.w_hat.min().item()} max={out.w_hat.max().item()}",
                        flush=True,
                    )

                for key, tensor in batch.items():
                    if torch.is_floating_point(tensor):
                        bad = int((~torch.isfinite(tensor)).sum().item())
                        if bad:
                            print(f"batch[{key}] has {bad} non-finite values", flush=True)

                # Recompute intermediates to localize
                with torch.no_grad():
                    c_h = model.context_encoder(
                        batch["head_edge_rel"],
                        batch["head_edge_dir"],
                        batch["head_edge_w"],
                        batch["head_edge_adj"],
                        batch["head_edge_mask"],
                    )
                    c_t = model.context_encoder(
                        batch["tail_edge_rel"],
                        batch["tail_edge_dir"],
                        batch["tail_edge_w"],
                        batch["tail_edge_adj"],
                        batch["tail_edge_mask"],
                    )
                    C = model.pair_fusion(c_h, c_t)
                    if model.use_relation_paths:
                        z = model.path_encoder(batch["paths_rel"], batch["paths_w"], batch["paths_len"], batch["paths_mask"])
                    else:
                        # Ablation (w/o RP): keep original line for reference.
                        # z = model.path_encoder(batch["paths_rel"], batch["paths_w"], batch["paths_len"], batch["paths_mask"])
                        z = torch.zeros((batch["paths_rel"].shape[0], batch["paths_rel"].shape[1], model.hidden_dim), device=device)

                    print(f"C finite={bool(torch.isfinite(C).all())} min={C.min().item()} max={C.max().item()}", flush=True)
                    print(f"z finite={bool(torch.isfinite(z).all())} min={z.min().item()} max={z.max().item()}", flush=True)

                    cand_ids = torch.arange(model.n_base_relations, device=device)
                    e_r = model.cand_rel_emb(cand_ids)
                    if model.use_relation_paths:
                        V, alpha = model.path_att(z=z, paths_mask=batch["paths_mask"], C=C, rel_emb_base=e_r)
                    else:
                        # Ablation (w/o RP): keep original attention line for reference.
                        # V, alpha = model.path_att(z=z, paths_mask=batch["paths_mask"], C=C, rel_emb_base=e_r)
                        V = torch.zeros((batch["paths_rel"].shape[0], model.n_base_relations, model.hidden_dim), device=device)
                        alpha = torch.zeros((batch["paths_rel"].shape[0], model.n_base_relations, batch["paths_rel"].shape[1]), device=device)

                    print(f"V finite={bool(torch.isfinite(V).all())} min={V.min().item()} max={V.max().item()}", flush=True)
                    print(f"alpha finite={bool(torch.isfinite(alpha).all())} min={alpha.min().item()} max={alpha.max().item()}", flush=True)

                raise RuntimeError("Non-finite loss detected (see debug_nan output above)")

            loss.backward()

            if args.debug_nan:
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        g = p.grad
                        bad = int((~torch.isfinite(g)).sum().item())
                        print("\n[debug_nan] Non-finite gradient detected", flush=True)
                        print(f"epoch={epoch} step={step} param={name} bad_grad_count={bad}", flush=True)
                        print(f"grad min={g.min().item()} max={g.max().item()}", flush=True)
                        raise RuntimeError("Non-finite gradient detected (see debug_nan output above)")
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            bs = batch["r"].shape[0]
            loss_meter.update(loss.item(), n=bs)
            rel_meter.update(rel_loss.item(), n=bs)
            if conf_loss is not None:
                conf_meter.update(conf_loss.item(), n=bs)

            if log_every and (step % log_every == 0 or step == n_steps):
                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                steps_per_sec = float(log_every) / dt if step != n_steps else 1.0 / dt
                elapsed = now - epoch_start
                log(
                    f"Epoch {epoch:03d} step {step:05d}/{n_steps:05d} "
                    f"| loss={loss_meter.avg:.4f} rel={rel_meter.avg:.4f} conf={conf_meter.avg:.4f} "
                    f"| {steps_per_sec:.2f} step/s elapsed={elapsed:.1f}s",
                )
                last_log_time = now

            if max_steps and step >= max_steps:
                log(f"Reached --max_steps={max_steps}; stopping early.")
                if log_fp is not None:
                    log_fp.close()
                return

        val_metrics, _, _ = evaluate(model, valid_loader, device=device, lambda_conf=lambda_conf)
        key = "mrr" if select_best_by == "mrr" else "accuracy"
        current = float(val_metrics[key])

        msg = (
            f"Epoch {epoch:03d} | loss={loss_meter.avg:.4f} rel={rel_meter.avg:.4f} conf={conf_meter.avg:.4f} | "
            f"val mrr={val_metrics['mrr']:.4f} h1={val_metrics['hits@1']:.4f} h3={val_metrics['hits@3']:.4f} h10={val_metrics['hits@10']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['macro_f1']:.4f}"
        )
        # Original behavior (kept for reference):
        # if use_conf_head and lambda_conf > 0 and ("mse" in val_metrics) and ("mae" in val_metrics):
        #     msg += f" mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f}"
        msg += f" mse={val_metrics.get('mse', 0.0):.4f} mae={val_metrics.get('mae', 0.0):.4f}"
        log(msg)

        if current > best_metric:
            best_metric = current
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": cfg,
                    "n_entities": n_entities,
                    "n_base_rel": n_base_rel,
                },
                best_path,
            )

    # final test with best checkpoint
    if best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            # Older PyTorch versions do not support weights_only.
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    test_metrics, _, _ = evaluate(model, test_loader, device=device, lambda_conf=lambda_conf)

    log("\nTest results")
    log(
        f"mrr={test_metrics['mrr']:.4f} h1={test_metrics['hits@1']:.4f} h3={test_metrics['hits@3']:.4f} h10={test_metrics['hits@10']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}"
    )
    # Original behavior (kept for reference):
    # if use_conf_head and lambda_conf > 0:
    #     print(f"mse={test_metrics.get('mse', 0.0):.4f} mae={test_metrics.get('mae', 0.0):.4f}", flush=True)
    log(f"mse={test_metrics.get('mse', 0.0):.4f} mae={test_metrics.get('mae', 0.0):.4f}")

    if log_fp is not None:
        log_fp.close()


if __name__ == "__main__":
    main()

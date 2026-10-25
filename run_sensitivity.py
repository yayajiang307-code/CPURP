#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DATASETS_ALL = ["CN15K", "NL27K", "PPI5K"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    csv_name: str
    config_key: str  # YAML path under model: <key>
    values: Sequence[float]


PARAM_SPECS = {
    "Lc": ParamSpec(
        name="Lc",
        csv_name="Lc",
        config_key="num_layers",
        values=[1, 2, 3, 4],
    ),
    "Lp": ParamSpec(
        name="Lp",
        csv_name="Lp",
        config_key="max_path_len",
        values=[1, 2, 3, 4],
    ),
    "B": ParamSpec(
        name="B",
        csv_name="B",
        config_key="max_paths",
        values=[5, 10, 20, 30, 50],
    ),
    "lambda_conf": ParamSpec(
        name="lambda_conf",
        csv_name="lambda_conf",
        config_key="lambda_conf",
        values=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    ),
}


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_values(values: Optional[str]) -> Optional[List[float]]:
    items = _parse_csv_list(values)
    if not items:
        return None
    out: List[float] = []
    for x in items:
        out.append(float(x))
    return out


def _now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _write_yaml_config(path: Path, seed: int, model_key: str, model_value: float, epochs: Optional[int], max_steps: Optional[int], log_every: Optional[int]) -> None:
    # Only write the single-variable override + a few training knobs for reproducibility.
    lines: List[str] = []
    lines.append(f"seed: {seed}")
    lines.append("")
    if epochs is not None or max_steps is not None or log_every is not None:
        lines.append("training:")
        if epochs is not None:
            lines.append(f"  epochs: {int(epochs)}")
        if max_steps is not None:
            lines.append(f"  max_steps: {int(max_steps)}")
        if log_every is not None:
            lines.append(f"  log_every: {int(log_every)}")
        lines.append("")

    lines.append("model:")
    if model_key in {"num_layers", "max_path_len", "max_paths"}:
        lines.append(f"  {model_key}: {int(model_value)}")
    else:
        # keep float formatting stable
        lines.append(f"  {model_key}: {float(model_value)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cmd: List[str], *, cwd: Path, stdout_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}\nSee log: {stdout_path}")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_metrics_json(json_path: Path) -> Tuple[float, float, float, float, float, float]:
    import json

    obj = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = obj.get("metrics") or {}
    mrr = float(metrics.get("mrr", 0.0))
    h1 = float(metrics.get("hits@1", 0.0))
    h3 = float(metrics.get("hits@3", 0.0))
    h10 = float(metrics.get("hits@10", 0.0))
    mse = float(metrics.get("mse", 0.0))
    mae = float(metrics.get("mae", 0.0))
    return mrr, h1, h3, h10, mse, mae


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _append_csv_row(csv_path: Path, row: List[object]) -> None:
    _ensure_parent(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["dataset", "parameter", "param_value", "MRR", "Hits@1", "Hits@3", "Hits@10", "MSE", "MAE"])
        w.writerow(row)


def _datasets_arg(s: str) -> List[str]:
    s = s.strip()
    if s.lower() in {"all", "*"}:
        return list(DATASETS_ALL)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPURP sensitivity experiments (single-variable control)")

    p.add_argument(
        "--datasets",
        type=_datasets_arg,
        default=list(DATASETS_ALL),
        help="Datasets to run, comma-separated (CN15K,NL27K,PPI5K) or 'all' (default: all)",
    )
    p.add_argument(
        "--parameter",
        type=str,
        default="all",
        choices=["all"] + sorted(PARAM_SPECS.keys()),
        help="Which parameter group to run (default: all)",
    )
    p.add_argument(
        "--values",
        type=str,
        default=None,
        help="Override parameter values, comma-separated (e.g., '1,2,3')",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value passed to train/evaluate")

    p.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Which split to record into CSV (default: valid). Note: train.py always runs a final test eval, but we record only the selected split.",
    )

    p.add_argument("--epochs", type=int, default=None, help="Override epochs (default: use configs/default.yaml)")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after N optimizer steps (debug/fast mode)")
    p.add_argument("--log_every", type=int, default=None, help="Log every N steps")

    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast debug mode: sets --epochs=2 and --max_steps=200 if not explicitly provided",
    )

    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: results/sensitivity_<split>.csv)",
    )
    p.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Where to store per-run artifacts (default: runs)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo = _repo_root()
    os.chdir(repo)

    if args.fast:
        if args.epochs is None:
            args.epochs = 2
        if args.max_steps is None:
            args.max_steps = 200
        if args.log_every is None:
            args.log_every = 50

    split = str(args.split)
    out_csv = Path(args.out_csv) if args.out_csv else (repo / "results" / f"sensitivity_{split}.csv")

    if args.parameter == "all":
        param_names = ["Lc", "Lp", "B", "lambda_conf"]
    else:
        param_names = [args.parameter]

    datasets: List[str] = list(args.datasets)

    # Basic validation
    unknown = [d for d in datasets if d not in DATASETS_ALL]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Expected one of {DATASETS_ALL}")

    # Run
    for dataset in datasets:
        for pname in param_names:
            spec = PARAM_SPECS[pname]
            override_values = _parse_values(args.values)
            values: Sequence[float] = override_values if override_values is not None else spec.values

            for v in values:
                ts = _now_ts()
                run_dir = repo / args.runs_dir / f"SENS_CPURP_{dataset}_{spec.csv_name}{v}_seed{args.seed}_{ts}"
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg_path = run_dir / "config.yaml"
                _write_yaml_config(
                    cfg_path,
                    seed=int(args.seed),
                    model_key=spec.config_key,
                    model_value=v,
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    log_every=args.log_every,
                )

                # Backup baseline checkpoint to avoid permanent overwrite.
                ckpt_dir = repo / "checkpoints" / dataset
                best_ckpt = ckpt_dir / "best.pt"
                backup_ckpt = run_dir / "_backup_best.pt"
                _copy_if_exists(best_ckpt, backup_ckpt)

                train_log = run_dir / "train.out"
                eval_log = run_dir / f"eval_{split}.out"
                metrics_json = run_dir / f"metrics_{split}.json"

                cmd_train = [
                    sys.executable,
                    "-u",
                    "train.py",
                    "--dataset",
                    dataset,
                    "--config",
                    str(cfg_path),
                    "--seed",
                    str(args.seed),
                ]
                if args.gpu is not None:
                    cmd_train += ["--gpu", str(args.gpu)]
                if args.epochs is not None:
                    cmd_train += ["--epochs", str(args.epochs)]
                if args.max_steps is not None:
                    cmd_train += ["--max_steps", str(args.max_steps)]
                if args.log_every is not None:
                    cmd_train += ["--log_every", str(args.log_every)]

                cmd_eval = [
                    sys.executable,
                    "-u",
                    "evaluate.py",
                    "--dataset",
                    dataset,
                    "--config",
                    str(cfg_path),
                    "--seed",
                    str(args.seed),
                    "--split",
                    split,
                    "--ckpt",
                    str(run_dir / "best.pt"),
                    "--save_json",
                    str(metrics_json),
                ]
                if args.gpu is not None:
                    cmd_eval += ["--gpu", str(args.gpu)]

                try:
                    _run(cmd_train, cwd=repo, stdout_path=train_log)

                    # Archive produced checkpoint.
                    if not best_ckpt.exists():
                        raise RuntimeError(f"Training finished but checkpoint not found: {best_ckpt}")
                    shutil.copy2(best_ckpt, run_dir / "best.pt")

                    _run(cmd_eval, cwd=repo, stdout_path=eval_log)

                    mrr, h1, h3, h10, mse, mae = _read_metrics_json(metrics_json)
                    _append_csv_row(
                        out_csv,
                        [
                            dataset,
                            spec.csv_name,
                            v,
                            f"{mrr:.6f}",
                            f"{h1:.6f}",
                            f"{h3:.6f}",
                            f"{h10:.6f}",
                            f"{mse:.6f}",
                            f"{mae:.6f}",
                        ],
                    )
                finally:
                    # Restore baseline checkpoint if it existed.
                    if backup_ckpt.exists():
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_ckpt, best_ckpt)


if __name__ == "__main__":
    main()

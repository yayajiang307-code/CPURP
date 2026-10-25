#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CPURP sensitivity curves from CSV")
    p.add_argument("--csv", type=str, required=True, help="CSV produced by run_sensitivity.py")
    p.add_argument("--out_dir", type=str, default="plots/sensitivity", help="Output directory")
    p.add_argument(
        "--mode",
        type=str,
        default="dual",
        choices=["dual", "split"],
        help="dual: twin y-axis (MRR+MSE, Hits@1+MAE); split: two separate plots per pair (default: dual)",
    )
    return p.parse_args()


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _group(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    g: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        g[(row["dataset"], row["parameter"])].append(row)
    return g


def _plot_dual(x: List[float], y_left: List[float], y_right: List[float], *, xlabel: str, left_label: str, right_label: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(x, y_left, marker="o", label=left_label)
    ax2.plot(x, y_right, marker="s", linestyle="--", label=right_label)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(left_label)
    ax2.set_ylabel(right_label)
    ax1.set_title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_single(x: List[float], y: List[float], *, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows = _read_rows(csv_path)
    groups = _group(rows)

    for (dataset, param), items in groups.items():
        items_sorted = sorted(items, key=lambda r: _to_float(r["param_value"]))

        x = [_to_float(r["param_value"]) for r in items_sorted]
        mrr = [_to_float(r["MRR"]) for r in items_sorted]
        h1 = [_to_float(r["Hits@1"]) for r in items_sorted]
        mse = [_to_float(r["MSE"]) for r in items_sorted]
        mae = [_to_float(r["MAE"]) for r in items_sorted]

        xlabel = f"{param}"

        if args.mode == "dual":
            _plot_dual(
                x,
                mrr,
                mse,
                xlabel=xlabel,
                left_label="MRR",
                right_label="MSE",
                title=f"{dataset} sensitivity: {param} (MRR vs MSE)",
                out_path=out_dir / f"{dataset}_{param}_mrr_mse.png",
            )
            _plot_dual(
                x,
                h1,
                mae,
                xlabel=xlabel,
                left_label="Hits@1",
                right_label="MAE",
                title=f"{dataset} sensitivity: {param} (Hits@1 vs MAE)",
                out_path=out_dir / f"{dataset}_{param}_hits1_mae.png",
            )
        else:
            _plot_single(
                x,
                mrr,
                xlabel=xlabel,
                ylabel="MRR",
                title=f"{dataset} sensitivity: {param} (MRR)",
                out_path=out_dir / f"{dataset}_{param}_mrr.png",
            )
            _plot_single(
                x,
                h1,
                xlabel=xlabel,
                ylabel="Hits@1",
                title=f"{dataset} sensitivity: {param} (Hits@1)",
                out_path=out_dir / f"{dataset}_{param}_hits1.png",
            )
            _plot_single(
                x,
                mse,
                xlabel=xlabel,
                ylabel="MSE",
                title=f"{dataset} sensitivity: {param} (MSE)",
                out_path=out_dir / f"{dataset}_{param}_mse.png",
            )
            _plot_single(
                x,
                mae,
                xlabel=xlabel,
                ylabel="MAE",
                title=f"{dataset} sensitivity: {param} (MAE)",
                out_path=out_dir / f"{dataset}_{param}_mae.png",
            )


if __name__ == "__main__":
    main()

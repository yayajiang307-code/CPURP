#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

OUT_CSV="results/sensitivity_valid_e10_rerun_20260602_180408.csv"

python -u run_sensitivity.py \
  --datasets PPI5K \
  --parameter Lp \
  --values 4 \
  --gpu 0 \
  --epochs 10 \
  --split valid \
  --out_csv "${OUT_CSV}"

python -u run_sensitivity.py \
  --datasets PPI5K \
  --parameter B \
  --gpu 0 \
  --epochs 10 \
  --split valid \
  --out_csv "${OUT_CSV}"

python -u run_sensitivity.py \
  --datasets PPI5K \
  --parameter lambda_conf \
  --gpu 0 \
  --epochs 10 \
  --split valid \
  --out_csv "${OUT_CSV}"

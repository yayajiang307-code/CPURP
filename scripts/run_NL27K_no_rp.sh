#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="NL27K"
GPU_ID="0"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs

# Backup existing checkpoint to avoid overwriting prior results
if [[ -f "checkpoints/${DATASET}/best.pt" ]]; then
  cp -f "checkpoints/${DATASET}/best.pt" "checkpoints/${DATASET}/best.prev_${TS}.pt"
fi

TRAIN_LOG="logs/train_${DATASET}_no_rp_${TS}.out"
VALID_LOG="logs/eval_${DATASET}_no_rp_valid_${TS}.out"
TEST_LOG="logs/eval_${DATASET}_no_rp_test_${TS}.out"

# Train (w/o RP only; keep RC)
python -u train.py --dataset "${DATASET}" --gpu "${GPU_ID}" --no_rp --epochs 20 >"${TRAIN_LOG}" 2>&1

# Evaluate valid/test with the newly produced checkpoint
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --no_rp --split valid >"${VALID_LOG}" 2>&1
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --no_rp --split test >"${TEST_LOG}" 2>&1

echo "DONE ${DATASET} no_rp TS=${TS}" | tee -a "${TRAIN_LOG}"

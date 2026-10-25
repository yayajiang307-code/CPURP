#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="PPI5K"
GPU_ID="${GPU_ID:-0}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs

# Backup existing checkpoint to avoid overwriting prior results
if [[ -f "checkpoints/${DATASET}/best.pt" ]]; then
  cp -f "checkpoints/${DATASET}/best.pt" "checkpoints/${DATASET}/best.prev_${TS}.pt"
fi

RESULT_JSON="results_ablation_wo_unc_${DATASET}.json"
if [[ -f "${RESULT_JSON}" ]]; then
  cp -f "${RESULT_JSON}" "${RESULT_JSON%.json}.prev_${TS}.json"
fi

TRAIN_LOG="logs/train_${DATASET}_wo_unc_${TS}.out"
VALID_LOG="logs/eval_${DATASET}_wo_unc_valid_${TS}.out"
TEST_LOG="logs/eval_${DATASET}_wo_unc_test_${TS}.out"

# Train (Full modules; w/o Unc evidence only)
# Original behavior (kept for reference):
# python -u train.py --dataset "${DATASET}" --gpu "${GPU_ID}" --ablation wo_unc >"${TRAIN_LOG}" 2>&1
python -u train.py --dataset "${DATASET}" --gpu "${GPU_ID}" --ablation wo_unc --epochs 20 >"${TRAIN_LOG}" 2>&1

# Evaluate valid/test with the newly produced checkpoint
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --ablation wo_unc --split valid >"${VALID_LOG}" 2>&1
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --ablation wo_unc --split test --save_json "${RESULT_JSON}" >"${TEST_LOG}" 2>&1

echo "DONE ${DATASET} wo_unc TS=${TS}" | tee -a "${TRAIN_LOG}"

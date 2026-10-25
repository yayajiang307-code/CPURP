#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="CN15K"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs

RUN_DIR="runs/CPURP_woGQ_${DATASET}_seed${SEED}"
mkdir -p "${RUN_DIR}"

# Backup existing run checkpoint / json to avoid overwriting prior results
if [[ -f "${RUN_DIR}/best.pt" ]]; then
  cp -f "${RUN_DIR}/best.pt" "${RUN_DIR}/best.prev_${TS}.pt"
fi

RESULT_JSON="${RUN_DIR}/results_${DATASET}_wo_gq_test.json"
if [[ -f "${RESULT_JSON}" ]]; then
  cp -f "${RESULT_JSON}" "${RESULT_JSON%.json}.prev_${TS}.json"
fi

TRAIN_LOG="logs/train_${DATASET}_wo_gq_${TS}.out"
VALID_LOG="logs/eval_${DATASET}_wo_gq_valid_${TS}.out"
TEST_LOG="logs/eval_${DATASET}_wo_gq_test_${TS}.out"

# Train (Full modules kept; only disable Gate & Query-aware path attention)
python -u train.py --dataset "${DATASET}" --gpu "${GPU_ID}" --seed "${SEED}" --ablation wo_gq --epochs "${EPOCHS}" >"${TRAIN_LOG}" 2>&1

# Evaluate valid/test with the newly produced checkpoint (auto-resolve from runs/)
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --seed "${SEED}" --ablation wo_gq --split valid >"${VALID_LOG}" 2>&1
python -u evaluate.py --dataset "${DATASET}" --gpu "${GPU_ID}" --seed "${SEED}" --ablation wo_gq --split test --save_json "${RESULT_JSON}" >"${TEST_LOG}" 2>&1

echo "DONE ${DATASET} wo_gq seed=${SEED} epochs=${EPOCHS} TS=${TS}" | tee -a "${TRAIN_LOG}"

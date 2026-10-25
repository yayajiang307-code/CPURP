#!/usr/bin/env bash
set -euo pipefail

# Minimal sensitivity experiment runner.
#
# Usage (examples):
#   TAG=layers3 NUM_LAYERS=3 DATASET=CN15K GPU_ID=0 SEED=42 EPOCHS=20 bash scripts/run_sensitivity.sh
#   TAG=mpl6 MAX_PATH_LEN=6 MAX_PATHS=32 LAMBDA_CONF=0.5 bash scripts/run_sensitivity.sh
#
# Notes:
# - Does NOT modify code; only uses a per-run YAML config.
# - Archives logs + checkpoint + metrics into a unique runs/SENS_* directory.
# - Backs up the underlying best.pt that train.py would overwrite.

cd "$(dirname "$0")/.."

DATASET="${DATASET:-CN15K}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
TAG="${TAG:-}"
ABLATION="${ABLATION:-}"

MAX_STEPS="${MAX_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"

NUM_LAYERS="${NUM_LAYERS:-}"
MAX_HOP="${MAX_HOP:-}"
MAX_PATH_LEN="${MAX_PATH_LEN:-}"
MAX_PATHS="${MAX_PATHS:-}"
LAMBDA_CONF="${LAMBDA_CONF:-}"

if [[ -z "${TAG}" ]]; then
	echo "ERROR: TAG is required (e.g., TAG=layers3)" >&2
	exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs runs

RUN_DIR="runs/SENS_${DATASET}_${TAG}_seed${SEED}_${TS}"
mkdir -p "${RUN_DIR}"

SENS_CFG="${RUN_DIR}/config.yaml"

{
	echo "seed: ${SEED}"
	echo
	if [[ -n "${ABLATION}" ]]; then
		echo "ablation: ${ABLATION}"
		echo
	fi
	echo "training:"
	echo "  epochs: ${EPOCHS}"
	echo
	echo "model:"
	if [[ -n "${NUM_LAYERS}" ]]; then
		echo "  num_layers: ${NUM_LAYERS}"
	fi
	if [[ -n "${MAX_HOP}" ]]; then
		echo "  max_hop: ${MAX_HOP}"
	fi
	if [[ -n "${MAX_PATH_LEN}" ]]; then
		echo "  max_path_len: ${MAX_PATH_LEN}"
	fi
	if [[ -n "${MAX_PATHS}" ]]; then
		echo "  max_paths: ${MAX_PATHS}"
	fi
	if [[ -n "${LAMBDA_CONF}" ]]; then
		echo "  lambda_conf: ${LAMBDA_CONF}"
	fi
} >"${SENS_CFG}"

# Determine where train.py will write best.pt, and back it up.
if [[ "${ABLATION}" == "wo_gq" ]]; then
	BASE_CKPT_DIR="runs/CPURP_woGQ_${DATASET}_seed${SEED}"
else
	BASE_CKPT_DIR="checkpoints/${DATASET}"
fi
mkdir -p "${BASE_CKPT_DIR}"

if [[ -f "${BASE_CKPT_DIR}/best.pt" ]]; then
	cp -f "${BASE_CKPT_DIR}/best.pt" "${BASE_CKPT_DIR}/best.prev_${TS}.pt"
fi

TRAIN_LOG="logs/train_${DATASET}_${TAG}_seed${SEED}_${TS}.out"
VALID_LOG="logs/eval_${DATASET}_${TAG}_valid_seed${SEED}_${TS}.out"
TEST_LOG="logs/eval_${DATASET}_${TAG}_test_seed${SEED}_${TS}.out"

echo "[SENS] RUN_DIR=${RUN_DIR}"
echo "[SENS] CONFIG=${SENS_CFG}"
echo "[SENS] BASE_CKPT_DIR=${BASE_CKPT_DIR}"
echo "[SENS] TRAIN_LOG=${TRAIN_LOG}"
echo "[SENS] VALID_LOG=${VALID_LOG}"
echo "[SENS] TEST_LOG=${TEST_LOG}"

# Train
echo "[SENS] Training... (output redirected to ${TRAIN_LOG})"
python -u train.py \
	--dataset "${DATASET}" \
	--gpu "${GPU_ID}" \
	--seed "${SEED}" \
	--epochs "${EPOCHS}" \
	--config "${SENS_CFG}" \
	${MAX_STEPS:+--max_steps "${MAX_STEPS}"} \
	${LOG_EVERY:+--log_every "${LOG_EVERY}"} \
	${ABLATION:+--ablation "${ABLATION}"} \
	>"${TRAIN_LOG}" 2>&1

if [[ ! -f "${BASE_CKPT_DIR}/best.pt" ]]; then
	echo "ERROR: training finished but checkpoint not found: ${BASE_CKPT_DIR}/best.pt" >&2
	exit 3
fi

# Archive checkpoint + config + logs
echo "[SENS] Archiving checkpoint + logs into ${RUN_DIR}"
cp -f "${BASE_CKPT_DIR}/best.pt" "${RUN_DIR}/best.pt"
cp -f "${TRAIN_LOG}" "${RUN_DIR}/train.out"

# Evaluate valid/test using the archived checkpoint and the same config
echo "[SENS] Evaluating valid... (output redirected to ${VALID_LOG})"
python -u evaluate.py \
	--dataset "${DATASET}" \
	--gpu "${GPU_ID}" \
	--seed "${SEED}" \
	--config "${SENS_CFG}" \
	${ABLATION:+--ablation "${ABLATION}"} \
	--ckpt "${RUN_DIR}/best.pt" \
	--split valid \
	--save_json "${RUN_DIR}/results_valid.json" \
	>"${VALID_LOG}" 2>&1

echo "[SENS] Evaluating test... (output redirected to ${TEST_LOG})"
python -u evaluate.py \
	--dataset "${DATASET}" \
	--gpu "${GPU_ID}" \
	--seed "${SEED}" \
	--config "${SENS_CFG}" \
	${ABLATION:+--ablation "${ABLATION}"} \
	--ckpt "${RUN_DIR}/best.pt" \
	--split test \
	--save_json "${RUN_DIR}/results_test.json" \
	>"${TEST_LOG}" 2>&1

cp -f "${VALID_LOG}" "${RUN_DIR}/eval_valid.out"
cp -f "${TEST_LOG}" "${RUN_DIR}/eval_test.out"

echo "DONE DATASET=${DATASET} TAG=${TAG} seed=${SEED} epochs=${EPOCHS} RUN_DIR=${RUN_DIR}"
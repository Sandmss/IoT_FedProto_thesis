#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

NUM_CLIENTS="${NUM_CLIENTS:-20}"
DEFAULT_CLIENT_RATIO="5:5"
CLIENT_RATIO="${CLIENT_RATIO:-$DEFAULT_CLIENT_RATIO}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-1000}"
SERVER_EPOCHS="${SERVER_EPOCHS:-100}"
EVAL_GAP="${EVAL_GAP:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-100}"
IFS=':' read -r MLP_RATIO CNN_RATIO <<< "$CLIENT_RATIO"
RATIO_SUM=$((MLP_RATIO + CNN_RATIO))
if (( RATIO_SUM <= 0 )); then
  echo "Invalid CLIENT_RATIO '$CLIENT_RATIO': sum must be positive." >&2
  exit 1
fi
if (( NUM_CLIENTS % RATIO_SUM != 0 )); then
  echo "Invalid CLIENT_RATIO '$CLIENT_RATIO' for NUM_CLIENTS=$NUM_CLIENTS: cannot convert ratio to integer client counts." >&2
  exit 1
fi
SCALE=$((NUM_CLIENTS / RATIO_SUM))
CLIENT_MODEL_RATIOS="$((MLP_RATIO * SCALE)):$((CNN_RATIO * SCALE))"
RATIO_TAG="${CLIENT_RATIO//:/_}"
RATIO_DIR_TAG="${CLIENT_RATIO//:/}"
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models${RATIO_DIR_TAG}/LGFedAvg}"
GOAL="${GOAL:-mix_mlp_cnn1d_ratio_${RATIO_TAG}}"
mkdir -p "$RESULT_DIR/logs"

python -u main.py \
  -t 1 \
  -lr 0.01 \
  -jr 1 \
  -lbs 10 \
  -ls "$LOCAL_EPOCHS" \
  -gr "$GLOBAL_ROUNDS" \
  -eg "$EVAL_GAP" \
  -nw "$NUM_WORKERS" \
  -nc "$NUM_CLIENTS" \
  -nb 15 \
  -dataset IoT \
  -model_family IoT_MIX_MLP_CNN1D \
  --client_model_ratios "$CLIENT_MODEL_RATIOS" \
  --input_dim 77 \
  -fd 64 \
  -did 0 \
  -algo LGFedAvg \
  -lam 1.0 \
  --lg_shared_param_prefixes head. \
  --algorithm_result_dir "$RESULT_DIR" \
  -go "$GOAL" \
  -se "$SERVER_EPOCHS" \
  -mart 100 \
  -ab True \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  > "$RESULT_DIR/logs/iot_mix_mlp_cnn1d_lgfedavg_${RATIO_TAG}.out" 2>&1

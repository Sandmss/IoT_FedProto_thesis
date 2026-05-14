#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

NUM_CLIENTS="${NUM_CLIENTS:-20}"
DEFAULT_CLIENT_RATIO="5:5"
CLIENT_RATIO="${CLIENT_RATIO:-$DEFAULT_CLIENT_RATIO}"
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
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models${RATIO_DIR_TAG}/FML}"
GOAL="${GOAL:-mix_mlp_cnn1d_ratio_${RATIO_TAG}}"
mkdir -p "$RESULT_DIR/logs"

python -u main.py \
  -t 1 \
  -lr 0.01 \
  -jr 1 \
  -lbs 10 \
  -ls 1 \
  -gr 1000 \
  -eg 1 \
  -nw 4 \
  -nc "$NUM_CLIENTS" \
  -nb 15 \
  -dataset IoT \
  -model_family IoT_MIX_MLP_CNN1D \
  --client_model_ratios "$CLIENT_MODEL_RATIOS" \
  --input_dim 77 \
  -fd 64 \
  -did 0 \
  -algo FML \
  --fml_alpha 0.5 \
  --fml_beta 0.5 \
  --fml_temperature 1.0 \
  --algorithm_result_dir "$RESULT_DIR" \
  -go "$GOAL" \
  -se 100 \
  -mart 100 \
  -ab True \
  --early_stop_patience 100 \
  > "$RESULT_DIR/logs/iot_mix_mlp_cnn1d_fml_${RATIO_TAG}.out" 2>&1

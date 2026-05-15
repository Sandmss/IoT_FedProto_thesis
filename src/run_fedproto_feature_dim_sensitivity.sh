#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="${MODEL_FAMILY:-IoT_MIX_MLP_CNN1D}"
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models/FedProto}"
GOAL_PREFIX="${GOAL_PREFIX:-feature_dim_sensitivity_${MODEL_FAMILY}}"
FEATURE_DIMS=(${FEATURE_DIMS:-128 256 512 1024})
LAMDA="${LAMDA:-1.0}"

mkdir -p "$RESULT_DIR/logs"

for FD in "${FEATURE_DIMS[@]}"; do
  GOAL="${GOAL_PREFIX}_fd_${FD}"
  LOG_FILE="$RESULT_DIR/logs/${GOAL}.out"

  echo "Running FedProto feature-dimension sensitivity with feature_dim=${FD}, lamda=${LAMDA} ..."
  python -u main.py \
    -t 1 \
    -lr 0.01 \
    -jr 1 \
    -lbs 10 \
    -ls 1 \
    -gr 1000 \
    -eg 1 \
    -nw 4 \
    -nc 20 \
    -nb 15 \
    -dataset IoT \
    -model_family "$MODEL_FAMILY" \
    --input_dim 77 \
    -fd "$FD" \
    -did 0 \
    -algo FedProto \
    -lam "$LAMDA" \
    --proto_eval_mode classifier \
    --algorithm_result_dir "$RESULT_DIR" \
    -go "$GOAL" \
    -se 100 \
    -mart 100 \
    -ab True \
    --early_stop_patience 100 \
    --skip_figures \
    > "$LOG_FILE" 2>&1
done

echo "Feature-dimension sensitivity sweep finished."

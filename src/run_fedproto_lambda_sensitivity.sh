#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODEL_FAMILY="${MODEL_FAMILY:-IoT_MIX_MLP_CNN1D}"
FEATURE_DIM="${FEATURE_DIM:-512}"
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models/FedProto}"
GOAL_PREFIX="${GOAL_PREFIX:-lambda_sensitivity_${MODEL_FAMILY}}"
LAMBDAS=(${LAMBDAS:-0 0.25 0.5 1 2 3 4})

mkdir -p "$RESULT_DIR/logs"

for LAM in "${LAMBDAS[@]}"; do
  LAM_TAG="${LAM//./p}"
  GOAL="${GOAL_PREFIX}_lam_${LAM_TAG}"
  LOG_FILE="$RESULT_DIR/logs/${GOAL}.out"

  echo "Running FedProto lambda sensitivity with lamda=${LAM} ..."
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
    -fd "$FEATURE_DIM" \
    -did 0 \
    -algo FedProto \
    -lam "$LAM" \
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

echo "Lambda sensitivity sweep finished."

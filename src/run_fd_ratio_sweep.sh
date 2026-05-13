#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RATIOS=(${RATIOS:-"5:5" "4:6" "3:7"})
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models/FD}"
GOAL_PREFIX="${GOAL_PREFIX:-hetero_ratio_fd}"

mkdir -p "$RESULT_DIR/logs"

for RATIO in "${RATIOS[@]}"; do
  RATIO_TAG="${RATIO//:/_}"
  GOAL="${GOAL_PREFIX}_${RATIO_TAG}"

  echo "Running FD heterogeneous ratio experiment with MLP:CNN=${RATIO} ..."
  CLIENT_MODEL_RATIOS="$RATIO" GOAL="$GOAL" RESULT_DIR="$RESULT_DIR" \
    bash ./run_fd_mix_mlp_cnn1d.sh
done

echo "FD heterogeneous ratio sweep finished."

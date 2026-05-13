#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RATIOS_STR="${RATIOS:-5:5 4:6 3:7}"
read -r -a RATIOS <<< "$RATIOS_STR"
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models/FML}"
GOAL_PREFIX="${GOAL_PREFIX:-hetero_ratio_fml}"

mkdir -p "$RESULT_DIR/logs"

for RATIO in "${RATIOS[@]}"; do
  RATIO_TAG="${RATIO//:/_}"
  GOAL="${GOAL_PREFIX}_${RATIO_TAG}"

  echo "Running FML heterogeneous ratio experiment with MLP:CNN=${RATIO} ..."
  CLIENT_MODEL_RATIOS="$RATIO" GOAL="$GOAL" RESULT_DIR="$RESULT_DIR" \
    bash ./run_fml_mix_mlp_cnn1d.sh
done

echo "FML heterogeneous ratio sweep finished."

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RATIOS_STR="${RATIOS:-5:5 4:6 3:7}"
read -r -a RATIOS <<< "$RATIOS_STR"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
RESULT_DIR="${RESULT_DIR:-../results/heterogeneous_models/FedProto}"
GOAL_PREFIX="${GOAL_PREFIX:-hetero_ratio_fedproto}"

mkdir -p "$RESULT_DIR/logs"

for RATIO in "${RATIOS[@]}"; do
  IFS=':' read -r MLP_RATIO CNN_RATIO <<< "$RATIO"
  RATIO_SUM=$((MLP_RATIO + CNN_RATIO))
  if (( RATIO_SUM <= 0 )); then
    echo "Invalid ratio '$RATIO': sum must be positive." >&2
    exit 1
  fi
  if (( NUM_CLIENTS % RATIO_SUM != 0 )); then
    echo "Invalid ratio '$RATIO' for NUM_CLIENTS=$NUM_CLIENTS: cannot convert ratio to integer client counts." >&2
    exit 1
  fi
  SCALE=$((NUM_CLIENTS / RATIO_SUM))
  CLIENT_COUNTS="$((MLP_RATIO * SCALE)):$((CNN_RATIO * SCALE))"
  RATIO_TAG="${RATIO//:/_}"
  GOAL="${GOAL_PREFIX}_${RATIO_TAG}"

  echo "Running FedProto heterogeneous ratio experiment with MLP:CNN=${RATIO} ..."
  CLIENT_MODEL_RATIOS="$CLIENT_COUNTS" GOAL="$GOAL" RESULT_DIR="$RESULT_DIR" \
    bash ./run_fedproto_mix_mlp_cnn1d.sh
done

echo "FedProto heterogeneous ratio sweep finished."

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RATIOS="${RATIOS:-"5:5 4:6 3:7"}"

RATIOS="$RATIOS" bash ./run_fedproto_ratio_sweep.sh
RATIOS="$RATIOS" bash ./run_fd_ratio_sweep.sh
RATIOS="$RATIOS" bash ./run_lgfedavg_ratio_sweep.sh
RATIOS="$RATIOS" bash ./run_fml_ratio_sweep.sh

echo "All heterogeneous ratio sweeps finished."

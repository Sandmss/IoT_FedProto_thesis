#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Output log file name: iot_transformer_local.out  (local_epochs=1000 with auto break patience=100)
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1000 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_Transformer1D --input_dim 77 -fd 512 -did 0 -algo Local --normal_class 0 -lam 0.1 -se 100 -mart 100 -ab True --early_stop_patience 100 > iot_transformer_local.out 2>&1

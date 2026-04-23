#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Output log file name: iot_transformer_fedavg.out  (global_rounds=1000 with auto break patience=100)
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_Transformer1D --input_dim 77 -fd 512 -did 0 -algo FedAvg -lam 0.1 -se 100 -mart 100 -ab True --early_stop_patience 100 --skip_figures > iot_transformer_fedavg.out 2>&1

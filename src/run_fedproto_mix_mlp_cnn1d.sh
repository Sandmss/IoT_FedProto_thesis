#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RESULT_DIR="../results/heterogeneous_models/FedProto"
mkdir -p "$RESULT_DIR/logs"

# Output log file name: iot_mix_mlp_cnn1d_fedproto.out  (global_rounds=1000 with auto break patience=100)
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100 > "$RESULT_DIR/logs/iot_mix_mlp_cnn1d_fedproto.out" 2>&1


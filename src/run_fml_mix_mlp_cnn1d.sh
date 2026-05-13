#!/usr/bin/env bash

set -euo pipefail

RESULT_DIR="../results/heterogeneous_models/FML"
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
  -nc 20 \
  -nb 15 \
  -dataset IoT \
  -model_family IoT_MIX_MLP_CNN1D \
  --input_dim 77 \
  -fd 64 \
  -did 0 \
  -algo FML \
  --fml_alpha 0.5 \
  --fml_beta 0.5 \
  --fml_temperature 1.0 \
  -se 100 \
  -mart 100 \
  -ab True \
  --early_stop_patience 100 \
  > "$RESULT_DIR/logs/iot_mix_mlp_cnn1d_fml.out" 2>&1


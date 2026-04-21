#!/usr/bin/env bash
set -euo pipefail

mkdir -p ../results/第九次结果

COMMON_ARGS="-t 1 -lr 0.01 -jr 1 -lbs 10 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT_k40_c20_noniid --input_dim 77 -fd 512 -did 0 -lam 1.0 -se 100 -mart 100 -ab True --skip_figures"

for MODEL in IoT_MLP IoT_CNN1D IoT_Transformer1D; do
  MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')

  python -u main.py $COMMON_ARGS -ls 1 -gr 1000 -algo FedProto -model_family "$MODEL" --proto_eval_mode classifier \
    > "../results/第九次结果/iot_k40_c20_noniid_${MODEL_TAG}_fedproto.out" 2>&1

  python -u main.py $COMMON_ARGS -ls 1 -gr 1000 -algo FedAvg -model_family "$MODEL" \
    > "../results/第九次结果/iot_k40_c20_noniid_${MODEL_TAG}_fedavg.out" 2>&1

  python -u main.py $COMMON_ARGS -ls 1000 -gr 100 -algo Local -model_family "$MODEL" \
    > "../results/第九次结果/iot_k40_c20_noniid_${MODEL_TAG}_local.out" 2>&1
done

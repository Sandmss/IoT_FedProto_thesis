#!/usr/bin/env bash
set -euo pipefail

python scripts/data_preprocess_sampling_style.py \
  --data-dir data/raw_data \
  --output-dir data/processed_data_k40_c20_noniid \
  --num-clients 20 \
  --classes-per-client 6 \
  --k-per-class 40 \
  --target-total 20000 \
  --train-ratio 0.75 \
  --seed 42

python scripts/repack_processed_to_dataset_layout.py \
  --input-dir data/processed_data_k40_c20_noniid \
  --output-root dataset/IoT_k40_c20_noniid \
  --num-clients 20

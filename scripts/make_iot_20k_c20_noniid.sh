#!/usr/bin/env bash
set -euo pipefail

# Generate IoT_20k_c20_noniid with classes_per_client n=4.
python scripts/data_preprocess_sampling_style.py --data-dir data/raw_data --output-dir data/processed_data_20k_c20_noniid --num-clients 20 --classes-per-client 3 --k-per-class 100 --target-total 20000 --train-ratio 0.75 --seed 42

python scripts/repack_processed_to_dataset_layout.py --input-dir data/processed_data_20k_c20_noniid --output-root dataset/IoT_20k_c20_noniid --num-clients 20

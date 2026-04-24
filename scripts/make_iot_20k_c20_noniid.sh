#!/usr/bin/env bash
set -euo pipefail

# Generate dataset/IoT from data/processed_data with 20k selected samples over 20 clients.
python scripts/data_preprocess.py --data-dir data/raw_data --output-dir data/processed_data --num-clients 20 --classes-per-client 5 --target-total 20000 --train-ratio 0.75 --seed 42 --consume-all-target-samples

python scripts/repack_to_dataset.py --input-dir data/processed_data --output-root dataset/IoT

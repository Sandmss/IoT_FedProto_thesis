#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Report IoT backbone parameter counts.")
    parser.add_argument("--input-dim", type=int, default=77)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)
    return parser.parse_args()


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from flcore.trainmodel.models import CNN1D_IoT, MLP_IoT, Transformer1D_IoT

    models = {
        "IoT_MLP": MLP_IoT(
            dim_in=args.input_dim,
            dim_hidden=128,
            dim_out=args.feature_dim,
            num_classes=args.num_classes,
        ),
        "IoT_CNN1D": CNN1D_IoT(
            dim_in=args.input_dim,
            dim_out=args.feature_dim,
            num_classes=args.num_classes,
        ),
        "IoT_Transformer1D": Transformer1D_IoT(
            dim_in=args.input_dim,
            dim_out=args.feature_dim,
            num_classes=args.num_classes,
            d_model=args.transformer_d_model,
            num_heads=args.transformer_num_heads,
            num_layers=args.transformer_num_layers,
            dropout=args.transformer_dropout,
        ),
    }

    print("model,total_params,trainable_params")
    for name, model in models.items():
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"{name},{count_params(model)},{trainable}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report IoT model size and paper-style communication cost."
    )
    parser.add_argument("--dataset", type=str, default="IoT_20k_c20_noniid")
    parser.add_argument("--model-family", type=str, default="IoT_MLP")
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--input-dim", type=int, default=77)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--bytes-per-param", type=int, default=4)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def model_size_bytes(model):
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def estimate_flops(model, input_dim):
    was_training = model.training
    model.eval()
    flops = {"value": 0.0}
    handles = []

    def linear_hook(module, inputs, output):
        output_elements = output.numel()
        flops["value"] += output_elements * module.in_features
        if module.bias is not None:
            flops["value"] += output_elements

    def conv1d_hook(module, inputs, output):
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_length = output.shape[2]
        kernel_ops = module.kernel_size[0] * (module.in_channels / module.groups)
        bias_ops = 1 if module.bias is not None else 0
        flops["value"] += batch * out_channels * out_length * (kernel_ops + bias_ops)

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, torch.nn.Conv1d):
            handles.append(module.register_forward_hook(conv1d_hook))

    try:
        with torch.no_grad():
            model(torch.zeros(1, input_dim))
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    return float(flops["value"])


def build_model_counts(args):
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
    return {
        name: {
            "params": count_params(model),
            "size_bytes": model_size_bytes(model),
            "flops": estimate_flops(model, args.input_dim),
        }
        for name, model in models.items()
    }


def load_client_class_counts(dataset_name):
    repo_root = Path(__file__).resolve().parents[1]
    train_root = repo_root / "dataset" / dataset_name / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Dataset train directory not found: {train_root}")

    counts = []
    for client_dir in sorted(train_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        y_path = client_dir / "y.npy"
        if not y_path.is_file():
            continue
        y = np.load(y_path)
        counts.append(int(len(np.unique(y))))

    if not counts:
        raise FileNotFoundError(f"No client y.npy files found under: {train_root}")
    return counts


def resolve_client_model_counts(model_family, num_clients, model_counts):
    if model_family == "IoT_MIX_MLP_CNN_TRANS":
        order = ["IoT_MLP", "IoT_CNN1D", "IoT_Transformer1D"]
        return [model_counts[order[cid % len(order)]] for cid in range(num_clients)]
    if model_family not in model_counts:
        raise ValueError(
            "Unsupported model family. Use IoT_MLP, IoT_CNN1D, "
            "IoT_Transformer1D, or IoT_MIX_MLP_CNN_TRANS."
        )
    return [model_counts[model_family]] * num_clients


def main():
    args = parse_args()
    model_counts = build_model_counts(args)
    class_counts = load_client_class_counts(args.dataset)
    num_clients = args.num_clients or len(class_counts)
    class_counts = class_counts[:num_clients]
    active_clients = max(1, int(num_clients * args.join_ratio))

    client_model_stats = resolve_client_model_counts(args.model_family, num_clients, model_counts)
    client_model_counts = [item["params"] for item in client_model_stats]
    client_model_sizes = [item["size_bytes"] for item in client_model_stats]
    client_model_flops = [item["flops"] for item in client_model_stats]

    # FedProto paper Table 1 style: per-round communicated parameters.
    # FedAvg counts uploaded model parameters from participating clients.
    # FedProto counts uploaded local class prototypes from participating clients.
    fedavg_params_per_round = int(sum(client_model_counts[:active_clients]))
    fedproto_params_per_round = int(sum(class_counts[:active_clients]) * args.feature_dim)

    rows = [
        {
            "method": "Local",
            "model_family": args.model_family,
            "model_params_mean": float(np.mean(client_model_counts)),
            "model_params_min": int(np.min(client_model_counts)),
            "model_params_max": int(np.max(client_model_counts)),
            "model_size_bytes_mean": float(np.mean(client_model_sizes)),
            "model_size_bytes_min": int(np.min(client_model_sizes)),
            "model_size_bytes_max": int(np.max(client_model_sizes)),
            "estimated_flops_mean": float(np.mean(client_model_flops)),
            "estimated_flops_min": int(np.min(client_model_flops)),
            "estimated_flops_max": int(np.max(client_model_flops)),
            "avg_classes_per_client": float(np.mean(class_counts)),
            "comm_params_per_round": 0,
            "comm_params_total": 0,
            "comm_bytes_per_round": 0,
            "comm_bytes_total": 0,
        },
        {
            "method": "FedAvg",
            "model_family": args.model_family,
            "model_params_mean": float(np.mean(client_model_counts)),
            "model_params_min": int(np.min(client_model_counts)),
            "model_params_max": int(np.max(client_model_counts)),
            "model_size_bytes_mean": float(np.mean(client_model_sizes)),
            "model_size_bytes_min": int(np.min(client_model_sizes)),
            "model_size_bytes_max": int(np.max(client_model_sizes)),
            "estimated_flops_mean": float(np.mean(client_model_flops)),
            "estimated_flops_min": int(np.min(client_model_flops)),
            "estimated_flops_max": int(np.max(client_model_flops)),
            "avg_classes_per_client": float(np.mean(class_counts)),
            "comm_params_per_round": fedavg_params_per_round,
            "comm_params_total": fedavg_params_per_round * args.global_rounds,
            "comm_bytes_per_round": fedavg_params_per_round * args.bytes_per_param,
            "comm_bytes_total": fedavg_params_per_round * args.global_rounds * args.bytes_per_param,
        },
        {
            "method": "FedProto",
            "model_family": args.model_family,
            "model_params_mean": float(np.mean(client_model_counts)),
            "model_params_min": int(np.min(client_model_counts)),
            "model_params_max": int(np.max(client_model_counts)),
            "model_size_bytes_mean": float(np.mean(client_model_sizes)),
            "model_size_bytes_min": int(np.min(client_model_sizes)),
            "model_size_bytes_max": int(np.max(client_model_sizes)),
            "estimated_flops_mean": float(np.mean(client_model_flops)),
            "estimated_flops_min": int(np.min(client_model_flops)),
            "estimated_flops_max": int(np.max(client_model_flops)),
            "avg_classes_per_client": float(np.mean(class_counts)),
            "comm_params_per_round": fedproto_params_per_round,
            "comm_params_total": fedproto_params_per_round * args.global_rounds,
            "comm_bytes_per_round": fedproto_params_per_round * args.bytes_per_param,
            "comm_bytes_total": fedproto_params_per_round * args.global_rounds * args.bytes_per_param,
        },
    ]

    fieldnames = list(rows[0].keys())
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import contextlib
import copy
import io
import json
import os
import sys

import numpy as np
import torch

from main import build_parser, build_server, configure_stdio, resolve_models
from flcore.clients.clientbase import load_item


def build_args(cli_args):
    args = build_parser().parse_args([])
    args.dataset = cli_args.dataset
    args.algorithm = cli_args.algorithm
    args.model_family = cli_args.model_family
    args.device = cli_args.device
    args.device_id = cli_args.device_id
    args.num_clients = cli_args.num_clients
    args.join_ratio = 1.0
    args.times = 1
    args.goal = cli_args.goal
    args.input_dim = cli_args.input_dim
    args.num_classes = cli_args.num_classes
    args.normal_class = cli_args.normal_class
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.feature_dim = cli_args.feature_dim
    args.transformer_d_model = cli_args.transformer_d_model
    args.transformer_num_heads = cli_args.transformer_num_heads
    args.transformer_num_layers = cli_args.transformer_num_layers
    args.transformer_dropout = cli_args.transformer_dropout
    args.local_epochs = 1
    args.global_rounds = 1
    args.eval_gap = 1
    args.lamda = cli_args.lamda
    args.save_folder_name = cli_args.save_root
    args.skip_figures = True
    resolve_models(args)
    return args


def load_best_artifacts(server):
    save_folder = server.save_folder_name

    if server.algorithm == "FedProto":
        best_global_protos = load_item("Server", "best_global_protos", save_folder)
        if best_global_protos is None:
            best_global_protos = load_item("Server", "global_protos", save_folder)
        server.global_protos = best_global_protos
        for client in server.clients:
            best_model = load_item(
                client.role,
                "best_model",
                save_folder,
                model_template=copy.deepcopy(client.model).to(server.device),
            )
            if best_model is None:
                best_model = load_item(
                    client.role,
                    "model",
                    save_folder,
                    model_template=copy.deepcopy(client.model).to(server.device),
                )
            if best_model is not None:
                client.model = best_model.to(server.device)
            client.global_protos = server.global_protos
        return

    if server.algorithm == "FedAvg":
        best_global_model = load_item(
            "Server",
            "best_global_model",
            save_folder,
            model_template=copy.deepcopy(server.global_model).to(server.device),
        )
        if best_global_model is None:
            best_global_model = load_item(
                "Server",
                "global_model",
                save_folder,
                model_template=copy.deepcopy(server.global_model).to(server.device),
            )
        if best_global_model is not None:
            server.global_model = best_global_model.to(server.device)
            server.set_global_model_to_clients()
        return

    if server.algorithm == "Local":
        for client in server.clients:
            best_model = load_item(
                client.role,
                "best_model",
                save_folder,
                model_template=copy.deepcopy(client.model).to(server.device),
            )
            if best_model is None:
                best_model = load_item(
                    client.role,
                    "model",
                    save_folder,
                    model_template=copy.deepcopy(client.model).to(server.device),
                )
            if best_model is not None:
                client.model = best_model.to(server.device)


def aggregate_metrics(server):
    stats = server.test_metrics()
    client_stats = stats["clients"]
    global_stats = stats["global"]
    client_map = json.loads(os.environ.get("IOT_FEDPROTO_CLIENT_MAP_JSON", "[]") or "[]")

    return {
        "accuracy": float(global_stats["accuracy"]),
        "aucMacro": float(global_stats["auc_macro"]),
        "aucMicro": float(global_stats["auc_micro"]),
        "precision": float(global_stats["precision"]),
        "recall": float(global_stats["recall"]),
        "f1": float(global_stats["f1"]),
        "fnr": float(global_stats["fnr"]),
        "fpr": float(global_stats["fpr"]),
        "inferenceLatencyMs": float(global_stats["latency_ms"]),
        "confusionMatrix": np.asarray(global_stats["confusion_matrix"]).tolist(),
        "perClient": [
            {
                "clientId": str(client_map[index] if index < len(client_map) else row["id"]),
                "accuracy": float(row["correct"] / max(row["samples"], 1)),
                "samples": int(row["samples"]),
            }
            for index, row in enumerate(client_stats)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--goal", default="desktop_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--input_dim", type=int, default=77)
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--transformer_d_model", type=int, default=64)
    parser.add_argument("--transformer_num_heads", type=int, default=4)
    parser.add_argument("--transformer_num_layers", type=int, default=2)
    parser.add_argument("--transformer_dropout", type=float, default=0.2)
    cli_args = parser.parse_args()

    configure_stdio()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.device_id
    if cli_args.device == "cuda" and not torch.cuda.is_available():
        cli_args.device = "cpu"

    args = build_args(cli_args)
    with contextlib.redirect_stdout(io.StringIO()):
        server = build_server(args, 0)
        load_best_artifacts(server)
        payload = aggregate_metrics(server)
    payload.update(
        {
            "modelLabel": f"{cli_args.algorithm} / {cli_args.model_family}",
            "dataset": cli_args.dataset,
            "testClients": json.loads(os.environ.get("IOT_FEDPROTO_CLIENT_MAP_JSON", "[]") or "[]"),
        }
    )
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

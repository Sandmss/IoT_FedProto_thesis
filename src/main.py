#!/usr/bin/env python

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch

from flcore.clients.clientbase import debug_log
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverlocal import Local
from flcore.servers.serverproto import FedProto
from utils.result_utils import average_data


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
warnings.simplefilter("ignore")
torch.manual_seed(0)


def configure_stdio():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


def print_runtime_config(args):
    print("=" * 50)
    print("Runtime configuration")
    print(f"Algorithm: {args.algorithm}")
    print(f"Dataset: {args.dataset}")
    print(f"Model family: {args.model_family}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Transformer d_model: {args.transformer_d_model}")
    print(f"Transformer heads: {args.transformer_num_heads}")
    print(f"Transformer layers: {args.transformer_num_layers}")
    print(f"Transformer dropout: {args.transformer_dropout}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Normal class label: {args.normal_class}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Local batch size: {args.batch_size}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Pin memory: {args.pin_memory}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Local learning rate: {args.local_learning_rate}")
    print(f"Client join ratio: {args.join_ratio}")
    print(f"Random join ratio: {args.random_join_ratio}")
    print(f"Client drop rate: {args.client_drop_rate}")
    print(f"Time-based client selection: {args.time_select}")
    if args.time_select:
        print(f"Time threshold: {args.time_threthold}")
    print(f"Runs: {args.times}")
    print(f"Device: {args.device}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Lambda: {args.lamda}")
    print(f"FedProto eval mode: {args.proto_eval_mode}")
    print(f"Packet weight: {args.packet_weight}")
    print(f"Auto break: {args.auto_break}")
    print(f"Early stop patience: {args.early_stop_patience}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Eval gap (rounds): {args.eval_gap}")
    print(f"Skip figures: {args.skip_figures}")
    print("=" * 50)


def resolve_models(args):
    if args.model_family == "IoT_MLP":
        args.models = [
            f"MLP_IoT(dim_in={args.input_dim}, dim_hidden=128, dim_out={args.feature_dim}, num_classes={args.num_classes})",
        ]
        args.heads = [f"nn.Linear({args.feature_dim}, {args.num_classes})"]
    elif args.model_family == "IoT_CNN1D":
        args.models = [
            f"CNN1D_IoT(dim_in={args.input_dim}, dim_out={args.feature_dim}, num_classes={args.num_classes})",
        ]
        args.heads = [f"nn.Linear({args.feature_dim}, {args.num_classes})"]
    elif args.model_family == "IoT_Transformer1D":
        args.models = [
            (
                "Transformer1D_IoT("
                f"dim_in={args.input_dim}, dim_out={args.feature_dim}, "
                f"num_classes={args.num_classes}, d_model={args.transformer_d_model}, "
                f"num_heads={args.transformer_num_heads}, num_layers={args.transformer_num_layers}, "
                f"dropout={args.transformer_dropout})"
            ),
        ]
        args.heads = [f"nn.Linear({args.feature_dim}, {args.num_classes})"]
    elif args.model_family == "IoT_MIX_MLP_CNN1D":
        args.models = [
            f"MLP_IoT(dim_in={args.input_dim}, dim_hidden=128, dim_out={args.feature_dim}, num_classes={args.num_classes})",
            f"CNN1D_IoT(dim_in={args.input_dim}, dim_out={args.feature_dim}, num_classes={args.num_classes})",
        ]
        args.heads = [
            f"nn.Linear({args.feature_dim}, {args.num_classes})",
            f"nn.Linear({args.feature_dim}, {args.num_classes})",
        ]
    elif args.model_family == "IoT_MIX_MLP_CNN_TRANS":
        args.models = [
            f"MLP_IoT(dim_in={args.input_dim}, dim_hidden=128, dim_out={args.feature_dim}, num_classes={args.num_classes})",
            f"CNN1D_IoT(dim_in={args.input_dim}, dim_out={args.feature_dim}, num_classes={args.num_classes})",
            (
                "Transformer1D_IoT("
                f"dim_in={args.input_dim}, dim_out={args.feature_dim}, "
                f"num_classes={args.num_classes}, d_model={args.transformer_d_model}, "
                f"num_heads={args.transformer_num_heads}, num_layers={args.transformer_num_layers}, "
                f"dropout={args.transformer_dropout})"
            ),
        ]
        args.heads = [
            f"nn.Linear({args.feature_dim}, {args.num_classes})",
            f"nn.Linear({args.feature_dim}, {args.num_classes})",
            f"nn.Linear({args.feature_dim}, {args.num_classes})",
        ]
    else:
        raise NotImplementedError(
            f"Unsupported model_family '{args.model_family}'. "
            "Available options: IoT_MLP, IoT_CNN1D, IoT_Transformer1D, "
            "IoT_MIX_MLP_CNN1D, IoT_MIX_MLP_CNN_TRANS"
        )


def build_server(args, run_idx):
    if args.algorithm == "FedProto":
        return FedProto(args, run_idx)
    if args.algorithm == "FedAvg":
        return FedAvg(args, run_idx)
    if args.algorithm == "Local":
        return Local(args, run_idx)
    raise NotImplementedError(
        f"Unsupported algorithm '{args.algorithm}'. "
        "Available options: FedAvg, FedProto, Local"
    )


def run(args):
    time_list = []
    for run_idx in range(args.prev, args.times):
        print(f"\n============= Run {run_idx} =============")
        print("Creating server and clients...")
        start = time.time()
        print(f"Model family: {args.model_family}")

        resolve_models(args)

        print("Resolved models:")
        for model in args.models:
            print(f"  {model}")

        server = build_server(args, run_idx)
        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    average_data(
        dataset=args.dataset,
        algorithm=args.algorithm,
        goal=args.goal,
        times=args.times,
        model_family=args.model_family,
    )
    print("All done!")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model_family",
        type=str,
        default="IoT_MLP",
        help="Backbone to use: IoT_MLP, IoT_CNN1D, IoT_Transformer1D, IoT_MIX_MLP_CNN1D, or IoT_MIX_MLP_CNN_TRANS",
    )
    parser.add_argument("-dataset", type=str, default="IoT", help="Dataset name")
    parser.add_argument("--input_dim", type=int, default=77, help="Input feature dimension for each sample")
    parser.add_argument(
        "--transformer_d_model",
        type=int,
        default=64,
        help="Token embedding dimension for IoT_Transformer1D",
    )
    parser.add_argument(
        "--transformer_num_heads",
        type=int,
        default=4,
        help="Number of attention heads for IoT_Transformer1D",
    )
    parser.add_argument(
        "--transformer_num_layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers for IoT_Transformer1D",
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.2,
        help="Dropout for IoT_Transformer1D",
    )
    parser.add_argument("-go", "--goal", type=str, default="test", help="Experiment goal tag")
    parser.add_argument("-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-nb", "--num_classes", type=int, default=15)
    parser.add_argument(
        "--normal_class",
        type=int,
        default=0,
        help="Label id treated as the normal class when computing FNR",
    )
    parser.add_argument("-lbs", "--batch_size", type=int, default=10)
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes per client",
    )
    parser.add_argument(
        "-pm",
        "--pin_memory",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to use pinned memory for DataLoader",
    )
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.005,
        help="Local learning rate",
    )
    parser.add_argument("-gr", "--global_rounds", type=int, default=100)
    parser.add_argument(
        "-ls",
        "--local_epochs",
        type=int,
        default=1,
        help="Local training epochs per communication round",
    )
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        help="Fraction of clients joining each round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to randomize the client join ratio",
    )
    parser.add_argument("-nc", "--num_clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("-pv", "--prev", type=int, default=0, help="Starting run index")
    parser.add_argument("-t", "--times", type=int, default=1, help="Number of repeated runs")
    parser.add_argument(
        "-eg",
        "--eval_gap",
        type=int,
        default=1,
        help="Evaluation interval in rounds (e.g. 10 cuts eval cost ~10x)",
    )
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Skip t-SNE / prototype figure generation at end (saves CPU time)",
    )
    parser.add_argument(
        "-sfn",
        "--save_folder_name",
        type=str,
        default="temp",
        help="Directory name for intermediate outputs",
    )
    parser.add_argument(
        "-ab",
        "--auto_break",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Early stop after consecutive evaluations without strictly higher averaged test accuracy",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="Number of consecutive evaluations without a new best averaged test accuracy before early stopping",
    )
    parser.add_argument("-fd", "--feature_dim", type=int, default=64, help="Prototype / representation dimension")
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Fraction of clients that drop after local training",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="Fraction of slow clients during local training",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="Fraction of slow clients during model upload",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to select clients based on time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="Time threshold for filtering slow clients",
    )
    parser.add_argument("-lam", "--lamda", type=float, default=1.0, help="Prototype regularization weight")
    parser.add_argument("-pw", "--packet_weight", type=float, default=1.0, help="Reserved weighting coefficient")
    parser.add_argument("-mart", "--margin_threthold", type=float, default=100.0, help="Margin threshold")
    parser.add_argument("-se", "--server_epochs", type=int, default=1000, help="Server-side optimization epochs")
    parser.add_argument("--fixed_margin", type=float, default=0.5, help="Fixed margin value for ablation runs")
    parser.add_argument(
        "--proto_eval_mode",
        type=str,
        default="classifier",
        choices=["classifier", "prototype"],
        help="FedProto evaluation mode: classifier head for the main metric, or prototype nearest-neighbor for ablation",
    )
    parser.add_argument("-mcl", "--margin_contrastive", type=float, default=1.0, help="Contrastive margin weight")
    parser.add_argument("-cc", "--classifier_calibration", type=float, default=1.0, help="Classifier calibration weight")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    configure_stdio()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\nCUDA is not available. Falling back to CPU.\n")
        args.device = "cpu"

    print_runtime_config(args)

    debug_log(
        "src/main.py:runtime",
        "runtime args snapshot",
        {
            "algorithm": args.algorithm,
            "dataset": args.dataset,
            "model_family": args.model_family,
            "device": args.device,
            "device_id": args.device_id,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "feature_dim": args.feature_dim,
            "transformer_d_model": args.transformer_d_model,
            "transformer_num_heads": args.transformer_num_heads,
            "transformer_num_layers": args.transformer_num_layers,
            "transformer_dropout": args.transformer_dropout,
            "lamda": args.lamda,
            "global_rounds": args.global_rounds,
            "early_stop_patience": args.early_stop_patience,
            "num_workers": args.num_workers,
        },
        run_id=f"{args.algorithm}_runtime",
        hypothesis_id="H1",
    )

    run(args)

#!/usr/bin/env python

import argparse
import csv
import os
import re
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize FedProto lambda sensitivity runs into CSV and a plot."
    )
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    parser.add_argument("--algorithm", default="FedProto")
    parser.add_argument("--model_family", default="IoT_MIX_MLP_CNN1D")
    parser.add_argument("--goal_prefix", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def get_model_result_category(model_family):
    mapping = {
        "IoT_MLP": "MLP",
        "IoT_CNN1D": "CNN1D",
        "IoT_Transformer1D": "Transformer",
    }
    return mapping.get(model_family, "heterogeneous_models")


def parse_lambda_from_goal(goal):
    match = re.search(r"_lam_([0-9]+(?:p[0-9]+)?)$", goal)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def safe_stat(values, reducer, default=0.0):
    if values.size == 0:
        return default
    return float(reducer(values))


def read_optional_dataset(hf, name):
    dataset = hf.get(name)
    if dataset is None:
        return np.array([])
    return np.array(dataset)


def parse_identity_from_stem(file_stem, algorithm, model_family):
    pattern = re.compile(
        rf"^(?P<dataset>.+?)_{re.escape(algorithm)}_{re.escape(model_family)}_"
        rf"(?P<goal>.+)_(?P<run_idx>\d+)$"
    )
    match = pattern.match(file_stem)
    if not match:
        return None
    return match.group("goal"), match.group("run_idx")


def summarize_file(file_path, algorithm, model_family):
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    parsed = parse_identity_from_stem(file_stem, algorithm, model_family)
    if parsed is None:
        return None
    goal, run_idx = parsed
    lamda = parse_lambda_from_goal(goal)
    if lamda is None:
        return None

    with h5py.File(file_path, "r") as hf:
        rs_test_acc = read_optional_dataset(hf, "rs_test_acc")
        rs_test_f1 = read_optional_dataset(hf, "rs_test_f1")
        rs_test_auc_macro = read_optional_dataset(hf, "rs_test_auc_macro")
        if rs_test_auc_macro.size == 0:
            rs_test_auc_macro = read_optional_dataset(hf, "rs_test_auc")
        rs_proto_loss = read_optional_dataset(hf, "rs_proto_loss")

    best_acc = safe_stat(rs_test_acc, np.max)
    best_f1 = safe_stat(rs_test_f1, np.max)
    best_auc_macro = safe_stat(rs_test_auc_macro, np.max)
    best_acc_round = int(np.argmax(rs_test_acc)) if rs_test_acc.size > 0 else -1
    final_proto_loss = float(rs_proto_loss[-1]) if rs_proto_loss.size > 0 else 0.0
    mean_proto_loss = safe_stat(rs_proto_loss, np.mean)
    min_proto_loss = safe_stat(rs_proto_loss, np.min)

    return {
        "Lambda": lamda,
        "Goal": goal,
        "RunIndex": run_idx,
        "BestAcc": round(best_acc, 6),
        "BestAccRoundIndex": best_acc_round,
        "BestF1": round(best_f1, 6),
        "BestAucMacro": round(best_auc_macro, 6),
        "FinalProtoLoss": round(final_proto_loss, 6),
        "MeanProtoLoss": round(mean_proto_loss, 6),
        "MinProtoLoss": round(min_proto_loss, 6),
        "RecordedRounds": int(rs_test_acc.size),
        "ResultFile": file_path,
    }


def write_csv(rows, output_path, fieldnames):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Lambda"], []).append(row)

    aggregated = []
    for lamda in sorted(grouped.keys()):
        group = grouped[lamda]
        best_acc = np.array([row["BestAcc"] for row in group], dtype=np.float64)
        best_f1 = np.array([row["BestF1"] for row in group], dtype=np.float64)
        best_auc_macro = np.array([row["BestAucMacro"] for row in group], dtype=np.float64)
        final_proto_loss = np.array([row["FinalProtoLoss"] for row in group], dtype=np.float64)
        mean_proto_loss = np.array([row["MeanProtoLoss"] for row in group], dtype=np.float64)
        aggregated.append(
            {
                "Lambda": lamda,
                "Runs": len(group),
                "BestAccMean": round(float(np.mean(best_acc)), 6),
                "BestAccStd": round(float(np.std(best_acc)), 6),
                "BestF1Mean": round(float(np.mean(best_f1)), 6),
                "BestAucMacroMean": round(float(np.mean(best_auc_macro)), 6),
                "FinalProtoLossMean": round(float(np.mean(final_proto_loss)), 6),
                "FinalProtoLossStd": round(float(np.std(final_proto_loss)), 6),
                "MeanProtoLossMean": round(float(np.mean(mean_proto_loss)), 6),
            }
        )
    return aggregated


def plot_summary(rows, output_path):
    lambdas = [row["Lambda"] for row in rows]
    accs = [row["BestAccMean"] for row in rows]
    proto_losses = [row["FinalProtoLossMean"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, accs, marker="o", color="#1f77b4", linewidth=2, label="Best accuracy")
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Best accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        lambdas,
        proto_losses,
        marker="s",
        color="#d62728",
        linewidth=2,
        label="Final proto loss",
    )
    ax2.set_ylabel("Final proto loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    plt.title("FedProto lambda sensitivity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    goal_prefix = args.goal_prefix or f"lambda_sensitivity_{args.model_family}"
    model_category = get_model_result_category(args.model_family)
    metrics_dir = os.path.join(
        os.path.abspath(args.results_root),
        model_category,
        args.algorithm,
        "metrics",
    )
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    pattern = os.path.join(metrics_dir, "*.h5")
    rows = []
    for file_path in sorted(glob(pattern)):
        summary = summarize_file(file_path, args.algorithm, args.model_family)
        if summary is None:
            continue
        if not summary["Goal"].startswith(goal_prefix):
            continue
        summary["ResultFile"] = os.path.relpath(file_path, args.results_root).replace("\\", "/")
        rows.append(summary)

    if not rows:
        raise RuntimeError(
            f"No matching lambda sensitivity results found for goal_prefix='{goal_prefix}'."
        )

    rows.sort(key=lambda row: (row["Lambda"], row["RunIndex"]))
    aggregated = aggregate_rows(rows)

    output_dir = args.output_dir or os.path.join(
        os.path.abspath(args.results_root),
        "summary",
        "lambda_sensitivity",
        args.algorithm,
        args.model_family,
    )
    os.makedirs(output_dir, exist_ok=True)

    raw_csv = os.path.join(output_dir, "lambda_sensitivity_runs.csv")
    summary_csv = os.path.join(output_dir, "lambda_sensitivity_summary.csv")
    plot_png = os.path.join(output_dir, "lambda_sensitivity_plot.png")

    write_csv(
        rows,
        raw_csv,
        [
            "Lambda",
            "Goal",
            "RunIndex",
            "BestAcc",
            "BestAccRoundIndex",
            "BestF1",
            "BestAucMacro",
            "FinalProtoLoss",
            "MeanProtoLoss",
            "MinProtoLoss",
            "RecordedRounds",
            "ResultFile",
        ],
    )
    write_csv(
        aggregated,
        summary_csv,
        [
            "Lambda",
            "Runs",
            "BestAccMean",
            "BestAccStd",
            "BestF1Mean",
            "BestAucMacroMean",
            "FinalProtoLossMean",
            "FinalProtoLossStd",
            "MeanProtoLossMean",
        ],
    )
    plot_summary(aggregated, plot_png)

    print(f"Matched runs: {len(rows)}")
    print(f"Per-run CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Plot: {plot_png}")


if __name__ == "__main__":
    main()

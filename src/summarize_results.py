#!/usr/bin/env python

import argparse
import csv
import os
from glob import glob

import h5py
import numpy as np


RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SUMMARY_DIR = os.path.join(RESULTS_ROOT, "summary")
ALGORITHMS = {"FedAvg", "FedProto", "Local"}
STANDARD_SUBDIRS = {"metrics", "figures", "logs"}
MODEL_CATEGORY_ALIASES = {
    "MLP": "MLP",
    "MLP_Model": "MLP_Model",
    "MLP模型": "MLP_Model",
    "CNN1D": "CNN1D",
    "CNN1D_Model": "CNN1D_Model",
    "CNN1D模型": "CNN1D_Model",
    "Transformer": "Transformer",
    "Transformer_Model": "Transformer_Model",
    "transformer模型": "Transformer_Model",
    "heterogeneous_models": "heterogeneous_models",
    "Heterogeneous_Model": "Heterogeneous_Model",
    "异构模型": "Heterogeneous_Model",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment .h5 results into a unified table.")
    parser.add_argument(
        "--results_root",
        type=str,
        default=RESULTS_ROOT,
        help="Root directory containing experiment result files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(SUMMARY_DIR, "experiment_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=os.path.join(SUMMARY_DIR, "experiment_summary.md"),
        help="Output Markdown path.",
    )
    return parser.parse_args()


def read_optional_dataset(hf, name):
    dataset = hf.get(name)
    if dataset is None:
        return np.array([])
    return np.array(dataset)


def safe_stat(array, reducer, default=""):
    if array.size == 0:
        return default
    return float(reducer(array))


def safe_round(value, ndigits=6):
    if value == "":
        return ""
    return round(float(value), ndigits)


def normalize_model_category(model_category):
    return MODEL_CATEGORY_ALIASES.get(model_category, model_category)


def infer_model_family(model_category, file_stem):
    stem = file_stem.lower()
    normalized_category = normalize_model_category(model_category)
    if "iot_mlp" in stem or normalized_category in {"MLP", "MLP_Model"}:
        return "IoT_MLP"
    if "iot_cnn1d" in stem or normalized_category in {"CNN1D", "CNN1D_Model"}:
        return "IoT_CNN1D"
    if "iot_transformer1d" in stem or normalized_category in {"Transformer", "Transformer_Model"}:
        return "IoT_Transformer1D"
    if normalized_category in {"heterogeneous_models", "Heterogeneous_Model"}:
        if "mix_mlp_cnn1d" in stem:
            return "IoT_MIX_MLP_CNN1D"
        if "mix_mlp_cnn_trans" in stem:
            return "IoT_MIX_MLP_CNN_TRANS"
    return ""


def infer_model_category_from_path(parts):
    joined = "/".join(parts).lower()
    if "cnn1d/" in joined or "cnn1d_model" in joined or "cnn1d模型" in joined:
        return "CNN1D"
    if "transformer/" in joined or "transformer_model" in joined or "transformer模型" in joined:
        return "Transformer"
    if "heterogeneous_models" in joined or "heterogeneous_model" in joined or "异构模型" in "/".join(parts):
        return "heterogeneous_models"
    if "mlp/" in joined or "mlp_model" in joined or "mlp模型" in joined:
        return "MLP"
    return ""


def parse_new_style_stem(file_stem):
    known_model_families = [
        "IoT_MIX_MLP_CNN_TRANS",
        "IoT_MIX_MLP_CNN1D",
        "IoT_Transformer1D",
        "IoT_CNN1D",
        "IoT_MLP",
    ]
    tokens = file_stem.split("_")
    if len(tokens) < 4 or tokens[1] not in ALGORITHMS:
        return None

    dataset = tokens[0]
    algorithm = tokens[1]
    run_idx = tokens[-1]
    prefix = f"{dataset}_{algorithm}_"
    suffix = f"_{run_idx}"
    if not file_stem.startswith(prefix) or not file_stem.endswith(suffix):
        return None

    middle = file_stem[len(prefix):-len(suffix)]
    for model_family in known_model_families:
        family_prefix = f"{model_family}_"
        if middle.startswith(family_prefix):
            return {
                "dataset": dataset,
                "algorithm": algorithm,
                "model_family": model_family,
                "goal": middle[len(family_prefix):],
                "run_idx": run_idx,
            }
    return None


def parse_result_identity(rel_path):
    parts = rel_path.split(os.sep)
    file_stem = os.path.splitext(parts[-1])[0]
    dataset = ""
    algorithm = ""
    model_family = ""
    goal = ""
    run_idx = ""
    setting = ""
    model_category = infer_model_category_from_path(parts)

    if parts and model_category:
        if len(parts) >= 3 and parts[1] in ALGORITHMS:
            algorithm = parts[1]
            if parts[2] in STANDARD_SUBDIRS:
                setting = "standard"
            else:
                setting = parts[2]
        elif len(parts) >= 2:
            setting = parts[1]

    parsed_new_style = parse_new_style_stem(file_stem)
    if parsed_new_style is not None:
        dataset = parsed_new_style["dataset"]
        algorithm = parsed_new_style["algorithm"]
        model_family = parsed_new_style["model_family"]
        goal = parsed_new_style["goal"]
        run_idx = parsed_new_style["run_idx"]
    else:
        tokens = file_stem.split("_")
        if len(tokens) >= 4 and tokens[1] in ALGORITHMS:
            dataset = tokens[0]
            algorithm = tokens[1]
            goal = "_".join(tokens[2:-1])
            run_idx = tokens[-1]
            if not model_family:
                model_family = infer_model_family(model_category, file_stem)

    if not setting:
        setting = "legacy_root"

    if not model_family:
        model_family = infer_model_family(model_category, file_stem)

    return {
        "dataset": dataset,
        "setting": setting,
        "model": model_family,
        "algorithm": algorithm,
        "model_category": normalize_model_category(model_category),
        "goal": goal,
        "run_idx": run_idx,
    }


def summarize_h5_file(file_path, results_root):
    rel_path = os.path.relpath(file_path, results_root)
    identity = parse_result_identity(rel_path)

    with h5py.File(file_path, "r") as hf:
        rs_test_acc = read_optional_dataset(hf, "rs_test_acc")
        rs_test_auc_macro = read_optional_dataset(hf, "rs_test_auc_macro")
        if rs_test_auc_macro.size == 0:
            rs_test_auc_macro = read_optional_dataset(hf, "rs_test_auc")
        rs_test_auc_micro = read_optional_dataset(hf, "rs_test_auc_micro")
        rs_test_precision = read_optional_dataset(hf, "rs_test_precision")
        rs_test_recall = read_optional_dataset(hf, "rs_test_recall")
        rs_test_f1 = read_optional_dataset(hf, "rs_test_f1")
        rs_test_fnr = read_optional_dataset(hf, "rs_test_fnr")
        rs_test_fpr = read_optional_dataset(hf, "rs_test_fpr")
        rs_inference_latency_ms = read_optional_dataset(hf, "rs_inference_latency_ms")
        rs_comm_params_per_round = read_optional_dataset(hf, "rs_comm_params_per_round")
        rs_comm_params_cumulative = read_optional_dataset(hf, "rs_comm_params_cumulative")
        rs_model_params_mean = read_optional_dataset(hf, "rs_model_params_mean")
        rs_model_size_bytes_mean = read_optional_dataset(hf, "rs_model_size_bytes_mean")
        rs_model_flops_mean = read_optional_dataset(hf, "rs_model_flops_mean")

    best_acc = safe_stat(rs_test_acc, np.max)
    best_acc_round = int(np.argmax(rs_test_acc)) if rs_test_acc.size > 0 else ""

    return {
        "Dataset": identity["dataset"],
        "Setting": identity["setting"],
        "Model": identity["model"],
        "Algorithm": identity["algorithm"],
        "ModelCategory": identity["model_category"],
        "Goal": identity["goal"],
        "RunIndex": identity["run_idx"],
        "Acc": safe_round(best_acc),
        "BestAccRoundIndex": best_acc_round,
        "AUC Macro": safe_round(safe_stat(rs_test_auc_macro, np.max)),
        "AUC Micro": safe_round(safe_stat(rs_test_auc_micro, np.max)),
        "Precision": safe_round(safe_stat(rs_test_precision, np.max)),
        "Recall": safe_round(safe_stat(rs_test_recall, np.max)),
        "F1": safe_round(safe_stat(rs_test_f1, np.max)),
        "FNR": safe_round(safe_stat(rs_test_fnr, np.min)),
        "FPR": safe_round(safe_stat(rs_test_fpr, np.min)),
        "InferenceLatencyMs": safe_round(safe_stat(rs_inference_latency_ms, np.mean)),
        "CommPerRound": safe_round(safe_stat(rs_comm_params_per_round, np.mean)),
        "CommCumulative": safe_round(safe_stat(rs_comm_params_cumulative, np.max)),
        "MeanModelParams": safe_round(safe_stat(rs_model_params_mean, np.mean)),
        "MeanModelSizeBytes": safe_round(safe_stat(rs_model_size_bytes_mean, np.mean)),
        "MeanFLOPs": safe_round(safe_stat(rs_model_flops_mean, np.mean)),
        "RecordedRounds": int(rs_test_acc.size) if rs_test_acc.size > 0 else 0,
        "ResultFile": rel_path.replace("\\", "/"),
    }


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "Dataset",
        "Setting",
        "Model",
        "Algorithm",
        "ModelCategory",
        "Goal",
        "RunIndex",
        "Acc",
        "BestAccRoundIndex",
        "AUC Macro",
        "AUC Micro",
        "Precision",
        "Recall",
        "F1",
        "FNR",
        "FPR",
        "InferenceLatencyMs",
        "CommPerRound",
        "CommCumulative",
        "MeanModelParams",
        "MeanModelSizeBytes",
        "MeanFLOPs",
        "RecordedRounds",
        "ResultFile",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(rows):
    headers = ["Model", "Algorithm", "Setting", "Acc", "AUC Macro", "F1", "FNR", "FPR", "ResultFile"]
    lines = [
        "# Experiment Summary",
        "",
        f"- Total summarized `.h5` files: `{len(rows)}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Model"]),
                    str(row["Algorithm"]),
                    str(row["Setting"]),
                    str(row["Acc"]),
                    str(row["AUC Macro"]),
                    str(row["F1"]),
                    str(row["FNR"]),
                    str(row["FPR"]),
                    str(row["ResultFile"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(rows, output_md):
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(build_markdown_table(rows))


def main():
    args = parse_args()
    pattern = os.path.join(args.results_root, "**", "*.h5")
    files = sorted(glob(pattern, recursive=True))

    rows = []
    skipped = []
    for file_path in files:
        if os.path.basename(os.path.dirname(file_path)) == "summary":
            continue
        try:
            rows.append(summarize_h5_file(file_path, args.results_root))
        except Exception as exc:
            skipped.append((os.path.relpath(file_path, args.results_root), str(exc)))

    rows.sort(key=lambda row: (row["ModelCategory"], row["Model"], row["Algorithm"], row["Setting"], row["ResultFile"]))
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)

    print(f"Scanned h5 files: {len(files)}")
    print(f"Summarized rows: {len(rows)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"CSV saved to: {args.output_csv}")
    print(f"Markdown saved to: {args.output_md}")

    if skipped:
        print("Skipped details:")
        for rel_path, message in skipped:
            print(f"  - {rel_path}: {message}")


if __name__ == "__main__":
    main()

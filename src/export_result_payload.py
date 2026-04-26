#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


METRIC_KEYS = {
    "Accuracy": "rs_test_acc",
    "AUC Macro": "rs_test_auc_macro",
    "AUC Micro": "rs_test_auc_micro",
    "Precision": "rs_test_precision",
    "Recall": "rs_test_recall",
    "F1": "rs_test_f1",
    "FNR": "rs_test_fnr",
    "FPR": "rs_test_fpr",
    "Inference Latency (ms)": "rs_inference_latency_ms",
    "Train Loss": "rs_train_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a single result payload as JSON for the Electron desktop app.")
    parser.add_argument("--results_root", required=True, type=str)
    parser.add_argument("--relative_path", required=True, type=str)
    return parser.parse_args()


def infer_related_assets(file_path: Path) -> dict[str, list[Path]]:
    stem = file_path.stem
    parent = file_path.parent.parent if file_path.parent.name == "metrics" else file_path.parent
    figures_dir = parent / "figures"
    logs_dir = parent / "logs"

    figures = sorted(figures_dir.glob(f"{stem}*.png")) if figures_dir.exists() else []
    logs = sorted(logs_dir.glob("*.out")) if logs_dir.exists() else []
    return {"figures": figures, "logs": logs}


def read_log_preview(log_files: list[Path]) -> list[str]:
    if not log_files:
        return []

    try:
        lines = log_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    return lines[-60:]


def build_payload(results_root: Path, relative_path: str) -> dict[str, object]:
    file_path = results_root / relative_path
    series: dict[str, list[float]] = {}
    best_round: int | None = None
    confusion_matrix: list[list[int]] | None = None

    with h5py.File(file_path, "r") as handle:
        for label, key in METRIC_KEYS.items():
            if key in handle:
                series[label] = np.array(handle[key]).astype(float).tolist()

        accuracy_series = series.get("Accuracy", [])
        if accuracy_series:
            best_round = int(np.argmax(np.array(accuracy_series)))

        confusion = handle.get("rs_confusion_matrices")
        if confusion is not None and best_round is not None and len(confusion) > best_round:
            confusion_matrix = np.array(confusion[best_round]).astype(int).tolist()

    assets = infer_related_assets(file_path)

    return {
        "path": str(file_path.resolve()),
        "relativePath": relative_path.replace("\\", "/"),
        "series": series,
        "bestRound": best_round,
        "confusionMatrix": confusion_matrix,
        "figures": [str(path.resolve()) for path in assets["figures"]],
        "logPreview": read_log_preview(assets["logs"]),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    payload = build_payload(results_root, args.relative_path)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

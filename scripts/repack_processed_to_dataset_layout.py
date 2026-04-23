import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repack processed client npy files into dataset/IoT train/test layout."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(script_dir / "../data/processed_data").resolve(),
        help="Directory containing processed client npy files (default: data/processed_data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(script_dir / "../dataset/IoT").resolve(),
        help="Root directory for repacked dataset layout (default: dataset/IoT).",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help=(
            "Number of clients to repack. Defaults to split_stats.json num_clients; "
            "if unavailable, infers it from client_*_X.npy files."
        ),
    )
    return parser.parse_args()


def infer_num_clients(input_dir):
    stats_path = input_dir / "split_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        num_clients = stats.get("num_clients")
        if num_clients is not None:
            return int(num_clients)

    client_ids = []
    pattern = re.compile(r"client_(\d+)_X\.npy$")
    for path in input_dir.glob("client_*_X.npy"):
        match = pattern.match(path.name)
        if match:
            client_ids.append(int(match.group(1)))

    if not client_ids:
        raise FileNotFoundError(
            f"Cannot infer client count: no split_stats.json num_clients and no client_*_X.npy files in {input_dir}"
        )

    expected_ids = set(range(max(client_ids) + 1))
    missing_ids = sorted(expected_ids - set(client_ids))
    if missing_ids:
        raise ValueError(
            f"Cannot infer contiguous client ids from processed files. Missing client ids: {missing_ids}"
        )

    return max(client_ids) + 1


def get_client_file_paths(input_dir, client_id):
    return {
        "train_X": input_dir / f"client_{client_id}_X.npy",
        "train_y": input_dir / f"client_{client_id}_y.npy",
        "test_X": input_dir / f"client_{client_id}_X_test.npy",
        "test_y": input_dir / f"client_{client_id}_y_test.npy",
    }


def count_samples(npy_path):
    array = np.load(npy_path, mmap_mode="r")
    if array.ndim == 0:
        return 1
    return int(array.shape[0])


def validate_client_files(client_paths, client_id):
    missing_files = [path for path in client_paths.values() if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Client {client_id} is missing required files: {missing_text}")

    train_count = count_samples(client_paths["train_X"])
    train_label_count = count_samples(client_paths["train_y"])
    test_count = count_samples(client_paths["test_X"])
    test_label_count = count_samples(client_paths["test_y"])

    if train_count != train_label_count:
        raise ValueError(
            f"Client {client_id} train sample mismatch: X has {train_count}, y has {train_label_count}"
        )
    if test_count != test_label_count:
        raise ValueError(
            f"Client {client_id} test sample mismatch: X has {test_count}, y has {test_label_count}"
        )

    return train_count, test_count


def repack_client(client_id, client_paths, output_root):
    train_dir = output_root / "train" / str(client_id)
    test_dir = output_root / "test" / str(client_id)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(client_paths["train_X"], train_dir / "X.npy")
    shutil.copy2(client_paths["train_y"], train_dir / "y.npy")
    shutil.copy2(client_paths["test_X"], test_dir / "X.npy")
    shutil.copy2(client_paths["test_y"], test_dir / "y.npy")


def main():
    args = parse_args()
    if args.num_clients is None:
        args.num_clients = infer_num_clients(args.input_dir)

    if args.num_clients <= 0:
        raise ValueError("--num-clients must be > 0")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")

    total_train = 0
    total_test = 0

    print(f"Input directory: {args.input_dir}")
    print(f"Output root: {args.output_root}")

    for client_id in range(args.num_clients):
        client_paths = get_client_file_paths(args.input_dir, client_id)
        train_count, test_count = validate_client_files(client_paths, client_id)
        repack_client(client_id, client_paths, args.output_root)

        total_train += train_count
        total_test += test_count
        print(
            f"Client {client_id}: migrated train={train_count} samples, "
            f"test={test_count} samples"
        )

    print(
        f"Finished repacking {args.num_clients} clients. "
        f"Total train={total_train}, total test={total_test}, total={total_train + total_test}"
    )


if __name__ == "__main__":
    main()

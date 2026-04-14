import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]


def load_and_merge_csv(data_dir):
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        dataframes.append(df)
        print(f"Loaded {csv_file.name}: {len(df)} rows")

    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")
    return merged_df


def clean_dataframe(df):
    df = df.copy()
    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    if "Label" not in df.columns:
        if df.columns[-1] == "Label":
            pass
        else:
            raise KeyError("Label column not found after stripping column names.")

    feature_columns = [column for column in df.columns if column != "Label"]
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(axis=0).reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")
    return df


def encode_and_scale(df):
    label_column = "Label" if "Label" in df.columns else df.columns[-1]
    y = df[label_column]
    X = df.drop(columns=[label_column])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled.astype(np.float32), y_encoded.astype(np.int64), label_encoder


def dirichlet_distribute_by_class(y, num_clients=10, alpha=0.1, seed=42):
    """
    仿照 datalabelgen-noniid.py：
    - 逐类别采样 Dirichlet 比例；
    - 将每个类别的样本分配到各客户端；
    - 每个客户端内部保留该类别样本列表，供后续再切 train/test。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    categories = np.unique(y)

    client_class_indices = {
        client_id: {int(class_id): [] for class_id in categories}
        for client_id in range(num_clients)
    }

    for class_id in categories:
        class_indices = np.where(y == class_id)[0]
        class_indices = np.array(class_indices, dtype=np.int64, copy=True)
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        client_counts = (proportions * len(class_indices)).astype(int)

        remaining = int(len(class_indices) - np.sum(client_counts))
        if remaining > 0:
            extra_clients = rng.choice(num_clients, size=remaining, replace=False)
            for client_id in extra_clients:
                client_counts[int(client_id)] += 1

        if int(np.sum(client_counts)) != len(class_indices):
            raise RuntimeError(
                f"Class {int(class_id)} allocation mismatch: "
                f"{int(np.sum(client_counts))} != {len(class_indices)}"
            )

        start = 0
        for client_id in range(num_clients):
            count = int(client_counts[client_id])
            assigned = class_indices[start : start + count]
            client_class_indices[client_id][int(class_id)] = assigned.tolist()
            start += count

    return client_class_indices


def split_client_data_train_test(client_class_indices, train_ratio=0.75, seed=42):
    """
    仿照 datalabelgen-noniid.py：
    - 对每个客户端、每个类别内的数据再次打乱；
    - 按 train_ratio 划分 train/test；
    - 若该类别在该客户端仅有 1 条样本，则保底放入训练集。
    """
    rng = np.random.default_rng(seed)
    split_indices = {}
    stats_summary = {}

    for client_id, class_dict in client_class_indices.items():
        train_parts = []
        test_parts = []
        details = {}

        for class_id, indices in class_dict.items():
            class_indices = np.array(indices, dtype=np.int64, copy=True)
            if class_indices.size == 0:
                continue

            rng.shuffle(class_indices)
            num_samples = int(class_indices.size)
            if num_samples == 1:
                num_train = 1
            else:
                num_train = max(1, int(num_samples * train_ratio))

            train_idx = class_indices[:num_train]
            test_idx = class_indices[num_train:]

            train_parts.append(train_idx)
            test_parts.append(test_idx)
            details[int(class_id)] = {
                "train": int(train_idx.size),
                "test": int(test_idx.size),
            }

        train_indices = (
            np.concatenate(train_parts).astype(np.int64, copy=False)
            if train_parts
            else np.array([], dtype=np.int64)
        )
        test_indices = (
            np.concatenate(test_parts).astype(np.int64, copy=False)
            if test_parts
            else np.array([], dtype=np.int64)
        )

        if train_indices.size > 0:
            rng.shuffle(train_indices)
        if test_indices.size > 0:
            rng.shuffle(test_indices)

        split_indices[client_id] = {
            "train": train_indices,
            "test": test_indices,
        }
        stats_summary[client_id] = {
            "total_train": int(train_indices.size),
            "total_test": int(test_indices.size),
            "details": details,
        }

    return split_indices, stats_summary


def save_client_data(X, y, split_indices, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, split in split_indices.items():
        train_indices = split["train"]
        test_indices = split["test"]

        np.save(output_dir / f"client_{client_id}_X.npy", X[train_indices])
        np.save(output_dir / f"client_{client_id}_y.npy", y[train_indices])
        np.save(output_dir / f"client_{client_id}_X_test.npy", X[test_indices])
        np.save(output_dir / f"client_{client_id}_y_test.npy", y[test_indices])

        print(
            f"Client {client_id}: "
            f"train={len(train_indices)} samples, test={len(test_indices)} samples"
        )


def save_metadata(output_dir, label_encoder, args, stats_summary, total_samples, feature_dim):
    metadata = {
        "num_clients": int(args.num_clients),
        "train_ratio": float(args.train_ratio),
        "dirichlet_alpha": float(args.alpha),
        "seed": int(args.seed),
        "total_samples": int(total_samples),
        "feature_dim": int(feature_dim),
        "labels": {
            str(label_id): class_name
            for label_id, class_name in enumerate(label_encoder.classes_.tolist())
        },
        "client_stats": stats_summary,
    }
    metadata_path = output_dir / "split_stats.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved split summary to: {metadata_path}")


def print_stats_summary(stats_summary):
    print("\n" + "=" * 80)
    print(f"{'客户端数据分布统计 (Client Data Distribution Statistics)':^80}")
    print("=" * 80)
    print(f"{'Client ID':<10} | {'Total Train':<12} | {'Total Test':<12} | {'Category Breakdown (Train/Test)'}")
    print("-" * 80)

    for client_id in sorted(stats_summary):
        info = stats_summary[client_id]
        detail_parts = []
        for class_id, counts in sorted(info["details"].items(), key=lambda item: int(item[0])):
            detail_parts.append(f"{class_id}:{counts['train']}/{counts['test']}")
        detail_str = ", ".join(detail_parts) if detail_parts else "No Data"
        print(
            f"{client_id:<10} | {info['total_train']:<12} | "
            f"{info['total_test']:<12} | {detail_str}"
        )

    print("=" * 80)
    print("注意: 详情列格式为 '类别ID:训练集数量/测试集数量'")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CICIDS CSV files and split them with class-wise Dirichlet non-IID partition."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(script_dir / "../data/raw_data").resolve(),
        help="Directory containing source CSV files (default: data/raw_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(script_dir / "../data/processed_data").resolve(),
        help="Directory to save processed client npy files (default: data/processed_data).",
    )
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Train split ratio inside each client/class partition.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Dirichlet alpha. Smaller values produce stronger non-IID partitions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_clients <= 0:
        raise ValueError("--num-clients must be > 0")
    if not (0.0 < args.train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in the range (0, 1]")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")

    merged_df = load_and_merge_csv(args.data_dir)
    cleaned_df = clean_dataframe(merged_df)
    X, y, label_encoder = encode_and_scale(cleaned_df)

    client_class_indices = dirichlet_distribute_by_class(
        y,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    split_indices, stats_summary = split_client_data_train_test(
        client_class_indices,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    save_client_data(X, y, split_indices, args.output_dir)
    save_metadata(
        args.output_dir,
        label_encoder,
        args,
        stats_summary,
        total_samples=len(y),
        feature_dim=X.shape[1],
    )
    print_stats_summary(stats_summary)

    total_train = sum(info["total_train"] for info in stats_summary.values())
    total_test = sum(info["total_test"] for info in stats_summary.values())
    print(f"最终参与训练的特征维度: {X.shape[1]}")
    print(f"总样本数: {len(y)} | 训练样本数: {total_train} | 测试样本数: {total_test}")


if __name__ == "__main__":
    main()

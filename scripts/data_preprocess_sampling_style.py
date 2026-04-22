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

    if "Label" not in df.columns and df.columns[-1] != "Label":
        raise KeyError("Label column not found after stripping column names.")

    feature_columns = [column for column in df.columns if column != "Label"]
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before_dropna = len(df)
    df = df.dropna(axis=0).reset_index(drop=True)
    after_dropna = len(df)
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_dedup = len(df)
    print(
        "Rows after cleaning: {} "
        "(dropped NaN/inf rows: {}, dropped duplicate rows: {})".format(
            len(df),
            before_dropna - after_dropna,
            before_dedup - after_dedup,
        )
    )
    return df


def encode_labels_and_extract_features(df):
    label_column = "Label" if "Label" in df.columns else df.columns[-1]
    y = df[label_column]
    X = df.drop(columns=[label_column]).to_numpy(dtype=np.float32, copy=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded.astype(np.int64), label_encoder


def balanced_subsample(X, y, target_total, seed=42):
    """
    从全体样本中尽量均衡地抽取 target_total 条，保证每一类都能被采到。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    X = np.asarray(X)
    n = int(y.shape[0])
    if target_total >= n:
        if target_total > n:
            print(
                f"提示: target_total={target_total} 大于可用样本数 {n}，将使用全部样本。"
            )
        return X, y

    classes = np.sort(np.unique(y))
    c_count = int(classes.size)
    if c_count == 0:
        raise ValueError("No class labels found in y.")

    base = target_total // c_count
    rem = target_total % c_count
    quota = {}
    for i, c in enumerate(classes):
        quota[int(c)] = base + (1 if i < rem else 0)

    selected_parts = []
    for c in classes:
        c = int(c)
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        take = min(int(idx.size), quota[c])
        if take > 0:
            selected_parts.append(idx[:take])

    selected = (
        np.concatenate(selected_parts).astype(np.int64, copy=False)
        if selected_parts
        else np.array([], dtype=np.int64)
    )

    need = int(target_total - selected.size)
    if need > 0:
        used = np.zeros(n, dtype=bool)
        used[selected] = True
        pool = np.where(~used)[0]
        rng.shuffle(pool)
        take_extra = min(need, int(pool.size))
        if take_extra > 0:
            selected = np.concatenate(
                [selected, pool[:take_extra].astype(np.int64, copy=False)]
            )
        if selected.size < target_total:
            print(
                f"警告: 仅能抽取 {selected.size} 条样本（目标 {target_total}），"
                "可能因清洗后可用样本过少。"
            )

    if selected.size > target_total:
        selected = rng.choice(selected, size=target_total, replace=False)

    rng.shuffle(selected)
    return X[selected], y[selected]


def sampling_style_distribute_by_class(
    y,
    num_clients,
    classes_per_client,
    k_per_class,
    seed=42,
):
    """
    更贴近 FedProto 论文 / sampling.py 的 n-way k-shot 思路：
    1. 先把样本按类别排序；
    2. 每个客户端只持有若干类别（n-way）；
    3. 每个被分配的“客户端-类别”固定取 k 条样本；
    4. 某个类别剩余样本不足 k 时，不再继续分配该类别（与你的要求一致）。

    说明：由于真实数据是长尾分布，某些客户端可能拿不到满 n 个类别，这是因为
    可用类别都已不足 k 条样本，属于论文式 k-shot 在长尾数据上的自然约束。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    classes = np.sort(np.unique(y))
    num_classes = int(classes.size)

    if classes_per_client <= 0:
        raise ValueError("--classes-per-client must be > 0")
    if classes_per_client > num_classes:
        raise ValueError(
            f"--classes-per-client={classes_per_client} exceeds num_classes={num_classes}"
        )
    if k_per_class <= 0:
        raise ValueError("--k-per-class must be > 0")

    sorted_indices = np.argsort(y, kind="stable")
    sorted_labels = y[sorted_indices]

    class_sorted_indices = {}
    label_begin = {}
    for class_id in classes:
        class_mask = sorted_labels == int(class_id)
        class_block = sorted_indices[class_mask]
        if class_block.size == 0:
            continue
        class_sorted_indices[int(class_id)] = class_block.astype(np.int64, copy=False)
        label_begin[int(class_id)] = int(np.where(class_mask)[0][0])

    class_counts = {
        int(class_id): int(class_sorted_indices[int(class_id)].size)
        for class_id in classes
    }
    class_cursor = {int(class_id): 0 for class_id in classes}
    client_classes = {client_id: set() for client_id in range(num_clients)}
    client_class_indices = {
        client_id: {int(class_id): [] for class_id in classes}
        for client_id in range(num_clients)
    }
    all_classes = classes.astype(int).tolist()
    insufficient_clients = []

    for client_id in range(num_clients):
        while len(client_classes[client_id]) < classes_per_client:
            selectable = [
                class_id
                for class_id in all_classes
                if class_id not in client_classes[client_id]
                and (class_counts[class_id] - class_cursor[class_id]) >= k_per_class
            ]
            if not selectable:
                insufficient_clients.append(client_id)
                break

            weights = np.array(
                [
                    class_counts[class_id] - class_cursor[class_id]
                    for class_id in selectable
                ],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            chosen = int(rng.choice(selectable, p=weights))

            start = class_cursor[chosen]
            end = start + k_per_class
            picked = class_sorted_indices[chosen][start:end]
            client_class_indices[client_id][chosen] = picked.astype(
                np.int64, copy=False
            ).tolist()
            client_classes[client_id].add(chosen)
            class_cursor[chosen] = end

    class_to_clients = {int(class_id): [] for class_id in classes}
    for client_id in range(num_clients):
        for class_id in sorted(client_classes[client_id]):
            class_to_clients[class_id].append(client_id)

    for class_id in classes.astype(int).tolist():
        remaining = class_counts[class_id] - class_cursor[class_id]
        print(
            f"Class {class_id}: total={class_counts[class_id]} samples, "
            f"assigned_clients={sorted(class_to_clients[class_id])}, "
            f"k={k_per_class}, consumed={class_cursor[class_id]}, remaining={remaining}, "
            f"label_begin={label_begin[class_id]}"
        )

    if insufficient_clients:
        uniq = sorted(set(insufficient_clients))
        print(
            "Warning: some clients could not reach the target n-way because all remaining "
            f"classes had fewer than k={k_per_class} samples: {uniq}"
        )

    classes_list = {
        client_id: sorted(int(class_id) for class_id in client_classes[client_id])
        for client_id in range(num_clients)
    }
    return client_class_indices, classes_list


def split_client_data_train_test(client_class_indices, train_ratio=0.75, seed=42):
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


def fit_global_train_scaler(X, split_indices):
    train_indices = [
        split["train"] for split in split_indices.values() if split["train"].size > 0
    ]
    if not train_indices:
        raise ValueError("No training samples were assigned across clients.")

    train_indices = np.concatenate(train_indices).astype(np.int64, copy=False)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X[train_indices])
    print(f"Scaler fitted on {len(train_indices)} training samples only.")
    return scaler


def save_client_data(X, y, split_indices, output_dir, scaler):
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, split in split_indices.items():
        train_indices = split["train"]
        test_indices = split["test"]

        train_X = scaler.transform(X[train_indices]).astype(np.float32, copy=False)
        test_X = scaler.transform(X[test_indices]).astype(np.float32, copy=False)
        train_y = y[train_indices]
        test_y = y[test_indices]

        np.save(output_dir / f"client_{client_id}_X.npy", train_X)
        np.save(output_dir / f"client_{client_id}_y.npy", train_y)
        np.save(output_dir / f"client_{client_id}_X_test.npy", test_X)
        np.save(output_dir / f"client_{client_id}_y_test.npy", test_y)

        print(
            f"Client {client_id}: "
            f"train={len(train_indices)} samples, test={len(test_indices)} samples"
        )


def save_metadata(
    output_dir,
    label_encoder,
    args,
    stats_summary,
    total_samples,
    feature_dim,
    scaler_train_samples,
    classes_list,
):
    metadata = {
        "num_clients": int(args.num_clients),
        "train_ratio": float(args.train_ratio),
        "client_split": "sampling_style",
        "classes_per_client": int(args.classes_per_client),
        "k_per_class": int(args.k_per_class),
        "target_total": int(args.target_total),
        "seed": int(args.seed),
        "total_samples": int(total_samples),
        "feature_dim": int(feature_dim),
        "scaler_scope": "global_train_only",
        "scaler_train_samples": int(scaler_train_samples),
        "labels": {
            str(label_id): class_name
            for label_id, class_name in enumerate(label_encoder.classes_.tolist())
        },
        "client_classes": {str(k): v for k, v in classes_list.items()},
        "client_stats": stats_summary,
    }
    metadata_path = output_dir / "split_stats.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved split summary to: {metadata_path}")


def print_stats_summary(stats_summary, classes_list):
    print("\n" + "=" * 100)
    print(f"{'客户端数据分布统计 (Sampling-Style Split)':^100}")
    print("=" * 100)
    print(
        f"{'Client ID':<10} | {'Classes':<20} | {'Total Train':<12} | "
        f"{'Total Test':<12} | {'Category Breakdown (Train/Test)'}"
    )
    print("-" * 100)

    for client_id in sorted(stats_summary):
        info = stats_summary[client_id]
        detail_parts = []
        for class_id, counts in sorted(info["details"].items(), key=lambda item: int(item[0])):
            detail_parts.append(f"{class_id}:{counts['train']}/{counts['test']}")
        detail_str = ", ".join(detail_parts) if detail_parts else "No Data"
        class_str = ",".join(str(c) for c in classes_list[client_id])
        print(
            f"{client_id:<10} | {class_str:<20} | {info['total_train']:<12} | "
            f"{info['total_test']:<12} | {detail_str}"
        )

    print("=" * 100)
    print("注意: 详情列格式为 '类别ID:训练集数量/测试集数量'")
    print("=" * 100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess CICIDS CSV files and split them with a sampling.py-style "
            "class-block non-IID partition."
        )
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
        help=(
            "Directory to save processed client npy files. "
            "Point this to an existing processed_data directory if you want to overwrite it."
        ),
    )
    parser.add_argument(
        "--num-clients", type=int, default=20, help="Number of clients."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Train split ratio inside each client/class partition.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=20000,
        help="Total number of cleaned samples to keep before client partitioning.",
    )
    parser.add_argument(
        "--classes-per-client",
        type=int,
        default=4,
        help=(
            "How many classes each client keeps in the sampling-style non-IID split. "
            "Paper-like FedProto settings are closer to small overlapping class subsets "
            "(e.g. 4) rather than full 15-class coverage on every client."
        ),
    )
    parser.add_argument(
        "--k-per-class",
        type=int,
        default=100,
        help=(
            "Paper-style k-shot setting: each assigned client-class pair gets exactly k samples. "
            "If a class has fewer than k remaining samples, it will no longer be assigned."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_clients <= 0:
        raise ValueError("--num-clients must be > 0")
    if not (0.0 < args.train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in the range (0, 1]")
    if args.target_total <= 0:
        raise ValueError("--target-total must be > 0")

    merged_df = load_and_merge_csv(args.data_dir)
    cleaned_df = clean_dataframe(merged_df)
    X, y, label_encoder = encode_labels_and_extract_features(cleaned_df)
    X, y = balanced_subsample(X, y, args.target_total, seed=args.seed)

    client_class_indices, classes_list = sampling_style_distribute_by_class(
        y,
        num_clients=args.num_clients,
        classes_per_client=args.classes_per_client,
        k_per_class=args.k_per_class,
        seed=args.seed,
    )
    split_indices, stats_summary = split_client_data_train_test(
        client_class_indices,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    scaler = fit_global_train_scaler(X, split_indices)
    total_train = sum(info["total_train"] for info in stats_summary.values())
    total_test = sum(info["total_test"] for info in stats_summary.values())
    save_client_data(X, y, split_indices, args.output_dir, scaler)
    save_metadata(
        args.output_dir,
        label_encoder,
        args,
        stats_summary,
        total_samples=len(y),
        feature_dim=X.shape[1],
        scaler_train_samples=total_train,
        classes_list=classes_list,
    )
    print_stats_summary(stats_summary, classes_list)

    print(f"最终参与划分的特征维度: {X.shape[1]}")
    print(f"总样本数: {len(y)} | 训练样本数: {total_train} | 测试样本数: {total_test}")


if __name__ == "__main__":
    main()

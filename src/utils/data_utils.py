# --- START OF FILE data_utils.py ---

import numpy as np
import os
import torch


def get_partitioned_data_root(dataset_name):
    dataset_name = str(dataset_name).strip().strip("/\\") or "IoT"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(current_dir, "..", "..", "dataset", dataset_name)
    )


def _empty_client_arrays():
    return {
        'x': np.array([], dtype=np.float32),
        'y': np.array([], dtype=np.int64),
    }

########################################
# 1. 读取原始客户端数据
########################################
def read_data(dataset_name, idx, is_train=True):
    """
    读取指定客户端 (idx) 的 npy 数据。

    Args:
        dataset_name (str): 数据集名称，目录结构为 ../dataset/{dataset_name}/[train/test]/{client_id}/。
        idx (int): 客户端ID。
        is_train (bool): 是否加载训练数据 (True 为 train，False 为 test)。

    Returns:
        dict: 包含 'x' (特征数组) 和 'y' (标签数组)。
    """
    partitioned_data_root = get_partitioned_data_root(dataset_name)
    split = "train" if is_train else "test"
    client_data_path = os.path.join(partitioned_data_root, split, str(idx))
    x_path = os.path.join(client_data_path, "X.npy")
    y_path = os.path.join(client_data_path, "y.npy")

    if not os.path.isdir(client_data_path):
        print(
            f"警告: client {idx} 的 {split} 数据目录不存在: '{client_data_path}'。返回空数组。"
        )
        return _empty_client_arrays()

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        print(
            f"警告: client {idx} 的 {split} 数据文件缺失: X='{x_path}', y='{y_path}'。返回空数组。"
        )
        return _empty_client_arrays()

    try:
        x_data = np.load(x_path, allow_pickle=False)
        y_data = np.load(y_path, allow_pickle=False)
    except Exception as e:
        print(
            f"警告: 读取 client {idx} 的 {split} npy 数据失败: {e}。返回空数组。"
        )
        return _empty_client_arrays()

    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.int64)

    return {'x': x_data, 'y': y_data}

########################################
# 2. 客户端数据读取
########################################
def read_client_data(dataset, idx, is_train=True):
    """
    读取并准备客户端数据，返回适合 DataLoader 的列表 [(特征, 标签), ...]。
    """
    client_raw_data = read_data(dataset, idx, is_train)
    split = "train" if is_train else "test"

    if len(client_raw_data['x']) == 0:
        print(f"警告: client {idx} 的 {split} 数据为空。")
        return []

    x_array = client_raw_data['x']
    y_array = client_raw_data['y']

    if x_array.ndim == 1:
        x_array = np.expand_dims(x_array, axis=0)
    if y_array.ndim == 0:
        y_array = np.expand_dims(y_array, axis=0)
    else:
        y_array = y_array.reshape(-1)

    if len(x_array) != len(y_array):
        sample_count = min(len(x_array), len(y_array))
        print(
            f"警告: client {idx} 的 {split} 数据样本数不一致: "
            f"len(x)={len(x_array)}, len(y)={len(y_array)}。将截断到 {sample_count}。"
        )
        x_array = x_array[:sample_count]
        y_array = y_array[:sample_count]

    if len(x_array) == 0:
        print(f"警告: client {idx} 的 {split} 数据在对齐后为空。")
        return []

    X_data = torch.tensor(x_array, dtype=torch.float32)
    y_data = torch.tensor(y_array, dtype=torch.int64)

    return [(x, y) for x, y in zip(X_data, y_data)]

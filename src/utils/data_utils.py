import os

import numpy as np
import torch


def get_partitioned_data_root(dataset_name):
    dataset_name = str(dataset_name).strip().strip("/\\") or "IoT"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", "..", "dataset", dataset_name))


def _empty_client_arrays():
    return {
        "x": np.array([], dtype=np.float32),
        "y": np.array([], dtype=np.int64),
    }


def read_data(dataset_name, idx, is_train=True):
    """
    Read per-client npy data from:
    ../dataset/{dataset_name}/[train|test]/{client_id}/X.npy and y.npy
    """
    partitioned_data_root = get_partitioned_data_root(dataset_name)
    split = "train" if is_train else "test"
    client_data_path = os.path.join(partitioned_data_root, split, str(idx))
    x_path = os.path.join(client_data_path, "X.npy")
    y_path = os.path.join(client_data_path, "y.npy")

    if not os.path.isdir(client_data_path):
        print(
            f"Warning: client {idx} {split} directory does not exist: '{client_data_path}'. Returning empty arrays."
        )
        return _empty_client_arrays()

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        print(
            f"Warning: client {idx} {split} files are missing: X='{x_path}', y='{y_path}'. Returning empty arrays."
        )
        return _empty_client_arrays()

    try:
        x_data = np.load(x_path, allow_pickle=False)
        y_data = np.load(y_path, allow_pickle=False)
    except Exception as exc:
        print(
            f"Warning: failed to load client {idx} {split} npy files: {exc}. Returning empty arrays."
        )
        return _empty_client_arrays()

    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.int64)
    return {"x": x_data, "y": y_data}


def read_client_data(dataset, idx, is_train=True):
    """
    Read and prepare client data for a PyTorch DataLoader.
    Returns a list of (feature, label) tuples.
    """
    client_raw_data = read_data(dataset, idx, is_train)
    split = "train" if is_train else "test"

    if len(client_raw_data["x"]) == 0:
        print(f"Warning: client {idx} {split} data is empty.")
        return []

    x_array = client_raw_data["x"]
    y_array = client_raw_data["y"]

    if x_array.ndim == 1:
        x_array = np.expand_dims(x_array, axis=0)
    if y_array.ndim == 0:
        y_array = np.expand_dims(y_array, axis=0)
    else:
        y_array = y_array.reshape(-1)

    if len(x_array) != len(y_array):
        sample_count = min(len(x_array), len(y_array))
        print(
            f"Warning: client {idx} {split} sample counts do not match. "
            f"len(x)={len(x_array)}, len(y)={len(y_array)}. Truncating to {sample_count}."
        )
        x_array = x_array[:sample_count]
        y_array = y_array[:sample_count]

    if len(x_array) == 0:
        print(f"Warning: client {idx} {split} data is empty after alignment.")
        return []

    x_data = torch.tensor(x_array, dtype=torch.float32)
    y_data = torch.tensor(y_array, dtype=torch.int64)
    return [(x, y) for x, y in zip(x_data, y_data)]

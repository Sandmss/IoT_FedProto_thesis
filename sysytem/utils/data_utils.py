# --- START OF FILE data_utils.py ---

import numpy as np
import os
import torch
from PIL import Image
import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

PARTITIONED_DATA_ROOT = "../dataset_v3/dataset_v3/"


def get_partitioned_data_root(dataset_name):
    dataset_name = str(dataset_name).strip().strip("/\\")
    if not dataset_name:
        return PARTITIONED_DATA_ROOT

    return os.path.join("..", dataset_name, dataset_name)
# 这个字典现在是所有数据加载的唯一真实来
CLASS_TO_LABEL = {
    "DDoS": 0, "DoS": 1,
    "Mirai": 2,
    "Benign": 3,
    "Web-based": 4,
    "Recon": 5,
    "Spoofing": 6,
    "BruteForce": 7
}

# 假设 CLASS_TO_LABEL 是全局定义的类别映射字典
# 例如：CLASS_TO_LABEL = {'classA': 0, 'classB': 1, ...}
# 并且 PARTITIONED_DATA_ROOT 是全局定义的数据根目录
# 例如：PARTITIONED_DATA_ROOT = "../dataset"

########################################
# 1. 读取原始客户端数据
########################################
def read_data(dataset_name, idx, is_train=True):
    """
    读取指定客户端 (idx) 的图像数据。
    它会从 PARTITIONED_DATA_ROOT/[train/test]/[client_id]/[class_name] 结构中加载PNG图像。

    Args:
        dataset_name (str): 数据集名称 (例如 "IoT_MFR")。主要用于兼容性。
        idx (int): 客户端ID。
        is_train (bool): 是否加载训练数据 (True 为 train，False 为 test)。

    Returns:
        dict: 包含 'x' (图像数组) 和 'y' (标签数组)。
    """
    # 确定训练或测试子目录
    partitioned_data_root = get_partitioned_data_root(dataset_name)
    data_type_subdir = "train" if is_train else "test"

    # 构造当前客户端的路径，例如 ../dataset/train/0/ 或 ../dataset/test/0/
    client_data_path = os.path.join(partitioned_data_root, data_type_subdir, str(idx))

    if not os.path.exists(client_data_path):
        print(f"警告: 客户端 {idx} 的数据目录 '{client_data_path}' 不存在。返回空数据。")
        return {'x': [], 'y': []}

    all_images_x = []
    all_labels_y = []

    # 遍历该客户端的所有类别子目录
    for class_name in os.listdir(client_data_path):
        class_dir_path = os.path.join(client_data_path, class_name)

        if os.path.isdir(class_dir_path):
            label = CLASS_TO_LABEL.get(class_name)
            if label is None:
                #print(f"警告: 客户端 {idx} 中发现未知类别 '{class_name}'，已跳过。")
                continue

            # 遍历类别文件夹中的所有 PNG 文件
            for img_file in glob.glob(os.path.join(class_dir_path, "*.png")):
                try:
                    # 读取为灰度图
                    img = Image.open(img_file).convert('L')

                    # 如果不是 40x40，则调整尺寸
                    if img.size != (40, 40):
                        img = img.resize((40, 40), Image.Resampling.LANCZOS)

                    # 转换为 numpy 数组
                    img_array = np.array(img, dtype=np.float32)
                    all_images_x.append(img_array)
                    all_labels_y.append(label)

                except Exception as e:
                    print(f"警告: 处理图像 '{img_file}' (客户端 {idx}, 类别 {class_name}) 时出错: {e}")
                    continue

    return {'x': np.array(all_images_x), 'y': np.array(all_labels_y)}

########################################
# 2. 客户端数据读取 (含归一化 + 添加通道维度)
########################################
def read_client_data(dataset, idx, is_train=True):
    """
    读取并准备客户端数据，返回适合 DataLoader 的列表 [(特征, 标签), ...]。
    """
    client_raw_data = read_data(dataset, idx, is_train)

    if len(client_raw_data['x']) == 0:
        print(f"客户端 {idx} 在 {'训练' if is_train else '测试'} 集中没有数据。")
        return []

    # 转为张量并归一化到 [0,1]
    X_data = torch.Tensor(client_raw_data['x']).type(torch.float32)
    y_data = torch.Tensor(client_raw_data['y']).type(torch.int64)

    # # 添加通道维度 (N, 1, H, W)
    # X_data = X_data.unsqueeze(1)
    #4.1为了htfe8修改
    # 这会把维度从 (N, 40, 40) 变成 (N, 3, 40, 40)
    X_data = X_data.unsqueeze(1).repeat(1, 3, 1, 1)

    return [(x, y) for x, y in zip(X_data, y_data)]

########################################
# 3. 全局测试集读取 (与客户端一致的处理)
########################################
# def read_global_test_data(global_test_root,batch_size):
#     """
#     读取全局测试集，数据结构为:
#         global_test_root/[class_name]/*.png

#     返回 [(特征, 标签), ...]，与 read_client_data 输出一致。
#     """
#     all_images_x = []
#     all_labels_y = []

#     # 遍历每个类别文件夹
#     for class_name in os.listdir(global_test_root):
#         class_dir_path = os.path.join(global_test_root, class_name)
#         if not os.path.isdir(class_dir_path):
#             continue

#         label = CLASS_TO_LABEL.get(class_name)
#         if label is None:
#             print(f"警告: 全局测试集中发现未知类别 '{class_name}'，已跳过。")
#             continue

#         for img_file in glob.glob(os.path.join(class_dir_path, "*.png")):
#             try:
#                 img = Image.open(img_file).convert('L')

#                 # 调整到 40x40
#                 if img.size != (40, 40):
#                     img = img.resize((40, 40), Image.Resampling.LANCZOS)

#                 img_array = np.array(img, dtype=np.float32)
#                 all_images_x.append(img_array)
#                 all_labels_y.append(label)

#             except Exception as e:
#                 print(f"警告: 处理全局测试集图像 '{img_file}' 时出错: {e}")
#                 continue

#     if len(all_images_x) == 0:
#         print(f"警告: 全局测试集中没有找到任何图像。")
#         return []

#     # 转为张量并归一化
#     X_data = torch.Tensor(np.array(all_images_x)).type(torch.float32)
#     y_data = torch.Tensor(np.array(all_labels_y)).type(torch.int64)

#     # 添加通道维度
#     X_data = X_data.unsqueeze(1)
#     dataset = [(x, y) for x, y in zip(X_data, y_data)]

#     return DataLoader(dataset, batch_size=batch_size, shuffle=False)

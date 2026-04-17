import os
import glob
import shutil
import numpy as np
import random
from tqdm import tqdm
random.seed(42); np.random.seed(42)

def makedir(path):
    """
    创建目录。如果目录已存在，则忽略错误。
    使用 os.makedirs 可以递归创建所有不存在的父目录。
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"警告: 无法创建目录 '{path}': {e}")


def distribute_data_to_clients(
        mfr_base_directory,
        output_base_directory,
        num_clients=10,
        train_ratio=0.75,
        dirichlet_alpha=0.5
):
    """
    将MFR图像数据根据迪利克雷分布分配给客户端，并划分为训练集和测试集。

    Args:
        mfr_base_directory (str): 包含类别子目录的MFR图像数据的根目录路径。
                                  例如: "mfr" (内部结构如 mfr/Benign/img.png)
        output_base_directory (str): 分布数据存储的根目录路径。
                                     例如: "distributed_mfr"
        num_clients (int): 客户端的数量。
        train_ratio (float): 用于训练集的数据比例 (0.0 到 1.0)。
        dirichlet_alpha (float): 迪利克雷分布的alpha参数。
                                 值越小，数据在客户端间分布越不均匀 (Non-IID)。
    """
    print(f"开始数据分配，客户端数量: {num_clients}, 训练比例: {train_ratio * 100}%, Dirichlet Alpha: {dirichlet_alpha}")

    # 定义所有类别名，确保顺序与实际目录一致或通过glob获取
    # 这里假设用户给出的类别名是准确的
    categories = ["Benign", "DDoS", "DoS", "Mirai", "Web-based", "Spoofing", "Recon", "BruteForce"]

    # 存储每个类别的所有图片路径
    all_category_images = {cat: [] for cat in categories}

    print("收集所有类别的图片文件...")
    for category in categories:
        category_path = os.path.join(mfr_base_directory, category)
        if not os.path.isdir(category_path):
            print(f"警告: 未找到类别目录: {category_path}，跳过此类别。")
            continue
        # 获取所有 .png 图片文件
        image_files = glob.glob(os.path.join(category_path, "*.png"))
        all_category_images[category] = sorted(image_files)  # 排序以保证每次运行结果一致性
        print(f"  类别 '{category}': 找到 {len(image_files)} 张图片。")

    # 初始化客户端数据存储
    # client_data[client_id][category_name] = list_of_image_paths
    client_data_distribution = {
        client_id: {cat: [] for cat in categories} for client_id in range(num_clients)
    }

    print("开始根据迪利克雷分布分配数据到客户端...")
    for category, images in all_category_images.items():
        if not images:
            continue

        num_images_in_category = len(images)
        # 生成迪利克雷分布的比例
        # dirichlet_alpha 越小，分配越不均匀
        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients)
        # 将比例转换为每个客户端应获得的数量
        client_counts = (proportions * num_images_in_category).astype(int)

        # 调整计数以确保总数等于原始图片数量
        # 由于向下取整，总和可能小于 num_images_in_category
        # 将剩余的图片随机分配给客户端
        remaining_images = num_images_in_category - np.sum(client_counts)
        if remaining_images > 0:
            # 随机选择客户端增加数量
            for i in random.sample(range(num_clients), k=remaining_images):
                client_counts[i] += 1

        # 确保总和精确等于原始图片数量
        assert np.sum(client_counts) == num_images_in_category, \
            f"迪利克雷分配计数不匹配: {np.sum(client_counts)} != {num_images_in_category}"

        # 随机打乱图片，然后按计算的计数分配给客户端
        random.shuffle(images)
        current_idx = 0
        for client_id in range(num_clients):
            count_for_client = client_counts[client_id]
            client_data_distribution[client_id][category].extend(
                images[current_idx: current_idx + count_for_client]
            )
            current_idx += count_for_client
        print(f"  类别 '{category}' 已分配到 {num_clients} 个客户端。")

    print("创建输出目录并复制文件...")
    # 创建训练和测试的根目录
    makedir(os.path.join(output_base_directory, "train"))
    makedir(os.path.join(output_base_directory, "test"))
    stats_summary = {client_id: {"total_train": 0, "total_test": 0, "details": {}} for client_id in range(num_clients)}

    for client_id in tqdm(range(num_clients), desc="处理客户端数据"):
        for category in categories:
            images_for_client_category = client_data_distribution[client_id][category]
            if not images_for_client_category:
                continue

            # 划分训练集和测试集
            num_images = len(images_for_client_category)
            # 增加保底逻辑：如果该类别有图片，训练集至少分到 1 张
            if num_images == 1:
                num_train = 1
            else:
                num_train = max(1, int(num_images * train_ratio))

            random.shuffle(images_for_client_category)  # 再次打乱以确保随机划分
            train_images = images_for_client_category[:num_train]
            test_images = images_for_client_category[num_train:]
            # >>>>> 修改点 2: 记录当前类别在该客户端的训练/测试数量 <<<<<
            stats_summary[client_id]["total_train"] += len(train_images)
            stats_summary[client_id]["total_test"] += len(test_images)
            stats_summary[client_id]["details"][category] = {
                "train": len(train_images),
                "test": len(test_images)
            }

            # 为训练集创建目录并复制文件
            train_output_path = os.path.join(
                output_base_directory, "train", str(client_id), category
            )
            makedir(train_output_path)
            for img_path in train_images:
                try:
                    shutil.copy(img_path, train_output_path)
                except Exception as e:
                    print(f"复制文件失败: {img_path} 到 {train_output_path}。错误: {e}")

            # 为测试集创建目录并复制文件
            test_output_path = os.path.join(
                output_base_directory, "test", str(client_id), category
            )
            makedir(test_output_path)
            for img_path in test_images:
                try:
                    shutil.copy(img_path, test_output_path)
                except Exception as e:
                    print(f"复制文件失败: {img_path} 到 {test_output_path}。错误: {e}")

    print(f"\n数据分配和复制完成。结果保存在: {output_base_directory}")
    print(f"\n数据分配和复制完成。结果保存在: {output_base_directory}")

    # >>>>> 修改点 3: 格式化并打印统计信息 <<<<<
    print("\n" + "=" * 80)
    print(f"{'客户端数据分布统计 (Client Data Distribution Statistics)':^80}")
    print("=" * 80)

    # 打印表头
    print(f"{'Client ID':<10} | {'Total Train':<12} | {'Total Test':<12} | {'Category Breakdown (Train/Test)'}")
    print("-" * 80)

    for client_id in range(num_clients):
        s = stats_summary[client_id]
        total_train = s['total_train']
        total_test = s['total_test']

        # 将详情字典转换为字符串: "Benign:100/30, DDoS:50/10"
        details_str_list = []
        for cat, count in s['details'].items():
            if count['train'] > 0 or count['test'] > 0:
                details_str_list.append(f"{cat}:{count['train']}/{count['test']}")

        details_str = ", ".join(details_str_list)
        if not details_str:
            details_str = "No Data"

        print(f"{client_id:<10} | {total_train:<12} | {total_test:<12} | {details_str}")

    print("=" * 80)
    print("注意: 详情列格式为 '类别名:训练集数量/测试集数量'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 配置你的路径和参数
    # 请根据你实际的MFR图片数据的根目录进行修改
    mfr_base_directory = "./mfr"  # 假设你的mfr目录在当前脚本的同级
    # 你希望将处理后的数据保存到的根目录
    output_base_directory = "./dataset_noniid0.5_v1/"

    NUM_CLIENTS = 10
    TRAIN_RATIO = 0.75  # 75% 训练，25% 测试
    DIRICHLET_ALPHA = 0.5  # 迪利克雷参数，值越小，客户端数据分布越不均匀

    distribute_data_to_clients(
        mfr_base_directory,
        output_base_directory,
        num_clients=NUM_CLIENTS,
        train_ratio=TRAIN_RATIO,
        dirichlet_alpha=DIRICHLET_ALPHA
    )
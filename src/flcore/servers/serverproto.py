

import time
import torch
import torch.nn as nn
import numpy as np
import hashlib
from flcore.clients.clientproto import clientproto  # 客户端代码可以保持不变
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


# 保留哈希验证函数，用于调试
def protos_hash(protos):
    """计算原型字典的哈希值。"""
    if not protos:
        return "empty_or_none"
    s = b''
    for key in sorted(protos.keys()):
        tensor = protos[key]
        if isinstance(tensor, torch.Tensor):
            s += tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(s).hexdigest()


class FedProto(Server):



    # ### MODIFIED: 支持打印 Class 0-3 的调试函数 ###
    def debug_print_protos(self, round_num):
        """
        调试专用：打印全局原型和 Client 0 本地原型的部分数据（Class 0, 1, 2, 3）。
        """
        print(f"\n+++ DEBUG: Round {round_num} 原型检查 (前10维) +++")

        # 1. 预先加载文件 (避免在循环中重复读取硬盘，提高效率)
        g_protos = load_item(self.role, 'global_protos', self.save_folder_name)
        c0_protos = load_item('Client_0', 'protos', self.save_folder_name)

        # 2. 指定要检查的类别列表
        target_classes = [0, 1, 2, 3]

        for cls in target_classes:
            # 如果数据集类别很少(例如只有2类)，跳过不存在的类别
            if cls >= self.num_classes:
                continue

            print(f"--- [Class {cls}] ---")

            # (A) 打印全局原型
            if g_protos is None:
                print(f"  Global   : 文件不存在")
            elif cls in g_protos:
                # 获取数据 -> 展平 -> 转numpy -> 取前10个
                g_data = g_protos[cls].flatten().detach().cpu().numpy()[:10]
                g_str = ", ".join([f"{x:.4f}" for x in g_data])
                print(f"  Global   : [{g_str}, ...]")
            else:
                print(f"  Global   : (字典中无此类别)")

            # (B) 打印 Client 0 本地原型
            if c0_protos is None:
                print(f"  Client 0 : 文件不存在 (可能未参与训练)")
            elif cls in c0_protos:
                c0_data = c0_protos[cls].flatten().detach().cpu().numpy()[:10]
                c0_str = ", ".join([f"{x:.4f}" for x in c0_data])
                print(f"  Client 0 : [{c0_str}, ...]")
            else:
                print(f"  Client 0 : (本地无此类别数据)")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientproto)
        # 预先落盘一个空的 global_protos，避免首轮聚合前客户端反复加载时报“文件未找到”。
        save_item(None, self.role, 'global_protos', self.save_folder_name)

        print(f"\n参与比例 / 客户端总数: {self.join_ratio} / {self.num_clients}")
        print("消融实验服务器和客户端创建完成。")

        self.Budget = []
        self.num_classes = args.num_classes

        # ### 新增代码 START ###
        # 用于跟踪迄今为止的最佳准确率和轮次
        self.best_test_acc = (0, 0)  # (准确率, 轮次)
        self.best_global_test_acc = (0, 0)
        #self.best_global_test_acc_local_proto = (0, 0)
        self.best_local_proto_test_acc = (0, 0)
        # ### 新增代码 END ###
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

    def train(self):
        """
        消融实验的主训练循环。
        """
        print(f"\n------------- 全局轮次: 0 (初始状态) -------------")
        self.evaluate()
        print("--------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            print(f"\n------------- 全局轮次: {i} -------------")

            # 1. 选择客户端
            self.selected_clients = self.select_clients()

            # 2. 客户端本地训练 (这部分不变)
            for client in self.selected_clients:
                client.train()


            self.aggregate_protos()
            if i % self.eval_gap == 0:
                print(f"--- 轮次 {i} 评估 ---")
                global_protos = load_item(self.role, 'global_protos', self.save_folder_name)
                print(f"当前 global_protos 哈希值: {protos_hash(global_protos)}")

                self.evaluate()  # 评估本地测试集

                # ### 新增代码 START ###
                # 更新迄今为止的最佳准确率
                if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                    self.best_test_acc = (self.rs_test_acc[-1], i)
                    # --- [新增] 保存最佳状态 ---
                    print(f"检测到新最佳准确率，正在保存最佳模型检查点 (Round {i})...")
                    self.save_best_checkpoint()

                # ### 新增代码 START ###
                # 每 100 轮打印一次最佳结果
            if i % 50 == 0 and i > 0:
                print("\n--- 迄今为止最佳准确率 ---")
                if self.best_test_acc[1] > 0:
                    print(f"本地测试集: {self.best_test_acc[0]:.4f} (在第 {self.best_test_acc[1]} 轮)")


            self.Budget.append(time.time() - s_t)
            print(f"本轮耗时: {self.Budget[-1]:.2f} 秒")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n实验完成。最佳准确率:")
        # 打印最终结果的逻辑保持不变
        if self.rs_test_acc:
            print(f"  - 本地测试集: {max(self.rs_test_acc):.4f}")

        if self.Budget:
            print("平均每轮训练时间:", sum(self.Budget) / len(self.Budget))
        self.draw_tsne()
        self.draw_proto_distribution_tsne()
        self.save_results()

    # ### MODIFIED START: 新的核心服务器逻辑 ###
    def aggregate_protos(self):
        """
        接收来自客户端的原型，聚合，并计算类间距离。
        """
        assert (len(self.selected_clients) > 0)

        uploaded_protos_per_client = []
        uploaded_weights_per_client = []  # 【新增】存放每个客户端传来的类别权重
        client_data_sizes = []
        for client in self.selected_clients:
            protos = load_item(client.role, 'protos', client.save_folder_name)
            # 【新增】加载对应的类别样本数权重
            weights = load_item(client.role, 'proto_weights', client.save_folder_name)
            if protos:
                uploaded_protos_per_client.append(protos)
                uploaded_weights_per_client.append(weights)

        if not uploaded_protos_per_client:
            print("警告：本轮没有收到任何客户端原型，跳过服务器更新。")
            return

    # 1. 按类别加权聚合原型 (得到 global_protos)
        global_protos = proto_aggregation_with_weights(uploaded_protos_per_client, uploaded_weights_per_client)

        # 2. 保存聚合结果
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
        print(f"服务器已基于【类别样本量加权】聚合了来自 {len(uploaded_protos_per_client)} 个客户端的原型。")
        # 3. ### ADDED: 计算类间距离 (Gap Calculation) ###
        # 初始化 gap 为无穷大
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9

        # 获取当前存在的类别列表
        existing_classes = list(global_protos.keys())

        # 双重循环计算两两之间的距离
        for k1 in existing_classes:
            for k2 in existing_classes:
                if k1 > k2:  # 避免重复计算 (A,B) 和 (B,A) 以及 (A,A)
                    # 计算欧氏距离 (L2 Norm)
                    dis = torch.norm(global_protos[k1] - global_protos[k2], p=2)

                    # 更新 k1 和 k2 的最小边界距离
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        # 找到全局最小 Gap
        self.min_gap = torch.min(self.gap)

        # 处理异常值：如果有类别从未出现过，或者gap仍为初始值1e9，将其设为min_gap
        # 这样打印出来好看一些，不至于出现 100000000.0
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap

        self.max_gap = torch.max(self.gap)

        # 打印调试信息
        print('\n--- 类间距离统计 (Inter-class Margins) ---')
        print('各类别最小类间距离:', self.gap.detach().cpu().numpy()) # 如果类别太多可以注释掉这一行
        print(f'全局最小类间距离 (Min Gap): {self.min_gap.item():.4f}')
        print(f'全局最大类间距离 (Max Gap): {self.max_gap.item():.4f}')
        print('------------------------------------------')
    def save_best_checkpoint(self):
        """保存当前全局原型和所有客户端模型为'最佳'版本"""
        # 1. 保存最佳全局原型
        global_protos = load_item(self.role, 'global_protos', self.save_folder_name)
        save_item(global_protos, self.role, 'best_global_protos', self.save_folder_name)

        # 2. 通知所有客户端保存他们的最佳模型
        # 注意：这里我们假设所有客户端在这一轮都刚刚更新过，或者我们保存当前时刻的状态
        for client in self.clients:
            client.save_best_model()

    def _get_figure_output_dir(self):
        result_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "figures")
        )
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def _build_figure_prefix(self):
        return f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}"

    def _load_item_if_exists(self, role, item_name):
        file_path = os.path.join(self.save_folder_name, f"{role}_{item_name}.pt")
        if not os.path.isfile(file_path):
            return None
        return load_item(role, item_name, self.save_folder_name)

    def draw_tsne(self):
        """
        为最佳客户端特征生成 t-SNE 可视化。
        如果样本不足或降维失败，仅打印提示，不影响训练收尾。
        """
        features_list = []
        labels_list = []

        for client in self.clients:
            if not hasattr(client, "extract_features"):
                continue
            features, labels = client.extract_features()
            if features.size == 0 or labels.size == 0:
                continue
            features_list.append(features)
            labels_list.append(labels)

        if not features_list:
            print("跳过特征 t-SNE：没有可用的客户端特征。")
            return

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if len(features) < 2:
            print("跳过特征 t-SNE：样本数不足 2。")
            return

        perplexity = min(30, len(features) - 1)
        if perplexity < 1:
            print("跳过特征 t-SNE：perplexity 不合法。")
            return

        try:
            embedded = TSNE(
                n_components=2,
                random_state=0,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
            ).fit_transform(features)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                embedded[:, 0],
                embedded[:, 1],
                c=labels,
                cmap="tab20",
                s=10,
                alpha=0.8,
            )
            plt.title("FedProto Feature t-SNE")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(scatter, label="Class")
            plt.tight_layout()

            output_path = os.path.join(
                self._get_figure_output_dir(),
                f"{self._build_figure_prefix()}_feature_tsne.png",
            )
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"已保存特征 t-SNE 图: {output_path}")
        except Exception as exc:
            plt.close("all")
            print(f"跳过特征 t-SNE：生成失败 ({exc})。")

    def draw_proto_distribution_tsne(self):
        """
        为最佳全局原型生成二维可视化。
        原型类别不足或降维失败时仅记录提示，避免训练结束报错。
        """
        protos = self._load_item_if_exists(self.role, 'best_global_protos')
        if protos is None:
            protos = self._load_item_if_exists(self.role, 'global_protos')

        if not protos:
            print("跳过原型可视化：没有可用的全局原型。")
            return

        class_ids = sorted(protos.keys())
        proto_vectors = []
        valid_class_ids = []
        for class_id in class_ids:
            proto = protos[class_id]
            if isinstance(proto, torch.Tensor):
                proto_vectors.append(proto.detach().cpu().numpy().reshape(-1))
                valid_class_ids.append(class_id)

        if len(proto_vectors) < 2:
            print("跳过原型可视化：有效原型数量不足 2。")
            return

        proto_vectors = np.stack(proto_vectors, axis=0)

        try:
            if proto_vectors.shape[0] == 2:
                embedded = np.column_stack(
                    [proto_vectors[:, 0], np.zeros(proto_vectors.shape[0], dtype=np.float32)]
                )
            else:
                perplexity = min(5, proto_vectors.shape[0] - 1)
                if perplexity < 1:
                    print("跳过原型可视化：perplexity 不合法。")
                    return
                embedded = TSNE(
                    n_components=2,
                    random_state=0,
                    init="pca",
                    learning_rate="auto",
                    perplexity=perplexity,
                ).fit_transform(proto_vectors)

            plt.figure(figsize=(8, 6))
            plt.scatter(embedded[:, 0], embedded[:, 1], s=80, c=valid_class_ids, cmap="tab20")
            for idx, class_id in enumerate(valid_class_ids):
                plt.annotate(str(class_id), (embedded[idx, 0], embedded[idx, 1]))
            plt.title("FedProto Global Prototype Distribution")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.tight_layout()

            output_path = os.path.join(
                self._get_figure_output_dir(),
                f"{self._build_figure_prefix()}_prototype_distribution.png",
            )
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"已保存原型分布图: {output_path}")
        except Exception as exc:
            plt.close("all")
            print(f"跳过原型可视化：生成失败 ({exc})。")



# ### MODIFIED: 辅助函数更新为支持加权平均 ###
def proto_aggregation_with_weights(protos_list, weights_list):
    """
    按类别（Class-specific）进行加权平均聚合。
    权重是该类别在该客户端上的具体样本数量。
    """
    proto_clusters = defaultdict(list)
    weight_clusters = defaultdict(list)

    # 按类别收集所有原型及其对应的【该类别的样本数】
    for i, protos in enumerate(protos_list):
        weights = weights_list[i]
        for k in protos.keys():
            proto_clusters[k].append(protos[k])
            # 取出第 i 个客户端上，类别 k 的具体样本数作为权重
            weight_clusters[k].append(weights[k])

    aggregated_protos = defaultdict(list)
    for k in proto_clusters.keys():
        class_protos = torch.stack(proto_clusters[k])  # (N, D)
        target_device = class_protos.device

        # 将该类别的权重转为 Tensor (N, 1)
        class_weights = torch.tensor(weight_clusters[k], dtype=torch.float32, device=target_device).view(-1, 1)

        total_weight = torch.sum(class_weights)

        if total_weight == 0:
            aggregated_protos[k] = torch.mean(class_protos, dim=0).detach()
        else:
            # 加权平均计算：sum(原型 * 权重) / 总权重
            weighted_sum = torch.sum(class_protos * class_weights, dim=0)
            aggregated_protos[k] = (weighted_sum / total_weight).detach()

    return aggregated_protos


import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from flcore.clients.clientavg import clientAvg
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item


class FedAvg(Server):
    """
    FedAvg 算法的服务器实现。
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAvg)

        print(f"\n参与比例 / 客户端总数: {self.join_ratio} / {self.num_clients}")
        print("FedAvg 服务器和客户端创建完成。")

        self.Budget = []
        self.num_classes = args.num_classes
        self.best_test_acc = (0, 0)  # (Acc, Round)

        # 初始化全局模型
        # 借用第一个客户端的模型结构来初始化
        client_model = load_item(self.clients[0].role, 'model', self.save_folder_name)
        self.global_model = copy.deepcopy(client_model)
        save_item(self.global_model, self.role, 'global_model', self.save_folder_name)

    # >>>>> 新增方法: 将全局模型强制分发给所有客户端 <<<<<
    def set_global_model_to_all_clients(self):
        print("正在将全局模型同步到所有客户端以进行测试...")
        # 1. 加载最新的全局模型
        global_model = load_item(self.role, 'global_model', self.save_folder_name)

        # 2. 覆盖所有客户端的本地模型文件
        # 这样当调用 clientbase.test_metrics() 时，读取的就是全局模型
        for client in self.clients:
            save_item(global_model, client.role, 'model', client.save_folder_name)

    # >>>>> 结束 <<<<<

    def train(self):
        """
        FedAvg 主训练循环
        """
        print(f"\n------------- 全局轮次: 0 (初始状态) -------------")
        # >>> 修改点 1: 初始评估前同步模型 <<<
        self.set_global_model_to_all_clients()
        self.evaluate()
        print("--------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()

            # 1. 选择客户端
            self.selected_clients = self.select_clients()

            # 2. 客户端本地训练
            for client in self.selected_clients:
                client.train()

            # 3. 服务器聚合
            self.aggregate_models()

            # 4. 评估
            if i % self.eval_gap == 0:
                print(f"\n------------- 全局轮次: {i} (评估) -------------")
                # >>> 修改点 2: 评估前同步模型 <<<
                # 这一步至关重要，它保证了 self.evaluate() 测的是全局模型
                self.set_global_model_to_all_clients()

                self.evaluate()

                if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                    self.best_test_acc = (self.rs_test_acc[-1], i)
                    print(f"检测到新最佳准确率，正在保存最佳模型检查点 (Round {i})...")
                    self.save_best_checkpoint()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n实验完成。最佳准确率:")
        if len(self.rs_test_acc) > 0:
            print(f"  测试集准确率: {self.best_test_acc[0]:.4f} (在第 {self.best_test_acc[1]} 轮)")

        if len(self.Budget) > 0:
            print("平均每轮训练时间:", sum(self.Budget) / len(self.Budget))

        self.draw_tsne()
        self.save_results()

    def aggregate_models(self):
        assert (len(self.selected_clients) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()

        total_train_samples = 0
        for client in self.selected_clients:
            total_train_samples += client.train_samples

        for client in self.selected_clients:
            client_model = load_item(client.role, 'model', client.save_folder_name)
            weight = client.train_samples / total_train_samples
            for gl_param, cl_param in zip(global_model.parameters(), client_model.parameters()):
                gl_param.data += cl_param.data * weight

        self.global_model = global_model
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def save_best_checkpoint(self):
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        save_item(global_model, self.role, 'best_global_model', self.save_folder_name)
        # 注意：这里如果调用 client.save_best_model()，保存的也将是全局模型副本
        # 因为我们刚刚在 set_global_model_to_all_clients 里覆盖了它。这是合理的。
        for client in self.clients:
            client.save_best_model()

    def draw_tsne(self):
        # 保持原样，省略...
        # 略 (代码与原来一致)
        pass
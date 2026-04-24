import time
import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientavg import clientAvg
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import save_item


class FedAvg(Server):
    """
    FedAvg server implementation.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAvg)

        print(f"\n参与比例 / 客户端总数: {self.join_ratio} / {self.num_clients}")
        print("FedAvg 服务端和客户端创建完成。")

        self.Budget = []
        self.num_classes = args.num_classes
        self.best_test_acc = (0, 0)

        client_model = self.clients[0].model
        self.global_model = copy.deepcopy(client_model).to(self.device)
        save_item(self.global_model, self.role, 'global_model', self.save_folder_name)

    def set_global_model_to_clients(self, clients=None):
        target_clients = self.clients if clients is None else clients
        print("正在将全局模型同步到客户端以内存评估/训练...")
        for client in target_clients:
            client.model.load_state_dict(self.global_model.state_dict())

    def train(self):
        print("\n------------- 全局轮次: 0 (初始状态) -------------")
        self.set_global_model_to_clients()
        self.evaluate()
        print("--------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            stop_training = False

            self.selected_clients = self.select_clients()

            self.set_global_model_to_clients(self.selected_clients)
            for client in self.selected_clients:
                client.train()

            self.aggregate_models()
            self.record_round_overheads()

            if i % self.eval_gap == 0:
                print(f"\n------------- 全局轮次: {i} (评估) -------------")
                self.set_global_model_to_clients()
                self.evaluate()

                if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                    self.best_test_acc = (self.rs_test_acc[-1], i)
                    print(f"检测到新最优准确率，正在保存最佳模型检查点 (Round {i})...")
                    self.save_best_checkpoint()

                if self.auto_break and self.patience_should_stop_after_eval():
                    stop_training = True

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if stop_training:
                break

        print("\n实验完成。最优准确率:")
        if len(self.rs_test_acc) > 0:
            print(f"  测试集准确率: {self.best_test_acc[0]:.4f} (在第 {self.best_test_acc[1]} 轮)")

        if len(self.Budget) > 0:
            print("平均每轮训练时间:", sum(self.Budget) / len(self.Budget))

        if getattr(self.args, "skip_figures", False):
            print("已跳过 t-SNE (--skip_figures)。")
        else:
            self.draw_tsne()
        self.save_results()

    def aggregate_models(self):
        assert (len(self.selected_clients) > 0)

        global_model = copy.deepcopy(self.global_model)
        for param in global_model.parameters():
            param.data.zero_()

        total_train_samples = sum(max(client.train_samples, 0) for client in self.selected_clients)

        for client in self.selected_clients:
            client_model = client.model
            weight = (
                client.train_samples / total_train_samples
                if total_train_samples > 0
                else 1.0 / len(self.selected_clients)
            )
            for gl_param, cl_param in zip(global_model.parameters(), client_model.parameters()):
                gl_param.data += cl_param.data * weight

        self.global_model = global_model

    def save_best_checkpoint(self):
        save_item(self.global_model, self.role, 'best_global_model', self.save_folder_name)
        for client in self.clients:
            client.save_best_model()

    def draw_tsne(self):
        self.draw_feature_tsne(title="FedAvg Feature t-SNE")

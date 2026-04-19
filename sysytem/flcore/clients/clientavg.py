import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item


class clientAvg(Client):
    """
    FedAvg 算法的客户端实现。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        # FedAvg 不需要 MSELoss 用于原型，只需要标准的分类 Loss (父类通常已定义 self.loss)
        # 如果父类没定义，通常是 nn.CrossEntropyLoss()

    def train(self):
        """
        执行本地训练：
        1. 加载全局模型参数覆盖本地模型。
        2. 进行 SGD 训练。
        3. 保存更新后的本地模型。
        """
        trainloader = self.load_train_data()

        # 1. 加载本地模型结构
        model = load_item(self.role, 'model', self.save_folder_name)

        # 2. 【关键】加载 Server 端的全局模型，并覆盖本地参数 (Global Synchronization)
        # 注意：第一次训练时，Server 需要先保存一个 global_model
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        if global_model is not None:
            model.load_state_dict(global_model.state_dict())

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        # 模拟掉队/慢客户端
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 前向传播
                output = model(x)
                loss = self.loss(output, y)

                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        # 3. 保存训练后的模型供 Server 聚合
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def save_best_model(self):
        """
        保存最佳模型副本。
        """
        model = load_item(self.role, 'model', self.save_folder_name)
        save_item(model, self.role, 'best_model', self.save_folder_name)

    def extract_features(self):
        """
        加载最佳模型，提取本地测试集的特征向量和标签，用于 t-SNE 可视化。
        """
        model = load_item(self.role, 'best_model', self.save_folder_name)
        if model is None:
            model = load_item(self.role, 'model', self.save_folder_name)

        testloader = self.load_test_data()
        model.eval()

        features_list = []
        labels_list = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 假设 model.base 是提取特征的部分，model.head 是分类器
                # 如果模型结构不同，这里需要调整，例如 model.features(x)
                if hasattr(model, 'base'):
                    rep = model.base(x)
                else:
                    # 如果没有显式的 base，可能需要通过 hook 或者修改模型结构来获取
                    # 这里假设是个简单的 CNN，取倒数第二层输出
                    # 简便起见，这里假设使用了与 FedTGP 相同的模型结构
                    rep = model.base(x)

                features_list.append(rep.detach().cpu().numpy())
                labels_list.append(y.detach().cpu().numpy())

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            return features, labels
        else:
            return np.array([]), np.array([])
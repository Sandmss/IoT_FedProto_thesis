# --- START OF FILE client_protonet.py ---

import copy
import torch
import torch.nn as nn
import numpy as np
import time
import os
from flcore.clients.clientbase import Client, save_item, debug_log
from collections import defaultdict
from sklearn import metrics

class clientproto(Client):
    """
    FedTGP 算法的客户端实现。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.global_protos = None
        self.local_protos = None
        self.local_proto_weights = None
        self.best_protos = None
        self.current_round = 0
        self.proto_eval_mode = getattr(args, "proto_eval_mode", "classifier")

    def train(self):
        """
        执行本地训练，包含原型正则化。
        """
        trainloader = self.load_train_data()
        model = self.model
        global_protos = self.global_protos
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max(2, max_local_epochs // 2 + 1)))

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)
                ce_loss_value = float(loss.item())
                proto_loss_value = 0.0

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        proto = global_protos.get(y_c)
                        if isinstance(proto, torch.Tensor):
                            proto_new[i, :] = proto.data
                    proto_loss_tensor = self.loss_mse(proto_new, rep)
                    proto_loss_value = float(proto_loss_tensor.item())
                    loss += proto_loss_tensor * self.lamda

                if (
                    self.id == 0
                    and step == 0
                    and i == 0
                    and self.current_round in {1, 10, 50, 98, 100}
                ):
                    available_proto_classes = 0 if global_protos is None else len(global_protos)
                    batch_labels, batch_counts = torch.unique(y.detach().cpu(), return_counts=True)
                    # region agent log
                    debug_log(
                        "src/flcore/clients/clientproto.py:72",
                        "client train loss snapshot",
                        {
                            "round": int(self.current_round),
                            "client_id": int(self.id),
                            "ce_loss": ce_loss_value,
                            "proto_loss": proto_loss_value,
                            "lamda": float(self.lamda),
                            "feature_norm_mean": float(rep.detach().norm(dim=1).mean().item()),
                            "available_proto_classes": int(available_proto_classes),
                            "batch_label_hist": {
                                int(label): int(count)
                                for label, count in zip(batch_labels.tolist(), batch_counts.tolist())
                            },
                        },
                        run_id="fedproto-runtime",
                        hypothesis_id="H2",
                    )
                    # endregion

                optimizer.zero_grad()
                loss.backward()
                # max_norm 通常设置为 5.0 或 10.0，能有效防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        self.model = model
        self.collect_protos(model=model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self, model=None):
        """
        在本地训练数据上计算每个类别的平均特征表示（即局部原型），并保存。
        """
        trainloader = self.load_train_data()
        if model is None:
            model = self.model
        model.eval()

        protos = defaultdict(list)
        class_counts = defaultdict(int)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
                    class_counts[y_c] += 1

        self.local_protos = agg_func(protos)
        self.local_proto_weights = dict(class_counts)

    def collect_test_outputs(self):
        """
        标准测试方法：在【本地测试集】上使用【全局原型】进行分类。
        """
        if self.proto_eval_mode == "classifier":
            return super().collect_test_outputs()

        testloader = self.load_test_data()
        model = self.model
        global_protos = self.global_protos
        model.eval()

        test_acc, test_num = 0, 0
        y_prob = []
        y_true = []
        y_pred = []
        inference_time = 0.0

        if global_protos is not None:
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(testloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    rep = model.base(x)

                    output = torch.full(
                        (y.shape[0], self.num_classes),
                        float("inf"),
                        device=self.device,
                    )
                    valid_proto_items = [
                        (int(class_id), proto)
                        for class_id, proto in global_protos.items()
                        if isinstance(proto, torch.Tensor)
                    ]
                    if valid_proto_items:
                        class_ids = [class_id for class_id, _ in valid_proto_items]
                        proto_matrix = torch.stack(
                            [proto.to(self.device) for _, proto in valid_proto_items],
                            dim=0,
                        )
                        # rep: [B, D], proto_matrix: [C, D]
                        # 计算每个样本到每个原型的均方距离，避免逐样本逐类别 Python 循环。
                        distances = torch.mean(
                            (rep.unsqueeze(1) - proto_matrix.unsqueeze(0)) ** 2,
                            dim=2,
                        )
                        output[:, class_ids] = distances

                    proto_pred = torch.argmin(output, dim=1)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    inference_time += time.perf_counter() - start_time
                    proto_correct = torch.sum(proto_pred == y).item()
                    test_acc += proto_correct
                    test_num += y.shape[0]
                    y_prob.append(torch.softmax(-output, dim=1).detach().cpu().numpy())
                    y_true.append(y.detach().cpu().numpy())
                    y_pred.append(proto_pred.detach().cpu().numpy())

            if y_prob:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                y_pred = np.concatenate(y_pred, axis=0)
            else:
                y_prob = np.zeros((0, self.num_classes), dtype=np.float32)
                y_true = np.array([], dtype=np.int64)
                y_pred = np.array([], dtype=np.int64)

            auc_macro, auc_micro = self.compute_multiclass_auc(y_true, y_prob)
            precision, recall, f1, fpr = self.compute_classification_metrics(y_true, y_pred)
            fnr = self.compute_false_negative_rate(y_true, y_pred)
            if self.id in {0, 2} and self.current_round in {1, 10, 50, 98, 100}:
                pred_labels, pred_counts = np.unique(y_pred, return_counts=True)
                true_labels, true_counts = np.unique(y_true, return_counts=True)
                # region agent log
                debug_log(
                    "src/flcore/clients/clientproto.py:180",
                    "client eval prediction snapshot",
                    {
                        "round": int(self.current_round),
                        "client_id": int(self.id),
                        "test_acc": float(test_acc / max(test_num, 1)),
                        "auc_macro": float(auc_macro),
                        "auc_micro": float(auc_micro),
                        "fnr": float(fnr),
                        "available_proto_classes": 0 if global_protos is None else len(global_protos),
                        "pred_label_hist": {
                            int(label): int(count)
                            for label, count in zip(pred_labels.tolist(), pred_counts.tolist())
                        },
                        "true_label_hist": {
                            int(label): int(count)
                            for label, count in zip(true_labels.tolist(), true_counts.tolist())
                        },
                    },
                    run_id="fedproto-runtime",
                    hypothesis_id="H4",
                )
                # endregion
            return {
                "test_acc": int(test_acc),
                "test_num": int(test_num),
                "y_prob": y_prob,
                "y_true": y_true,
                "y_pred": y_pred,
                "inference_time": float(inference_time),
            }
        else:
            return {
                "test_acc": 0,
                "test_num": 0,
                "y_prob": np.zeros((0, self.num_classes), dtype=np.float32),
                "y_true": np.array([], dtype=np.int64),
                "y_pred": np.array([], dtype=np.int64),
                "inference_time": 0.0,
            }

    # def evaluate_on_global_test_set(self, global_protos=None):
    #     """
    #     [新增] 在全局测试集上使用全局原型进行评估。
    #     参数:
    #         global_protos (dict, optional): 如果为 None，则尝试从磁盘加载 Server 发来的最新原型。
    #     返回:
    #         (correct_count, total_count)
    #     """
    #     # 1. 加载全局测试数据
    #     testloader = self.load_global_test_data()
    #
    #     # 如果没有全局测试集数据，直接返回
    #     if testloader is None:
    #         return 0, 0
    #
    #     # 2. 加载模型
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     model.eval()
    #
    #     # 3. 确定使用的全局原型
    #     # 如果调用时没传，就去读文件
    #     if global_protos is None:
    #         global_protos = load_item('Server', 'global_protos', self.save_folder_name)
    #
    #     test_acc = 0
    #     test_num = 0
    #
    #     # 只有在存在全局原型的情况下才能进行测试
    #     if global_protos is not None:
    #         with torch.no_grad():
    #             for x, y in testloader:
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #
    #                 # 提取特征
    #                 rep = model.base(x)
    #
    #                 # 初始化距离矩阵 (Batch Size x Num Classes)
    #                 output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #
    #                 # 计算每个样本特征与每个类别全局原型的距离
    #                 for i, r in enumerate(rep):
    #                     for j, pro in global_protos.items():
    #                         # 确保原型数据有效（非空列表）
    #                         if type(pro) != type([]):
    #                             # 使用 MSE 计算距离，保持与 clienttgp1113 风格一致
    #                             output[i, j] = self.loss_mse(r, pro)
    #
    #                 # 预测距离最小的原型对应的类别
    #                 test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
    #                 test_num += y.shape[0]
    #
    #     return test_acc, test_num
    ### MODIFIED START ###
    # def test_with_local_prototype(self, test_loader=None):
    #     """
    #     新增的测试方法：在给定的测试集（即全局测试集）上使用【本地原型】进行分类。
    #     这个方法由服务器调用。
    #     如果服务器调用时不传 test_loader，则默认加载全局测试集。
    #     """
    #     # 1. 自动处理数据加载
    #     if test_loader is None:
    #         # 优先尝试加载全局测试集 (通常用于 Server 评估 Global Performance)
    #         if hasattr(self, 'load_global_test_data'):
    #             test_loader = self.load_global_test_data()
    #         else:
    #             # 如果没有实现 load_global_test_data，则回退到加载本地测试集
    #             test_loader = self.load_test_data()
    #
    #     # 如果依然没有数据，直接返回 0
    #     if test_loader is None:
    #         return 0, 0
    #
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     # 加载这个客户端自己的原型
    #     local_protos = load_item(self.role, 'protos', self.save_folder_name)
    #     model.eval()
    #
    #     test_acc, test_num = 0, 0
    #
    #     # 检查本地原型是否存在
    #     if local_protos:
    #         with torch.no_grad():
    #             for x, y in test_loader:
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #
    #                 rep = model.base(x)
    #
    #                 output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #                 for i, r in enumerate(rep):
    #                     # 使用本地原型进行分类
    #                     for j, pro in local_protos.items():
    #                         if type(pro) != type([]):
    #                             output[i, j] = self.loss_mse(r, pro)
    #
    #                 test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
    #                 test_num += y.shape[0]
    #
    #     return test_acc, test_num

    ### MODIFIED END ###

    def train_metrics(self):
        """
        重写训练评估方法，以包含原型正则化损失。
        """
        trainloader = self.load_train_data()
        model = self.model
        global_protos = self.global_protos
        model.eval()

        train_num, losses = 0, 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        proto = global_protos.get(y_c)
                        if isinstance(proto, torch.Tensor):
                            proto_new[i, :] = proto.data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    # def test_local_proto_on_local_test(self):
    #     """
    #     [新增] 在本地测试集上使用本地原型进行评估。
    #     返回: (正确样本数, 总样本数)
    #     """
    #     # 1. 加载本地测试数据
    #     testloader = self.load_test_data()
    #
    #     # 2. 加载模型和本地原型
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     local_protos = load_item(self.role, 'protos', self.save_folder_name)
    #
    #     model.eval()
    #
    #     test_acc = 0
    #     test_num = 0
    #
    #     # 如果没有本地原型（可能是还没训练或该客户端数据为空），直接返回
    #     if not local_protos:
    #         return 0, 0
    #
    #     with torch.no_grad():
    #         for x, y in testloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #
    #             # 提取特征
    #             rep = model.base(x)
    #
    #             # 初始化距离矩阵，默认为无穷大
    #             # 形状: [batch_size, num_classes]
    #             output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #
    #             for i, r in enumerate(rep):
    #                 # 遍历本地拥有的原型类别
    #                 for j, pro in local_protos.items():
    #                     # 确保 pro 是 tensor 数据而非列表
    #                     if type(pro) != type([]):
    #                         # 计算特征 r 与原型 pro 的距离 (使用 MSE，与类中定义一致)
    #                         output[i, j] = self.loss_mse(r, pro)
    #
    #             # 预测类别为距离最近的原型对应的类别
    #             pred = torch.argmin(output, dim=1)
    #
    #             test_acc += (torch.sum(pred == y)).item()
    #             test_num += y.shape[0]
    #
    #     return test_acc, test_num

    def save_best_model(self):
        """
        将当前模型保存为最佳模型副本。
        【修改】同时保存当前的原型为最佳原型副本。
        """
        # 1. 保存最佳模型
        self.best_model = copy.deepcopy(self.model)
        save_item(self.best_model, self.role, 'best_model', self.save_folder_name)

        # 2. 保存最佳原型 (新增)
        if self.local_protos:
            self.best_protos = copy.deepcopy(self.local_protos)
            save_item(self.best_protos, self.role, 'best_protos', self.save_folder_name)
    def extract_features(self, max_samples=None):
        """
        加载最佳模型，提取本地测试集的特征向量和标签，用于 t-SNE 可视化。
        返回: (features, labels) 的 numpy 数组
        """
        model = self.best_model if self.best_model is not None else self.model

        testloader = self.load_test_data()
        model.eval()

        features_list = []
        labels_list = []
        collected = 0
        rng = np.random.default_rng(0)

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 提取特征 (Rep)
                rep = model.base(x)

                rep_np = rep.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                if max_samples is not None:
                    remaining = max_samples - collected
                    if remaining <= 0:
                        break
                    if len(rep_np) > remaining:
                        sample_idx = rng.choice(len(rep_np), size=remaining, replace=False)
                        rep_np = rep_np[sample_idx]
                        y_np = y_np[sample_idx]

                features_list.append(rep_np)
                labels_list.append(y_np)
                collected += len(rep_np)
                if max_samples is not None and collected >= max_samples:
                    break

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            return features, labels
        else:
            return np.array([]), np.array([])



def agg_func(protos):
    """
    对字典中每个键对应的值（一个张量列表）进行平均。
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos

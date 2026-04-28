# --- START OF FILE clientbase.py ---

import copy
import json
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
# 从项目工具中导入数据读取和模型定义
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit

# region agent log
DEBUG_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "debug-bfc949.log")
)


def debug_log(location, message, data=None, run_id="fedproto-debug", hypothesis_id="H?"):
    payload = {
        "sessionId": "bfc949",
        "id": f"log_{time.time_ns()}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion


class Client(object):
    """
    联邦学习中所有客户端的基类 (Base Class)。
    它定义了客户端应具备的核心属性和方法，如数据加载、模型评估等。
    具体的联邦学习算法客户端（如 clientAvg, clientTGP）应继承自这个类。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 设置随机种子以保证可复现性
        torch.manual_seed(0)
        # 从参数中初始化核心属性
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # 客户端的唯一标识符 (整数)
        self.role = 'Client_' + str(self.id)  # 客户端的角色名，用于文件保存
        self.save_folder_name = args.save_folder_name_full  # 结果保存的文件夹路径

        self.num_classes = args.num_classes  # 数据集的类别数
        self.normal_class = getattr(args, "normal_class", 0)
        self.train_samples = train_samples  # 训练样本数量
        self.test_samples = test_samples  # 测试样本数量
        self.batch_size = args.batch_size  # 批处理大小
        self.num_workers = args.num_workers
        self.pin_memory = bool(args.pin_memory) and self.device == "cuda"
        self.learning_rate = args.local_learning_rate  # 本地学习率
        self.local_epochs = args.local_epochs  # 本地训练轮数
        self.model = None
        self.best_model = None

        model_path = os.path.join(self.save_folder_name, f"{self.role}_model.pt")
        model_template = self._build_model_template()
        if os.path.isfile(model_path):
            self.model = load_item(
                self.role,
                'model',
                self.save_folder_name,
                model_template=model_template,
            )
            if self.model is None:
                self.model = model_template
                save_item(self.model, self.role, 'model', self.save_folder_name)
        else:
            model = model_template
            save_item(model, self.role, 'model', self.save_folder_name)
            self.model = model

        # 从关键字参数中获取慢客户端标志，用于模拟异构性
        self.train_slow = kwargs['train_slow']  # 标记是否为训练慢的客户端
        self.send_slow = kwargs['send_slow']  # 标记是否为通信慢的客户端
        # 用于记录时间和开销的字典
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 定义默认的损失函数
        self.loss = nn.CrossEntropyLoss()

    def _build_model_template(self):
        return BaseHeadSplit(self.args, self.id).to(self.device)

    def _load_local_model(self, item_name='model'):
        return load_item(
            self.role,
            item_name,
            self.save_folder_name,
            model_template=self._build_model_template(),
        )

    # def load_global_test_data(self, batch_size=None):
    #     """让客户端有能力自行加载全局测试数据集。"""
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     # 假设全局测试数据对所有客户端都是可访问的
    #     # 注意: read_global_test_data 需要从 utils 中导入

    #     # 我们需要确保 args 中有 global_test_dir 这个属性
    #     # 在 __init__ 中添加 self.global_test_dir = args.global_test_dir
    #     # 为简单起见，这里我们直接使用 self.dataset
    #     # 在实际项目中，应确保路径正确
    #     global_test_path = f"../dataset/mfr_split/global_test"
    #     if hasattr(self, 'global_test_dir') and self.global_test_dir:
    #         global_test_path = self.global_test_dir

    #     return read_global_test_data(global_test_path, batch_size)

    def load_train_data(self, batch_size=None):
        """加载当前客户端的训练数据集。"""
        if batch_size == None:
            batch_size = self.batch_size
        # 从文件中读取属于该客户端ID的训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # 返回一个PyTorch的DataLoader对象
        return DataLoader(
            train_data,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
    def load_test_data(self, batch_size=None):
        """加载当前客户端的测试数据集。"""
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(
            test_data,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def clone_model(self, model, target):
        """一个工具函数，用于将一个模型的参数复制到另一个模型。"""
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # 梯度通常不需要克隆
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        """一个工具函数，用于将新的参数列表更新到现有模型中。"""
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def set_parameters(self, global_model=None):
        """
        Sync server parameters into the client's local model.

        Falls back to the persisted server checkpoint so the generic server
        helper remains usable even outside the in-memory FedAvg path.
        """
        if global_model is None:
            global_model = load_item(
                'Server',
                'global_model',
                self.save_folder_name,
                model_template=self._build_model_template(),
            )
        if global_model is None:
            raise FileNotFoundError("Server_global_model.pt not found for client parameter sync.")

        if self.model is None:
            self.model = self._build_model_template()
        self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))

    def collect_test_outputs(self):
        """
        在本地测试集上评估模型的性能，计算准确率和AUC。
        这是标准的评估方法，使用模型的分类头进行预测。
        """
        testloaderfull = self.load_test_data()
        model = self.model if self.model is not None else self._load_local_model('model')
        if self.algorithm == "Local" and self.id == 0 and getattr(self, "current_round", 0) in {1, 10, 50, 99, 100}:
            disk_model = self._load_local_model('model')
            # region agent log
            debug_log(
                "src/flcore/clients/clientbase.py:138",
                "local eval model source snapshot",
                {
                    "round": int(getattr(self, "current_round", 0)),
                    "client_id": int(self.id),
                    "using_self_model": self.model is not None,
                    "self_model_id": id(self.model) if self.model is not None else None,
                    "disk_model_id": id(disk_model) if disk_model is not None else None,
                    "self_model_norm": float(
                        sum(p.detach().float().norm().item() for p in self.model.parameters())
                    ) if self.model is not None else None,
                    "disk_model_norm": float(
                        sum(p.detach().float().norm().item() for p in disk_model.parameters())
                    ) if disk_model is not None else None,
                },
                run_id="local-runtime",
                hypothesis_id="L1",
            )
            # endregion
        # 将模型设置为评估模式 (关闭dropout等)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []  # 存储模型输出的概率
        y_true = []  # 存储真实的标签
        y_pred = []  # 存储预测标签
        inference_time = 0.0

        # 禁用梯度计算以加速评估
        with torch.no_grad():
            for x, y in testloaderfull:
                # 将数据移动到计算设备
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 前向传播
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                output = model(x)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                inference_time += time.perf_counter() - start_time
                pred = torch.argmax(output, dim=1)

                # 计算正确预测的数量
                test_acc += (torch.sum(pred == y)).item()
                test_num += y.shape[0]

                # 收集概率和标签用于计算AUC
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())

        # 将所有批次的概率和标签连接起来
        if y_prob:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
        else:
            y_prob = np.zeros((0, self.num_classes), dtype=np.float32)
            y_true = np.array([], dtype=np.int64)
            y_pred = np.array([], dtype=np.int64)

        return {
            "test_acc": int(test_acc),
            "test_num": int(test_num),
            "y_prob": y_prob,
            "y_true": y_true,
            "y_pred": y_pred,
            "inference_time": float(inference_time),
        }

    def test_metrics(self):
        outputs = self.collect_test_outputs()
        test_acc = outputs["test_acc"]
        test_num = outputs["test_num"]
        y_prob = outputs["y_prob"]
        y_true = outputs["y_true"]
        y_pred = outputs["y_pred"]
        inference_time = outputs["inference_time"]

        if test_num == 0:
            confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, confusion_matrix, 0.0

        auc_macro, auc_micro = self.compute_multiclass_auc(y_true, y_prob)
        precision, recall, f1, fpr = self.compute_classification_metrics(y_true, y_pred)
        fnr = self.compute_false_negative_rate(y_true, y_pred)
        confusion_matrix = metrics.confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
        )
        latency_ms = 1000.0 * inference_time / max(test_num, 1)

        return test_acc, test_num, auc_macro, auc_micro, fnr, precision, recall, f1, fpr, confusion_matrix, latency_ms

    def compute_multiclass_auc(self, y_true, y_prob):
        """
        计算真实的多分类 AUC，并同时返回 macro / micro。
        当某些类别在当前客户端测试集中完全缺失时，自动跳过这些无效列。
        """
        if len(y_true) == 0:
            return 0.0, 0.0

        y_true = np.asarray(y_true, dtype=np.int64)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
            return 0.0, 0.0

        y_true_bin = np.eye(self.num_classes, dtype=np.int32)[y_true]
        valid_columns = np.logical_and(
            y_true_bin.sum(axis=0) > 0,
            y_true_bin.sum(axis=0) < y_true_bin.shape[0],
        )

        if not np.any(valid_columns):
            return 0.0, 0.0

        y_true_bin = y_true_bin[:, valid_columns]
        y_prob = y_prob[:, valid_columns]

        try:
            if y_true_bin.shape[1] == 1:
                auc_value = metrics.roc_auc_score(y_true_bin[:, 0], y_prob[:, 0])
                return float(auc_value), float(auc_value)

            auc_macro = metrics.roc_auc_score(y_true_bin, y_prob, average='macro')
            auc_micro = metrics.roc_auc_score(y_true_bin, y_prob, average='micro')
            return float(auc_macro), float(auc_micro)
        except ValueError:
            return 0.0, 0.0

    def train_metrics(self):
        """在本地训练集上评估模型的总损失。"""
        trainloader = self.load_train_data()
        model = self.model if self.model is not None else self._load_local_model('model')
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)

                # 累加总样本数和总损失
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def compute_false_negative_rate(self, y_true, y_pred):
        """
        计算 IoT 检测任务中的漏报率(FNR)。
        默认将标签 0 视为正常类，非 0 视为攻击类；
        漏报定义为“攻击样本被预测为正常类”。
        """
        if len(y_true) == 0:
            return 0.0

        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        positive_mask = y_true != self.normal_class
        positive_count = int(np.sum(positive_mask))
        if positive_count == 0:
            return 0.0

        false_negatives = int(np.sum(np.logical_and(positive_mask, y_pred == self.normal_class)))
        return float(false_negatives / positive_count)

    def compute_classification_metrics(self, y_true, y_pred):
        """
        Return macro Precision/Recall/F1 and binary false positive rate.

        FPR treats normal_class as benign traffic and any non-normal label as
        attack traffic, measuring benign samples incorrectly flagged as attacks.
        """
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0

        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        precision = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)

        normal_mask = y_true == self.normal_class
        normal_count = int(np.sum(normal_mask))
        if normal_count == 0:
            fpr = 0.0
        else:
            false_positives = int(np.sum(np.logical_and(normal_mask, y_pred != self.normal_class)))
            fpr = false_positives / normal_count

        return float(precision), float(recall), float(f1), float(fpr)


# ---- 全局工具函数 ----

def save_item(item, role, item_name, item_path=None):
    """
    将一个PyTorch对象（如模型、张量）保存到文件。

    Args:
        item: 要保存的对象。
        role (str): 角色名 (如 'Server' 或 'Client_0')。
        item_name (str): 对象的名称 (如 'model', 'protos')。
        item_path (str): 保存路径。
    """
    # 如果路径不存在，则创建它
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    payload = item
    if isinstance(item, nn.Module):
        payload = {
            "__kind__": "module_state_dict",
            "state_dict": item.state_dict(),
        }
    torch.save(payload, os.path.join(item_path, role + "_" + item_name + ".pt"))


def load_item(role, item_name, item_path=None, model_template=None):
    """
    从文件中加载一个PyTorch对象。
    """
    file_path = os.path.join(item_path, role + "_" + item_name + ".pt")
    try:
        payload = torch.load(file_path, weights_only=True)
    except FileNotFoundError:
        print(f"文件未找到: {role}_{item_name}.pt")
        return None
    except Exception:
        payload = torch.load(file_path, weights_only=False)

    if isinstance(payload, dict) and payload.get("__kind__") == "module_state_dict":
        if model_template is None:
            raise ValueError(
                f"Loading module '{role}_{item_name}.pt' requires a model_template."
            )
        model_template.load_state_dict(payload["state_dict"])
        return model_template

    return payload

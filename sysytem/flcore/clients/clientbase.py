# --- START OF FILE clientbase.py ---

import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 从 scikit-learn 导入用于计算多分类 AUC 的工具
from sklearn.preprocessing import label_binarize
from sklearn import metrics
# 从项目工具中导入数据读取和模型定义
from utils.data_utils import read_client_data
from flcore.trainmodel.models import BaseHeadSplit


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
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # 客户端的唯一标识符 (整数)
        self.role = 'Client_' + str(self.id)  # 客户端的角色名，用于文件保存
        self.save_folder_name = args.save_folder_name_full  # 结果保存的文件夹路径

        self.num_classes = args.num_classes  # 数据集的类别数
        self.train_samples = train_samples  # 训练样本数量
        self.test_samples = test_samples  # 测试样本数量
        self.batch_size = args.batch_size  # 批处理大小
        self.learning_rate = args.local_learning_rate  # 本地学习率
        self.local_epochs = args.local_epochs  # 本地训练轮数

        # 如果不是从已有的临时文件夹恢复，则为该客户端创建一个新的模型实例
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # BaseHeadSplit会根据配置动态创建 "base" + "head" 结构的模型
            model = BaseHeadSplit(args, self.id).to(self.device)
            # 将新创建的模型保存到文件
            save_item(model, self.role, 'model', self.save_folder_name)

        # 从关键字参数中获取慢客户端标志，用于模拟异构性
        self.train_slow = kwargs['train_slow']  # 标记是否为训练慢的客户端
        self.send_slow = kwargs['send_slow']  # 标记是否为通信慢的客户端
        # 用于记录时间和开销的字典
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 定义默认的损失函数
        self.loss = nn.CrossEntropyLoss()

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
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        
    def load_test_data(self, batch_size=None):
        """加载当前客户端的测试数据集。"""
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

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

    def test_metrics(self):
        """
        在本地测试集上评估模型的性能，计算准确率和AUC。
        这是标准的评估方法，使用模型的分类头进行预测。
        """
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # 将模型设置为评估模式 (关闭dropout等)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []  # 存储模型输出的概率
        y_true = []  # 存储真实的标签

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
                output = model(x)

                # 计算正确预测的数量
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # 收集概率和标签用于计算AUC
                y_prob.append(output.detach().cpu().numpy())
                # 将标签进行one-hot编码 (binarize) 以计算多分类AUC
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1  # scikit-learn binarize的特殊处理
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # 将所有批次的概率和标签连接起来
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # 使用 micro-average 计算AUC分数
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        """在本地训练集上评估模型的总损失。"""
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
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
    # 使用torch.save保存
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))


def load_item(role, item_name, item_path=None):
    """
    从文件中加载一个PyTorch对象。
    """
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"))
    except FileNotFoundError:
        print(f"文件未找到: {role}_{item_name}.pt")
        return None




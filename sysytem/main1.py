#!/usr/bin/env python
# 这是一个 shebang 行，用于告诉操作系统使用 python 环境来执行此脚本

# 导入 PyTorch 核心库
import torch
# 导入 argparse 模块，用于解析命令行参数
import argparse
# 导入 os 模块，用于与操作系统交互，如设置环境变量
import os
# 导入 time 模块，用于计时
import time
# 导入 warnings 模块，用于控制警告信息的显示
import warnings
# 导入 NumPy 库，用于高效的数值运算，如此处的平均值计算
import numpy as np
# 导入 logging 模块，用于记录日志信息
import logging
from flcore.servers.serverproto import FedProto
from flcore.servers.serveravg import FedAvg
# 从 utils.result_utils 模块导入 average_data 函数，用于对多次实验结果求平均
from utils.result_utils import average_data
# 从 utils.mem_utils 模块导入 MemReporter 类，用于报告内存使用情况
from utils.mem_utils import MemReporter

# 获取日志记录器对象
logger = logging.getLogger()
# 设置日志级别为 ERROR，即只显示错误及以上级别的日志信息
logger.setLevel(logging.ERROR)

# 简单过滤警告信息，使其不显示
warnings.simplefilter("ignore")
# 设置 PyTorch 的随机种子为0，以确保实验的可复现性
torch.manual_seed(0)


def run(args):
    """
    主运行函数，负责执行整个联邦学习的流程。
    """
    time_list = []  # 初始化一个列表，用于存储每次实验的运行时间
    reporter = MemReporter()  # 创建一个内存报告器实例

    # 循环指定的实验次数 (args.times)，`args.prev` 用于从中间次数开始
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")  # 打印当前是第几次实验
        print("Creating server and clients ...")  # 打印提示信息
        start = time.time()  # 记录本次实验的开始时间
        # 打印调试信息，显示当前使用的模型家族
        print(f"DEBUG: The value of args.model_family is: {args.model_family}")

        # 根据命令行参数 `args.model_family` 的值，动态地构建模型和分类头的字符串表示
        if args.model_family == 'Transformer_MFR':
            args.models = [
                # MFRVisionTransformer 模型的构造字符串
                f"MFRVisionTransformer(in_channels=1, patch_size={args.patch_size}, embed_dim={args.feature_dim}, num_heads={args.num_heads}, num_layers={args.num_layers}, num_classes={args.num_classes}, img_size={args.img_size})",
                f"MFRMamba(in_channels=1, patch_size={args.patch_size}, embed_dim={args.feature_dim},  num_layers={args.num_layers}, num_classes={args.num_classes}, img_size={args.img_size})",
                f"CNN(in_channels=1, num_classes={args.num_classes}, img_size={args.img_size}, num_cov=2, feature_dim={args.feature_dim})",
            ]
            # 对应每个模型的分类头（一个简单的线性层）
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
                f"nn.Linear({args.feature_dim}, {args.num_classes})"
            ]
            print(f"模型家族设置为: {args.model_family}")
            # 打印出每个模型和其对应的分类头
            for m, h in zip(args.models, args.heads):
                print(f"模型: {m}, Head: {h}")
        elif args.model_family == 'Transformer_Avg':
            # 如果是 'Transformer_Avg' 家族，只使用 MFRVisionTransformer
            args.models = [
                f"MFRVisionTransformer(in_channels=1, patch_size={args.patch_size}, embed_dim={args.feature_dim}, num_heads={args.num_heads}, num_layers={args.num_layers}, num_classes={args.num_classes}, img_size={args.img_size})",
            ]
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
            ]
            for m, h in zip(args.models, args.heads):
                print(f"模型: {m}, Head: {h}")
        elif args.model_family == 'CNN':
            # 如果是 'CNN' 家族，调用 models.py 中的 CNN 类
            # 注意：in_channels=1 对应灰度图(如MNIST)，如果是彩色图(如CIFAR-10)请改为 3
            # num_cov=2 是卷积层的数量，你可以根据需要修改，或者在 args 中添加参数 args.num_cov
            args.models = [
                f"CNN(in_channels=1, num_classes={args.num_classes}, img_size={args.img_size}, num_cov=2, feature_dim={args.feature_dim})",
            ]
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
            ]
            for m, h in zip(args.models, args.heads):
                print(f"模型: {m}, Head: {h}")
        elif args.model_family == 'V-Mamba':
            # 如果是 'V-Mamba' 家族
            # Vmamba_efficient 模型的特征维度是其 `dims` 列表的最后一个元素。
            # 为了与其他模型保持一致（它们的特征维度是 embed_dim），我们动态构造 dims 列表，
            # 使其最后一个元素等于 args.embed_dim。
            # 这里我们采用一个常见的金字塔结构，例如 [D/8, D/4, D/2, D]。
            # 这要求 embed_dim 最好是 8 的倍数（例如 128, 256, 512）。
            # 动态构建 V-Mamba 模型的维度列表
            dim_list = [args.embed_dim // 8, args.embed_dim // 4, args.embed_dim // 2, args.embed_dim]
            args.models = [
                # 我们调用在 models.py 中创建的包装类 Vmamba_efficient
                # 将命令行参数传递给它
                # 构建 Vmamba_efficient 模型的构造字符串
                f"Vmamba_efficient(in_chans=1, patch_size={args.patch_size}, num_classes={args.num_classes}, "
                f"img_size={args.img_size}, dims={dim_list}, d_state={args.d_state})"
            ]
            # 由于模型的最终特征维度是 args.embed_dim，所以 Head 的输入维度也是它
            args.heads = [
                f"nn.Linear({args.embed_dim}, {args.num_classes})"
            ]
            print(f"模型家族设置为: {args.model_family}")
            for m, h in zip(args.models, args.heads):
                print(f"模型: {m}, Head: {h}")
        elif args.model_family == 'Mamba_Test':
            # 如果是 'Mamba_Test' 家族
            args.models = [
                f"MFRMamba(in_channels=1, patch_size={args.patch_size}, embed_dim={args.feature_dim},  num_layers={args.num_layers}, num_classes={args.num_classes}, img_size={args.img_size})"
            ]
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
            ]
            for m, h in zip(args.models, args.heads):
                print(f"模型: {m}, Head: {h}")
        elif args.model_family == 'HCNNs8':
            # 定义 HCNNs8 的 8 种异构配置
            # 对应论文中的配置：
            # 1. num_cov=1, hidden_dims=[]
            # 2. num_cov=2, hidden_dims=[]
            # 3. num_cov=1, hidden_dims=[512]
            # ... 以此类推
            
            # 基础参数字符串，减少重复代码
            base_args = f"in_channels=1, num_classes={args.num_classes}, img_size={args.img_size}, feature_dim={args.feature_dim}"
            
            args.models = [
                f"CNN({base_args}, num_cov=1, hidden_dims=[])",
                f"CNN({base_args}, num_cov=2, hidden_dims=[])",
                f"CNN({base_args}, num_cov=1, hidden_dims=[512])",
                f"CNN({base_args}, num_cov=2, hidden_dims=[512])",
                f"CNN({base_args}, num_cov=1, hidden_dims=[1024])",
                f"CNN({base_args}, num_cov=2, hidden_dims=[1024])",
                f"CNN({base_args}, num_cov=1, hidden_dims=[1024, 512])",
                f"CNN({base_args}, num_cov=2, hidden_dims=[1024, 512])",
            ]
            
            # 为这 8 个模型分别配置 Head
            # 因为所有的 CNN 最终输出维度都被设置为 args.feature_dim，所以 Head 是一样的
            args.heads = [f"nn.Linear({args.feature_dim}, {args.num_classes})" for _ in range(8)]

            print(f"模型家族设置为: {args.model_family} (HCNNs8)")
            for i, (m, h) in enumerate(zip(args.models, args.heads)):
                print(f"Client {i} -> 模型: {m}")
        elif args.model_family == "HtFE2":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
            ]
        elif args.model_family == "HtFE3":
            args.models = [
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]
        elif args.model_family == "HtFE4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)', 
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
            ]
        elif args.model_family == "HtFE8":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)', 
                # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)', 
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]
            
        elif args.model_family == "HtFE9":
            args.models = [
                'resnet4(num_classes=args.num_classes)', 
                'resnet6(num_classes=args.num_classes)', 
                'resnet8(num_classes=args.num_classes)', 
                'resnet10(num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)', 
            ]

        elif args.model_family == "HtFE8-HtC4":
            args.models = [
                'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)', 
                'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)', 
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)', 
                'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)'
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]

        elif args.model_family == "Res34-HtC4":
            args.models = [
                'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)', 
            ]
            args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=3136)'
            args.heads = [
                'Head(hidden_dims=[512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 512], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 256], num_classes=args.num_classes)', 
                'Head(hidden_dims=[512, 128], num_classes=args.num_classes)', 
            ]
        else:
            # 如果 `args.model_family` 不是以上任何一个，则抛出未实现错误
            raise NotImplementedError

        # 打印最终确定的模型列表
        for model in args.models:
            print(model)

        # 根据命令行参数 `args.algorithm` 选择并实例化服务器
        if args.algorithm == "FedProto":
            server = FedProto(args, i)
        elif args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        else:
            # 如果算法未实现，则抛出错误
            raise NotImplementedError

        # 调用服务器的 `train` 方法，开始整个联邦学习训练过程
        server.train()

        # 将本次实验的运行时长（当前时间 - 开始时间）添加到列表中
        time_list.append(time.time() - start)

    # 所有实验运行完毕后，打印平均运行时长
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # 调用 average_data 函数，对所有保存的实验结果文件进行汇总和平均
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")  # 打印完成信息

    # 报告最终的内存使用情况
    reporter.report()


if __name__ == "__main__":
    # 记录整个脚本开始执行的时间
    total_start = time.time()

    # 创建一个 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # general: 定义通用的命令行参数
    # 添加 Transformer 相关的参数
    parser.add_argument('-model_family', type=str, default='Transformer_MFR',
                        help='Model family to use (e.g., Transformer_MFR, MLP, HtFE2)')
    parser.add_argument('-patch_size', type=int, default=4, help='Patch size for Vision Transformer')
    parser.add_argument("--stride", type=int, default=2, help="Stride for overlapping patches.")
    parser.add_argument('-embed_dim', type=int, default=128, help='Embedding dimension for Transformer')
    parser.add_argument('-num_heads', type=int, default=4, help='Number of attention heads in Transformer')
    parser.add_argument('-num_layers', type=int, default=2, help='Number of Transformer Encoder layers')
    parser.add_argument('-img_size', type=int, default=40, help='Image size (height/width) for MFR images')
    # 添加 Mamba 相关的参数
    parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='1D Conv kernel size for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='Block expansion factor for Mamba')

    # 将 dataset 参数默认值设置为你的MFR数据集名称
    parser.add_argument('-dataset', type=str, default='IoT', help='Dataset used for training')

    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    # parser.add_argument('-m', "--model_family", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical: 定义与实际联邦学习环境相关的参数
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedTGP: 定义 FedTGP 算法特有的参数
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    parser.add_argument('-pw', "--packet_weight", type=float, default=1.0)
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument("--fixed_margin", type=float, default=0.5,
                        help="Value for the fixed margin in FM ablation study")

    # FedSA
    parser.add_argument('-mcl', "--margin_contrastive", type=float, default=1.0)
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0)

    # =====================================================================
    # 解析命令行传入的参数
    args = parser.parse_args()

    # 设置 `CUDA_VISIBLE_DEVICES` 环境变量，以指定使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    # 检查如果设备设置为 "cuda"，但 CUDA 又不可用，则自动切换到 "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    # 打印分隔线
    print("=" * 50)

    # 打印所有重要的实验参数配置
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_family))
    print("Using device: {}".format(args.device))
    print("Lamda: {}".format(args.lamda))
    print("Packet weight: {}".format(args.packet_weight))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    print("=" * 50)

    # 下面是被注释掉的数据生成部分，如果需要，可以取消注释以自动生成数据集划分
    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)
    # if args.dataset == "IoT":
    #     generate_IoT('../dataset/IoT',args.num_clients,10,args.nidd)

    # 下面是被注释掉的性能分析工具 (PyTorch Profiler) 的代码
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:

    # 调用主运行函数，传入解析好的参数
    run(args)

    # 下面是用于打印性能分析结果的代码
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")

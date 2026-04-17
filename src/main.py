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
from flcore.servers.serverlocal import Local
from flcore.clients.clientbase import debug_log
# 从 utils.result_utils 模块导入 average_data 函数，用于对多次实验结果求平均
from utils.result_utils import average_data

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
    # 循环指定的实验次数 (args.times)，`args.prev` 用于从中间次数开始
    for i in range(args.prev, args.times):
        print(f"\n============= Run {i} =============")
        print("Creating server and clients...")
        start = time.time()  # 记录本次实验的开始时间
        print(f"Model family: {args.model_family}")

        # 仅保留一维 IoT 任务可用的模型分支
        if args.model_family == 'IoT_MLP':
            args.models = [
                f"MLP_IoT(dim_in={args.input_dim}, dim_hidden=128, dim_out={args.feature_dim}, num_classes={args.num_classes})",
            ]
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
            ]
        elif args.model_family == 'IoT_CNN1D':
            args.models = [
                f"CNN1D_IoT(dim_in={args.input_dim}, dim_out={args.feature_dim}, num_classes={args.num_classes})",
            ]
            args.heads = [
                f"nn.Linear({args.feature_dim}, {args.num_classes})",
            ]
        else:
            raise NotImplementedError(
                f"Unsupported model_family '{args.model_family}'. "
                "Available options: IoT_MLP, IoT_CNN1D"
            )

        # 打印最终确定的模型列表
        print("Resolved models:")
        for model in args.models:
            print(f"  {model}")

        # 根据命令行参数 `args.algorithm` 选择并实例化服务器
        if args.algorithm == "FedProto":
            server = FedProto(args, i)
        elif args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "Local":
            server = Local(args, i)
        else:
            # 如果算法未实现，则抛出错误
            raise NotImplementedError(
                f"Unsupported algorithm '{args.algorithm}'. "
                "Available options: FedAvg, FedProto, Local"
            )

        # 调用服务器的 `train` 方法，开始整个联邦学习训练过程
        server.train()

        # 将本次实验的运行时长（当前时间 - 开始时间）添加到列表中
        time_list.append(time.time() - start)

    # 所有实验运行完毕后，打印平均运行时长
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # 调用 average_data 函数，对所有保存的实验结果文件进行汇总和平均
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")  # 打印完成信息


if __name__ == "__main__":
    # 记录整个脚本开始执行的时间
    total_start = time.time()

    # 创建一个 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # general: 一维 IoT 任务专用参数
    parser.add_argument('-model_family', type=str, default='IoT_MLP',
                        help='Tabular backbone to use: IoT_MLP or IoT_CNN1D')
    parser.add_argument('-dataset', type=str, default='IoT', help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=77, help='Input feature dimension for each sample')

    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="Experiment goal tag")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-nb', "--num_classes", type=int, default=15)
    parser.add_argument('--normal_class', type=int, default=0,
                        help="Label id treated as the normal class when computing FNR")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-nw', "--num_workers", type=int, default=4,
                        help="DataLoader worker processes per client")
    parser.add_argument('-pm', "--pin_memory", type=bool, default=True,
                        help="Whether to use pinned memory for DataLoader")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Local training epochs per communication round")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Fraction of clients joining each round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Whether to randomize the client join ratio")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Starting run index")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Number of repeated runs")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Evaluation interval in rounds")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp',
                        help="Directory name for intermediate outputs")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False,
                        help="Reserved flag, currently disabled")
    parser.add_argument('-fd', "--feature_dim", type=int, default=64,
                        help="Prototype / representation dimension")
    # practical: 联邦学习环境参数
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Fraction of clients that drop after local training")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="Fraction of slow clients during local training")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="Fraction of slow clients during model upload")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to select clients based on time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="Time threshold for filtering slow clients")
    # FedProto / FedAvg 相关参数
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Prototype regularization weight")
    parser.add_argument('-pw', "--packet_weight", type=float, default=1.0,
                        help="Reserved weighting coefficient")
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0,
                        help="Margin threshold")
    parser.add_argument('-se', "--server_epochs", type=int, default=1000,
                        help="Server-side optimization epochs")
    parser.add_argument("--fixed_margin", type=float, default=0.5,
                        help="Fixed margin value for ablation runs")

    # FedSA
    parser.add_argument('-mcl', "--margin_contrastive", type=float, default=1.0,
                        help="Contrastive margin weight")
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0,
                        help="Classifier calibration weight")

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
    print("Dataset: {}".format(args.dataset))
    print("Model family: {}".format(args.model_family))
    print("Input dimension: {}".format(args.input_dim))
    print("Number of classes: {}".format(args.num_classes))
    print("Normal class label: {}".format(args.normal_class))
    print("Number of clients: {}".format(args.num_clients))
    print("Local batch size: {}".format(args.batch_size))
    print("DataLoader workers: {}".format(args.num_workers))
    print("Pin memory: {}".format(args.pin_memory))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Client join ratio: {}".format(args.join_ratio))
    print("Random join ratio: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time-based client selection: {}".format(args.time_select))
    if args.time_select:
        print("Time threshold: {}".format(args.time_threthold))
    print("Runs: {}".format(args.times))
    print("Device: {}".format(args.device))
    print("Feature dimension: {}".format(args.feature_dim))
    print("Lambda: {}".format(args.lamda))
    print("Packet weight: {}".format(args.packet_weight))
    print("Auto break: {}".format(args.auto_break))
    print("Global rounds: {}".format(args.global_rounds))
    print("=" * 50)

    # region agent log
    debug_log(
        "src/main.py:220",
        "runtime args snapshot",
        {
            "algorithm": args.algorithm,
            "dataset": args.dataset,
            "model_family": args.model_family,
            "device": args.device,
            "device_id": args.device_id,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "feature_dim": args.feature_dim,
            "lamda": args.lamda,
            "global_rounds": args.global_rounds,
            "num_workers": args.num_workers,
        },
        run_id=f"{args.algorithm}_runtime",
        hypothesis_id="H1",
    )
    # endregion

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

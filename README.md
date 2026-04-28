# 基于 FedProto 的 IoT 恶意流量联邦检测系统

本项目面向 IoT 恶意流量检测场景，构建了一个可运行、可复现实验结果、可进行桌面化展示的联邦学习原型系统。系统以 `FedProto` 为核心方法，同时保留 `Local` 与 `FedAvg` 作为对照方案，支持同构模型对比实验，以及当前项目正式采用的异构客户端实验配置。

当前仓库已经包含：

- 联邦训练主入口
- 本地训练、FedAvg、FedProto 三类训练流程
- MLP、CNN1D、Transformer1D 三种同构模型
- `MLP + CNN1D` 异构客户端组合
- 结果汇总、历史模型评估与可视化导出脚本
- Electron 桌面端原型

## 项目结构

```text
IoT_FedProto/
├─ src/                  # 训练入口、联邦客户端/服务端、模型定义、评估脚本
├─ dataset/              # 当前训练直接读取的客户端数据目录
├─ data/                 # 原始数据与预处理阶段中间文件
├─ results/              # 指标文件、图像、日志、汇总表
├─ artifacts/            # 桌面端保存的模型与运行清单
├─ scripts/              # 数据处理、参数统计、效率分析脚本
├─ desktop-app/          # Electron 桌面端
├─ reference/            # 设计文档与过程材料
├─ requirements.txt
└─ README.md
```

## 环境要求

- Python 3.10
- 建议使用项目根目录下的 `.venv`
- Node.js 18 及以上，仅桌面端需要

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 已按项目现有环境固定版本：

- `torch==2.5.1`
- `numpy==2.0.1`
- `scikit-learn==1.7.2`
- `scipy==1.15.3`
- `h5py==3.16.0`
- `matplotlib==3.10.8`
- `pandas==2.3.3`

## 当前数据组织

当前项目代码默认读取的不是多个候选数据集，而是仓库内现有的这一套客户端划分数据：

- `dataset/IoT`

实际读取方式由 [src/utils/data_utils.py](/e:/IoT_FedProto/src/utils/data_utils.py:1) 决定，目录结构必须为：

```text
dataset/IoT/
├─ train/
│  ├─ 0/
│  │  ├─ X.npy
│  │  └─ y.npy
│  ├─ 1/
│  └─ ...
└─ test/
   ├─ 0/
   │  ├─ X.npy
   │  └─ y.npy
   ├─ 1/
   └─ ...
```

也就是说，当前训练代码读取的是：

- `dataset/IoT/train/<client_id>/X.npy`
- `dataset/IoT/train/<client_id>/y.npy`
- `dataset/IoT/test/<client_id>/X.npy`
- `dataset/IoT/test/<client_id>/y.npy`

不是 CSV，不是 HDF5，也不是我之前文档里写过的其他候选目录结构。

训练前需要确认：

- `train` 与 `test` 下客户端编号对应
- 每个客户端目录下都同时存在 `X.npy` 和 `y.npy`
- `-dataset IoT` 与实际目录名一致
- `-nc 20` 与当前客户端划分数量一致

## 训练入口

统一训练入口为 [src/main.py](/e:/IoT_FedProto/src/main.py:1)。

建议在 `src/` 目录中执行训练命令：

```bash
cd src
```

主程序当前支持的核心参数包括：

- `-dataset`：数据集名称，当前项目使用 `IoT`
- `-algo`：`Local`、`FedAvg`、`FedProto`
- `-model_family`：`IoT_MLP`、`IoT_CNN1D`、`IoT_Transformer1D`、`IoT_MIX_MLP_CNN1D`
- `-nc`：客户端数量
- `-gr`：全局轮数
- `-ls`：本地训练轮数
- `-lr`：本地学习率
- `-fd`：特征维度
- `-lbs`：batch size
- `-nw`：DataLoader worker 数
- `--early_stop_patience`：早停阈值
- `--skip_figures`：跳过图像生成

如果指定 `cuda` 但环境不可用，程序会自动回退到 `cpu`。

## 快速开始

### 1. 同构实验

当前仓库内保留的同构实验脚本为：

- [src/run_local.sh](/e:/IoT_FedProto/src/run_local.sh:1)
- [src/run_fedavg.sh](/e:/IoT_FedProto/src/run_fedavg.sh:1)
- [src/run_fedproto.sh](/e:/IoT_FedProto/src/run_fedproto.sh:1)

这三组脚本默认基于 `IoT_MLP` 运行：

```bash
cd src
bash run_local.sh
bash run_fedavg.sh
bash run_fedproto.sh
```

### 2. 异构实验

当前项目正式保留的异构实验脚本为：

- [src/run_fedproto_mix_mlp_cnn1d.sh](/e:/IoT_FedProto/src/run_fedproto_mix_mlp_cnn1d.sh:1)

运行方式：

```bash
cd src
bash run_fedproto_mix_mlp_cnn1d.sh
```

该脚本对应当前项目的异构展示主方案：

- 算法：`FedProto`
- 模型组合：`IoT_MIX_MLP_CNN1D`

## 常用命令

### MLP + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### Transformer1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_Transformer1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 异构 MLP + CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### PowerShell 示例

```powershell
Set-Location .\src
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

## 结果输出

训练结果当前统一写入 `results/`，按模型类别与算法分类组织。现有目录结构包括：

- `results/MLP/Local`
- `results/MLP/FedAvg`
- `results/MLP/FedProto`
- `results/CNN1D/Local`
- `results/CNN1D/FedAvg`
- `results/CNN1D/FedProto`
- `results/Transformer/Local`
- `results/Transformer/FedAvg`
- `results/Transformer/FedProto`
- `results/heterogeneous_models/FedProto`

标准子目录通常包括：

- `metrics/`：`.h5` 指标文件
- `figures/`：t-SNE 与原型分布图
- `logs/`：训练日志

结果文件命名格式为：

```text
{dataset}_{algorithm}_{model_family}_{goal}_{run_idx}.h5
```

当前仓库中已经存在的命名示例包括：

- `IoT_FedAvg_IoT_MLP_test_0.h5`
- `IoT_FedProto_IoT_CNN1D_test_0.h5`
- `IoT_FedProto_IoT_MIX_MLP_CNN1D_heterogeneous_demo_0.h5`

## 结果汇总

在项目根目录执行：

```bash
python src/summarize_results.py
```

会生成：

- `results/summary/experiment_summary.csv`
- `results/summary/experiment_summary.md`

汇总脚本会扫描 `results/` 下的 `.h5` 文件，并提取：

- Accuracy
- AUC Macro
- AUC Micro
- Precision
- Recall
- F1
- FNR
- FPR
- 推理时延
- 通信量
- 参数量与 FLOPs

## 已保存模型评估

[src/evaluate_saved_model.py](/e:/IoT_FedProto/src/evaluate_saved_model.py:1) 用于重新加载历史训练模型并在测试客户端上执行评估。该能力目前主要由桌面端调用，但脚本本身已经独立可用。

## 数据处理与分析脚本

### 数据预处理

```bash
python scripts/data_preprocess.py
```

### 数据重打包

```bash
python scripts/repack_to_dataset.py
```

### 模型参数统计

```bash
python scripts/report_iot_model_params.py
```

### 联邦效率分析

```bash
python scripts/report_iot_efficiency.py --model-family IoT_CNN1D
python scripts/report_iot_efficiency.py --model-family IoT_MIX_MLP_CNN1D
```

## 当前实验口径

当前项目代码与结果目录最贴合的实验范围是：

- 同构 `IoT_MLP`：`Local / FedAvg / FedProto`
- 同构 `IoT_CNN1D`：`Local / FedAvg / FedProto`
- 同构 `IoT_Transformer1D`：`Local / FedAvg / FedProto`
- 异构 `IoT_MIX_MLP_CNN1D`：`FedProto`

## 桌面端

桌面端说明见 [desktop-app/README.md](/e:/IoT_FedProto/desktop-app/README.md:1)。

桌面端当前可以完成：

- 扫描数据集目录
- 选择训练客户端
- 启动联邦训练
- 实时显示训练日志
- 保存模型运行记录
- 加载历史模型并执行测试评估
- 显示实验图像与汇总结果

## 说明

- 当前 README 以仓库内现有代码、现有数据目录和现有结果目录为准
- 当前默认数据集是 `dataset/IoT`
- 当前训练读取的最小数据单元是每个客户端目录下的 `X.npy` 与 `y.npy`
- 如果后续你调整了数据目录结构，`src/utils/data_utils.py` 和桌面端的数据扫描逻辑也需要同步修改

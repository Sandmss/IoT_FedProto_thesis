# 基于轻量化联邦学习的物联网恶意流量检测系统

## 项目简介

本项目实现了一个面向 IoT 恶意流量检测场景的轻量化联邦学习原型系统。

系统以 `FedProto` 为核心联邦方案，同时保留 `Local` 和 `FedAvg` 作为对照方法，支持多种轻量化本地模型，并支持同构与部分异构实验。

当前系统重点包括：

- 轻量化恶意流量检测模型
- 联邦训练流程实现
- 多指标结果评估
- 结果归档与总结果表汇总

## 当前支持内容

### 算法

- `Local`
- `FedAvg`
- `FedProto`

### 模型

- `IoT_MLP`
- `IoT_CNN1D`
- `IoT_Transformer1D`
- `IoT_MIX_MLP_CNN1D`
- `IoT_MIX_MLP_CNN_TRANS`

说明：

- `Local`、`FedAvg`、`FedProto` 适用于同构模型实验。
- 异构模型实验当前优先建议使用 `FedProto`。
- 当前实现下，不建议直接将异构模型用于 `FedAvg`。

## 项目目录

- `src`：最终主代码目录，包含训练入口、客户端、服务端、模型实现与结果汇总脚本。
- `dataset`：联邦划分后的客户端数据目录。
- `data`：原始数据与处理中间产物。
- `results`：实验日志、指标结果、图像输出和汇总表。
- `reference`：开题报告、任务书、计划文档等参考材料。
- `scripts`：参数统计、效率分析等辅助脚本。

## 环境依赖

建议 Python 版本：

- `Python 3.10+`

主要依赖：

- `torch`
- `numpy`
- `scikit-learn`
- `h5py`
- `matplotlib`

安装示例：

```bash
pip install torch numpy scikit-learn h5py matplotlib
```

## 数据说明

项目当前使用已经处理好的 IoT 恶意流量特征数据，并已构建联邦学习客户端划分目录。

常用数据目录包括：

- `dataset/IoT`
- `dataset/IoT_20k_c20`
- `dataset/IoT_20k_c20_noniid`
- `dataset/IoT_k40_c20_noniid`

运行实验前，需要确认 `main.py` 读取的数据目录与当前准备好的数据集一致。

## 主入口

统一主入口为 [src/main.py](C:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/main.py:1)。

建议始终在 `src` 目录下执行命令，这样相对路径最稳定。

## 快速开始

### 1. 进入主代码目录

```bash
cd src
```

### 2. 运行同构实验

当前仓库中的三个正式脚本为同构实验入口：

- [run_local.sh](C:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/run_local.sh:1)
- [run_fedavg.sh](C:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/run_fedavg.sh:1)
- [run_fedproto.sh](C:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/run_fedproto.sh:1)

运行示例：

```bash
bash run_local.sh
bash run_fedavg.sh
bash run_fedproto.sh
```

### 3. 运行异构实验

异构 `MLP + CNN1D` 的 `FedProto` 入口脚本：

- [run_fedproto_mix_mlp_cnn1d.sh](C:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/run_fedproto_mix_mlp_cnn1d.sh:1)

运行方式：

```bash
bash run_fedproto_mix_mlp_cnn1d.sh
```

## 手动运行命令示例

### 同构 MLP + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 同构 CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 同构 Transformer + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_Transformer1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 异构 MLP + CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

## PowerShell 运行方式

```powershell
cd src
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

## 常用参数说明

- `-model_family`：选择模型家族，例如 `IoT_MLP`、`IoT_CNN1D`、`IoT_Transformer1D`。
- `-algo`：选择训练算法，例如 `Local`、`FedAvg`、`FedProto`。
- `-fd`：特征表示维度，也作为原型维度。
- `-gr`：全局轮数。
- `-ls`：本地训练轮数。
- `-nc`：客户端数量。
- `--early_stop_patience`：早停容忍轮数。
- `--skip_figures`：跳过图片生成，仅在需要节省时间时手动开启。

## 结果目录说明

当前结果目录按“模型类别 / 算法 / 文件类型”统一整理。

### 标准目录结构

- `results/MLP模型/Local/`
- `results/MLP模型/FedAvg/`
- `results/MLP模型/FedProto/`
- `results/CNN1D模型/Local/`
- `results/CNN1D模型/FedAvg/`
- `results/CNN1D模型/FedProto/`
- `results/transformer模型/Local/`
- `results/transformer模型/FedAvg/`
- `results/transformer模型/FedProto/`
- `results/异构模型/FedProto/`

每个算法目录下进一步分为：

- `logs/`：保存 `.out` 日志文件。
- `metrics/`：保存 `.h5` 指标结果文件。
- `figures/`：保存 t-SNE 图、原型分布图等图片。

### 结果文件命名

新的标准 `.h5` 文件命名格式为：

```text
{dataset}_{algorithm}_{model_family}_{goal}_{run_idx}.h5
```

例如：

```text
IoT_FedAvg_IoT_MLP_test_0.h5
IoT_FedProto_IoT_MLP_smoke_proto_0.h5
```

## 总结果表使用命令

训练完成后，可以运行统一汇总脚本扫描 `results/**/*.h5`，自动生成总结果表。

在项目根目录运行：

```bash
python src/summarize_results.py
```

生成文件：

- `results/summary/experiment_summary.csv`
- `results/summary/experiment_summary.md`

该脚本会：

- 兼容扫描旧结果目录和新标准目录。
- 提取 `Acc`、`AUC Macro`、`AUC Micro`、`Precision`、`Recall`、`F1`、`FNR`、`FPR`。
- 汇总推理延迟、通信量、参数量、模型大小和 FLOPs。
- 输出适合后续论文整理的统一结果表。

## 结果输出说明

训练结束后，系统会输出：

- `.out` 日志文件
- `.h5` 结果文件
- `.png` 图片文件
- 汇总后的 `.csv` 和 `.md` 总结果表

## 轻量化与效率分析脚本

### 模型参数量统计

```bash
python scripts/report_iot_model_params.py
```

### 联邦效率统计

```bash
python scripts/report_iot_efficiency.py --model-family IoT_CNN1D
```

异构统计示例：

```bash
python scripts/report_iot_efficiency.py --model-family IoT_MIX_MLP_CNN1D
```

## 当前系统建议实验结构

### 主实验

- 同构 `MLP`：`Local / FedAvg / FedProto`
- 同构 `CNN1D`：`Local / FedAvg / FedProto`
- 同构 `Transformer`：`Local / FedAvg / FedProto`

### 扩展实验

- 异构 `MLP + CNN1D`：`FedProto`

## 注意事项

- 建议始终在 `src` 目录运行主程序，避免相对路径问题。
- `FedAvg` 当前仅用于同构实验。
- 正式脚本默认会生成图片；若只想快速验证，可手动添加 `--skip_figures`。
- 如果 GPU 不可用，程序会自动切换到 CPU。

## 当前系统定位

该项目当前已经具备：

- 轻量化本地检测模型
- 联邦原型聚合训练框架
- 同构与异构客户端实验入口
- 多指标结果评估与效率分析能力
- 统一结果目录与总结果表生成能力

因此，它已经具备“轻量化联邦学习恶意流量检测原型系统”的核心结构，后续工作主要是继续整理结果、补充展示材料并完成最终论文交付。

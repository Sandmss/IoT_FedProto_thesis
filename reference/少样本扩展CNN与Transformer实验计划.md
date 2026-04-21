# 少样本下扩展 1D-CNN / Transformer 实验计划

## Summary

目标是按学姐建议验证：当本地模型容量变大、每客户端样本更少时，`Local` 不一定继续明显占优。路线选择为当前 77 维序列输入适配的 `IoT_CNN1D` 和轻量 `IoT_Transformer1D`，不直接迁入旧 `HtFE8` 的 ResNet/MobileNet 图像模型。

## Key Changes

- 保留 `IoT_MLP` 作为原基线，使用已有 `IoT_CNN1D` 作为第一组更大模型。
- 新增 `IoT_Transformer1D`：
  - 输入：`[batch, 77]`
  - 视为长度 77 的一维序列，每个特征作为一个 token。
  - 使用线性 embedding、可学习位置编码、`TransformerEncoder`、mean pooling、投影到 `feature_dim`。
  - 输出仍保持 `extract_features()` + `nn.Linear(feature_dim, num_classes)`，兼容 FedProto 原型聚合。
- 扩展 `model_family`：
  - `IoT_MLP`
  - `IoT_CNN1D`
  - `IoT_Transformer1D`
  - 可选后续：`IoT_MIX_MLP_CNN_TRANS`，用于异构客户端实验。

## Experiment Design

- 建议先生成少样本数据集：`IoT_k40_c20_noniid`
  - `num_clients=20`
  - 每客户端 `5` 个类别
  - 每个客户端-类别 `40` 条样本
  - `train_ratio=0.75`，即每类约 `30` 训练、`10` 测试。
- 对每个模型分别跑：
  - `Local`
  - `FedAvg`
  - `FedProto`
- 主对比表：
  - MLP / CNN1D / Transformer1D
  - Local Acc
  - FedAvg Acc
  - FedProto Acc
  - AUC Macro / Micro
  - FNR
  - 参数量
  - 每轮时间或总训练时间
- 核心判断：
  - 如果 CNN1D/Transformer 下 Local 下降或波动变大，而 FedProto 相对 FedAvg 更稳，就能支撑“少样本 + 较大模型时，本地孤立训练泛化不足，原型协同更有意义”。

## Test Plan

- 先做 smoke test：
  - 每个新模型跑 `FedProto -gr 2 -nw 0 -dev cpu`，确认前向、训练、原型提取、评估都不报 shape 错。
- 再跑短实验：
  - `-gr 30 -eg 5`，看曲线是否正常上升。
- 最后跑正式实验：
  - `-gr 1000 -eg 1 -ab True --skip_figures`
  - 每组保存 `.out` 日志，放入新的结果目录，例如 `results/第九次结果/`。
- 验收标准：
  - 三种模型均能跑完 Local / FedAvg / FedProto。
  - FedProto 使用 `--proto_eval_mode classifier` 作为主指标。
  - 论文表格能解释 Local、FedAvg、FedProto 在少样本较大模型下的差异。

## Assumptions

- 不直接使用旧 `HtFE8` 的 ResNet/MobileNet，因为旧代码面向 `3x40x40` 图像输入，不适合当前 77 维序列，强行转换会引入额外干扰。
- 少样本默认先用 `k_per_class=40`，比当前每类约 100 条更能体现样本不足，但测试集仍不至于太小。
- Transformer 采用轻量配置，默认 `d_model=64`、`num_heads=4`、`num_layers=2`、`dropout=0.2`，避免模型过大导致训练不稳定。

## Current Implementation Notes

- 新增模型：`src/flcore/trainmodel/models.py` 中的 `Transformer1D_IoT`。
- 新增模型族：`IoT_Transformer1D` 和 `IoT_MIX_MLP_CNN_TRANS`。
- 少样本数据生成脚本：`scripts/make_iot_k40_c20_noniid.sh`。
- 正式实验脚本：`src/run_k40_model_suite.sh`。
- 参数量统计脚本：`scripts/report_iot_model_params.py`。


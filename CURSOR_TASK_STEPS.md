# IoT_FedProto 改造执行指令（给 Cursor）

下面内容是可直接发给 Cursor 的分步指令，按顺序执行即可。  
目标是：保持 `sh run_me.sh` 主流程不变，把项目完全切换到一维流量特征（`npy`）处理，不再保留图像处理逻辑。

---

## Step 0：先修运行入口（必须先做）

```text
请检查并修复 `src/run_me.sh` 的入口脚本引用。

要求：
1) 当前 run_me.sh 使用的是 `python -u main1.py`，请改为实际存在的 `main.py`。
2) 输出文件统一为 `iot.out`（小写）或保持 `IoT.out`，但请在脚本注释中说明最终文件名。
3) 其他参数先不改，只修正可运行性。
4) 修改后返回最终 run_me.sh 内容。
```

---

## Step 1：按目录规范改 `data_utils`（核心）

```text
请只修改 `src/utils/data_utils.py`，目标是彻底替换为一维npy数据读取逻辑。

目录规范：
- `../dataset/IoT/train/{idx}/X.npy`
- `../dataset/IoT/train/{idx}/y.npy`
- `../dataset/IoT/test/{idx}/X.npy`
- `../dataset/IoT/test/{idx}/y.npy`

要求：
1) 保留函数名：`read_data(dataset_name, idx, is_train=True)` 和 `read_client_data(dataset, idx, is_train=True)`，但内部实现全部重写为npy读取。
2) `read_data` 按上述目录读取，返回 `{'x': np_array, 'y': np_array}`。
3) `read_client_data` 输出 `[(x,y), ...]`，其中：
   - x -> `torch.float32`
   - y -> `torch.int64`
   - x 先保持二维输入样式 `(feature_dim,)`，不要 `unsqueeze` 和 `repeat`。
4) 删除图像读取逻辑：PIL/glob/png/resize/CLASS_TO_LABEL。
5) 路径不存在时返回空数组并给出清晰警告日志（包含client id和train/test）。
6) 删除与图像任务相关的常量和注释，文件中不再出现“图像/PNG/resize/通道扩展”等逻辑描述。
7) 不改其他文件。

验收：
- 客户端0能读取train/test；
- DataLoader可正常迭代；
- 不再依赖图像文件。
```

---

## Step 2：新增数据重排脚本（把 processed_data 转成 dataset 目录格式）

```text
请新增一个脚本 `scripts/repack_processed_to_dataset_layout.py`，
把 `data/processed_data` 重排为 `dataset/IoT/train/{cid}` 和 `dataset/IoT/test/{cid}` 目录结构。

输入文件（已有）：
- client_{cid}_X.npy
- client_{cid}_y.npy
- client_{cid}_X_test.npy
- client_{cid}_y_test.npy

输出文件（新结构）：
- dataset/IoT/train/{cid}/X.npy
- dataset/IoT/train/{cid}/y.npy
- dataset/IoT/test/{cid}/X.npy
- dataset/IoT/test/{cid}/y.npy

要求：
1) 支持参数：`--input-dir`（默认 data/processed_data）, `--output-root`（默认 dataset/IoT）, `--num-clients`（默认10）。
2) 自动创建目录。
3) 每个客户端打印迁移后的样本数。
4) 最后打印总计信息。
5) 不改动原始 processed_data 文件。
```

---

## Step 3：替换模型为一维版本（不保留图像模型）

```text
请修改 `src/flcore/trainmodel/models.py`，将当前图像模型主逻辑替换为一维流量特征模型。

要求：
1) 新增 `IoTMLPBase`：
   - 输入 `input_dim`（默认77）
   - 输出 `feature_dim`（从参数传入）
   - forward 输出 `(B, feature_dim)`
2) 改造 `BaseHeadSplit`：
   - 若 base 不含 heads/head/fc/classifier，不报错，直接把 base 视为特征提取器。
   - head 使用 args.heads 或默认 `nn.Linear(feature_dim, num_classes)`。
3) 删除或停用当前与二维图像强绑定的模型定义（Conv2d图像分支、ViT图像分支等），避免后续误用。
4) 保留 FedProto/FedAvg 所需的 `base + head` 调用范式。
5) 提供 main.py 中可直接 eval 的模型字符串示例。
```

---

## Step 4：修改 main.py 对齐一维任务

```text
请修改 `src/main.py`，将入口配置切换为一维任务专用版本。

要求：
1) 将 `model_family` 精简为一维任务可用分支（至少保留 `IoT_MLP`，可选保留 `IoT_CNN1D`）。
2) 增加参数：`--input_dim` 默认77。
3) 默认参数改为：
   - num_classes = 15
   - num_clients = 10
   - dataset = IoT
   - model_family = IoT_MLP
4) 删除/停用图像相关分支（如 Transformer_MFR、HtFE*、V-Mamba 等）和对应图像参数依赖（如 `img_size/patch_size/in_channels`）。
5) 启动日志里打印 input_dim、num_classes、num_clients。
```

---

## Step 5：先跑 FedAvg 冒烟，再跑 FedProto 冒烟

```text
请不要继续大改代码，先验证可运行性。

步骤A：FedAvg最小冒烟
- global_rounds=3, local_epochs=1, batch_size=64, device=cpu
- 检查每个client是否有数据、训练是否完成、评估是否输出

步骤B：FedProto最小冒烟
- 同样小轮次
- 检查protos保存、global_protos聚合、MSE正则项是否shape一致

输出：
1) 两次运行的命令
2) 关键日志摘要
3) 若失败，精确定位文件和函数并最小修复
```

---

## Step 6：补论文需要的评估指标

```text
请在当前一维任务流程下，补充更适合类别不平衡场景的评估指标。

目标：
1) 在测试阶段增加 Macro-F1（至少）。
2) 可选增加每类Recall或混淆矩阵保存。
3) 保持原有Acc/AUC输出。

要求：
- 改动尽量集中在 server/client 的评估函数。
- 保存结果到现有results路径（格式可扩展）。
- 给出新增指标在日志中的示例输出。
```

---

## 推荐执行顺序与停止条件

1. 先做 Step 0（入口可运行）  
2. 再做 Step 1（数据读取）  
3. 做 Step 2（数据目录重排）  
4. 做 Step 3 + Step 4（模型和入口参数）  
5. 跑 Step 5（FedAvg/FedProto 冒烟）  
6. 最后做 Step 6（指标增强）

**停止条件（每步都要满足）：**
- 不出现 shape/dtype/path 报错；
- 客户端训练与测试都能跑；
- 输出日志与结果文件正常生成。


# IoT FedProto Desktop Prototype

本目录提供毕业设计配套的桌面端原型，用于把联邦学习训练流程、历史模型管理和结果展示整合为一个本地可操作的可视化界面。桌面端不重复实现训练算法，而是直接调用主工程中的 Python 脚本完成训练、汇总与测试评估。

## 功能概览

当前版本已经打通以下核心流程：

- 扫描 `dataset/` 目录，读取数据集及客户端划分信息
- 选择参与训练的客户端，并生成训练配置
- 通过 Electron 主进程调用 `python src/main.py` 启动训练
- 实时接收并显示训练日志
- 将训练产物保存到 `artifacts/models/`
- 加载已保存模型，并重新在测试客户端上执行评估
- 读取 `results/` 下的图像、指标文件和汇总表进行展示

## 与主工程的关系

桌面端以仓库根目录作为运行上下文，默认复用以下脚本与目录：

- [../src/main.py](/e:/IoT_FedProto/src/main.py:1)：联邦训练入口
- [../src/summarize_results.py](/e:/IoT_FedProto/src/summarize_results.py:1)：结果汇总脚本
- [../src/evaluate_saved_model.py](/e:/IoT_FedProto/src/evaluate_saved_model.py:1)：已保存模型测试评估
- `../dataset/`：客户端数据集根目录
- `../results/`：实验结果目录
- `../artifacts/models/`：桌面端管理的模型保存目录

因此，桌面端本质上是主工程训练链路之上的一层可视化调度与展示界面。

## 目录结构

```text
desktop-app/
├─ electron/
│  ├─ main.ts
│  └─ preload.ts
├─ scripts/
│  └─ copy-static.mjs
├─ src/
│  ├─ renderer/
│  │  ├─ global.d.ts
│  │  ├─ index.html
│  │  ├─ main.ts
│  │  └─ styles.css
│  └─ shared/
│     └─ types.ts
├─ package.json
├─ package-lock.json
└─ tsconfig.json
```

## 运行环境

建议环境如下：

- Node.js 18 及以上
- npm
- 主工程 Python 环境已安装完成

桌面端默认按以下规则定位 Python 解释器：

- Windows：`../.venv/python.exe`
- Linux / macOS：`../.venv/bin/python`

如果本地虚拟环境路径不同，可在 [electron/main.ts](/e:/IoT_FedProto/desktop-app/electron/main.ts:1) 中调整默认解释器路径。

## 安装与启动

在 `desktop-app/` 目录下执行：

```bash
npm install
```

启动应用：

```bash
npm start
```

开发阶段也可以使用：

```bash
npm run dev
```

构建命令：

```bash
npm run build
```

## 界面流程

桌面端当前采用三段式工作流：

### 1. 选择客户端

- 扫描数据集根目录
- 读取 `train/` 与 `test/` 下的客户端列表
- 选择参与训练的客户端

### 2. 运行训练

- 固定以 `FedProto + IoT_MIX_MLP_CNN1D` 作为默认展示方案
- 在表单中设置训练轮数、学习率、批大小、特征维度等参数
- 调用主工程训练入口执行训练
- 将日志实时回传到界面

### 3. 加载模型测试

- 读取 `artifacts/models/` 中的历史模型清单
- 选择测试数据集与测试客户端
- 调用评估脚本生成 Accuracy、AUC、F1、FNR、FPR、混淆矩阵等指标
- 联动显示训练生成的图像结果

## 训练产物管理

每次由桌面端启动训练后，都会在 `../artifacts/models/` 下创建独立运行目录。目录中通常包含：

- `manifest.json`：模型清单与训练配置
- `metrics/`：关联的结果指标文件
- `figures/`：训练阶段生成的可视化图像
- `logs/`：训练日志副本

桌面端会自动把 `results/` 中对应训练的核心产物同步到该运行目录，便于后续查询、复测和展示。

## 当前展示范围

- 同构模型：`IoT_MLP`、`IoT_CNN1D`、`IoT_Transformer1D`
- 异构展示主方案：`IoT_MIX_MLP_CNN1D`
- 默认训练算法：`FedProto`

其中训练面板当前将 `FedProto + IoT_MIX_MLP_CNN1D` 固定为主要展示组合，以保证界面流程与实验展示口径一致。

## 说明

- 桌面端默认假设 `desktop-app/` 与主工程同仓库嵌套存在
- 若主工程依赖未安装完成，训练、汇总和测试功能将无法正常执行
- 若数据集目录结构不完整，客户端扫描与测试映射会受到影响
- 若训练时勾选跳过图像生成，结果页可能不会显示可视化图像

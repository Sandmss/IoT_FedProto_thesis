# Electron 可视化联邦学习原型系统设计

## 1. 系统定位

本系统定位为一个基于 `Electron + Node.js + TypeScript` 的本地可视化联邦学习原型系统，目标不是重写联邦学习算法，而是在现有 `Python` 实验系统之上提供一个统一、可操作、可展示的桌面软件界面。

系统重点覆盖三条完整链路：

1. 本地数据集与客户端目录选择
2. 联邦训练任务发起、日志观察与结果保存
3. 历史实验结果读取、指标对比与图像展示

## 2. 设计原则

- 最小侵入式集成：继续复用现有 [src/main.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/main.py) 与 [src/summarize_results.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/summarize_results.py)
- 本地化运行：所有数据、模型、结果均来自本地文件目录
- 桌面端负责调度与展示：训练逻辑仍由 `Python` 执行，`Electron` 不参与算法计算
- 面向答辩展示：界面突出“流程完整、结果直观、结构清晰”

## 3. 总体架构

系统采用四层结构：

### 3.1 表现层

由 `Electron Renderer + TypeScript` 构成，负责：

- 左侧导航与多页面切换
- 数据集与客户端可视化选择
- 训练参数表单
- 实时日志显示
- 历史结果表格与单次实验详情页

### 3.2 桌面调度层

由 `Electron Main Process` 构成，负责：

- 本地窗口创建
- 目录选择对话框
- 读取 `dataset/`、`results/` 目录结构
- 调用 `Python` 训练脚本
- 转发训练日志到前端
- 调用结果导出脚本解析 `.h5`

### 3.3 Python 算法层

继续复用仓库内现有联邦学习代码，负责：

- 数据加载
- 客户端训练
- 服务端聚合
- 指标统计
- 模型与实验结果输出

### 3.4 本地资源层

直接依赖现有仓库目录：

- `dataset/`
- `results/`
- `src/`
- `results/summary/experiment_summary.csv`

## 4. 功能模块设计

## 4.1 数据与客户端模块

目标是让用户明确知道当前实验使用哪个数据集、包含哪些客户端。

主要功能：

- 扫描 `dataset/` 下的数据集目录
- 自动识别 `train/` 与 `test/` 子目录
- 展示训练客户端与测试客户端数量
- 支持勾选参与训练的客户端

## 4.2 联邦训练模块

目标是把原本命令行训练流程转为界面可配置、可观察的桌面交互流程。

主要功能：

- 选择算法：`Local / FedAvg / FedProto`
- 选择模型：`IoT_MLP / IoT_CNN1D / IoT_Transformer1D / IoT_MIX_MLP_CNN1D / IoT_MIX_MLP_CNN_TRANS`
- 配置轮数、学习率、特征维度、早停参数等
- 通过 `child_process.spawn` 调用 `python src/main.py`
- 实时展示标准输出日志
- 支持训练停止
- 训练完成后自动刷新结果汇总

## 4.3 结果解析模块

目标是对单次实验结果做可视化阅读，而不是让用户直接打开 `.h5` 文件。

主要功能：

- 遍历 `results/` 下的 `.h5` 文件
- 调用新增的 [src/export_result_payload.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/export_result_payload.py)
- 读取 Accuracy、AUC、F1、FNR、FPR、推理时延等序列
- 提取最佳轮次混淆矩阵
- 加载关联图像，如 `t-SNE` 图
- 预览训练日志摘要

## 4.4 历史结果模块

目标是让系统具备论文展示与实验对比能力。

主要功能：

- 读取 `results/summary/experiment_summary.csv`
- 表格展示模型、算法、Acc、AUC Macro、F1 等核心指标
- 支持快速查看当前最优组合
- 为后续扩展“筛选、导出、答辩截图”预留结构

## 5. 界面结构设计

界面采用“左侧导航 + 右侧主工作区”的单窗口布局。

左侧导航包含：

- 数据与客户端
- 联邦训练
- 结果解析
- 历史结果

主工作区包含：

- 顶部系统介绍与快捷按钮
- 中部功能面板
- 卡片式指标区
- 日志区与图表区

视觉风格采用暖色底色配合青绿色强调色，突出“实验工作台”而不是普通后台页面，适合毕业设计展示场景。

## 6. 关键技术实现

## 6.1 Electron 与 TypeScript

新增桌面子项目目录：

```text
desktop-app/
  electron/
  src/
    renderer/
    shared/
  scripts/
  package.json
  tsconfig.json
```

其中：

- `electron/main.ts`：主进程，负责调度
- `electron/preload.ts`：暴露安全 API
- `src/renderer/main.ts`：前端页面逻辑
- `src/shared/types.ts`：统一类型定义

## 6.2 Python 集成方式

使用 `spawn` 而不是重写训练逻辑：

- 训练：调用 `python src/main.py`
- 汇总：调用 `python src/summarize_results.py`
- 单文件解析：调用 `python src/export_result_payload.py`

这种方式可以最大限度复用现有实验资产，同时保持系统结构清晰。

## 6.3 结果读取策略

系统采用“两级读取”：

1. 汇总页直接读取 `CSV`
2. 单次详情页通过 Python 解析 `.h5` 再返回 `JSON`

这样兼顾实现成本与可扩展性，也避免在 `Node.js` 里直接处理 `h5` 文件。

## 7. 当前原型实现落点

当前已新增桌面端原型骨架：

- [desktop-app/package.json](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/desktop-app/package.json)
- [desktop-app/electron/main.ts](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/desktop-app/electron/main.ts)
- [desktop-app/electron/preload.ts](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/desktop-app/electron/preload.ts)
- [desktop-app/src/renderer/main.ts](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/desktop-app/src/renderer/main.ts)
- [desktop-app/src/renderer/styles.css](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/desktop-app/src/renderer/styles.css)
- [src/export_result_payload.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/export_result_payload.py)

这套结构已经能支撑“桌面原型系统”的主体设计。

## 8. 后续可增强方向

- 增加模型测试专页，把“训练结果查看”和“独立模型测试”彻底分开
- 增加参数模板保存与加载
- 增加实验对比筛选器与排序器
- 增加图像导出与报告导出
- 增加打包发布流程，形成可独立运行的 `.exe`

## 9. 结论

该方案适合作为你的毕业设计最终系统形态，因为它：

- 满足“可视化界面 + 可操作流程 + 结果展示”的要求
- 与现有 `Python` 联邦学习代码天然衔接
- 实现成本可控
- 便于答辩展示与后续扩展

从工程实现角度看，这是一种风险低、表达力强、最符合当前项目阶段的桌面化原型设计方案。

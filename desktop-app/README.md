# IoT FedProto Desktop Prototype

这是为当前毕业设计仓库准备的桌面端原型骨架，目标是把现有 `Python` 联邦学习训练系统组织成一个本地可操作、可展示、可答辩的桌面界面。

## 设计原则

- 不重写训练算法，继续复用 [src/main.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/main.py)
- 不重写结果汇总逻辑，继续复用 [src/summarize_results.py](/c:/Users/朱世豪/Desktop/毕业设计/IoT_FedProto_thesis/src/summarize_results.py)
- Electron 只负责本地目录选择、训练调度、日志展示、结果读取和历史对比

## 目录结构

```text
desktop-app/
  electron/
    main.ts
    preload.ts
  scripts/
    copy-static.mjs
  src/
    renderer/
      index.html
      main.ts
      styles.css
      global.d.ts
    shared/
      types.ts
  package.json
  tsconfig.json
```

## 当前原型能力

1. 扫描 `dataset/` 目录并展示训练端、测试端客户端。
2. 从表单生成训练配置，并通过 Electron Main Process 调用 `python src/main.py`。
3. 实时接收训练日志，训练结束后自动刷新历史结果索引。
4. 浏览 `results/` 下的 `.h5` 文件，并调用 Python 辅助脚本读取指标、混淆矩阵、图像与日志摘要。
5. 从 `results/summary/experiment_summary.csv` 加载历史实验对比表。

## 启动方式

先在 `desktop-app/` 安装依赖：

```bash
npm install
```

然后构建并启动：

```bash
npm start
```

## 说明

- 默认假设 `desktop-app/` 与论文仓库根目录同级嵌套，即当前结构下 `desktop-app/..` 就是项目根目录。
- 默认 Python 解释器为项目根目录下的 `.venv`。
  Windows: `.venv/python.exe`
  Linux/macOS: `.venv/bin/python`
- 如果你的环境路径不同，可以在 `electron/main.ts` 里调整 `pythonExecutable`。
- 当前版本是“原型系统骨架”，重点在完整链路打通与界面结构设计，后续可以继续增强任务队列、模型测试页、导出报告页等功能。

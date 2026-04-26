type TrainingStatus = "idle" | "running" | "success" | "failed" | "stopped";

interface ProjectContext {
  projectRoot: string;
  datasetRoot: string;
  resultsRoot: string;
  srcRoot: string;
  pythonExecutable: string;
}

interface DatasetInfo {
  name: string;
  path: string;
  trainDir: string;
  testDir: string;
  trainClients: string[];
  testClients: string[];
}

interface TrainingConfig {
  dataset: string;
  algorithm: string;
  modelFamily: string;
  numClients: number;
  globalRounds: number;
  localEpochs: number;
  localLearningRate: number;
  joinRatio: number;
  featureDim: number;
  device: "cpu" | "cuda";
  deviceId: string;
  times: number;
  goal: string;
  inputDim: number;
  numClasses: number;
  normalClass: number;
  batchSize: number;
  numWorkers: number;
  earlyStopPatience: number;
  evalGap: number;
  lamda: number;
  skipFigures: boolean;
  transformerDModel: number;
  transformerNumHeads: number;
  transformerNumLayers: number;
  transformerDropout: number;
}

interface ResultSummaryRow {
  Dataset: string;
  Setting: string;
  Model: string;
  Algorithm: string;
  ModelCategory: string;
  Goal: string;
  RunIndex: string;
  Acc: string;
  "AUC Macro": string;
  "AUC Micro": string;
  Precision: string;
  Recall: string;
  F1: string;
  FNR: string;
  FPR: string;
  InferenceLatencyMs: string;
  ResultFile: string;
  [key: string]: string;
}

interface ResultDetail {
  path: string;
  relativePath: string;
  series: Record<string, number[]>;
  bestRound: number | null;
  confusionMatrix: number[][] | null;
  figures: string[];
  logPreview: string[];
}

interface TrainingEvent {
  type: "log" | "status";
  status?: TrainingStatus;
  message: string;
  timestamp: string;
}

const state: {
  context: ProjectContext | null;
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  selectedClients: string[];
  summaryRows: ResultSummaryRow[];
  resultFiles: string[];
  selectedResultFile: string | null;
  selectedResultDetail: ResultDetail | null;
  trainingStatus: TrainingStatus;
  logs: string[];
} = {
  context: null,
  datasets: [],
  selectedDataset: null,
  selectedClients: [],
  summaryRows: [],
  resultFiles: [],
  selectedResultFile: null,
  selectedResultDetail: null,
  trainingStatus: "idle",
  logs: [],
};

const ALGORITHMS = ["Local", "FedAvg", "FedProto"];
const MODEL_FAMILIES = [
  "IoT_MLP",
  "IoT_CNN1D",
  "IoT_Transformer1D",
  "IoT_MIX_MLP_CNN1D",
  "IoT_MIX_MLP_CNN_TRANS",
];

function fileUrl(absolutePath: string): string {
  const normalized = absolutePath.replace(/\\/g, "/");
  return normalized.startsWith("/") ? `file://${normalized}` : `file:///${normalized}`;
}

function renderShell(): void {
  const app = document.getElementById("app");
  if (!app) {
    return;
  }

  app.innerHTML = `
    <div class="shell">
      <aside class="sidebar">
        <div class="brand">
          <h1>IoT FedProto Studio</h1>
          <p>基于 Electron + Node.js + TypeScript 的本地化联邦学习原型系统，复用现有 Python 训练链路。</p>
        </div>
        <nav class="nav">
          <button class="nav-btn active" data-section="dataset">数据与客户端</button>
          <button class="nav-btn" data-section="training">联邦训练</button>
          <button class="nav-btn" data-section="results">结果解析</button>
          <button class="nav-btn" data-section="history">历史结果</button>
        </nav>
        <div class="context-card">
          <h2>项目上下文</h2>
          <div id="context-card-body"></div>
        </div>
      </aside>
      <main class="main">
        <section class="hero">
          <div>
            <h2>桌面化联邦学习工作台</h2>
            <p>围绕“数据选择、训练调度、结果读取、历史对比”四条主线组织现有实验系统，满足毕业设计答辩中的可视化操作与结果展示需求。</p>
          </div>
          <div class="hero-actions">
            <button id="refresh-summary" class="button secondary">刷新结果汇总</button>
            <button id="choose-dataset-root" class="button primary">选择数据目录</button>
          </div>
        </section>

        <section class="section active" id="section-dataset"></section>
        <section class="section" id="section-training"></section>
        <section class="section" id="section-results"></section>
        <section class="section" id="section-history"></section>
      </main>
    </div>
  `;

  bindNav();
  bindGlobalActions();
}

function bindNav(): void {
  document.querySelectorAll<HTMLButtonElement>(".nav-btn").forEach((button) => {
    button.addEventListener("click", () => {
      const sectionName = button.dataset.section;
      document.querySelectorAll<HTMLButtonElement>(".nav-btn").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll<HTMLElement>(".section").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`section-${sectionName}`)?.classList.add("active");
    });
  });
}

function bindGlobalActions(): void {
  document.getElementById("refresh-summary")?.addEventListener("click", async () => {
    if (!state.context) {
      return;
    }
    const refreshed = await window.desktopApi.refreshSummary(state.context);
    state.summaryRows = refreshed.rows;
    state.resultFiles = refreshed.resultFiles;
    if (!state.selectedResultFile && state.resultFiles.length > 0) {
      state.selectedResultFile = state.resultFiles[0];
    }
    await loadResultDetail();
    renderAll();
    window.alert(refreshed.output || "结果汇总已刷新。");
  });

  document.getElementById("choose-dataset-root")?.addEventListener("click", async () => {
    if (!state.context) {
      return;
    }
    const chosen = await window.desktopApi.chooseDirectory(state.context.datasetRoot);
    if (!chosen) {
      return;
    }
    state.context = { ...state.context, datasetRoot: chosen };
    await loadDatasets();
    renderAll();
  });
}

function renderContextCard(): void {
  const container = document.getElementById("context-card-body");
  if (!container || !state.context) {
    return;
  }

  container.innerHTML = `
    <div class="path-item">
      <label>Project Root</label>
      <code>${state.context.projectRoot}</code>
    </div>
    <div class="path-item">
      <label>Dataset Root</label>
      <code>${state.context.datasetRoot}</code>
    </div>
    <div class="path-item">
      <label>Results Root</label>
      <code>${state.context.resultsRoot}</code>
    </div>
    <div class="path-item">
      <label>Python</label>
      <code>${state.context.pythonExecutable}</code>
    </div>
  `;
}

function metricCard(label: string, value: string, note?: string): string {
  return `
    <div class="metric-card">
      <span>${label}</span>
      <strong>${value}</strong>
      ${note ? `<div class="caption">${note}</div>` : ""}
    </div>
  `;
}

function renderDatasetSection(): void {
  const root = document.getElementById("section-dataset");
  if (!root) {
    return;
  }

  const selectedDataset = state.selectedDataset;
  const datasetMetrics = selectedDataset
    ? `
      <div class="grid three">
        ${metricCard("训练客户端", String(selectedDataset.trainClients.length))}
        ${metricCard("测试客户端", String(selectedDataset.testClients.length))}
        ${metricCard("当前勾选", String(state.selectedClients.length))}
      </div>
    `
    : `<div class="panel">未发现可用数据集目录。</div>`;

  root.innerHTML = `
    <div class="grid two">
      <div class="panel">
        <div class="panel-title">数据集目录</div>
        <div class="dataset-list stack" id="dataset-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title">数据集摘要</div>
        ${datasetMetrics}
        ${
          selectedDataset
            ? `
              <div class="stack" style="margin-top: 16px;">
                <div><strong>Train</strong><div class="caption">${selectedDataset.trainDir}</div></div>
                <div><strong>Test</strong><div class="caption">${selectedDataset.testDir}</div></div>
              </div>
            `
            : ""
        }
      </div>
    </div>
    <div class="grid two" style="margin-top: 18px;">
      <div class="panel">
        <div class="panel-title">参与训练的客户端</div>
        <div style="margin-bottom: 12px;">
          <button id="select-all-clients" class="button secondary">全选</button>
          <button id="clear-all-clients" class="button secondary">清空</button>
        </div>
        <div class="client-list stack" id="train-client-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title">测试端客户端目录</div>
        <div class="client-list stack" id="test-client-list"></div>
      </div>
    </div>
  `;

  const datasetList = document.getElementById("dataset-list");
  datasetList!.innerHTML = state.datasets
    .map(
      (dataset) => `
        <div class="dataset-item ${dataset.name === selectedDataset?.name ? "active" : ""}">
          <button data-dataset="${dataset.name}">
            <strong>${dataset.name}</strong>
            <div class="caption">${dataset.trainClients.length} train / ${dataset.testClients.length} test</div>
          </button>
        </div>
      `,
    )
    .join("");

  datasetList!.querySelectorAll<HTMLButtonElement>("button[data-dataset]").forEach((button) => {
    button.addEventListener("click", () => {
      const datasetName = button.dataset.dataset!;
      const dataset = state.datasets.find((item) => item.name === datasetName) || null;
      state.selectedDataset = dataset;
      state.selectedClients = dataset ? [...dataset.trainClients] : [];
      renderDatasetSection();
      renderTrainingSection();
    });
  });

  const trainClientList = document.getElementById("train-client-list");
  const testClientList = document.getElementById("test-client-list");

  trainClientList!.innerHTML = selectedDataset
    ? selectedDataset.trainClients
        .map(
          (clientId) => `
            <label class="client-item">
              <input type="checkbox" data-client="${clientId}" ${state.selectedClients.includes(clientId) ? "checked" : ""} />
              客户端 ${clientId}
            </label>
          `,
        )
        .join("")
    : `<div class="muted">暂无客户端。</div>`;

  testClientList!.innerHTML = selectedDataset
    ? selectedDataset.testClients.map((clientId) => `<div class="client-item">客户端 ${clientId}</div>`).join("")
    : `<div class="muted">暂无客户端。</div>`;

  trainClientList!.querySelectorAll<HTMLInputElement>("input[data-client]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const clientId = checkbox.dataset.client!;
      state.selectedClients = checkbox.checked
        ? [...new Set([...state.selectedClients, clientId])]
        : state.selectedClients.filter((item) => item !== clientId);
      renderDatasetSection();
      renderTrainingSection();
    });
  });

  document.getElementById("select-all-clients")?.addEventListener("click", () => {
    state.selectedClients = selectedDataset ? [...selectedDataset.trainClients] : [];
    renderDatasetSection();
    renderTrainingSection();
  });

  document.getElementById("clear-all-clients")?.addEventListener("click", () => {
    state.selectedClients = [];
    renderDatasetSection();
    renderTrainingSection();
  });
}

function getDefaultTrainingConfig(): TrainingConfig {
  return {
    dataset: state.selectedDataset?.name || "IoT",
    algorithm: "FedProto",
    modelFamily: "IoT_MLP",
    numClients: Math.max(1, state.selectedClients.length || state.selectedDataset?.trainClients.length || 1),
    globalRounds: 100,
    localEpochs: 1,
    localLearningRate: 0.005,
    joinRatio: 1,
    featureDim: 64,
    device: "cpu",
    deviceId: "0",
    times: 1,
    goal: "desktop_prototype",
    inputDim: 77,
    numClasses: 15,
    normalClass: 0,
    batchSize: 10,
    numWorkers: 0,
    earlyStopPatience: 100,
    evalGap: 1,
    lamda: 1,
    skipFigures: false,
    transformerDModel: 64,
    transformerNumHeads: 4,
    transformerNumLayers: 2,
    transformerDropout: 0.2,
  };
}

function trainingStatusMarkup(): string {
  return `<div class="status-pill ${state.trainingStatus}">状态：${state.trainingStatus}</div>`;
}

function renderTrainingSection(): void {
  const root = document.getElementById("section-training");
  if (!root) {
    return;
  }

  const defaults = getDefaultTrainingConfig();

  root.innerHTML = `
    <div class="panel">
      <div class="panel-title">训练配置</div>
      <form id="training-form" class="stack">
        <div class="form-grid">
          ${selectField("algorithm", "算法", ALGORITHMS, defaults.algorithm)}
          ${selectField("modelFamily", "模型", MODEL_FAMILIES, defaults.modelFamily)}
          ${selectField("device", "设备", ["cpu", "cuda"], defaults.device)}
          ${inputField("goal", "Goal Tag", defaults.goal)}
          ${inputField("numClients", "客户端数量", String(defaults.numClients), "number")}
          ${inputField("globalRounds", "全局轮数", String(defaults.globalRounds), "number")}
          ${inputField("localEpochs", "本地 Epoch", String(defaults.localEpochs), "number")}
          ${inputField("localLearningRate", "学习率", String(defaults.localLearningRate), "number", "0.0001")}
          ${inputField("joinRatio", "Join Ratio", String(defaults.joinRatio), "number", "0.01")}
          ${inputField("featureDim", "Feature Dim", String(defaults.featureDim), "number")}
          ${inputField("batchSize", "Batch Size", String(defaults.batchSize), "number")}
          ${inputField("numWorkers", "Workers", String(defaults.numWorkers), "number")}
          ${inputField("inputDim", "Input Dim", String(defaults.inputDim), "number")}
          ${inputField("numClasses", "类别数", String(defaults.numClasses), "number")}
          ${inputField("normalClass", "Normal Class", String(defaults.normalClass), "number")}
          ${inputField("earlyStopPatience", "Early Stop", String(defaults.earlyStopPatience), "number")}
          ${inputField("evalGap", "Eval Gap", String(defaults.evalGap), "number")}
          ${inputField("lamda", "Lambda", String(defaults.lamda), "number", "0.1")}
          ${inputField("times", "重复次数", String(defaults.times), "number")}
          ${inputField("deviceId", "CUDA Device ID", defaults.deviceId)}
          ${inputField("transformerDModel", "Transformer d_model", String(defaults.transformerDModel), "number")}
          ${inputField("transformerNumHeads", "Transformer Heads", String(defaults.transformerNumHeads), "number")}
          ${inputField("transformerNumLayers", "Transformer Layers", String(defaults.transformerNumLayers), "number")}
          ${inputField("transformerDropout", "Transformer Dropout", String(defaults.transformerDropout), "number", "0.1")}
        </div>
        <label class="client-item">
          <input type="checkbox" name="skipFigures" />
          跳过图像生成，加快原型验证
        </label>
        <div style="display:flex; gap: 10px; align-items:center;">
          <button class="button primary" type="submit">启动训练</button>
          <button class="button danger" type="button" id="stop-training">停止训练</button>
          ${trainingStatusMarkup()}
        </div>
      </form>
    </div>
    <div class="grid two" style="margin-top: 18px;">
      <div class="panel">
        <div class="panel-title">训练说明</div>
        <div class="stack">
          ${metricCard("选中数据集", state.selectedDataset?.name || "-", "来自 dataset/ 下的本地目录")}
          ${metricCard("当前选中客户端", String(state.selectedClients.length), "原型阶段先做可视化确认，训练入口仍走 Python")}
          ${metricCard("训练入口", "src/main.py", "Electron Main Process 负责调度，Renderer 负责展示")}
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">实时日志</div>
        <div class="log-view" id="training-log">${state.logs.join("\n") || "等待训练任务启动..."}</div>
      </div>
    </div>
  `;

  document.getElementById("training-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.context) {
      return;
    }
    const config = readTrainingForm();
    state.logs = [];
    renderTrainingSection();
    try {
      await window.desktopApi.startTraining(state.context, config);
    } catch (error) {
      state.logs.push(String(error));
      state.trainingStatus = "failed";
      renderTrainingSection();
    }
  });

  document.getElementById("stop-training")?.addEventListener("click", async () => {
    await window.desktopApi.stopTraining();
  });
}

function selectField(name: string, label: string, options: string[], value: string): string {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <select id="${name}" name="${name}">
        ${options.map((option) => `<option value="${option}" ${option === value ? "selected" : ""}>${option}</option>`).join("")}
      </select>
    </div>
  `;
}

function inputField(name: string, label: string, value: string, type = "text", step?: string): string {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <input id="${name}" name="${name}" type="${type}" value="${value}" ${step ? `step="${step}"` : ""} />
    </div>
  `;
}

function readTrainingForm(): TrainingConfig {
  const form = document.getElementById("training-form") as HTMLFormElement;
  const data = new FormData(form);

  const number = (key: string): number => Number(data.get(key));
  const text = (key: string): string => String(data.get(key) ?? "");

  return {
    dataset: state.selectedDataset?.name || "IoT",
    algorithm: text("algorithm"),
    modelFamily: text("modelFamily"),
    numClients: number("numClients"),
    globalRounds: number("globalRounds"),
    localEpochs: number("localEpochs"),
    localLearningRate: number("localLearningRate"),
    joinRatio: number("joinRatio"),
    featureDim: number("featureDim"),
    device: text("device") as "cpu" | "cuda",
    deviceId: text("deviceId"),
    times: number("times"),
    goal: text("goal"),
    inputDim: number("inputDim"),
    numClasses: number("numClasses"),
    normalClass: number("normalClass"),
    batchSize: number("batchSize"),
    numWorkers: number("numWorkers"),
    earlyStopPatience: number("earlyStopPatience"),
    evalGap: number("evalGap"),
    lamda: number("lamda"),
    skipFigures: data.get("skipFigures") === "on",
    transformerDModel: number("transformerDModel"),
    transformerNumHeads: number("transformerNumHeads"),
    transformerNumLayers: number("transformerNumLayers"),
    transformerDropout: number("transformerDropout"),
  };
}

function renderLineChart(series: number[], stroke = "#0f766e"): string {
  if (series.length === 0) {
    return `<div class="muted">暂无可绘制曲线。</div>`;
  }

  const max = Math.max(...series);
  const min = Math.min(...series);
  const width = 600;
  const height = 220;
  const points = series
    .map((value, index) => {
      const x = (index / Math.max(series.length - 1, 1)) * width;
      const ratio = max === min ? 0.5 : (value - min) / (max - min);
      const y = height - ratio * height;
      return `${x},${y}`;
    })
    .join(" ");

  return `
    <div class="chart">
      <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        <polyline fill="none" stroke="${stroke}" stroke-width="3" points="${points}" />
      </svg>
      <div class="caption">min ${min.toFixed(4)} / max ${max.toFixed(4)} / rounds ${series.length}</div>
    </div>
  `;
}

function renderResultSection(): void {
  const root = document.getElementById("section-results");
  if (!root) {
    return;
  }

  root.innerHTML = `
    <div class="grid two">
      <div class="panel">
        <div class="panel-title">结果文件</div>
        <div class="result-list stack" id="result-file-list"></div>
      </div>
      <div class="panel" id="result-detail-panel"></div>
    </div>
  `;

  const list = document.getElementById("result-file-list");
  list!.innerHTML = state.resultFiles.length
    ? state.resultFiles
        .map(
          (file) => `
            <div class="result-item ${file === state.selectedResultFile ? "active" : ""}">
              <button data-result-file="${file}">
                <strong>${file}</strong>
              </button>
            </div>
          `,
        )
        .join("")
    : `<div class="muted">尚未发现 .h5 结果文件。</div>`;

  list!.querySelectorAll<HTMLButtonElement>("button[data-result-file]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedResultFile = button.dataset.resultFile!;
      await loadResultDetail();
      renderResultSection();
    });
  });

  renderResultDetailPanel();
}

function renderResultDetailPanel(): void {
  const panel = document.getElementById("result-detail-panel");
  if (!panel) {
    return;
  }

  const detail = state.selectedResultDetail;
  if (!detail) {
    panel.innerHTML = `<div class="panel-title">结果详情</div><div class="muted">请选择一个结果文件。</div>`;
    return;
  }

  const accuracy = detail.series["Accuracy"] || [];
  const auc = detail.series["AUC Macro"] || [];
  const f1 = detail.series["F1"] || [];
  const lastLog = detail.logPreview.join("\n");

  panel.innerHTML = `
    <div class="panel-title">结果详情</div>
    <div class="stack">
      ${metricCard("结果文件", detail.relativePath, "由 Python 侧解析 .h5 后返回")}
      ${metricCard("最佳轮次", detail.bestRound === null ? "-" : String(detail.bestRound))}
      <div class="grid three">
        ${metricCard("Best Acc", accuracy.length ? Math.max(...accuracy).toFixed(4) : "-")}
        ${metricCard("Best AUC Macro", auc.length ? Math.max(...auc).toFixed(4) : "-")}
        ${metricCard("Best F1", f1.length ? Math.max(...f1).toFixed(4) : "-")}
      </div>
      <div>
        <div class="panel-title">Accuracy 曲线</div>
        ${renderLineChart(accuracy)}
      </div>
      <div>
        <div class="panel-title">AUC Macro 曲线</div>
        ${renderLineChart(auc, "#b45309")}
      </div>
      ${
        detail.confusionMatrix
          ? `
            <div class="matrix">
              <div class="panel-title">最佳轮次混淆矩阵</div>
              <table>
                <tbody>
                  ${detail.confusionMatrix.map((row) => `<tr>${row.map((item) => `<td>${item}</td>`).join("")}</tr>`).join("")}
                </tbody>
              </table>
            </div>
          `
          : ""
      }
      ${
        detail.figures.length
          ? `
            <div>
              <div class="panel-title">关联图像</div>
              <div class="figure-strip">
                ${detail.figures.map((figure) => `<img src="${fileUrl(figure)}" alt="${figure}" />`).join("")}
              </div>
            </div>
          `
          : ""
      }
      <div>
        <div class="panel-title">日志摘录</div>
        <div class="log-view">${lastLog || "暂无日志预览。"}</div>
      </div>
    </div>
  `;
}

function renderHistorySection(): void {
  const root = document.getElementById("section-history");
  if (!root) {
    return;
  }

  const rows = [...state.summaryRows].sort((left, right) => Number(right.Acc || 0) - Number(left.Acc || 0));
  const top = rows[0];

  root.innerHTML = `
    <div class="grid three">
      ${metricCard("历史实验数", String(rows.length))}
      ${metricCard("当前最佳 Accuracy", top?.Acc || "-")}
      ${metricCard("最佳组合", top ? `${top.Algorithm} / ${top.Model}` : "-")}
    </div>
    <div class="panel" style="margin-top: 18px;">
      <div class="panel-title">历史实验汇总</div>
      <div class="history-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Algorithm</th>
              <th>Acc</th>
              <th>AUC Macro</th>
              <th>F1</th>
              <th>Goal</th>
              <th>ResultFile</th>
            </tr>
          </thead>
          <tbody>
            ${
              rows.length
                ? rows
                    .map(
                      (row) => `
                        <tr>
                          <td>${row.Model || "-"}</td>
                          <td>${row.Algorithm || "-"}</td>
                          <td>${row.Acc || "-"}</td>
                          <td>${row["AUC Macro"] || "-"}</td>
                          <td>${row.F1 || "-"}</td>
                          <td>${row.Goal || "-"}</td>
                          <td>${row.ResultFile || "-"}</td>
                        </tr>
                      `,
                    )
                    .join("")
                : `<tr><td colspan="7" class="muted">请先刷新结果汇总。</td></tr>`
            }
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderAll(): void {
  renderContextCard();
  renderDatasetSection();
  renderTrainingSection();
  renderResultSection();
  renderHistorySection();
}

async function loadDatasets(): Promise<void> {
  if (!state.context) {
    return;
  }

  state.datasets = await window.desktopApi.scanDatasets(state.context.datasetRoot);
  state.selectedDataset = state.datasets[0] || null;
  state.selectedClients = state.selectedDataset ? [...state.selectedDataset.trainClients] : [];
}

async function loadResults(): Promise<void> {
  if (!state.context) {
    return;
  }

  state.summaryRows = await window.desktopApi.listSummaryRows(state.context.resultsRoot);
  state.resultFiles = await window.desktopApi.listResultFiles(state.context.resultsRoot);
  state.selectedResultFile = state.resultFiles[0] || null;
}

async function loadResultDetail(): Promise<void> {
  if (!state.context || !state.selectedResultFile) {
    state.selectedResultDetail = null;
    return;
  }
  state.selectedResultDetail = await window.desktopApi.getResultDetail(
    state.context.resultsRoot,
    state.selectedResultFile,
    state.context,
  );
}

function subscribeTrainingEvents(): void {
  window.desktopApi.onTrainingEvent(async (event: TrainingEvent) => {
    if (event.type === "status" && event.status) {
      state.trainingStatus = event.status;
    }

    state.logs = [...state.logs, `[${event.timestamp}] ${event.message}`].slice(-300);
    renderTrainingSection();

    if (event.type === "status" && event.status === "success" && state.context) {
      const refreshed = await window.desktopApi.refreshSummary(state.context);
      state.summaryRows = refreshed.rows;
      state.resultFiles = refreshed.resultFiles;
      if (!state.selectedResultFile && state.resultFiles.length > 0) {
        state.selectedResultFile = state.resultFiles[0];
      }
      await loadResultDetail();
      renderResultSection();
      renderHistorySection();
    }
  });
}

async function bootstrap(): Promise<void> {
  renderShell();
  state.context = await window.desktopApi.getContext();
  await loadDatasets();
  await loadResults();
  await loadResultDetail();
  subscribeTrainingEvents();
  renderAll();
}

void bootstrap();

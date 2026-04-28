import type {
  DatasetInfo,
  ProjectContext,
  ResultDetail,
  ResultSummaryRow,
  SavedModelInfo,
  TestEvaluationResult,
  TrainingConfig,
  TrainingEvent,
  TrainingStatus,
} from "../shared/types";

const PRIMARY_ALGORITHM = "FedProto";
const PRIMARY_MODEL_FAMILY = "IoT_MIX_MLP_CNN1D";

const state: {
  context: ProjectContext | null;
  datasets: DatasetInfo[];
  selectedDataset: DatasetInfo | null;
  selectedClients: string[];
  selectedTestClients: string[];
  savedModels: SavedModelInfo[];
  selectedModelId: string | null;
  evaluationResult: TestEvaluationResult | null;
  selectedModelResultDetail: ResultDetail | null;
  summaryRows: ResultSummaryRow[];
  trainingStatus: TrainingStatus;
  logs: string[];
  testing: boolean;
  trainingLogScrollTop: number;
  trainingLogStickToBottom: boolean;
  datasetTrainClientScrollTop: number;
  testingClientScrollTop: number;
  trainingConfigDraft: TrainingConfig | null;
} = {
  context: null,
  datasets: [],
  selectedDataset: null,
  selectedClients: [],
  selectedTestClients: [],
  savedModels: [],
  selectedModelId: null,
  evaluationResult: null,
  selectedModelResultDetail: null,
  summaryRows: [],
  trainingStatus: "idle",
  logs: [],
  testing: false,
  trainingLogScrollTop: 0,
  trainingLogStickToBottom: true,
  datasetTrainClientScrollTop: 0,
  testingClientScrollTop: 0,
  trainingConfigDraft: null,
};

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
          <p>联邦恶意流量检测系统</p>
        </div>
        <nav class="nav">
          <button class="nav-btn active" data-section="dataset">1. 选择客户端</button>
          <button class="nav-btn" data-section="training">2. 运行训练</button>
          <button class="nav-btn" data-section="testing">3. 加载模型测试</button>
        </nav>
        <div class="context-card">
          <h2>项目上下文</h2>
          <div id="context-card-body"></div>
        </div>
      </aside>
      <main class="main">
        <section class="hero">
          <div>
            <h2>功能界面</h2>
          </div>
          <div class="hero-actions">
            <button id="refresh-summary" class="button secondary">刷新结果汇总</button>
            <button id="choose-dataset-root" class="button primary">选择数据目录</button>
          </div>
        </section>

        <section class="section active" id="section-dataset"></section>
        <section class="section" id="section-training"></section>
        <section class="section" id="section-testing"></section>
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
    state.savedModels = await window.desktopApi.listSavedModels(state.context.projectRoot);
    if (!state.selectedModelId) {
      state.selectedModelId = state.savedModels[0]?.id || null;
      resetDefaultTestClients();
    }
    renderTestingSection();
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

function inputField(name: string, label: string, value: string, type = "text", step?: string): string {
  return `
    <div class="field">
      <label for="${name}">${label}</label>
      <input id="${name}" name="${name}" type="${type}" value="${value}" ${step ? `step="${step}"` : ""} />
    </div>
  `;
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

function getSelectedSavedModel(): SavedModelInfo | null {
  return state.savedModels.find((item) => item.id === state.selectedModelId) || null;
}

function formatModelTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatModelStatus(status: SavedModelInfo["status"]): string {
  if (status === "ready") {
    return "已完成";
  }
  if (status === "running") {
    return "训练中";
  }
  if (status === "stopped") {
    return "已停止";
  }
  if (status === "failed") {
    return "失败";
  }
  return status;
}

async function handleDeleteSavedModel(modelId: string): Promise<void> {
  const targetModel = state.savedModels.find((item) => item.id === modelId);
  if (!targetModel) {
    return;
  }

  const confirmed = window.confirm(
    `确认删除这条已训练模型记录吗？\n\n${targetModel.label}\n轮次 ${targetModel.config?.globalRounds ?? "-"}\n时间 ${formatModelTimestamp(targetModel.createdAt)}`,
  );
  if (!confirmed) {
    return;
  }

  try {
    await window.desktopApi.deleteSavedModel(modelId);
    await loadSavedModels();
    if (!state.savedModels.some((item) => item.id === state.selectedModelId)) {
      state.selectedModelId = state.savedModels[0]?.id || null;
    }
    state.evaluationResult = null;
    resetDefaultTestClients();
    await loadSelectedModelResultDetail();
    renderTestingSection();
  } catch (error) {
    window.alert(String(error));
  }
}

function getFigureTitle(figurePath: string): string {
  const lower = figurePath.toLowerCase();
  if (lower.includes("feature_tsne")) {
    return "特征 t-SNE";
  }
  if (lower.includes("prototype_distribution")) {
    return "原型分布图";
  }
  const fileName = figurePath.split(/[\\/]/).pop() || figurePath;
  return fileName.replace(/\.[^.]+$/, "");
}

function resetDefaultTestClients(): void {
  const selectedModel = getSelectedSavedModel();
  const selectedDataset = state.selectedDataset;
  if (!selectedModel || !selectedDataset) {
    state.selectedTestClients = [];
    return;
  }

  const matched = selectedModel.trainingClients.filter((clientId) => selectedDataset.testClients.includes(clientId));
  if (matched.length === selectedModel.trainingClients.length) {
    state.selectedTestClients = matched;
    return;
  }

  state.selectedTestClients = selectedDataset.testClients.slice(0, selectedModel.trainingClients.length);
}

function renderDatasetSection(): void {
  const root = document.getElementById("section-dataset");
  if (!root) {
    return;
  }

  const previousTrainClientList = document.getElementById("train-client-list");
  if (previousTrainClientList) {
    state.datasetTrainClientScrollTop = previousTrainClientList.scrollTop;
  }

  const selectedDataset = state.selectedDataset;
  root.innerHTML = `
    <div class="grid two">
      <div class="panel">
        <div class="panel-title">数据集目录</div>
        <div class="dataset-list stack" id="dataset-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title">当前数据集</div>
        ${
          selectedDataset
            ? `
              <div class="grid three">
                ${metricCard("训练客户端", String(selectedDataset.trainClients.length))}
                ${metricCard("测试客户端", String(selectedDataset.testClients.length))}
                ${metricCard("已选训练端", String(state.selectedClients.length))}
              </div>
              <div class="stack" style="margin-top: 16px;">
                <div><strong>Train</strong><div class="caption">${selectedDataset.trainDir}</div></div>
                <div><strong>Test</strong><div class="caption">${selectedDataset.testDir}</div></div>
              </div>
            `
            : `<div class="muted">未发现可用数据集目录。</div>`
        }
      </div>
    </div>
    <div class="grid two" style="margin-top: 18px;">
      <div class="panel">
        <div class="panel-title">参与训练的客户端</div>
        <div style="display:flex; gap: 10px; margin-bottom: 12px;">
          <button id="select-all-clients" class="button secondary">全选</button>
          <button id="clear-all-clients" class="button secondary">清空</button>
        </div>
        <div class="client-list stack" id="train-client-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title">测试集文件夹</div>
        <div class="client-list stack" id="test-client-preview"></div>
      </div>
    </div>
  `;

  const datasetList = document.getElementById("dataset-list");
  if (!datasetList) {
    return;
  }

  datasetList.innerHTML = state.datasets
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

  datasetList.querySelectorAll<HTMLButtonElement>("button[data-dataset]").forEach((button) => {
    button.addEventListener("click", () => {
      const datasetName = button.dataset.dataset!;
      state.selectedDataset = state.datasets.find((item) => item.name === datasetName) || null;
      state.selectedClients = state.selectedDataset ? [...state.selectedDataset.trainClients] : [];
      resetDefaultTestClients();
      renderAll();
    });
  });

  const trainClientList = document.getElementById("train-client-list");
  const testClientPreview = document.getElementById("test-client-preview");
  if (!trainClientList || !testClientPreview) {
    return;
  }

  trainClientList.innerHTML = selectedDataset
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

  testClientPreview.innerHTML = selectedDataset
    ? selectedDataset.testClients.map((clientId) => `<div class="client-item">测试客户端 ${clientId}</div>`).join("")
    : `<div class="muted">暂无测试目录。</div>`;

  trainClientList.querySelectorAll<HTMLInputElement>("input[data-client]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const clientId = checkbox.dataset.client!;
      state.datasetTrainClientScrollTop = trainClientList.scrollTop;
      state.selectedClients = checkbox.checked
        ? [...new Set([...state.selectedClients, clientId])]
        : state.selectedClients.filter((item) => item !== clientId);
      renderDatasetSection();
      renderTrainingSection();
    });
  });

  document.getElementById("select-all-clients")?.addEventListener("click", () => {
    state.datasetTrainClientScrollTop = trainClientList.scrollTop;
    state.selectedClients = selectedDataset ? [...selectedDataset.trainClients] : [];
    renderDatasetSection();
    renderTrainingSection();
  });

  document.getElementById("clear-all-clients")?.addEventListener("click", () => {
    state.datasetTrainClientScrollTop = trainClientList.scrollTop;
    state.selectedClients = [];
    renderDatasetSection();
    renderTrainingSection();
  });

  trainClientList.scrollTop = state.datasetTrainClientScrollTop;
}

function getDefaultTrainingConfig(): TrainingConfig {
  return {
    dataset: state.selectedDataset?.name || "IoT",
    algorithm: PRIMARY_ALGORITHM,
    modelFamily: PRIMARY_MODEL_FAMILY,
    selectedClients: [...state.selectedClients],
    saveFolderRoot: state.context ? `${state.context.projectRoot}/artifacts/models/pending` : "artifacts/models/pending",
    numClients: Math.max(1, state.selectedClients.length || 1),
    globalRounds: 1000,
    localEpochs: 1,
    localLearningRate: 0.005,
    joinRatio: 1,
    featureDim: 64,
    device: "cpu",
    deviceId: "0",
    times: 1,
    goal: "heterogeneous_demo",
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

function getTrainingConfigDraft(): TrainingConfig {
  const defaults = getDefaultTrainingConfig();
  const draft = state.trainingConfigDraft;
  if (!draft) {
    state.trainingConfigDraft = defaults;
    return defaults;
  }

  const merged: TrainingConfig = {
    ...draft,
    dataset: state.selectedDataset?.name || draft.dataset || defaults.dataset,
    algorithm: PRIMARY_ALGORITHM,
    modelFamily: PRIMARY_MODEL_FAMILY,
    selectedClients: [...state.selectedClients],
    saveFolderRoot: state.context ? `${state.context.projectRoot}/artifacts/models/pending` : draft.saveFolderRoot || defaults.saveFolderRoot,
    numClients: Math.max(1, state.selectedClients.length || draft.numClients || defaults.numClients),
  };
  state.trainingConfigDraft = merged;
  return merged;
}

function getTrainingStatusLabel(status: TrainingStatus): string {
  if (status === "idle") return "空闲";
  if (status === "running") return "训练中";
  if (status === "success") return "已完成";
  if (status === "failed") return "失败";
  if (status === "stopped") return "已停止";
  return status;
}

function trainingStatusMarkup(): string {
  return `<div class="status-pill ${state.trainingStatus}">状态：${getTrainingStatusLabel(state.trainingStatus)}</div>`;
}

function refreshTrainingRuntimeView(): void {
  const logView = document.getElementById("training-log");
  if (logView) {
    const shouldStickToBottom =
      logView.scrollHeight - logView.clientHeight - logView.scrollTop < 24;
    logView.textContent = state.logs.join("\n") || "等待训练任务启动...";
    if (shouldStickToBottom || state.trainingLogStickToBottom) {
      logView.scrollTop = logView.scrollHeight;
    } else {
      logView.scrollTop = state.trainingLogScrollTop;
    }
  }

  const statusPill = document.querySelector(".status-pill");
  if (statusPill) {
    statusPill.className = `status-pill ${state.trainingStatus}`;
    statusPill.textContent = `状态：${getTrainingStatusLabel(state.trainingStatus)}`;
  }
}

function renderTrainingSection(): void {
  const root = document.getElementById("section-training");
  if (!root) {
    return;
  }

  const previousLogView = document.getElementById("training-log");
  if (previousLogView) {
    const maxScrollTop = previousLogView.scrollHeight - previousLogView.clientHeight;
    state.trainingLogScrollTop = previousLogView.scrollTop;
    state.trainingLogStickToBottom = maxScrollTop - previousLogView.scrollTop < 24;
  }

  const defaults = getTrainingConfigDraft();
  root.innerHTML = `
    <div class="panel">
      <div class="panel-title">训练配置</div>
      <form id="training-form" class="stack">
        <div class="grid three">
          ${metricCard("系统模式", "异构联邦检测")}
          ${metricCard("固定算法", PRIMARY_ALGORITHM)}
          ${metricCard("固定模型", PRIMARY_MODEL_FAMILY)}
        </div>
        <div class="form-grid">
          ${selectField("device", "设备", ["cpu", "cuda"], defaults.device)}
          ${inputField("goal", "任务标识", defaults.goal)}
          ${inputField("globalRounds", "全局轮数（建议≥1000）", String(defaults.globalRounds), "number")}
          ${inputField("localEpochs", "本地轮次", String(defaults.localEpochs), "number")}
          ${inputField("localLearningRate", "学习率", String(defaults.localLearningRate), "number", "0.0001")}
          ${inputField("joinRatio", "参与比例", String(defaults.joinRatio), "number", "0.01")}
          ${inputField("featureDim", "特征维度", String(defaults.featureDim), "number")}
          ${inputField("batchSize", "批大小", String(defaults.batchSize), "number")}
          ${inputField("numWorkers", "数据加载线程", String(defaults.numWorkers), "number")}
          ${inputField("inputDim", "输入维度", String(defaults.inputDim), "number")}
          ${inputField("numClasses", "类别数", String(defaults.numClasses), "number")}
          ${inputField("normalClass", "正常类标签", String(defaults.normalClass), "number")}
          ${inputField("earlyStopPatience", "早停阈值", String(defaults.earlyStopPatience), "number")}
          ${inputField("evalGap", "评估间隔", String(defaults.evalGap), "number")}
          ${inputField("lamda", "原型损失权重", String(defaults.lamda), "number", "0.1")}
          ${inputField("times", "重复次数", String(defaults.times), "number")}
          ${inputField("deviceId", "设备编号", defaults.deviceId)}
        </div>
        <label class="client-item">
          <input type="checkbox" name="skipFigures" />
          跳过图像生成
        </label>
        <div style="display:flex; gap: 10px; align-items:center;">
          <button class="button primary" type="submit">开始训练并保存模型</button>
          <button class="button danger" type="button" id="stop-training">停止训练</button>
          ${trainingStatusMarkup()}
        </div>
      </form>
    </div>
    <div class="grid two" style="margin-top: 18px;">
      <div class="panel">
        <div class="panel-title">训练概览</div>
        <div class="stack">
          ${metricCard("当前数据集", state.selectedDataset?.name || "-")}
          ${metricCard("选中客户端", String(state.selectedClients.length))}
          ${metricCard("模型保存", "artifacts/models")}
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">实时日志</div>
        <div class="log-view" id="training-log">${state.logs.join("\n") || "等待训练任务启动..."}</div>
      </div>
    </div>
  `;

  const logView = document.getElementById("training-log");
  if (logView) {
    if (state.trainingLogStickToBottom) {
      logView.scrollTop = logView.scrollHeight;
    } else {
      logView.scrollTop = state.trainingLogScrollTop;
    }

    logView.addEventListener("scroll", () => {
      const maxScrollTop = logView.scrollHeight - logView.clientHeight;
      state.trainingLogScrollTop = logView.scrollTop;
      state.trainingLogStickToBottom = maxScrollTop - logView.scrollTop < 24;
    });
  }

  document.getElementById("training-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.context) {
      return;
    }
    if (state.selectedClients.length === 0) {
      window.alert("请至少选择一个训练客户端。");
      return;
    }

    const config = readTrainingForm();
    if (
      config.globalRounds < 1000 &&
      !window.confirm("当前训练轮次低于推荐值 1000，展示结果可能不稳定。仍要继续训练吗？")
    ) {
      return;
    }
    state.logs = [];
    state.trainingLogScrollTop = 0;
    state.trainingLogStickToBottom = true;
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

  const trainingForm = document.getElementById("training-form") as HTMLFormElement | null;
  if (trainingForm) {
    trainingForm.addEventListener("input", () => {
      state.trainingConfigDraft = readTrainingForm();
    });
    trainingForm.addEventListener("change", () => {
      state.trainingConfigDraft = readTrainingForm();
    });
  }
}

function readTrainingForm(): TrainingConfig {
  const form = document.getElementById("training-form") as HTMLFormElement;
  const data = new FormData(form);
  const number = (key: string): number => Number(data.get(key));
  const text = (key: string): string => String(data.get(key) ?? "");

  const config: TrainingConfig = {
    dataset: state.selectedDataset?.name || "IoT",
    algorithm: PRIMARY_ALGORITHM,
    modelFamily: PRIMARY_MODEL_FAMILY,
    selectedClients: [...state.selectedClients],
    saveFolderRoot: state.context ? `${state.context.projectRoot}/artifacts/models/pending` : "artifacts/models/pending",
    numClients: state.selectedClients.length,
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
    transformerDModel: 64,
    transformerNumHeads: 4,
    transformerNumLayers: 2,
    transformerDropout: 0.2,
  };
  state.trainingConfigDraft = config;
  return config;
}

function renderTestingSection(): void {
  const root = document.getElementById("section-testing");
  if (!root) {
    return;
  }

  const selectedModel = getSelectedSavedModel();
  const selectedDataset = state.selectedDataset;
  root.innerHTML = `
    <div class="grid two-wide">
      <div class="panel">
        <div class="panel-title">已训练模型</div>
        <div class="stack" id="saved-model-list"></div>
      </div>
      <div class="panel">
        <div class="panel-title">测试设置</div>
        ${
          selectedModel
            ? `
              <div class="grid three">
                ${metricCard("当前模型", selectedModel.modelFamily)}
                ${metricCard("训练算法", selectedModel.algorithm)}
                ${metricCard("训练客户端数", String(selectedModel.trainingClients.length))}
              </div>
              <div class="stack" style="margin-top: 16px;">
                <div>
                  <strong>模型目录</strong>
                  <div class="caption">${selectedModel.saveFolder}</div>
                </div>
                <div>
                  <strong>测试数据集</strong>
                  <select id="test-dataset-select">
                    ${state.datasets
                      .map(
                        (dataset) =>
                          `<option value="${dataset.name}" ${dataset.name === selectedDataset?.name ? "selected" : ""}>${dataset.name}</option>`,
                      )
                      .join("")}
                  </select>
                </div>
                <div>
                  <strong>测试客户端</strong>
                  <div class="client-list stack" id="test-client-list"></div>
                </div>
                <div style="display:flex; gap: 10px; align-items:center;">
                  <button id="run-testing" class="button primary" ${state.testing ? "disabled" : ""}>加载模型并测试</button>
                  <span class="caption">${state.testing ? "测试运行中..." : "选择测试集后输出评估数据"}</span>
                </div>
              </div>
            `
            : `<div class="muted">请先完成一次训练，或从已保存模型中选择一项。</div>`
        }
      </div>
    </div>
    <div class="panel" style="margin-top: 18px;" id="testing-result-panel"></div>
  `;

  renderSavedModelList();
  renderTestClientList();
  renderTestingResultPanel();

  document.getElementById("test-dataset-select")?.addEventListener("change", (event) => {
    const datasetName = (event.target as HTMLSelectElement).value;
    state.selectedDataset = state.datasets.find((item) => item.name === datasetName) || null;
    resetDefaultTestClients();
    renderDatasetSection();
    renderTestingSection();
  });

  document.getElementById("run-testing")?.addEventListener("click", async () => {
    if (!state.context || !selectedModel || !state.selectedDataset) {
      return;
    }
    if (state.selectedTestClients.length !== selectedModel.trainingClients.length) {
      window.alert(`测试客户端数量需要与训练客户端数量一致，当前应选择 ${selectedModel.trainingClients.length} 个。`);
      return;
    }

    state.testing = true;
    renderTestingSection();
    try {
      state.evaluationResult = await window.desktopApi.evaluateSavedModel(state.context, {
        modelId: selectedModel.id,
        dataset: state.selectedDataset.name,
        testClients: state.selectedTestClients,
      });
      await loadSelectedModelResultDetail();
    } catch (error) {
      window.alert(String(error));
    } finally {
      state.testing = false;
      renderTestingSection();
    }
  });
}

function renderSavedModelList(): void {
  const container = document.getElementById("saved-model-list");
  if (!container) {
    return;
  }

  container.innerHTML = state.savedModels.length
    ? state.savedModels
        .map(
          (item) => `
            <div class="saved-model-item ${item.id === state.selectedModelId ? "active" : ""}">
              <button class="saved-model-main" data-model-id="${item.id}">
                <strong>异构检测模型</strong>
                <div class="caption">数据集 ${item.dataset} · 轮次 ${item.config?.globalRounds ?? "-"}</div>
                <div class="caption">客户端 ${item.trainingClients.length} · ${formatModelStatus(item.status)}</div>
                <div class="caption">${formatModelTimestamp(item.createdAt)}</div>
              </button>
              <button class="button secondary saved-model-delete" type="button" data-delete-model-id="${item.id}">删除</button>
            </div>
          `,
        )
        .join("")
    : `<div class="muted">暂无已保存模型。</div>`;

  container.querySelectorAll<HTMLButtonElement>("[data-model-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedModelId = button.dataset.modelId || null;
      resetDefaultTestClients();
      await loadSelectedModelResultDetail();
      renderTestingSection();
    });
  });

  container.querySelectorAll<HTMLButtonElement>("[data-delete-model-id]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const modelId = button.dataset.deleteModelId;
      if (!modelId) {
        return;
      }
      await handleDeleteSavedModel(modelId);
    });
  });
}

function renderTestClientList(): void {
  const container = document.getElementById("test-client-list");
  const selectedDataset = state.selectedDataset;
  if (!container) {
    return;
  }

  state.testingClientScrollTop = container.scrollTop;

  container.innerHTML = selectedDataset
    ? selectedDataset.testClients
        .map(
          (clientId) => `
            <label class="client-item">
              <input type="checkbox" data-test-client="${clientId}" ${state.selectedTestClients.includes(clientId) ? "checked" : ""} />
              测试客户端 ${clientId}
            </label>
          `,
        )
        .join("")
    : `<div class="muted">暂无测试客户端。</div>`;

  container.querySelectorAll<HTMLInputElement>("input[data-test-client]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const clientId = checkbox.dataset.testClient!;
      state.testingClientScrollTop = container.scrollTop;
      state.selectedTestClients = checkbox.checked
        ? [...new Set([...state.selectedTestClients, clientId])]
        : state.selectedTestClients.filter((item) => item !== clientId);
      renderTestingSection();
    });
  });

  container.scrollTop = state.testingClientScrollTop;
}

function renderTestingResultPanel(): void {
  const panel = document.getElementById("testing-result-panel");
  if (!panel) {
    return;
  }

  const result = state.evaluationResult;
  const resultDetail = state.selectedModelResultDetail;
  if (!result) {
    panel.innerHTML = `
      <div class="panel-title">测试结果</div>
      <div class="muted">加载训练好的模型并选定测试集后，这里会展示 Accuracy、AUC、F1、FNR、混淆矩阵等评估数据。</div>
    `;
    return;
  }

  const sortedPerClient = [...result.perClient].sort(
    (left, right) => Number(left.clientId) - Number(right.clientId),
  );

  panel.innerHTML = `
    <div class="panel-title">测试结果</div>
    <div class="stack">
      <div class="grid three">
        ${metricCard("Accuracy", result.accuracy.toFixed(4))}
        ${metricCard("AUC Macro", result.aucMacro.toFixed(4))}
        ${metricCard("AUC Micro", result.aucMicro.toFixed(4))}
      </div>
      <div class="grid three">
        ${metricCard("Precision", result.precision.toFixed(4))}
        ${metricCard("Recall", result.recall.toFixed(4))}
        ${metricCard("F1", result.f1.toFixed(4))}
      </div>
      <div class="grid three">
        ${metricCard("FNR", result.fnr.toFixed(4))}
        ${metricCard("FPR", result.fpr.toFixed(4))}
        ${metricCard("推理时延", `${result.inferenceLatencyMs.toFixed(4)} ms/sample`)}
      </div>
      <div class="matrix">
        <div class="panel-title">混淆矩阵</div>
        <table>
          <tbody>
            ${result.confusionMatrix.map((row) => `<tr>${row.map((item) => `<td>${item}</td>`).join("")}</tr>`).join("")}
          </tbody>
        </table>
      </div>
      <div>
        <div class="panel-title">训练结果图像</div>
        ${
          resultDetail?.figures?.length
            ? `
              <div class="figure-strip">
                ${resultDetail.figures
                  .map(
                    (figure) => `
                      <div class="figure-card">
                        <div class="caption">${getFigureTitle(figure)}</div>
                        <img src="${fileUrl(figure)}" alt="${getFigureTitle(figure)}" />
                      </div>
                    `,
                  )
                  .join("")}
              </div>
            `
            : `<div class="muted">当前模型没有可展示的图像结果，可能是训练时跳过了图像生成，或该结果文件下没有找到关联图片。</div>`
        }
      </div>
      <div>
        <div class="panel-title">各客户端测试表现</div>
        <div class="history-table">
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>Accuracy</th>
                <th>Samples</th>
              </tr>
            </thead>
            <tbody>
              ${sortedPerClient
                .map(
                  (item) => `
                    <tr>
                      <td>${item.clientId}</td>
                      <td>${item.accuracy.toFixed(4)}</td>
                      <td>${item.samples}</td>
                    </tr>
                  `,
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

function renderAll(): void {
  renderContextCard();
  renderDatasetSection();
  renderTrainingSection();
  renderTestingSection();
}

async function loadDatasets(): Promise<void> {
  if (!state.context) {
    return;
  }
  state.datasets = await window.desktopApi.scanDatasets(state.context.datasetRoot);
  state.selectedDataset = state.datasets[0] || null;
  state.selectedClients = state.selectedDataset ? [...state.selectedDataset.trainClients] : [];
  resetDefaultTestClients();
}

async function loadSavedModels(): Promise<void> {
  if (!state.context) {
    return;
  }
  state.savedModels = await window.desktopApi.listSavedModels(state.context.projectRoot);
  state.selectedModelId = state.savedModels[0]?.id || null;
  resetDefaultTestClients();
  await loadSelectedModelResultDetail();
}

async function loadSummary(): Promise<void> {
  if (!state.context) {
    return;
  }
  state.summaryRows = await window.desktopApi.listSummaryRows(state.context.resultsRoot);
}

async function loadSelectedModelResultDetail(): Promise<void> {
  const selectedModel = getSelectedSavedModel();
  if (!state.context || !selectedModel) {
    state.selectedModelResultDetail = null;
    return;
  }

  state.selectedModelResultDetail = await window.desktopApi.getSavedModelResultDetail(selectedModel.id, state.context);
}

function subscribeTrainingEvents(): void {
  window.desktopApi.onTrainingEvent(async (event: TrainingEvent) => {
    if (event.type === "status" && event.status) {
      state.trainingStatus = event.status;
    }

    state.logs = [...state.logs, `[${event.timestamp}] ${event.message}`].slice(-300);
    refreshTrainingRuntimeView();

    if (event.type === "status" && ["success", "failed", "stopped"].includes(event.status || "")) {
      if (state.context) {
        await loadSummary();
        await loadSavedModels();
      }
      refreshTrainingRuntimeView();
      renderTestingSection();
    }
  });
}

async function bootstrap(): Promise<void> {
  renderShell();
  state.context = await window.desktopApi.getContext();
  await loadDatasets();
  await loadSavedModels();
  await loadSummary();
  subscribeTrainingEvents();
  renderAll();
}

void bootstrap();

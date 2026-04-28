import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { spawn, type ChildProcessByStdio } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { Readable } from "node:stream";
import type {
  DatasetInfo,
  ProjectContext,
  ResultDetail,
  ResultSummaryRow,
  SavedModelInfo,
  TestEvaluationRequest,
  TestEvaluationResult,
  TrainingConfig,
  TrainingEvent,
  TrainingStatus,
} from "../src/shared/types";

const desktopRoot = path.resolve(__dirname, "..", "..");
const thesisRoot = path.resolve(desktopRoot, "..");
const savedModelRoot = path.join(thesisRoot, "artifacts", "models");
const defaultPythonExecutable =
  process.platform === "win32"
    ? path.join(thesisRoot, ".venv", "python.exe")
    : path.join(thesisRoot, ".venv", "bin", "python");

let mainWindow: BrowserWindow | null = null;
let trainingProcess: ChildProcessByStdio<null, Readable, Readable> | null = null;
let trainingStopRequested = false;
let activeTrainingRunRoot: string | null = null;
let activeTrainingManifestPath: string | null = null;

const DEFAULT_CONTEXT: ProjectContext = {
  projectRoot: thesisRoot,
  datasetRoot: path.join(thesisRoot, "dataset"),
  resultsRoot: path.join(thesisRoot, "results"),
  srcRoot: path.join(thesisRoot, "src"),
  pythonExecutable: defaultPythonExecutable,
};

function listSubdirs(dirPath: string): string[] {
  if (!fs.existsSync(dirPath)) {
    return [];
  }
  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right, "zh-CN", { numeric: true, sensitivity: "base" }));
}

function readDatasetCatalog(datasetRoot: string): DatasetInfo[] {
  if (!fs.existsSync(datasetRoot)) {
    return [];
  }

  return listSubdirs(datasetRoot).map((datasetName) => {
    const datasetPath = path.join(datasetRoot, datasetName);
    const trainDir = path.join(datasetPath, "train");
    const testDir = path.join(datasetPath, "test");
    return {
      name: datasetName,
      path: datasetPath,
      trainDir,
      testDir,
      trainClients: listSubdirs(trainDir),
      testClients: listSubdirs(testDir),
    };
  });
}

function collectResultFiles(resultsRoot: string): string[] {
  const resultFiles: string[] = [];

  function walk(currentPath: string): void {
    for (const entry of fs.readdirSync(currentPath, { withFileTypes: true })) {
      const fullPath = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "summary") {
          continue;
        }
        walk(fullPath);
        continue;
      }

      if (entry.isFile() && entry.name.endsWith(".h5")) {
        resultFiles.push(path.relative(resultsRoot, fullPath).replace(/\\/g, "/"));
      }
    }
  }

  if (fs.existsSync(resultsRoot)) {
    walk(resultsRoot);
  }

  return resultFiles.sort((left, right) => left.localeCompare(right, "zh-CN"));
}

function parseCsv(content: string): ResultSummaryRow[] {
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    return [];
  }

  const parseLine = (line: string): string[] => {
    const cells: string[] = [];
    let current = "";
    let insideQuotes = false;

    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      const next = line[index + 1];
      if (char === "\"" && insideQuotes && next === "\"") {
        current += "\"";
        index += 1;
        continue;
      }
      if (char === "\"") {
        insideQuotes = !insideQuotes;
        continue;
      }
      if (char === "," && !insideQuotes) {
        cells.push(current);
        current = "";
        continue;
      }
      current += char;
    }

    cells.push(current);
    return cells;
  };

  const headers = parseLine(lines[0].replace(/^\uFEFF/, ""));
  return lines.slice(1).map((line) => {
    const cells = parseLine(line);
    const row: ResultSummaryRow = {} as ResultSummaryRow;
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? "";
    });
    return row;
  });
}

function readSummaryRows(resultsRoot: string): ResultSummaryRow[] {
  const summaryPath = path.join(resultsRoot, "summary", "experiment_summary.csv");
  if (!fs.existsSync(summaryPath)) {
    return [];
  }
  return parseCsv(fs.readFileSync(summaryPath, "utf8"));
}

function ensureSavedModelRoot(): void {
  fs.mkdirSync(savedModelRoot, { recursive: true });
}

function buildRunId(config: TrainingConfig): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${stamp}_${config.dataset}_${config.algorithm}_${config.modelFamily}`;
}

function resultCategoryForModel(modelFamily: string): string {
  if (modelFamily === "IoT_MLP") {
    return "MLP";
  }
  if (modelFamily === "IoT_CNN1D") {
    return "CNN1D";
  }
  if (modelFamily === "IoT_Transformer1D") {
    return "Transformer";
  }
  return "heterogeneous_models";
}

function expectedResultRelativePath(config: TrainingConfig): string {
  return [
    resultCategoryForModel(config.modelFamily),
    config.algorithm,
    "metrics",
    `${config.dataset}_${config.algorithm}_${config.modelFamily}_${config.goal}_${config.times}.h5`,
  ].join("/");
}

function writeSavedModelManifest(runRoot: string, manifest: Omit<SavedModelInfo, "id"> & { id?: string }): void {
  ensureSavedModelRoot();
  const manifestPath = path.join(runRoot, "manifest.json");
  fs.mkdirSync(runRoot, { recursive: true });
  fs.writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        ...manifest,
        id: manifest.id ?? path.basename(runRoot),
      },
      null,
      2,
    ),
    "utf8",
  );
}

function readSavedModelManifest(manifestPath: string): SavedModelInfo | null {
  try {
    const raw = fs.readFileSync(manifestPath, "utf8");
    return JSON.parse(raw) as SavedModelInfo;
  } catch {
    return null;
  }
}

function listSavedModels(): SavedModelInfo[] {
  ensureSavedModelRoot();
  return fs
    .readdirSync(savedModelRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const manifestPath = path.join(savedModelRoot, entry.name, "manifest.json");
      const manifest = readSavedModelManifest(manifestPath);
      if (!manifest) {
        return null;
      }
      const normalized = normalizeSavedModelManifest(manifest, manifestPath, DEFAULT_CONTEXT.resultsRoot);
      return normalized;
    })
    .filter((entry): entry is SavedModelInfo => Boolean(entry))
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt, "zh-CN"));
}

function buildResultFileCandidates(manifest: SavedModelInfo): string[] {
  const config = manifest.config;
  const dataset = config?.dataset || manifest.dataset;
  const algorithm = config?.algorithm || manifest.algorithm;
  const modelFamily = config?.modelFamily || manifest.modelFamily;
  const goal = config?.goal || manifest.goal;
  const category = resultCategoryForModel(modelFamily);
  const metricsDir = path.join(DEFAULT_CONTEXT.resultsRoot, category, algorithm, "metrics");
  if (!fs.existsSync(metricsDir)) {
    return [];
  }

  const prefix = `${dataset}_${algorithm}_${modelFamily}_${goal}_`;
  return fs
    .readdirSync(metricsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.startsWith(prefix) && entry.name.endsWith(".h5"))
    .map((entry) => path.posix.join(category, algorithm, "metrics", entry.name))
    .sort((left, right) => right.localeCompare(left, "zh-CN", { numeric: true, sensitivity: "base" }));
}

function normalizeSavedModelManifest(
  manifest: SavedModelInfo,
  manifestPath: string,
  resultsRoot: string,
): SavedModelInfo {
  const saveRootResultPath = manifest.resultFile ? path.join(manifest.saveRoot, manifest.resultFile) : "";
  const globalResultPath = manifest.resultFile ? path.join(resultsRoot, manifest.resultFile) : "";

  if (
    manifest.resultFile &&
    (fs.existsSync(saveRootResultPath) || fs.existsSync(globalResultPath))
  ) {
    return manifest;
  }

  const candidates = buildResultFileCandidates(manifest);
  if (!candidates.length) {
    return manifest;
  }

  const nextManifest: SavedModelInfo = {
    ...manifest,
    resultFile: candidates[0],
  };
  fs.writeFileSync(manifestPath, JSON.stringify(nextManifest, null, 2), "utf8");
  return nextManifest;
}

function deleteSavedModelById(modelId: string): boolean {
  const targetPath = path.join(savedModelRoot, modelId);
  if (!fs.existsSync(targetPath)) {
    return false;
  }
  if (activeTrainingRunRoot && path.resolve(activeTrainingRunRoot) === path.resolve(targetPath)) {
    throw new Error("当前训练任务仍在运行，不能删除对应模型记录。");
  }
  fs.rmSync(targetPath, { recursive: true, force: true });
  return true;
}

async function getSavedModelResultDetail(modelId: string, context: ProjectContext): Promise<ResultDetail | null> {
  const manifest = listSavedModels().find((item) => item.id === modelId);
  if (!manifest?.resultFile) {
    return null;
  }

  const attempts: Array<{ root: string; relativePath: string }> = [
    { root: manifest.saveRoot, relativePath: manifest.resultFile },
    { root: context.resultsRoot, relativePath: manifest.resultFile },
  ];

  const basename = path.basename(manifest.resultFile);
  if (basename !== manifest.resultFile) {
    attempts.push({ root: manifest.saveRoot, relativePath: path.posix.join("metrics", basename) });
    attempts.push({ root: context.resultsRoot, relativePath: path.posix.join("metrics", basename) });
  }

  for (const attempt of attempts) {
    try {
      return await runJsonPython<ResultDetail>(
        "export_result_payload.py",
        ["--results_root", attempt.root, "--relative_path", attempt.relativePath],
        context,
      );
    } catch {
      continue;
    }
  }

  return null;
}

function copyFileIfExists(sourcePath: string, targetPath: string): void {
  if (!fs.existsSync(sourcePath)) {
    return;
  }
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.copyFileSync(sourcePath, targetPath);
}

function syncRunArtifactsFromResults(context: ProjectContext, manifest: SavedModelInfo): SavedModelInfo {
  if (!manifest.resultFile) {
    return manifest;
  }

  const sourceMetricsPath = path.join(context.resultsRoot, manifest.resultFile);
  if (!fs.existsSync(sourceMetricsPath)) {
    return manifest;
  }

  const metricFileName = path.basename(sourceMetricsPath);
  const metricStem = path.parse(metricFileName).name;
  const sourceMetricsDir = path.dirname(sourceMetricsPath);
  const sourceAlgorithmDir = path.dirname(sourceMetricsDir);
  const sourceFiguresDir = path.join(sourceAlgorithmDir, "figures");
  const sourceLogsDir = path.join(sourceAlgorithmDir, "logs");

  const targetMetricsDir = path.join(manifest.saveRoot, "metrics");
  const targetFiguresDir = path.join(manifest.saveRoot, "figures");
  const targetLogsDir = path.join(manifest.saveRoot, "logs");

  copyFileIfExists(sourceMetricsPath, path.join(targetMetricsDir, metricFileName));

  if (fs.existsSync(sourceFiguresDir)) {
    for (const entry of fs.readdirSync(sourceFiguresDir, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.startsWith(metricStem) || !entry.name.endsWith(".png")) {
        continue;
      }
      copyFileIfExists(
        path.join(sourceFiguresDir, entry.name),
        path.join(targetFiguresDir, entry.name),
      );
    }
  }

  if (fs.existsSync(sourceLogsDir)) {
    const logCandidates = fs
      .readdirSync(sourceLogsDir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith(".out"))
      .map((entry) => ({
        name: entry.name,
        fullPath: path.join(sourceLogsDir, entry.name),
        mtimeMs: fs.statSync(path.join(sourceLogsDir, entry.name)).mtimeMs,
      }))
      .sort((left, right) => right.mtimeMs - left.mtimeMs);
    if (logCandidates[0]) {
      copyFileIfExists(logCandidates[0].fullPath, path.join(targetLogsDir, logCandidates[0].name));
    }
  }

  return {
    ...manifest,
    resultFile: `metrics/${metricFileName}`,
  };
}

function buildTrainingArgs(config: TrainingConfig): string[] {
  const args = [
    "main.py",
    "-dataset",
    config.dataset,
    "-algo",
    config.algorithm,
    "-model_family",
    config.modelFamily,
    "-nc",
    String(config.numClients),
    "-gr",
    String(config.globalRounds),
    "-ls",
    String(config.localEpochs),
    "-lr",
    String(config.localLearningRate),
    "-jr",
    String(config.joinRatio),
    "-fd",
    String(config.featureDim),
    "-dev",
    config.device,
    "-did",
    config.deviceId,
    "-t",
    String(config.times),
    "-go",
    config.goal,
    "--input_dim",
    String(config.inputDim),
    "-nb",
    String(config.numClasses),
    "--normal_class",
    String(config.normalClass),
    "-lbs",
    String(config.batchSize),
    "-nw",
    String(config.numWorkers),
    "--early_stop_patience",
    String(config.earlyStopPatience),
    "-eg",
    String(config.evalGap),
    "-lam",
    String(config.lamda),
    "-sfn",
    config.saveFolderRoot || "temp",
  ];

  if (config.skipFigures) {
    args.push("--skip_figures");
  }

  if (config.modelFamily === "IoT_Transformer1D" || config.modelFamily === "IoT_MIX_MLP_CNN_TRANS") {
    args.push(
      "--transformer_d_model",
      String(config.transformerDModel),
      "--transformer_num_heads",
      String(config.transformerNumHeads),
      "--transformer_num_layers",
      String(config.transformerNumLayers),
      "--transformer_dropout",
      String(config.transformerDropout),
    );
  }

  return args;
}

function emitTrainingEvent(event: TrainingEvent): void {
  mainWindow?.webContents.send("training:event", event);
}

function splitOutputLines(chunk: Buffer): string[] {
  return chunk
    .toString()
    .split(/\r?\n/)
    .filter((line: string) => line.trim().length > 0);
}

function timestamp(): string {
  return new Date().toISOString();
}

function runJsonPython<T>(
  scriptName: string,
  args: string[],
  context: ProjectContext,
  extraEnv?: Record<string, string>,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const child = spawn(context.pythonExecutable, [scriptName, ...args], {
      cwd: context.srcRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        ...extraEnv,
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Python exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout) as T);
      } catch (error) {
        reject(new Error(`Failed to parse JSON output: ${String(error)}`));
      }
    });
  });
}

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1240,
    minHeight: 780,
    backgroundColor: "#efe7d8",
    title: "IoT FedProto Studio",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  await mainWindow.loadFile(path.join(__dirname, "..", "src", "renderer", "index.html"));
}

app.whenReady().then(() => {
  ipcMain.handle("app:get-context", () => DEFAULT_CONTEXT);

  ipcMain.handle("dialog:choose-directory", async (_event, defaultPath?: string) => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      defaultPath: defaultPath || DEFAULT_CONTEXT.projectRoot,
      properties: ["openDirectory"],
    });

    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle("dataset:scan", (_event, datasetRoot: string) => readDatasetCatalog(datasetRoot));
  ipcMain.handle("result:list-files", (_event, resultsRoot: string) => collectResultFiles(resultsRoot));
  ipcMain.handle("result:list-summary", (_event, resultsRoot: string) => readSummaryRows(resultsRoot));
  ipcMain.handle("result:detail", (_event, resultsRoot: string, relativePath: string, context: ProjectContext) => {
    return runJsonPython<ResultDetail>(
      "export_result_payload.py",
      ["--results_root", resultsRoot, "--relative_path", relativePath],
      context,
    );
  });

  ipcMain.handle("result:refresh-summary", async (_event, context: ProjectContext) => {
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(context.pythonExecutable, ["summarize_results.py"], {
        cwd: context.srcRoot,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
        },
      });

      let output = "";
      child.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.stderr.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(output.trim() || `summarize_results.py exited with code ${code}`));
          return;
        }
        resolve(output.trim());
      });
    });

    return {
      output,
      rows: readSummaryRows(context.resultsRoot),
      resultFiles: collectResultFiles(context.resultsRoot),
    };
  });

  ipcMain.handle("model:list-saved", () => listSavedModels());
  ipcMain.handle("model:delete-saved", (_event, modelId: string) => deleteSavedModelById(modelId));
  ipcMain.handle("model:result-detail", (_event, modelId: string, context: ProjectContext) =>
    getSavedModelResultDetail(modelId, context),
  );
  ipcMain.handle("model:evaluate", async (_event, context: ProjectContext, request: TestEvaluationRequest) => {
    const manifest = listSavedModels().find((item) => item.id === request.modelId);
    if (!manifest) {
      throw new Error("Saved model not found.");
    }
    if (request.testClients.length !== manifest.trainingClients.length) {
      throw new Error("测试客户端数量需要与训练客户端数量一致。");
    }

    const payload = await runJsonPython<TestEvaluationResult>(
      "evaluate_saved_model.py",
      [
        "--dataset",
        request.dataset,
        "--algorithm",
        manifest.algorithm,
        "--model_family",
        manifest.modelFamily,
        "--save_root",
        manifest.saveRoot,
        "--num_clients",
        String(manifest.trainingClients.length),
        "--goal",
        manifest.goal,
        "--input_dim",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.inputDim ?? 77),
        "--num_classes",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.numClasses ?? 15),
        "--normal_class",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.normalClass ?? 0),
        "--batch_size",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.batchSize ?? 10),
        "--num_workers",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.numWorkers ?? 0),
        "--feature_dim",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.featureDim ?? 64),
        "--lamda",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.lamda ?? 1),
        "--transformer_d_model",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.transformerDModel ?? 64),
        "--transformer_num_heads",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.transformerNumHeads ?? 4),
        "--transformer_num_layers",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.transformerNumLayers ?? 2),
        "--transformer_dropout",
        String((manifest as SavedModelInfo & { config?: TrainingConfig }).config?.transformerDropout ?? 0.2),
      ],
      {
        ...context,
        datasetRoot: context.datasetRoot,
      },
      {
        IOT_FEDPROTO_DATASET_BASE: context.datasetRoot,
        IOT_FEDPROTO_CLIENT_MAP_JSON: JSON.stringify(request.testClients),
      },
    );
    return payload;
  });

  ipcMain.handle("training:start", (_event, context: ProjectContext, config: TrainingConfig) => {
    if (trainingProcess) {
      throw new Error("A training job is already running.");
    }

    ensureSavedModelRoot();
    const runId = buildRunId(config);
    const runRoot = path.join(savedModelRoot, runId);
    const trainingConfig: TrainingConfig = {
      ...config,
      saveFolderRoot: runRoot,
    };
    const saveFolder = path.join(runRoot, trainingConfig.dataset, trainingConfig.algorithm);
    const manifest = {
      id: runId,
      label: `${trainingConfig.algorithm} / ${trainingConfig.modelFamily}`,
      dataset: trainingConfig.dataset,
      sourceDataset: trainingConfig.dataset,
      algorithm: trainingConfig.algorithm,
      modelFamily: trainingConfig.modelFamily,
      trainingClients: trainingConfig.selectedClients || [],
      saveRoot: runRoot,
      saveFolder,
      createdAt: new Date().toISOString(),
      status: "running" as const,
      goal: trainingConfig.goal,
      resultFile: expectedResultRelativePath(trainingConfig),
      config: trainingConfig,
    };
    writeSavedModelManifest(runRoot, manifest);

    const args = buildTrainingArgs(trainingConfig);
    trainingStopRequested = false;
    const child = spawn(context.pythonExecutable, args, {
      cwd: context.srcRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        IOT_FEDPROTO_DATASET_BASE: context.datasetRoot,
        IOT_FEDPROTO_CLIENT_MAP_JSON: JSON.stringify(trainingConfig.selectedClients || []),
      },
    });
    trainingProcess = child;
    activeTrainingRunRoot = runRoot;
    activeTrainingManifestPath = path.join(runRoot, "manifest.json");

    emitTrainingEvent({
      type: "status",
      status: "running",
      message: `Training started: ${context.pythonExecutable} ${args.join(" ")}`,
      timestamp: timestamp(),
    });

    child.stdout.on("data", (chunk: Buffer) => {
      const lines = splitOutputLines(chunk);
      lines.forEach((line: string) => {
        emitTrainingEvent({
          type: "log",
          message: line,
          timestamp: timestamp(),
        });
      });
    });

    child.stderr.on("data", (chunk: Buffer) => {
      const lines = splitOutputLines(chunk);
      lines.forEach((line: string) => {
        emitTrainingEvent({
          type: "log",
          message: line,
          timestamp: timestamp(),
        });
      });
    });

    child.on("close", (code) => {
      const status: TrainingStatus = trainingStopRequested ? "stopped" : code === 0 ? "success" : "failed";
      if (activeTrainingManifestPath && fs.existsSync(activeTrainingManifestPath)) {
        const currentManifest = readSavedModelManifest(activeTrainingManifestPath);
        if (currentManifest) {
          const nextManifest =
            code === 0 ? syncRunArtifactsFromResults(context, currentManifest) : currentManifest;
          writeSavedModelManifest(activeTrainingRunRoot!, {
            ...nextManifest,
            status: trainingStopRequested ? "stopped" : code === 0 ? "ready" : "failed",
          });
        }
      }
      emitTrainingEvent({
        type: "status",
        status,
        message: trainingStopRequested
          ? "Training stopped by user."
          : code === 0
            ? "Training completed successfully."
            : `Training exited with code ${code}.`,
        timestamp: timestamp(),
      });
      trainingStopRequested = false;
      trainingProcess = null;
      activeTrainingRunRoot = null;
      activeTrainingManifestPath = null;
    });

    return {
      command: `${context.pythonExecutable} ${args.join(" ")}`,
    };
  });

  ipcMain.handle("training:stop", () => {
    if (!trainingProcess) {
      return false;
    }

    trainingStopRequested = true;
    trainingProcess.kill();
    emitTrainingEvent({
      type: "status",
      status: "stopped",
      message: "Training stop requested.",
      timestamp: timestamp(),
    });
    trainingProcess = null;
    return true;
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

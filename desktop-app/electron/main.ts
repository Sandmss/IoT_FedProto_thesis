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
  TrainingConfig,
  TrainingEvent,
  TrainingStatus,
} from "../src/shared/types";

const desktopRoot = path.resolve(__dirname, "..", "..");
const thesisRoot = path.resolve(desktopRoot, "..");
const defaultPythonExecutable =
  process.platform === "win32"
    ? path.join(thesisRoot, ".venv", "python.exe")
    : path.join(thesisRoot, ".venv", "bin", "python");

let mainWindow: BrowserWindow | null = null;
let trainingProcess: ChildProcessByStdio<null, Readable, Readable> | null = null;
let trainingStopRequested = false;

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
    .sort((left, right) => left.localeCompare(right, "zh-CN"));
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

function runJsonPython<T>(scriptName: string, args: string[], context: ProjectContext): Promise<T> {
  return new Promise((resolve, reject) => {
    const child = spawn(context.pythonExecutable, [scriptName, ...args], {
      cwd: context.srcRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
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

  ipcMain.handle("training:start", (_event, context: ProjectContext, config: TrainingConfig) => {
    if (trainingProcess) {
      throw new Error("A training job is already running.");
    }

    const args = buildTrainingArgs(config);
    trainingStopRequested = false;
    const child = spawn(context.pythonExecutable, args, {
      cwd: context.srcRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
      },
    });
    trainingProcess = child;

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

import { contextBridge, ipcRenderer } from "electron";
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
} from "../src/shared/types";

const api = {
  getContext: (): Promise<ProjectContext> => ipcRenderer.invoke("app:get-context"),
  chooseDirectory: (defaultPath?: string): Promise<string | null> => ipcRenderer.invoke("dialog:choose-directory", defaultPath),
  scanDatasets: (datasetRoot: string): Promise<DatasetInfo[]> => ipcRenderer.invoke("dataset:scan", datasetRoot),
  listResultFiles: (resultsRoot: string): Promise<string[]> => ipcRenderer.invoke("result:list-files", resultsRoot),
  listSummaryRows: (resultsRoot: string): Promise<ResultSummaryRow[]> => ipcRenderer.invoke("result:list-summary", resultsRoot),
  getResultDetail: (resultsRoot: string, relativePath: string, context: ProjectContext): Promise<ResultDetail> =>
    ipcRenderer.invoke("result:detail", resultsRoot, relativePath, context),
  refreshSummary: (context: ProjectContext): Promise<{ output: string; rows: ResultSummaryRow[]; resultFiles: string[] }> =>
    ipcRenderer.invoke("result:refresh-summary", context),
  startTraining: (context: ProjectContext, config: TrainingConfig): Promise<{ command: string }> =>
    ipcRenderer.invoke("training:start", context, config),
  stopTraining: (): Promise<boolean> => ipcRenderer.invoke("training:stop"),
  listSavedModels: (projectRoot: string): Promise<SavedModelInfo[]> => ipcRenderer.invoke("model:list-saved", projectRoot),
  deleteSavedModel: (modelId: string): Promise<boolean> => ipcRenderer.invoke("model:delete-saved", modelId),
  getSavedModelResultDetail: (modelId: string, context: ProjectContext): Promise<ResultDetail | null> =>
    ipcRenderer.invoke("model:result-detail", modelId, context),
  evaluateSavedModel: (context: ProjectContext, request: TestEvaluationRequest): Promise<TestEvaluationResult> =>
    ipcRenderer.invoke("model:evaluate", context, request),
  onTrainingEvent: (handler: (event: TrainingEvent) => void): (() => void) => {
    const listener = (_event: unknown, payload: TrainingEvent) => handler(payload);
    ipcRenderer.on("training:event", listener);
    return () => ipcRenderer.removeListener("training:event", listener);
  },
};

contextBridge.exposeInMainWorld("desktopApi", api);

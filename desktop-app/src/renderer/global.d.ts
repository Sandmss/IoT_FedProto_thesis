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
} from "../shared/types";

declare global {
  interface Window {
    desktopApi: {
      getContext(): Promise<ProjectContext>;
      chooseDirectory(defaultPath?: string): Promise<string | null>;
      scanDatasets(datasetRoot: string): Promise<DatasetInfo[]>;
      listResultFiles(resultsRoot: string): Promise<string[]>;
      listSummaryRows(resultsRoot: string): Promise<ResultSummaryRow[]>;
      getResultDetail(resultsRoot: string, relativePath: string, context: ProjectContext): Promise<ResultDetail>;
      refreshSummary(context: ProjectContext): Promise<{ output: string; rows: ResultSummaryRow[]; resultFiles: string[] }>;
      startTraining(context: ProjectContext, config: TrainingConfig): Promise<{ command: string }>;
      stopTraining(): Promise<boolean>;
      listSavedModels(projectRoot: string): Promise<SavedModelInfo[]>;
      deleteSavedModel(modelId: string): Promise<boolean>;
      getSavedModelResultDetail(modelId: string, context: ProjectContext): Promise<ResultDetail | null>;
      evaluateSavedModel(context: ProjectContext, request: TestEvaluationRequest): Promise<TestEvaluationResult>;
      onTrainingEvent(handler: (event: TrainingEvent) => void): () => void;
    };
  }
}

export {};

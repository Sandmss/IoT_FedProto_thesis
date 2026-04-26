import type { DatasetInfo, ProjectContext, ResultDetail, ResultSummaryRow, TrainingConfig, TrainingEvent } from "../shared/types";

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
      onTrainingEvent(handler: (event: TrainingEvent) => void): () => void;
    };
  }
}

export {};

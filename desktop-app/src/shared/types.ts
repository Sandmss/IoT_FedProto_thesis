export type TrainingStatus = "idle" | "running" | "success" | "failed" | "stopped";

export interface ProjectContext {
  projectRoot: string;
  datasetRoot: string;
  resultsRoot: string;
  srcRoot: string;
  pythonExecutable: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  trainDir: string;
  testDir: string;
  trainClients: string[];
  testClients: string[];
}

export interface TrainingConfig {
  dataset: string;
  algorithm: string;
  modelFamily: string;
  selectedClients?: string[];
  saveFolderRoot?: string;
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

export interface ResultSummaryRow {
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

export interface ResultSeriesMap {
  [label: string]: number[];
}

export interface ResultDetail {
  path: string;
  relativePath: string;
  series: ResultSeriesMap;
  bestRound: number | null;
  confusionMatrix: number[][] | null;
  figures: string[];
  logPreview: string[];
}

export interface TrainingEvent {
  type: "log" | "status";
  status?: TrainingStatus;
  message: string;
  timestamp: string;
}

export interface SavedModelInfo {
  id: string;
  label: string;
  dataset: string;
  sourceDataset: string;
  algorithm: string;
  modelFamily: string;
  trainingClients: string[];
  saveRoot: string;
  saveFolder: string;
  createdAt: string;
  status: "running" | "ready" | "failed" | "stopped";
  goal: string;
  resultFile?: string;
  config?: TrainingConfig;
}

export interface TestEvaluationRequest {
  modelId: string;
  dataset: string;
  testClients: string[];
}

export interface TestEvaluationResult {
  modelLabel: string;
  dataset: string;
  testClients: string[];
  accuracy: number;
  aucMacro: number;
  aucMicro: number;
  precision: number;
  recall: number;
  f1: number;
  fnr: number;
  fpr: number;
  inferenceLatencyMs: number;
  confusionMatrix: number[][];
  perClient: Array<{
    clientId: string;
    accuracy: number;
    samples: number;
  }>;
}

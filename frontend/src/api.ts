import type { PipelineConfig, FolderInfo, FileInfo, TrainStatus, LRStage, EvalResponse } from "./types";

const BASE = "";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  return res.json();
}

export async function getConfig() {
  return fetchJSON<PipelineConfig & { device: string; model_loaded: boolean }>("/api/config");
}

export async function setConfig(config: PipelineConfig) {
  return fetchJSON<{ status?: string; error?: string }>("/api/config", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function getFolders() {
  return fetchJSON<{ folders: FolderInfo[]; data_dir: string; error?: string }>("/api/data/folders");
}

export async function getFiles(folder: string) {
  return fetchJSON<{ files: FileInfo[]; error?: string }>(`/api/data/files/${encodeURIComponent(folder)}`);
}

export interface TestFileInfo {
  name: string;
  folder: string;
  path: string;
  broke: boolean;
  n_samples: number;
  params: Record<string, number>;
}

export async function getTestFiles() {
  return fetchJSON<{ files: TestFileInfo[] }>("/api/train/test_files");
}

export async function startTraining(body: {
  folders: string[];
  test_split: number;
  lr_schedule: LRStage[];
  batch_size: number;
}) {
  return fetchJSON<any>("/api/train/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getTrainStatus() {
  return fetchJSON<TrainStatus>("/api/train/status");
}

export async function stopTraining() {
  return fetchJSON<any>("/api/train/stop", { method: "POST" });
}

export async function resetModel() {
  return fetchJSON<any>("/api/train/reset", { method: "POST" });
}

export async function evaluate(body: {
  source: string;
  folders?: string[];
  window_position: number;
}) {
  return fetchJSON<EvalResponse & { error?: string }>("/api/evaluate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

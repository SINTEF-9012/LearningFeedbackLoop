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
  return fetchJSON<PipelineConfig & { device: string; model_loaded: boolean; channel_names: string[] }>(
    "/api/config"
  );
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
  val_split: number;
  lr_schedule: LRStage[];
  batch_size: number;
  patience: number;
  n_windows: number;
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

export interface SavedModel {
  name: string;
  size_bytes: number;
  mtime: number;
}

export async function listSavedModels() {
  return fetchJSON<{ models: SavedModel[] }>("/api/model/list");
}

export async function saveModel(name: string) {
  return fetchJSON<{ status?: string; name?: string; path?: string; error?: string }>(
    "/api/model/save",
    { method: "POST", body: JSON.stringify({ name }) }
  );
}

export async function loadModel(name: string) {
  return fetchJSON<{ status?: string; name?: string; error?: string }>(
    "/api/model/load",
    { method: "POST", body: JSON.stringify({ name }) }
  );
}

export interface ModelWeights {
  pair_encoder_W0: number[][];          // (D, 2)  baseline first-layer weights
  pair_encoder_M: number[][][];         // (D, n_params, 2)  param modulation
  pair_encoder_b1: number[];            // (D,)
  pair_input_labels: string[];          // ["f_rel", "amp"]
  pair_embed_dim: number;
  param_mean: number[];
  param_std: number[];
  param_keys: string[];
  channel_names: string[];
  k_peaks: number;
  n_channels: number;
  n_params: number;
  error?: string;
}

export async function getModelWeights() {
  return fetchJSON<ModelWeights>("/api/model/weights");
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

// -- OF replay -------------------------------------------------------------

export interface MachineInfo {
  id: string;
  name: string;
  n_ofs: number;
  available: boolean;
}

export interface OFWindow {
  start: string;
  end: string;
  duration_sec: number;
  tool_number?: number;
  diameter_mm?: number | null;
  n_inserts?: number | null;
  n_rows?: number;
}

export async function getMachines() {
  return fetchJSON<{ machines: MachineInfo[] }>("/api/of/machines");
}

export async function getOFs(machineId: string) {
  return fetchJSON<{ ofs: string[]; error?: string }>(
    `/api/of/ofs/${encodeURIComponent(machineId)}`
  );
}

export async function getOFWindows(body: {
  machine_id: string;
  of: string;
}) {
  return fetchJSON<{
    windows: OFWindow[];
    files?: Record<string, string | null>;
    error?: string;
  }>("/api/of/windows", { method: "POST", body: JSON.stringify(body) });
}

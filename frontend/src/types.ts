export interface PipelineConfig {
  data_dir: string;
  fft_window: number;
  fft_step: number;
  sample_rate: number;
  k_peaks: number;
  f_max_rel: number | null;
  cnn_window: number;
  pair_embed_dim: number;
  conv_channels: number[];
  fc_hidden: number;
  kernel_size: number;
}

export const defaultConfig: PipelineConfig = {
  data_dir: "../lfl/testdata",
  fft_window: 4096,
  fft_step: 4096,
  sample_rate: 4096,
  k_peaks: 5,
  f_max_rel: 12,
  cnn_window: 16,
  pair_embed_dim: 16,
  conv_channels: [16, 16],
  fc_hidden: 32,
  kernel_size: 5,
};

export interface FolderInfo {
  name: string;
  n_files: number;
}

export interface FileInfo {
  name: string;
  broke: boolean;
  n_samples: number;
  params: Record<string, number>;
}

export interface TrainStatus {
  running: boolean;
  current_epoch: number;
  total_epochs: number;
  current_stage: number;
  total_stages: number;
  history: number[];
  val_history: number[];
  best_val_loss: number | null;
  best_epoch: number;
  epochs_since_improve: number;
  early_stopped: boolean;
}

export interface LRStage {
  lr: number;
  epochs: number;
}

export interface EvalResult {
  file: string;
  true_label: number;
  predicted: number;
  probability: number;
}

export interface EvalResponse {
  results: EvalResult[];
  confusion_matrix: number[][];
  classification_report: Record<string, any>;
  accuracy: number;
  n_samples: number;
  avg_inference_ms: number;
  error?: string;
}

// --- Streaming simulation (pair-input model) ---

export interface SimInit {
  type: "init";
  total_steps: number;
  n_channels: number;
  k_peaks: number;
  channel_names: string[];     // e.g. ["X", "Y"]
  spindle_freq: number;        // fg in Hz
  cnn_window: number;
  broke: boolean;
  params: Record<string, number>;
  file: string;
  f_max_rel: number | null;
}

// pairs[c][k] = [f_rel, amp] for channel c, peak slot k
export interface SimStep {
  type: "step";
  t: number;
  pairs: number[][][];
  prob: number | null;
  inference_ms: number | null;
}

/* ── Experiment Store — types + Zustand state ────────────────────────── */

import { create } from 'zustand'
import { api } from '../api/http'

/* ── API Response Types ──────────────────────────────────────────────── */

export interface RunSummary {
  run_id: string
  experiment_type?: string
  config: {
    train_ops: string[]
    test_op: string
    eval_op: string
    eval_variant?: string
    noise_rate?: number
    feedback_every_n?: number
    prediction_gap_s?: number
    features_csv?: string
    min_discrimination_ratio?: number
    negative_sampling_enabled?: boolean
    negative_sampling_rate?: number
    store_threshold?: number
    alert_threshold?: number
    critical_threshold?: number
  }
  eval_metrics: {
    f1?: number
    precision?: number
    recall?: number
    auc_roc?: number
    auc_pr?: number
    balanced_accuracy?: number
    n_samples?: number
    tp?: number
    fp?: number
    tn?: number
    fn?: number
  }
  test_metrics: {
    f1?: number
    precision?: number
    recall?: number
    auc_roc?: number
    n_samples?: number
    tp?: number
    fp?: number
    tn?: number
    fn?: number
  }
  feedback_stats: {
    n_events?: number
    n_confirms?: number
    n_dismissals?: number
    accuracy?: number
  }
  gap_s: number | null
  timestamp?: number | string
  error?: boolean | string
  error_message?: string
  live_status?: string
}

export interface SinditContext {
  spindle_speed?: number
  feed_rate?: number
  tool_id?: string
  feed_override?: number
  machine_state?: string
  power_level?: number
  [key: string]: any
}

export interface SinditContextSummary {
  n_normal?: number
  n_degraded?: number
  total?: number
  [key: string]: any
}

export interface ModelBreakdownSection {
  [key: string]: number | string | boolean | null | undefined
}

export interface ModelBreakdown {
  classical?: ModelBreakdownSection
  harmonic?: ModelBreakdownSection
  stoppage?: ModelBreakdownSection
  online?: ModelBreakdownSection
  available?: string[]
  feature_schema_version?: number
  feature_count?: number
}

export interface HarmonicContribution {
  label: string
  weight: number
  value: number
  contribution: number
}

export interface HarmonicExplainResponse {
  available: boolean
  reason?: string | null
  score?: number | null
  model_source?: string
  dataset?: string
  context_weights: number[]
  feature_labels: string[]
  harmonic_values: number[]
  contributions: HarmonicContribution[]
  top_weighted: HarmonicContribution[]
}

export interface SampleResult {
  score_trace?: Array<{ component: string; value: number; source: string }>
  sample_id: string
  label: string
  operation_id: string
  tool_number: string
  memory_id?: string | null
  significance_score: number
  action: string
  predicted_positive: boolean
  raw_model_score: number
  pattern_rule_score: number
  anomaly_z_score: number
  prior_boost: number
  multi_rule_bonus: number
  n_rules_triggered: number
  detected_patterns: string[]
  feedback_given: boolean
  feedback_action: string
  feedback_source: string
  counterfactual_score: number
  prediction_flipped: boolean
  prior_snapshot: Record<string, number>
  supervised_score: number
  unsupervised_score: number
  combined_score: number
  tool_prior: number
  tool_multiplier: number
  weight_supervised: number
  weight_unsupervised: number
  model_breakdown?: ModelBreakdown | null
  explanation: string | null
  explanation_source: string | null
  alert_line?: string | null
  alert_line_source?: string | null
  stored_in_memory: boolean
  co_occurring_pairs: string[]
  propagated_prior_deltas: Record<string, number>
  sindit_context: SinditContext | null
}

export interface FeedbackEventPatternUpdate {
  pattern_key: string
  polarity?: string | null
  weight?: number
  old_prior?: number
  new_prior?: number
  delta?: number
  confirm_count?: number
  dismiss_count?: number
}

export interface FeedbackEvent {
  source_sample_id: string
  source_operation_id?: string
  source_label?: string
  feedback_action: string
  feedback_source?: string
  was_significant?: boolean
  source_index?: number
  applied_at_index?: number
  applied_after_samples?: number
  memory_id?: string | null
  detected_patterns?: string[]
  pattern_updates?: FeedbackEventPatternUpdate[]
  propagated_prior_deltas?: Record<string, number>
  threshold_before?: number | null
  threshold_after?: number | null
  tool_prior_before?: number | null
  tool_prior_after?: number | null
  model_weights_before?: Record<string, number>
  model_weights_after?: Record<string, number>
  model_retrained?: boolean
}

export interface PatternFeedbackSummaryEntry {
  polarity?: string | null
  n_feedback_events?: number
  n_confirms?: number
  n_dismissals?: number
  total_prior_delta?: number
  mean_prior_delta?: number
  max_abs_prior_delta?: number
  last_prior?: number
}

export interface PhaseDetail {
  phase: string
  operation: string
  n_samples: number
  threshold: number
  adapted_threshold: number
  prior_history: Record<string, number[]>
  scores_positive: number[]
  scores_negative: number[]
  n_predictions_flipped: number
  weight_history: Record<string, number[]>
  tool_prior_history: Record<string, number[]>
  n_model_retrains: number
  co_occurrence_graph: Record<string, number>
  stored_memories_count: number
  all_propagated_deltas: any[]
  sindit_context_summary: SinditContextSummary
  feedback_events?: FeedbackEvent[]
  pattern_feedback_summary?: Record<string, PatternFeedbackSummaryEntry>
  n_propagation_events?: number
  n_discovered_patterns?: number
  n_suppression_patterns?: number
  discovered_pattern_keys?: string[]
  samples: SampleResult[]
}

export interface EvaluationDetail {
  run_id: string
  test: PhaseDetail
  eval: PhaseDetail
}

export interface FeatureData {
  total_rows: number
  offset: number
  limit: number
  columns: string[]
  feature_columns: string[]
  rows: Record<string, any>[]
}

/* ── Zustand Store ───────────────────────────────────────────────────── */

interface ExperimentState {
  runs: RunSummary[]
  selectedRunId: string
  evaluation: EvaluationDetail | null
  featureData: FeatureData | null
  runsLoading: boolean
  evalLoading: boolean
  featuresLoading: boolean
  runTriggering: boolean

  /* Computed */
  selectedRun: RunSummary | undefined

  /* Actions */
  fetchRuns: () => Promise<void>
  fetchEvaluation: (runId: string) => Promise<void>
  fetchFeatures: (runId: string, limit?: number) => Promise<void>
  triggerRun: (params: Record<string, any>) => Promise<{ success: boolean; stdout: string; stderr: string }>
  triggerExtraction: (params: Record<string, any>) => Promise<{ success: boolean; stdout: string; stderr: string }>
  setSelectedRunId: (id: string) => void
}

export const useExperimentStore = create<ExperimentState>((set, get) => ({
  runs: [],
  selectedRunId: '',
  evaluation: null,
  featureData: null,
  runsLoading: false,
  evalLoading: false,
  featuresLoading: false,
  runTriggering: false,

  get selectedRun() {
    const s = get()
    return s.runs.find(r => r.run_id === s.selectedRunId)
  },

  setSelectedRunId(id: string) {
    set({ selectedRunId: id, evaluation: null, featureData: null })
  },

  async fetchRuns() {
    set({ runsLoading: true })
    try {
      const res = await api<{ runs: RunSummary[] }>('/agent/memory/experiment/runs')
      set({ runs: res.runs || [] })
    } finally {
      set({ runsLoading: false })
    }
  },

  async fetchEvaluation(runId: string) {
    set({ evalLoading: true })
    try {
      const res = await api<EvaluationDetail>(`/agent/memory/experiment/runs/${encodeURIComponent(runId)}/evaluate`, 'POST')
      set({ evaluation: res })
    } finally {
      set({ evalLoading: false })
    }
  },

  async fetchFeatures(runId: string, limit = 500) {
    set({ featuresLoading: true })
    try {
      const res = await api<FeatureData>(`/agent/memory/experiment/features?run_id=${encodeURIComponent(runId)}&limit=${limit}`)
      set({ featureData: res })
    } finally {
      set({ featuresLoading: false })
    }
  },

  async triggerRun(params: Record<string, any>) {
    set({ runTriggering: true })
    try {
      return await api<{ success: boolean; stdout: string; stderr: string }>('/agent/memory/experiment/run', 'POST', params)
    } finally {
      set({ runTriggering: false })
    }
  },

  async triggerExtraction(params: Record<string, any>) {
    return api<{ success: boolean; stdout: string; stderr: string }>('/agent/memory/experiment/extract', 'POST', params)
  },
}))

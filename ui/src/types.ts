/**
 * Shared type definitions for the LFL UI.
 *
 * Centralises types previously spread across App.tsx, AppContext.tsx,
 * and component files.  Eliminates `any` usage in core data flows.
 */

/* ── Memory / Feedback / Traces ───────────────────────────── */

export interface MemorySummary {
  id: string
  session_id: string
  created_at: string
  patterns: string[]
  pattern_keys?: Array<string | { key: string; pattern_type?: string }>
  label?: string | null
  tags: string[]
  significance_score?: number | null
  annotation_preview?: string | null
  annotation_text?: string
  metadata?: MemoryMetadata
  created_by?: string
}

export interface DocLink {
  id?: string | null
  citation?: string | null
  score?: number | null
  page?: number | string | null
  file_name?: string | null
  source?: string | null
  usecase?: string | null
  machine?: string | null
  text?: string | null
  document_type?: string | null
  language?: string | null
  query_used: string
  pattern_key: string
  doc_feedback?: string | null
  helpful_count?: number
  not_helpful_count?: number
  feedback_score?: number
  evidence_entities?: Array<Record<string, unknown>>
}

export interface MemoryMetadata {
  significance_score?: number
  significance_action?: string
  cutting_context?: CuttingContext
  [key: string]: unknown
}

export interface CuttingContext {
  tool_type?: string
  spindle_speed?: number
  num_teeth?: number
  axial_depth?: number
  radial_depth?: number
  workpiece_material?: string
  operating_regime?: string
  machine_type?: string
  feed_rate?: number
  coolant?: string
  [key: string]: unknown
}

export interface FeedbackEvent {
  id?: string
  action: string
  user_id?: string
  created_at?: string
  timestamp?: string
  comment?: string
  reason?: string
  label?: string
  tags?: string[]
}

export interface FeedbackStats {
  net_significance?: number
  n_confirms?: number
  n_dismissals?: number
  [key: string]: unknown
}

export interface TracePayload {
  significance?: {
    score?: number
    action?: string
    reasons?: string[]
  }
  returned?: Array<{
    memory_id: string
    score?: number
    reasons?: string[]
  }>
  [key: string]: unknown
}

export interface Trace {
  id: string
  created_at: string
  trace_type: string
  payload?: TracePayload
}

/* ── Prior ─────────────────────────────────────────────────── */

export interface PriorRow {
  pattern: string
  prior: number
  effective_weight_total?: number
  passive_outcome_count?: number
  severity_correction_count?: number
  severity_calibration?: {
    average_delta?: number
    weight_total?: number
    targets?: Partial<Record<'info' | 'warning' | 'critical', number>>
  }
}

export interface PriorDiffRow {
  pattern: string
  before: number
  after: number
  delta: number
}

/* ── API responses ────────────────────────────────────────── */

export interface ListMemoriesResponse {
  memories: MemorySummary[]
  total_count: number
}

export interface PriorsResponse {
  priors: PriorRow[]
}

export interface MemoryDetailResponse {
  memory: MemorySummary
  feedback_stats: FeedbackStats
  doc_links: DocLink[]
}

export interface FeedbackHistoryResponse {
  events: FeedbackEvent[]
  stats: FeedbackStats
}

export interface TraceListResponse {
  traces: Trace[]
}

export interface SessionsResponse {
  sessions: string[]
}

export interface SessionInfo {
  running: boolean
  paused: boolean
  position: number
  last_error?: string
  config?: {
    speed?: number
    samples_per_tick?: number
    [key: string]: unknown
  }
  [key: string]: unknown
}

/* ── Alert context ────────────────────────────────────────── */

export interface AlertMetrics {
  rms?: number[]
  dominant_freq?: number[]
  total_energy?: number[]
  spectral_centroids?: number[]
  anomaly_detector_score?: number
  model_confidence?: number
  breakage_prediction?: number
  tool_wear_estimate?: number
  [key: string]: unknown
}

/* ── Severity helper ──────────────────────────────────────── */

export interface SeverityInfo {
  label: 'CRITICAL' | 'WARNING' | 'INFO'
  color: string
}

export function severity(score?: number): SeverityInfo {
  const s = typeof score === 'number' && Number.isFinite(score) ? score : 0
  if (s >= 0.9) return { label: 'CRITICAL', color: 'var(--danger)' }
  if (s >= 0.75) return { label: 'WARNING', color: 'var(--accent)' }
  return { label: 'INFO', color: 'var(--muted)' }
}

/* ── Review tracking ──────────────────────────────────────── */

export interface ReviewEntry {
  action: 'confirm' | 'dismiss'
  at: number
}

export interface ReviewHistoryEntry extends ReviewEntry {
  id: string
  reason?: string
}

/* ── Feedback form state ──────────────────────────────────── */

export interface FeedbackFormState {
  reason: string
  comment: string
  label: string
  tags: string
}

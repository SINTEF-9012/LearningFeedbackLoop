import { create } from 'zustand'

/**
 * Per-fault-type indicator with contributing signal scores.
 */
export type FaultSignals = Record<string, number>

export type FaultIndicator = {
  score: number
  signals: FaultSignals
}

export type FaultIndicators = {
  tool_breakage: FaultIndicator
  chatter: FaultIndicator
  chip_adhesion: FaultIndicator
  workpiece_slip: FaultIndicator
  dominant_fault: string | null
}

/**
 * A single inference score point emitted by the backend inference streamer.
 * One point is produced per sliding window.
 */
export type InferencePoint = {
  /** Plot/display time for the inference window, aligned to the playback clock when available. */
  t: number
  /** Optional display-time window start */
  t0?: number
  /** Optional display-time window end */
  t1?: number
  /** Optional window-centre time on the active clock */
  t_center?: number
  /** Centre sample index */
  i_center: number
  /** Window bounds [start, end) */
  window: [number, number]
  /** Sampling frequency */
  fs: number
  /** Window duration in seconds when emitted by the backend */
  window_seconds?: number
  /** Number of raw samples in the window */
  window_entries?: number
  /** Explicit backend sample rate field */
  sample_rate_hz?: number
  /** Per-model scores, all in [0, 1] where 1 = most anomalous */
  scores: {
    ensemble: number
    isolation_forest: number
    lof: number
    z_score: number
    /** Harmonic context-weighted CNN score (optional, present when model is loaded) */
    harmonic_context_score?: number
    /** Harmonic pair-amplitude CNN score (optional, present when pair model is loaded) */
    harmonic_pair_score?: number
  }
  /** Feature values used for this window (for detail view) */
  features?: Record<string, number>
  /** Per-fault-type likelihood indicators */
  fault_indicators?: FaultIndicators
  /** Learned context weights from HarmonicContextNet (w = params × Wᵀ) */
  harmonic_context_weights?: number[]
  /** Labels matching 1:1 with harmonic_context_weights, format "Group·Harmonic" */
  harmonic_feature_labels?: string[]
  /** Latest harmonic feature values aligned with harmonic_feature_labels */
  harmonic_values?: number[]
  /** Context-model labels (explicit; mirrors primary when context is primary) */
  harmonic_context_feature_labels?: string[]
  /** Context-model feature values */
  harmonic_context_values?: number[]
  /** Pair-model learned context weights */
  harmonic_pair_weights?: number[]
  /** Pair-model feature labels (per-peak amplitudes) */
  harmonic_pair_feature_labels?: string[]
  /** Pair-model feature values */
  harmonic_pair_values?: number[]
  /** Runtime status for omitted harmonic scores */
  harmonic_status?: {
    context?: string
    pair?: string
  }
  /** Model-specific harmonic decision thresholds emitted by the backend */
  harmonic_thresholds?: {
    context?: number
    pair?: number
  }
}

type InferenceState = {
  /** Time-series of inference scores (chronological order) */
  points: InferencePoint[]
  /** Maximum points to keep (ring buffer style) */
  maxPoints: number

  push: (p: InferencePoint) => void
  clear: () => void
}

export const useInferenceStore = create<InferenceState>((set, get) => ({
  points: [],
  maxPoints: 600, // keep the most recent 600 inference windows

  push: (p) => {
    set((s) => {
      const points = [...s.points, p]
      while (points.length > s.maxPoints) points.shift()
      return { points }
    })
  },

  clear: () => set({ points: [] }),
}))

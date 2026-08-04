/**
 * liveScoreStore — Zustand store for real-time live playback scoring data.
 *
 * Fed from scored_event messages on the alerts WebSocket. Maintains ring
 * buffers for the significance breakdown (for LiveSignificanceChart) and
 * prior snapshots (for LivePriorChart).
 */
import { create } from 'zustand'

/** One data-point extracted from a scored_event for real-time charting. */
export type LiveScorePoint = {
  /** Monotonic index (insertion order) */
  idx: number
  /** Epoch seconds from the event timestamp */
  t: number
  significance_score: number
  prior_boost: number
  n_rules_triggered: number
  anomaly_detector_score: number
}

/** A snapshot of the top-N pattern_priors at one point in time. */
export type PriorSnapshot = {
  t: number
  priors: Record<string, number>
}

/**
 * Latest abstracted process snapshot — the cutting context + raw metrics that
 * rode along on the most recent scored_event. Kept so overview surfaces (e.g.
 * the landing "process pulse") can show an at-a-glance state without re-deriving
 * it from the raw stream.
 */
export type LiveProcessSnapshot = {
  t: number
  context: Record<string, unknown>
  metrics: Record<string, unknown>
}

type LiveScoreState = {
  points: LiveScorePoint[]
  priorSnapshots: PriorSnapshot[]
  latest: LiveProcessSnapshot | null
  maxPoints: number
  _nextIdx: number

  push: (point: Omit<LiveScorePoint, 'idx'>) => void
  pushPriors: (snapshot: PriorSnapshot) => void
  setLatest: (snapshot: LiveProcessSnapshot) => void
  clear: () => void
}

export const useLiveScoreStore = create<LiveScoreState>((set, get) => ({
  points: [],
  priorSnapshots: [],
  latest: null,
  maxPoints: 600,
  _nextIdx: 0,

  push: (point) => {
    const idx = get()._nextIdx
    set((s) => {
      const points = [...s.points, { ...point, idx }]
      while (points.length > s.maxPoints) points.shift()
      return { points, _nextIdx: idx + 1 }
    })
  },

  pushPriors: (snapshot) => {
    set((s) => {
      const priorSnapshots = [...s.priorSnapshots, snapshot]
      while (priorSnapshots.length > s.maxPoints) priorSnapshots.shift()
      return { priorSnapshots }
    })
  },

  setLatest: (snapshot) => set({ latest: snapshot }),

  clear: () => set({ points: [], priorSnapshots: [], latest: null, _nextIdx: 0 }),
}))

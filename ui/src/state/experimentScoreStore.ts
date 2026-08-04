/**
 * experimentScoreStore — Zustand store for real-time experiment score streaming.
 *
 * Receives batches of ScorePoint from the experiment WS ("scores" phase events)
 * and maintains ring-buffered arrays for the test and eval phases.
 */
import { create } from 'zustand'

/**
 * A single sample's scoring breakdown, emitted by the evaluator and
 * buffered 50-at-a-time by the experiment runner.
 */
export type ScorePoint = {
  idx: number
  raw_model_score: number
  supervised_score: number
  combined_score: number
  significance_score: number
  anomaly_z_score: number
  prior_boost: number
  pattern_rule_score: number
  label: number
  predicted_positive: boolean
  threshold: number
}

type ExperimentScoreState = {
  testScores: ScorePoint[]
  evalScores: ScorePoint[]
  fold: number
  maxPoints: number

  push: (phase: 'test' | 'eval', points: ScorePoint[], fold?: number) => void
  setFold: (fold: number) => void
  clear: () => void
}

export const useExperimentScoreStore = create<ExperimentScoreState>((set, get) => ({
  testScores: [],
  evalScores: [],
  fold: 0,
  maxPoints: 5000,

  push: (phase, points, fold = 0) => {
    if (points.length === 0) return
    const key = phase === 'test' ? 'testScores' : 'evalScores'
    set((s) => {
      const nextFold = fold > 0 ? fold : s.fold
      const foldChanged = nextFold > 0 && nextFold !== s.fold
      const testScores = foldChanged ? [] : s.testScores
      const evalScores = foldChanged ? [] : s.evalScores

      let arr = [...(phase === 'test' ? testScores : evalScores)]
      const firstIncomingIdx = points[0]?.idx
      const lastExistingIdx = arr.length > 0 ? arr[arr.length - 1].idx : -Infinity

      // Score snapshots are indexed per phase/fold. When a new fold starts,
      // or a reconnected stream replays earlier samples, reset that series so
      // the chart x-axis remains monotonic instead of drawing backward lines.
      if (typeof firstIncomingIdx === 'number' && firstIncomingIdx <= lastExistingIdx) {
        arr = []
      }

      arr = [...arr, ...points]
      const max = s.maxPoints
      while (arr.length > max) arr.shift()
      return {
        fold: nextFold,
        testScores: phase === 'test' ? arr : testScores,
        evalScores: phase === 'eval' ? arr : evalScores,
      }
    })
  },

  setFold: (fold) => set({ fold }),

  clear: () => set({ testScores: [], evalScores: [], fold: 0 }),
}))

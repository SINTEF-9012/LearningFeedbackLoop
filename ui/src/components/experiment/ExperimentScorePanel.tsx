/**
 * ExperimentScorePanel — Real-time scoring charts for running experiments.
 *
 * Shows two RealtimeScoreChart instances:
 *   1. Model Scores — raw_model_score, supervised_score, combined_score,
 *      significance_score + threshold line.
 *   2. Pattern & Context — pattern_rule_score, prior_boost, anomaly_z_score.
 *
 * Phase toggle (test / eval) lets users switch between the two evaluation phases.
 */
import React, { useMemo, useState } from 'react'
import { useExperimentScoreStore, type ScorePoint } from '../../state/experimentScoreStore'
import { RealtimeScoreChart, type SeriesDef, type ThresholdLine } from '../charts/RealtimeScoreChart'

/* ── Series definitions ──────────────────────────────────── */

const MODEL_SERIES: SeriesDef[] = [
  { key: 'raw_model_score',   label: 'Raw Model',          color: '#7aa2f7', width: 1.5 },
  { key: 'supervised_score',  label: 'Supervised',         color: '#9ece6a', width: 1.5 },
  { key: 'combined_score',    label: 'Combined',           color: '#bb9af7', width: 2 },
  { key: 'significance_score', label: 'Significance',      color: '#e0af68', width: 2 },
]

const PATTERN_SERIES: SeriesDef[] = [
  { key: 'pattern_rule_score', label: 'Pattern Rule',     color: '#f7768e', width: 1.5 },
  { key: 'prior_boost',       label: 'Prior Boost',       color: '#73daca', width: 1.5 },
  { key: 'anomaly_z_score',   label: 'Anomaly Z',         color: '#ff9e64', width: 1.5 },
]

/* ── Helpers ──────────────────────────────────────────────── */

function buildData(points: ScorePoint[], series: SeriesDef[]): (Float64Array | number[])[] {
  const n = points.length
  if (n === 0) return []
  const xArr = new Float64Array(n)
  const arrs = series.map(() => new Float64Array(n))
  for (let i = 0; i < n; i++) {
    const p = points[i]
    xArr[i] = p.idx
    for (let j = 0; j < series.length; j++) {
      arrs[j][i] = (p as any)[series[j].key] ?? 0
    }
  }
  return [xArr, ...arrs]
}

/* ── Component ─────────────────────────────────────────────── */

export function ExperimentScorePanel() {
  const testScores = useExperimentScoreStore((s) => s.testScores)
  const evalScores = useExperimentScoreStore((s) => s.evalScores)
  const fold = useExperimentScoreStore((s) => s.fold)
  const [phase, setPhase] = useState<'test' | 'eval'>('test')

  const points = phase === 'test' ? testScores : evalScores
  const threshold = points.length > 0 ? points[points.length - 1].threshold : 0.5

  const modelData = useMemo(() => buildData(points, MODEL_SERIES), [points])
  const patternData = useMemo(() => buildData(points, PATTERN_SERIES), [points])

  const thresholdLines: ThresholdLine[] = useMemo(
    () => [{ value: threshold, label: `threshold ${threshold.toFixed(2)}`, color: 'rgba(247, 118, 142, 0.5)' }],
    [threshold],
  )

  if (testScores.length === 0 && evalScores.length === 0) return null

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)' }}>
          Real-Time Scores {fold > 0 ? `(Fold ${fold})` : ''}
        </span>
        <div style={{ display: 'flex', gap: 4 }}>
          {(['test', 'eval'] as const).map((p) => (
            <button
              key={p}
              onClick={() => setPhase(p)}
              style={{
                padding: '2px 10px', fontSize: 11, borderRadius: 3, cursor: 'pointer',
                border: phase === p ? '1px solid var(--accent)' : '1px solid transparent',
                background: phase === p ? 'rgba(52,152,219,0.15)' : 'rgba(255,255,255,0.04)',
                color: phase === p ? 'var(--accent)' : 'var(--muted)',
              }}
            >
              {p} ({p === 'test' ? testScores.length : evalScores.length})
            </button>
          ))}
        </div>
      </div>

      {/* Chart 1: Model Scores */}
      <div style={{ marginBottom: 6 }}>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 2 }}>Model Scores</div>
        <RealtimeScoreChart
          data={modelData}
          series={MODEL_SERIES}
          xLabel="Sample"
          yLabel="Score"
          thresholdLines={thresholdLines}
          height={170}
        />
      </div>

      {/* Chart 2: Pattern & Context */}
      <div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 2 }}>Pattern & Context Scores</div>
        <RealtimeScoreChart
          data={patternData}
          series={PATTERN_SERIES}
          xLabel="Sample"
          yLabel="Score"
          height={140}
          yMax={5}
        />
      </div>
    </div>
  )
}

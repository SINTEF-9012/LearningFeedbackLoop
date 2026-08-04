/**
 * LiveSignificanceChart — Real-time significance & pattern score chart
 * for the live playback / inference view.
 *
 * Shows significance_score, prior_boost, anomaly_detector_score, and
 * n_rules_triggered from scored_event messages over time.
 */
import React, { useMemo } from 'react'
import { useLiveScoreStore } from '../state/liveScoreStore'
import { RealtimeScoreChart, type SeriesDef, type ThresholdLine } from './charts/RealtimeScoreChart'

const SERIES: SeriesDef[] = [
  { key: 'significance_score',    label: 'Significance',     color: '#e0af68', width: 2 },
  { key: 'anomaly_detector_score', label: 'Anomaly Detector', color: '#7aa2f7', width: 1.5 },
  { key: 'prior_boost',           label: 'Prior Boost',      color: '#73daca', width: 1.5 },
  { key: 'n_rules_triggered',     label: 'Rules Triggered',  color: '#f7768e', width: 1, dash: [4, 3] },
]

const THRESHOLD: ThresholdLine[] = [
  { value: 0.7, label: 'score threshold (0.7)', color: 'rgba(247, 118, 142, 0.4)' },
]

export function LiveSignificanceChart({ height = 180 }: { height?: number }) {
  const points = useLiveScoreStore((s) => s.points)

  const data = useMemo(() => {
    const n = points.length
    if (n === 0) return []
    const t = new Float64Array(n)
    const sig = new Float64Array(n)
    const anom = new Float64Array(n)
    const boost = new Float64Array(n)
    const rules = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const p = points[i]
      t[i] = p.idx
      sig[i] = p.significance_score
      anom[i] = p.anomaly_detector_score
      boost[i] = p.prior_boost
      rules[i] = p.n_rules_triggered
    }
    return [t, sig, anom, boost, rules]
  }, [points])

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)', marginBottom: 6 }}>
        Live Significance Scores
      </div>
      <RealtimeScoreChart
        data={data}
        series={SERIES}
        xLabel="Event"
        yLabel="Score"
        thresholdLines={THRESHOLD}
        height={height}
      />
    </div>
  )
}

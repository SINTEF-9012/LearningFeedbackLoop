import React, { useMemo } from 'react'
import { clamp } from './chartUtils'

/**
 * Score distribution histogram with overlapping positive/negative bars
 * and an optional threshold line.
 */
export function ScoreHistogram({
  positive,
  negative,
  threshold,
  title,
}: {
  positive: number[]
  negative: number[]
  threshold: number
  title: string
}) {
  const BINS = 20
  const W = 480, H = 180
  const PAD = { t: 20, r: 12, b: 28, l: 40 }
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b

  const binCounts = useMemo(() => {
    const pos = new Array(BINS).fill(0)
    const neg = new Array(BINS).fill(0)
    for (const v of positive) {
      const b = Math.min(BINS - 1, Math.floor(clamp(v, 0, 0.9999) * BINS))
      pos[b]++
    }
    for (const v of negative) {
      const b = Math.min(BINS - 1, Math.floor(clamp(v, 0, 0.9999) * BINS))
      neg[b]++
    }
    return { pos, neg }
  }, [positive, negative])

  const maxCount = Math.max(1, ...binCounts.pos, ...binCounts.neg)
  const barW = plotW / BINS

  return (
    <div>
      <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6 }}>
        {binCounts.pos.map((c, i) => {
          const x = PAD.l + i * barW
          const h = (c / maxCount) * plotH
          return <rect key={`p${i}`} x={x + 1} y={PAD.t + plotH - h} width={barW / 2 - 1} height={h} fill="rgba(247,118,142,0.6)" />
        })}
        {binCounts.neg.map((c, i) => {
          const x = PAD.l + i * barW + barW / 2
          const h = (c / maxCount) * plotH
          return <rect key={`n${i}`} x={x} y={PAD.t + plotH - h} width={barW / 2 - 1} height={h} fill="rgba(122,162,247,0.6)" />
        })}
        {threshold > 0 && threshold < 1 && (() => {
          const tx = PAD.l + threshold * plotW
          return <line x1={tx} y1={PAD.t} x2={tx} y2={PAD.t + plotH} stroke="var(--accent)" strokeWidth={2} strokeDasharray="4,3" />
        })()}
      </svg>
      <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
        <span className="small">
          <span style={{ width: 12, height: 8, background: 'rgba(247,118,142,0.6)', borderRadius: 2, display: 'inline-block' }} /> Pre-break (n={positive.length})
        </span>
        <span className="small">
          <span style={{ width: 12, height: 8, background: 'rgba(122,162,247,0.6)', borderRadius: 2, display: 'inline-block' }} /> Normal (n={negative.length})
        </span>
        <span className="small" style={{ color: 'var(--accent)' }}>┊ Threshold = {threshold.toFixed(3)}</span>
      </div>
    </div>
  )
}

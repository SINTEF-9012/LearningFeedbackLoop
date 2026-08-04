import React from 'react'
import { num } from './chartUtils'

/**
 * Per-feature scatter plot — one row per feature column, with points
 * coloured by label (pre_stoppage vs normal).
 */
export function FeatureSignalSVG({
  rows,
  column,
  labelKey,
}: {
  rows: Record<string, any>[]
  column: string
  labelKey: string
}) {
  if (!rows.length) return null

  const vals = rows.map(r => num(r[column]))
  const labels = rows.map(r => r[labelKey])
  const minV = Math.min(...vals)
  const maxV = Math.max(...vals)
  const range = maxV - minV || 1
  const W = 640, H = 80
  const PAD = { t: 8, r: 8, b: 8, l: 56 }
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b

  return (
    <div style={{ marginBottom: 4 }}>
      <div className="small" style={{ color: 'var(--muted)', marginBottom: 2 }}>{column}</div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.1)', borderRadius: 4 }}>
        <text x={PAD.l - 4} y={PAD.t + 6} textAnchor="end" fill="var(--muted)" fontSize={8}>{maxV.toFixed(1)}</text>
        <text x={PAD.l - 4} y={PAD.t + plotH} textAnchor="end" fill="var(--muted)" fontSize={8}>{minV.toFixed(1)}</text>
        {vals.map((v, i) => {
          const x = PAD.l + (i / Math.max(1, vals.length - 1)) * plotW
          const y = PAD.t + plotH * (1 - (v - minV) / range)
          const color = (labels[i] === 'pre_stoppage' || labels[i] === 'pre_break') ? 'rgba(247,118,142,0.8)' : 'rgba(122,162,247,0.6)'
          return <circle key={i} cx={x} cy={y} r={2.5} fill={color} />
        })}
      </svg>
    </div>
  )
}

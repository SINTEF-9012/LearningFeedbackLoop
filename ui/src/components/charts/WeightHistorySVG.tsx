import React from 'react'
import { PAL } from './chartUtils'

/**
 * SVG polyline chart showing how model weights or priors evolve over time.
 */
export function WeightHistorySVG({
  history,
  title,
  keys: keysProp,
}: {
  history: Record<string, number[]>
  title: string
  keys?: string[]
}) {
  const allKeys = keysProp || Object.keys(history).filter(k => history[k]?.length > 1)
  if (!allKeys.length) return <div className="small" style={{ color: 'var(--muted)' }}>No weight history data.</div>

  const maxLen = Math.max(...allKeys.map(k => (history[k] || []).length))
  const allVals = allKeys.flatMap(k => history[k] || [])
  const minV = Math.min(...allVals, 0)
  const maxV = Math.max(...allVals, 1)
  const range = maxV - minV || 1
  const W = 480, H = 160
  const PAD = { t: 16, r: 12, b: 24, l: 50 }
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b

  return (
    <div>
      <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6 }}>
        {allKeys.map((k, ki) => {
          const vals = history[k] || []
          const pts = vals.map((v, i) => {
            const x = PAD.l + (i / Math.max(1, maxLen - 1)) * plotW
            const y = PAD.t + plotH * (1 - (v - minV) / range)
            return `${x},${y}`
          })
          return <polyline key={k} points={pts.join(' ')} fill="none" stroke={PAL[ki % PAL.length]} strokeWidth={1.8} />
        })}
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 4 }}>
        {allKeys.map((k, ki) => (
          <span key={k} className="small" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 12, height: 3, background: PAL[ki % PAL.length], borderRadius: 2 }} />
            {k}
          </span>
        ))}
      </div>
    </div>
  )
}

import React from 'react'
import { PAL, clamp } from './chartUtils'
import { humanPattern } from '../../utils/patternNames'

/**
 * SVG polyline chart showing how pattern priors evolve over feedback rounds.
 */
export function PriorEvolutionSVG({ evolution, title }: { evolution: Record<string, number[]>; title: string }) {
  const keys = Object.keys(evolution).filter(k => evolution[k]?.length > 1)
  if (!keys.length) return <div className="small" style={{ color: 'var(--muted)' }}>No prior evolution data.</div>

  const maxLen = Math.max(...keys.map(k => evolution[k].length))
  const W = 480, H = 220, PAD = { t: 16, r: 12, b: 24, l: 40 }
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b

  return (
    <div>
      <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6 }}>
        {[0, 0.25, 0.5, 0.75, 1.0].map(v => {
          const y = PAD.t + plotH * (1 - v)
          return (
            <g key={v}>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="rgba(255,255,255,0.08)" />
              <text x={PAD.l - 4} y={y + 3} textAnchor="end" fill="var(--muted)" fontSize={9}>{v.toFixed(2)}</text>
            </g>
          )
        })}
        {keys.map((k, ki) => {
          const vals = evolution[k]
          const pts = vals.map((v, i) => {
            const x = PAD.l + (i / Math.max(1, maxLen - 1)) * plotW
            const y = PAD.t + plotH * (1 - clamp(v, 0, 1))
            return `${x},${y}`
          })
          return <polyline key={k} points={pts.join(' ')} fill="none" stroke={PAL[ki % PAL.length]} strokeWidth={1.8} />
        })}
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 4 }}>
        {keys.map((k, ki) => (
          <span key={k} className="small" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 12, height: 3, background: PAL[ki % PAL.length], borderRadius: 2 }} />
            {humanPattern(k)}
          </span>
        ))}
      </div>
    </div>
  )
}

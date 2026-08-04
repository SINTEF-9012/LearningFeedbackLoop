import React from 'react'
import { PAL, clamp, f3 } from './chartUtils'
import { humanPattern } from '../../utils/patternNames'

/**
 * PriorDiffSVG — Grouped bar chart showing before/after prior values per pattern.
 *
 * Used in RunTab after a live experiment completes to visualise the sandbox diff.
 */
interface PriorDiff {
  [pattern: string]: { before: number; after: number }
}

export function PriorDiffSVG({ diff, title }: { diff: PriorDiff; title?: string }) {
  const keys = Object.keys(diff).filter(k => {
    const d = diff[k]
    return d && (d.before !== d.after)
  })
  if (!keys.length) return <div className="small" style={{ color: 'var(--muted)' }}>No prior changes.</div>

  const W = 520, H = Math.max(180, keys.length * 36 + 56)
  const PAD = { t: 20, r: 16, b: 32, l: 130 }
  const plotW = W - PAD.l - PAD.r
  const barH = 12
  const groupH = barH * 2 + 4  // two bars + gap
  const plotH = keys.length * (groupH + 8)

  const maxVal = Math.max(
    1,
    ...keys.flatMap(k => [diff[k].before, diff[k].after]),
  )

  const x = (v: number) => PAD.l + clamp(v / maxVal, 0, 1) * plotW

  return (
    <div>
      {title && <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>}
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6 }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
          const xPos = PAD.l + frac * plotW
          return (
            <g key={frac}>
              <line x1={xPos} y1={PAD.t} x2={xPos} y2={PAD.t + plotH} stroke="rgba(255,255,255,0.06)" />
              <text x={xPos} y={PAD.t + plotH + 14} textAnchor="middle" fill="var(--muted)" fontSize={9}>
                {(frac * maxVal).toFixed(2)}
              </text>
            </g>
          )
        })}

        {keys.map((k, i) => {
          const yBase = PAD.t + i * (groupH + 8)
          const bw = diff[k].before
          const aw = diff[k].after
          const delta = aw - bw
          return (
            <g key={k}>
              {/* Pattern label */}
              <text x={PAD.l - 6} y={yBase + groupH / 2 + 3} textAnchor="end" fill="#ccc" fontSize={10}>
                {humanPattern(k)}
              </text>
              {/* Before bar */}
              <rect x={PAD.l} y={yBase} width={Math.max(1, x(bw) - PAD.l)} height={barH} rx={2}
                fill={PAL[0]} opacity={0.7} />
              <text x={x(bw) + 4} y={yBase + barH - 2} fill={PAL[0]} fontSize={9}>{f3(bw)}</text>
              {/* After bar */}
              <rect x={PAD.l} y={yBase + barH + 2} width={Math.max(1, x(aw) - PAD.l)} height={barH} rx={2}
                fill={PAL[2]} opacity={0.8} />
              <text x={x(aw) + 4} y={yBase + barH * 2 + 2} fill={PAL[2]} fontSize={9}>{f3(aw)}</text>
              {/* Delta annotation */}
              <text x={W - PAD.r} y={yBase + groupH / 2 + 3} textAnchor="end"
                fill={delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--danger)' : 'var(--muted)'} fontSize={9}
                fontWeight={600}>
                {delta > 0 ? '+' : ''}{f3(delta)}
              </text>
            </g>
          )
        })}
      </svg>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 4, fontSize: 11 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 12, height: 8, background: PAL[0], borderRadius: 2, opacity: 0.7 }} />
          Before
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 12, height: 8, background: PAL[2], borderRadius: 2, opacity: 0.8 }} />
          After
        </span>
      </div>
    </div>
  )
}

/**
 * TimeSeriesChart — multi-channel SVG line chart for raw sensor waveforms.
 *
 * X-axis = timestep (within a single sample window),
 * Y-axis = sensor reading value.
 * Each channel is a colored line with an interactive legend.
 */
import React, { useMemo, useState } from 'react'
import { PAL, clamp, num } from './chartUtils'

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface ChannelData {
  name: string
  values: number[]
}

export interface SampleAnnotation {
  phase?: string
  true_label?: string
  predicted?: string
  combined_score?: number | null
  pattern_score?: number | null
  model_score?: number | null
  event_triggered?: boolean
  patterns_detected?: string[]
  correct?: boolean | null
  threshold?: number
}

export interface TimeSeriesChartProps {
  channels: ChannelData[]
  /** Total number of timesteps along X (defaults to longest channel). */
  nTimesteps?: number
  /** Chart width in px.  */
  width?: number
  /** Chart height in px. */
  height?: number
  /** Optional title above the chart. */
  title?: string
  /** Sample label (e.g. "normal" or "pre_stoppage"). */
  label?: string
  /** Per-sample scoring / event annotation. */
  annotation?: SampleAnnotation
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const MARGIN = { top: 24, right: 16, bottom: 32, left: 56 }

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function niceAxis(min: number, max: number, ticks: number): number[] {
  if (max === min) { max = min + 1 }
  const step = (max - min) / ticks
  const res: number[] = []
  for (let i = 0; i <= ticks; i++) res.push(min + step * i)
  return res
}

function shortenName(name: string): string {
  // "Vibration_ch1" → "Vib ch1", "Energy_ActiveEnergy" → "ActiveEnergy"
  return name
    .replace('Machine_State_', '')
    .replace('Axis_Power_', 'Power ')
    .replace('Vibration_', 'Vib ')
    .replace('Energy_', '')
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

const PHASE_COLORS: Record<string, string> = {
  train: '#7aa2f7',
  eval: '#e0af68',
  test: '#bb9af7',
  baseline: '#73daca',
}

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  channels,
  nTimesteps,
  width = 840,
  height = 340,
  title,
  label,
  annotation,
}) => {
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const toggle = (name: string) =>
    setHidden(prev => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })

  const plotW = width - MARGIN.left - MARGIN.right
  const plotH = height - MARGIN.top - MARGIN.bottom

  const nT = nTimesteps ?? Math.max(...channels.map(c => c.values.length), 1)

  // Compute global Y range across visible channels
  const { yMin, yMax } = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const ch of channels) {
      if (hidden.has(ch.name)) continue
      for (const v of ch.values) {
        const n = num(v)
        if (n < lo) lo = n
        if (n > hi) hi = n
      }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1 }
    const pad = (hi - lo) * 0.08 || 0.5
    return { yMin: lo - pad, yMax: hi + pad }
  }, [channels, hidden])

  const xScale = (i: number) => MARGIN.left + (i / Math.max(nT - 1, 1)) * plotW
  const yScale = (v: number) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH

  const yTicks = niceAxis(yMin, yMax, 5)
  const xTicks = niceAxis(0, nT - 1, Math.min(nT - 1, 10)).map(Math.round)

  return (
    <div style={{ marginBottom: 12 }}>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4, fontSize: 11 }}>
        {channels.map((ch, i) => (
          <button
            key={ch.name}
            onClick={() => toggle(ch.name)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '2px 8px', borderRadius: 4, border: '1px solid var(--border, #444)',
              background: hidden.has(ch.name) ? 'transparent' : `${PAL[i % PAL.length]}22`,
              opacity: hidden.has(ch.name) ? 0.4 : 1,
              cursor: 'pointer', fontSize: 11,
              color: 'var(--fg, #ccc)',
            }}
          >
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: PAL[i % PAL.length] }} />
            {shortenName(ch.name)}
          </button>
        ))}
      </div>

      <svg width={width} height={height} style={{ background: 'var(--surface, #1a1b26)', borderRadius: 6, display: 'block' }}>
        {/* Title */}
        {title && (
          <text x={width / 2} y={14} textAnchor="middle" fill="var(--fg, #ccc)" fontSize={12} fontWeight={600}>
            {title}{label ? ` — ${label}` : ''}
          </text>
        )}

        {/* Y grid + labels */}
        {yTicks.map((v, i) => {
          const y = clamp(yScale(v), MARGIN.top, MARGIN.top + plotH)
          return (
            <g key={`y${i}`}>
              <line x1={MARGIN.left} x2={MARGIN.left + plotW} y1={y} y2={y} stroke="var(--border, #333)" strokeWidth={0.5} />
              <text x={MARGIN.left - 6} y={y + 3} textAnchor="end" fill="var(--muted, #888)" fontSize={10}>
                {Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1)}
              </text>
            </g>
          )
        })}

        {/* X grid + labels */}
        {xTicks.map((t, i) => {
          const x = xScale(t)
          return (
            <g key={`x${i}`}>
              <line x1={x} x2={x} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke="var(--border, #333)" strokeWidth={0.5} />
              <text x={x} y={MARGIN.top + plotH + 16} textAnchor="middle" fill="var(--muted, #888)" fontSize={10}>
                {t}s
              </text>
            </g>
          )
        })}

        {/* Axis lines */}
        <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + plotH} stroke="var(--border, #555)" strokeWidth={1} />
        <line x1={MARGIN.left} x2={MARGIN.left + plotW} y1={MARGIN.top + plotH} y2={MARGIN.top + plotH} stroke="var(--border, #555)" strokeWidth={1} />

        {/* Event-triggered highlight — full plot background glow */}
        {annotation?.event_triggered && (
          <rect x={MARGIN.left} y={MARGIN.top} width={plotW} height={plotH}
                fill="rgba(247,118,142,0.06)" stroke="#f7768e" strokeWidth={2}
                strokeDasharray="6 3" rx={4} />
        )}

        {/* Phase colour bar along top */}
        {annotation?.phase && (
          <rect x={MARGIN.left} y={MARGIN.top - 4} width={plotW} height={4}
                fill={PHASE_COLORS[annotation.phase] || '#888'} rx={2} opacity={0.7} />
        )}

        {/* Channel lines */}
        {channels.map((ch, ci) => {
          if (hidden.has(ch.name)) return null
          const color = PAL[ci % PAL.length]
          const pts = ch.values.map((v, i) => `${xScale(i)},${clamp(yScale(num(v)), MARGIN.top, MARGIN.top + plotH)}`)
          if (pts.length < 2) return null
          return (
            <polyline
              key={ch.name}
              points={pts.join(' ')}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
              strokeLinejoin="round"
              opacity={0.85}
            />
          )
        })}

        {/* Combined-score bar (right gutter) */}
        {annotation?.combined_score != null && Number.isFinite(annotation.combined_score) && (
          <g>
            <rect x={MARGIN.left + plotW + 4} y={MARGIN.top} width={8} height={plotH}
                  fill="var(--border, #333)" rx={2} />
            <rect x={MARGIN.left + plotW + 4}
                  y={MARGIN.top + plotH * (1 - clamp(annotation.combined_score, 0, 1))}
                  width={8}
                  height={plotH * clamp(annotation.combined_score, 0, 1)}
                  fill={annotation.event_triggered ? '#f7768e' : '#7aa2f7'}
                  rx={2} />
            {annotation.threshold != null && Number.isFinite(annotation.threshold) && (
              <line
                x1={MARGIN.left + plotW + 2} x2={MARGIN.left + plotW + 14}
                y1={MARGIN.top + plotH * (1 - clamp(annotation.threshold, 0, 1))}
                y2={MARGIN.top + plotH * (1 - clamp(annotation.threshold, 0, 1))}
                stroke="#e0af68" strokeWidth={2} />
            )}
            <text x={MARGIN.left + plotW + 8} y={MARGIN.top - 6}
                  textAnchor="middle" fill="var(--muted, #888)" fontSize={8}>Score</text>
          </g>
        )}

        {/* Event triggered badge */}
        {annotation?.event_triggered && (
          <g>
            <rect x={MARGIN.left + plotW - 96} y={MARGIN.top + 6} width={92} height={18}
                  rx={4} fill="rgba(247,118,142,0.85)" />
            <text x={MARGIN.left + plotW - 50} y={MARGIN.top + 18}
                  textAnchor="middle" fill="#fff" fontSize={10} fontWeight={700}>
              ⚡ EVENT TRIGGERED
            </text>
          </g>
        )}
      </svg>

      {/* Annotation info bar below chart */}
      {annotation && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 4, fontSize: 11, alignItems: 'center' }}>
          {annotation.phase && (
            <span style={{ padding: '2px 8px', borderRadius: 4, background: `${PHASE_COLORS[annotation.phase] || '#888'}33`,
              color: PHASE_COLORS[annotation.phase] || '#888', fontWeight: 600 }}>
              {annotation.phase.toUpperCase()}
            </span>
          )}
          {annotation.combined_score != null && Number.isFinite(annotation.combined_score) && (
            <span style={{ color: 'var(--fg)' }}>
              Combined: <strong>{(annotation.combined_score * 100).toFixed(1)}%</strong>
            </span>
          )}
          {annotation.pattern_score != null && Number.isFinite(annotation.pattern_score) && (
            <span style={{ color: 'var(--muted)' }}>
              Pattern: {(annotation.pattern_score * 100).toFixed(1)}%
            </span>
          )}
          {annotation.model_score != null && Number.isFinite(annotation.model_score) && (
            <span style={{ color: 'var(--muted)' }}>
              Model: {(annotation.model_score * 100).toFixed(1)}%
            </span>
          )}
          {annotation.threshold != null && Number.isFinite(annotation.threshold) && (
            <span style={{ color: '#e0af68' }}>
              Threshold: {(annotation.threshold * 100).toFixed(1)}%
            </span>
          )}
          {annotation.predicted && (
            <span style={{ color: annotation.correct === false ? '#f7768e' : annotation.correct ? '#9ece6a' : 'var(--muted)' }}>
              Predicted: {annotation.predicted}{annotation.correct != null && (annotation.correct ? ' ✓' : ' ✗')}
            </span>
          )}
          {annotation.patterns_detected && annotation.patterns_detected.length > 0 && (
            <span style={{ color: '#bb9af7' }}>
              Patterns: {annotation.patterns_detected.join(', ')}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

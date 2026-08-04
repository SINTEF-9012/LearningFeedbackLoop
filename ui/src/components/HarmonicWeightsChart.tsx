/**
 * HarmonicWeightsChart — Visualises learned harmonic context weights.
 *
 * Adapted from the classical HarmonicBreakNet SimulationPanel (Plotly bar
 * charts) to use pure canvas rendering.  Weights are grouped by channel /
 * axis when the label format "Group·Harmonic" is detected, producing one
 * bar chart per group — matching the original X / Y / Z layout.
 *
 * Tag: [HARMONIC_CONTEXT_V1]
 */
import React, { useEffect, useMemo, useRef } from 'react'

/* ── Group colours (matching original) ─────────────────── */
const GROUP_COLORS = [
  '#3b82f6', // blue  — X
  '#10b981', // green — Y
  '#f59e0b', // amber — Z
  '#a78bfa', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
]
const NEG_COLOR = '#ef4444'

/* ── Types ─────────────────────────────────────────────── */
type WeightGroup = {
  name: string
  weights: number[]
  labels: string[]
}

/* ── Group detection ───────────────────────────────────── */
function groupWeights(weights: number[], labels: string[]): WeightGroup[] {
  const groups = new Map<string, { weights: number[]; labels: string[] }>()

  for (let i = 0; i < weights.length; i++) {
    const label = labels[i] || `F${i + 1}`
    const dotIdx = label.indexOf('·')
    let groupName: string
    let barLabel: string
    if (dotIdx > 0) {
      groupName = label.slice(0, dotIdx)
      barLabel = label.slice(dotIdx + 1)
    } else {
      groupName = 'Features'
      barLabel = label
    }
    if (!groups.has(groupName)) {
      groups.set(groupName, { weights: [], labels: [] })
    }
    groups.get(groupName)!.weights.push(weights[i])
    groups.get(groupName)!.labels.push(barLabel)
  }

  return Array.from(groups.entries()).map(([name, data]) => ({
    name,
    weights: data.weights,
    labels: data.labels,
  }))
}

/* ── Single bar-chart canvas ───────────────────────────── */
type BarChartProps = {
  weights: number[]
  labels: string[]
  color: string
  title: string
  seriesLabel: string
  height?: number
}

function BarChart({ weights, labels, color, title, seriesLabel, height = 160 }: BarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || weights.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    if (w === 0) return

    canvas.width = w * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    canvas.style.width = `${w}px`
    canvas.style.height = `${height}px`

    const n = weights.length
    const maxAbs = Math.max(...weights.map(Math.abs), 1e-4)
    const margin = { top: 8, right: 8, bottom: 32, left: 38 }
    const plotW = w - margin.left - margin.right
    const plotH = height - margin.top - margin.bottom
    const barW = Math.max(6, Math.min(28, (plotW / n) * 0.7))
    const step = plotW / n
    const zeroY = margin.top + plotH / 2

    ctx.clearRect(0, 0, w, height)

    // Faint grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    for (const frac of [-1, -0.5, 0.5, 1]) {
      const y = zeroY - frac * (plotH / 2)
      ctx.beginPath()
      ctx.moveTo(margin.left, y)
      ctx.lineTo(w - margin.right, y)
      ctx.stroke()
    }

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.25)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(margin.left, zeroY)
    ctx.lineTo(w - margin.right, zeroY)
    ctx.stroke()

    // Bars
    for (let i = 0; i < n; i++) {
      const val = weights[i]
      const barH = Math.max(1, Math.abs(val / maxAbs) * (plotH / 2))
      const x = margin.left + step * i + (step - barW) / 2

      ctx.fillStyle = val >= 0 ? color : NEG_COLOR
      if (val >= 0) {
        ctx.fillRect(x, zeroY - barH, barW, barH)
      } else {
        ctx.fillRect(x, zeroY, barW, barH)
      }

      // Bar label (rotated)
      ctx.save()
      ctx.fillStyle = 'rgba(169, 177, 214, 0.7)'
      ctx.font = `${Math.min(10, step * 0.8)}px monospace`
      ctx.textAlign = 'right'
      ctx.translate(x + barW / 2 + 3, height - margin.bottom + 8)
      ctx.rotate(-Math.PI / 4)
      ctx.fillText(labels[i] || `${i + 1}`, 0, 0)
      ctx.restore()
    }

    // Y-axis labels
    ctx.fillStyle = 'rgba(169, 177, 214, 0.5)'
    ctx.font = '9px monospace'
    ctx.textAlign = 'right'
    ctx.fillText(`+${maxAbs.toFixed(2)}`, margin.left - 4, margin.top + 10)
    ctx.fillText('0', margin.left - 4, zeroY + 3)
    ctx.fillText(`-${maxAbs.toFixed(2)}`, margin.left - 4, margin.top + plotH)
  }, [weights, labels, color, height])

  return (
    <div style={{ flex: '1 1 200px', minWidth: 160 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color, marginBottom: 4 }}>
        {seriesLabel} — {title}
      </div>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height,
          background: 'rgba(0,0,0,0.15)',
          borderRadius: 4,
          border: '1px solid var(--border)',
        }}
      />
    </div>
  )
}

/* ── Main component ────────────────────────────────────── */
export type HarmonicWeightsChartProps = {
  weights: number[]
  labels?: string[]
  score?: number
  height?: number
  kind?: 'weights' | 'outputs'
}

export function HarmonicWeightsChart({
  weights,
  labels: rawLabels,
  score,
  height = 160,
  kind = 'weights',
}: HarmonicWeightsChartProps) {
  const labels = rawLabels && rawLabels.length === weights.length
    ? rawLabels
    : weights.map((_, i) => `F${i + 1}`)

  const groups = useMemo(() => groupWeights(weights, labels), [weights, labels])
  const showingWeights = kind === 'weights'

  if (weights.length === 0) return null

  return (
    <div>
      {/* Title matching original — "Learned Harmonic Weights (w = params × Wᵀ)" */}
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: 12,
        marginBottom: 8,
      }}>
        <div style={{
          fontSize: 12, fontWeight: 600, color: 'var(--text)',
          letterSpacing: 0.3,
        }}>
          {showingWeights ? 'Learned Harmonic Weights' : 'Live Harmonic Outputs'}
          {showingWeights && (
            <span style={{ fontWeight: 400, color: 'var(--muted)', marginLeft: 6, fontSize: 11 }}>
              w = params × W<sup>T</sup>
            </span>
          )}
        </div>
        {showingWeights && typeof score === 'number' && (
          <div style={{
            fontSize: 11, fontFamily: 'monospace',
            color: score > 0.7 ? 'var(--danger)' : score > 0.4 ? '#f0a050' : 'var(--ok)',
            fontWeight: 600,
          }}>
            P = {score.toFixed(3)}
          </div>
        )}
      </div>

      {/* Bar charts — one per channel group */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {groups.map((g, gi) => (
          <BarChart
            key={g.name}
            weights={g.weights}
            labels={g.labels}
            color={GROUP_COLORS[gi % GROUP_COLORS.length]}
            title={g.name}
            seriesLabel={showingWeights ? 'w vector' : 'output'}
            height={height}
          />
        ))}
      </div>

      {/* Colour legend */}
      <div style={{
        display: 'flex', gap: 16, padding: '6px 0 0', fontSize: 10,
        color: 'var(--muted)',
      }}>
        {groups.map((g, gi) => (
          <span key={g.name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              display: 'inline-block', width: 10, height: 10, borderRadius: 2,
              background: GROUP_COLORS[gi % GROUP_COLORS.length],
            }} />
            {g.name} (positive)
          </span>
        ))}
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{
            display: 'inline-block', width: 10, height: 10, borderRadius: 2,
            background: NEG_COLOR,
          }} />
          Negative
        </span>
      </div>
    </div>
  )
}

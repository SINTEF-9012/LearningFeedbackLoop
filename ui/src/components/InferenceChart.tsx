import React, { useEffect, useMemo, useRef } from 'react'
import uPlot, { Options } from 'uplot'
import 'uplot/dist/uPlot.min.css'

import { useInferenceStore } from '../state/inferenceStore'
import { useStreamStore } from '../state/streamStore'

/* ── Colours per model ─────────────────────────────────────── */
const MODEL_COLORS: Record<string, string> = {
  ensemble: '#7aa2f7',        // blue
  isolation_forest: '#9ece6a', // green
  lof: '#bb9af7',             // purple
  z_score: '#ff9e64',         // orange
  harmonic_context_score: '#f7768e', // coral
  harmonic_pair_score: '#e0af68', // amber
}

const MODEL_LABELS: Record<string, string> = {
  ensemble: 'Ensemble (IF+LOF)',
  isolation_forest: 'Isolation Forest',
  lof: 'Local Outlier Factor',
  z_score: 'Z-Score',
  harmonic_context_score: 'Harmonic Context',
  harmonic_pair_score: 'Harmonic Pair',
}

const MODEL_KEYS = [
  'ensemble',
  'isolation_forest',
  'lof',
  'z_score',
  'harmonic_context_score',
  'harmonic_pair_score',
] as const

function lowerBound(arr: number[], x: number): number {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] < x) lo = mid + 1
    else hi = mid
  }
  return lo
}

function isAbsoluteTimestamp(value: number): boolean {
  return Number.isFinite(value) && value > 1_000_000_000
}

/* ── Component ─────────────────────────────────────────────── */
export type InferenceChartProps = {
  height?: number
}

export function InferenceChart({ height = 220 }: InferenceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotRef = useRef<uPlot | null>(null)
  const points = useInferenceStore((s) => s.points)
  const windowSeconds = useStreamStore((s) => s.windowSeconds)

  const plotState = useMemo(() => {
    if (points.length === 0) {
      return {
        data: Array.from({ length: MODEL_KEYS.length + 1 }, () => []),
        useAbsoluteTime: false,
        xAxisLabel: 'elapsed s',
        visibleCount: 0,
      }
    }

    let visiblePoints = points
    const allTimes = points.map((point) => point.t)
    const useAbsoluteTime = allTimes.some((value) => isAbsoluteTimestamp(value))
    const xAxisLabel = useAbsoluteTime ? 'timestamp' : 'elapsed s'

    if (points.length > 1 && windowSeconds > 0) {
      const cutoff = allTimes[allTimes.length - 1] - windowSeconds
      const start = lowerBound(allTimes, cutoff)
      visiblePoints = points.slice(start)
    }

    const n = visiblePoints.length
    const t = new Array<number>(n)
    const series = MODEL_KEYS.map(() => new Array<number | null>(n))
    for (let i = 0; i < n; i++) {
      const p = visiblePoints[i]
      t[i] = p.t
      MODEL_KEYS.forEach((key, seriesIndex) => {
        const score = p.scores[key]
        series[seriesIndex][i] = typeof score === 'number' && Number.isFinite(score)
          ? score
          : null
      })
    }
    return {
      data: [t, ...series],
      useAbsoluteTime,
      xAxisLabel,
      visibleCount: n,
    }
  }, [points, windowSeconds])
  const showPointMarkers = plotState.visibleCount <= 2

  // Create or re-create chart
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    // Destroy previous instance
    if (plotRef.current) {
      plotRef.current.destroy()
      plotRef.current = null
    }

    const opts: Options = {
      width: el.clientWidth || 800,
      height,
      cursor: { drag: { setScale: false } },
      legend: { show: false },
      scales: {
        x: { time: plotState.useAbsoluteTime },
        y: { min: 0, max: 1 },
      },
      series: [
        { label: plotState.xAxisLabel },
        ...MODEL_KEYS.map((key) => ({
          label: MODEL_LABELS[key],
          stroke: MODEL_COLORS[key],
          width: key === 'ensemble' || key === 'harmonic_context_score' ? 2 : 1,
          points: {
            show: showPointMarkers,
            size: 6,
            width: 2,
            stroke: MODEL_COLORS[key],
            fill: MODEL_COLORS[key],
          },
        })),
      ],
      axes: [
        {
          label: plotState.xAxisLabel,
          stroke: '#a9b1d6',
          grid: { stroke: 'rgba(255,255,255,0.08)' },
        },
        {
          label: 'Anomaly Score',
          stroke: '#a9b1d6',
          grid: { stroke: 'rgba(255,255,255,0.08)' },
        },
      ],
      hooks: {
        draw: [
          (u) => {
            // Draw model threshold line at 0.7
            const ctx = u.ctx
            const yPx = u.valToPos(0.7, 'y', true)
            const left = u.bbox.left
            const width = u.bbox.width
            ctx.save()
            ctx.strokeStyle = 'rgba(247, 118, 142, 0.4)'
            ctx.lineWidth = 1
            ctx.setLineDash([6, 4])
            ctx.beginPath()
            ctx.moveTo(left, yPx)
            ctx.lineTo(left + width, yPx)
            ctx.stroke()
            ctx.setLineDash([])
            // Label
            ctx.fillStyle = 'rgba(247, 118, 142, 0.6)'
            ctx.font = '10px sans-serif'
            ctx.fillText('model threshold', left + 4, yPx - 4)
            ctx.restore()
          },
        ],
      },
    }

    const u = new uPlot(opts, plotState.data as any, el)
    plotRef.current = u

    // Resize handler
    const ro = new ResizeObserver(() => {
      if (plotRef.current && el.clientWidth > 0) {
        plotRef.current.setSize({ width: el.clientWidth, height })
      }
    })
    ro.observe(el)

    return () => {
      ro.disconnect()
      if (plotRef.current) {
        plotRef.current.destroy()
        plotRef.current = null
      }
    }
    // Re-create chart only when there's a significant structural change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, showPointMarkers, plotState.useAbsoluteTime, plotState.xAxisLabel])

  // Efficient data update without recreating the chart
  useEffect(() => {
    if (plotRef.current) {
      plotRef.current.setData(plotState.data as any)
    }
  }, [plotState])

  return (
    <div style={{ position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%' }} />
      {plotState.visibleCount === 0 && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--muted)', fontSize: 13,
        }}>
          Waiting for inference data…
        </div>
      )}
      {plotState.visibleCount === 1 && (
        <div style={{ padding: '0 8px 4px', fontSize: 11, color: 'var(--muted)' }}>
          Showing the first inference window. The line will extend as more windows arrive.
        </div>
      )}
      {/* Legend */}
      {plotState.visibleCount > 0 && (
        <div style={{ display: 'flex', gap: 16, padding: '4px 8px', fontSize: 11 }}>
          {MODEL_KEYS.map((key) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                display: 'inline-block', width: 12, height: 3, borderRadius: 2,
                background: MODEL_COLORS[key],
              }} />
              <span style={{ color: 'var(--muted)' }}>{MODEL_LABELS[key]}</span>
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 'auto' }}>
            <span style={{
              display: 'inline-block', width: 12, height: 1, borderRadius: 1,
              borderTop: '1px dashed rgba(247, 118, 142, 0.6)',
            }} />
            <span style={{ color: 'var(--muted)' }}>Model threshold (0.7)</span>
          </div>
        </div>
      )}
    </div>
  )
}

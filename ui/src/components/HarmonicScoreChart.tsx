import React, { useEffect, useMemo, useRef } from 'react'
import uPlot, { Options } from 'uplot'
import 'uplot/dist/uPlot.min.css'

import { useInferenceStore } from '../state/inferenceStore'
import { useStreamStore } from '../state/streamStore'

const SCORE_SERIES = [
  { key: 'harmonic_pair_score', label: 'Harmonic Pair', color: '#e0af68' },
  { key: 'harmonic_context_score', label: 'Harmonic Context', color: '#f7768e' },
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

export type HarmonicScoreChartProps = {
  height?: number
}

export function HarmonicScoreChart({ height = 300 }: HarmonicScoreChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotRef = useRef<uPlot | null>(null)
  const points = useInferenceStore((s) => s.points)
  const windowSeconds = useStreamStore((s) => s.windowSeconds)

  const plotState = useMemo(() => {
    if (points.length === 0) {
      return {
        data: Array.from({ length: SCORE_SERIES.length + 1 }, () => []),
        useAbsoluteTime: false,
        xAxisLabel: 'elapsed s',
        visibleCount: 0,
        pairThreshold: undefined as number | undefined,
        contextThreshold: undefined as number | undefined,
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
    const scoreSeries = SCORE_SERIES.map(() => new Array<number | null>(n))
    for (let index = 0; index < n; index += 1) {
      const point = visiblePoints[index]
      t[index] = point.t
      SCORE_SERIES.forEach((series, seriesIndex) => {
        const value = point.scores[series.key]
        scoreSeries[seriesIndex][index] = typeof value === 'number' && Number.isFinite(value)
          ? value
          : null
      })
    }

    const latest = visiblePoints[visiblePoints.length - 1]
    return {
      data: [t, ...scoreSeries],
      useAbsoluteTime,
      xAxisLabel,
      visibleCount: n,
      pairThreshold: latest?.harmonic_thresholds?.pair,
      contextThreshold: latest?.harmonic_thresholds?.context,
    }
  }, [points, windowSeconds])

  const showPointMarkers = plotState.visibleCount <= 2

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    if (plotRef.current) {
      plotRef.current.destroy()
      plotRef.current = null
    }

    const opts: Options = {
      width: el.clientWidth || 900,
      height,
      cursor: { drag: { setScale: false } },
      legend: { show: false },
      scales: {
        x: { time: plotState.useAbsoluteTime },
        y: { min: 0, max: 1 },
      },
      series: [
        { label: plotState.xAxisLabel },
        ...SCORE_SERIES.map((series) => ({
          label: series.label,
          stroke: series.color,
          width: 2,
          points: {
            show: showPointMarkers,
            size: 6,
            width: 2,
            stroke: series.color,
            fill: series.color,
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
          label: 'Harmonic Score',
          stroke: '#a9b1d6',
          grid: { stroke: 'rgba(255,255,255,0.08)' },
        },
      ],
      hooks: {
        draw: [
          (u) => {
            const lines = [
              {
                value: plotState.pairThreshold,
                color: 'rgba(224, 175, 104, 0.7)',
                label: typeof plotState.pairThreshold === 'number'
                  ? `pair thr ${plotState.pairThreshold.toFixed(3)}`
                  : null,
              },
              {
                value: plotState.contextThreshold,
                color: 'rgba(247, 118, 142, 0.55)',
                label: typeof plotState.contextThreshold === 'number'
                  ? `context thr ${plotState.contextThreshold.toFixed(3)}`
                  : null,
              },
            ]

            const ctx = u.ctx
            const left = u.bbox.left
            const width = u.bbox.width

            lines.forEach((line, index) => {
              if (typeof line.value !== 'number' || !Number.isFinite(line.value)) return
              const yPx = u.valToPos(line.value, 'y', true)
              ctx.save()
              ctx.strokeStyle = line.color
              ctx.lineWidth = 1
              ctx.setLineDash([6, 4])
              ctx.beginPath()
              ctx.moveTo(left, yPx)
              ctx.lineTo(left + width, yPx)
              ctx.stroke()
              ctx.setLineDash([])
              if (line.label) {
                ctx.fillStyle = line.color
                ctx.font = '10px sans-serif'
                ctx.fillText(line.label, left + 4, yPx - 6 - index * 12)
              }
              ctx.restore()
            })
          },
        ],
      },
    }

    const u = new uPlot(opts, plotState.data as never, el)
    plotRef.current = u

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
  }, [height, showPointMarkers, plotState.contextThreshold, plotState.pairThreshold, plotState.useAbsoluteTime, plotState.xAxisLabel])

  useEffect(() => {
    if (plotRef.current) {
      plotRef.current.setData(plotState.data as never)
    }
  }, [plotState])

  return (
    <div style={{ position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%' }} />
      {plotState.visibleCount === 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--muted)',
            fontSize: 13,
          }}
        >
          Waiting for harmonic inference data…
        </div>
      )}
      {plotState.visibleCount > 0 && (
        <div style={{ display: 'flex', gap: 16, padding: '4px 8px', fontSize: 11, flexWrap: 'wrap' }}>
          {SCORE_SERIES.map((series) => (
            <div key={series.key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 12,
                  height: 3,
                  borderRadius: 2,
                  background: series.color,
                }}
              />
              <span style={{ color: 'var(--muted)' }}>{series.label}</span>
            </div>
          ))}
          {typeof plotState.pairThreshold === 'number' && (
            <div style={{ color: 'var(--muted)' }}>Pair threshold {plotState.pairThreshold.toFixed(3)}</div>
          )}
          {typeof plotState.contextThreshold === 'number' && (
            <div style={{ color: 'var(--muted)' }}>Context threshold {plotState.contextThreshold.toFixed(3)}</div>
          )}
        </div>
      )}
    </div>
  )
}
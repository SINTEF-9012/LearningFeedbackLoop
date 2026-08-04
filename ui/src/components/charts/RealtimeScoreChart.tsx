/**
 * RealtimeScoreChart — Reusable uPlot wrapper for streaming score data.
 *
 * Renders a canvas-based time-series chart with configurable series, optional
 * horizontal threshold lines, and efficient incremental data updates via
 * `uPlot.setData()`.
 */
import React, { useEffect, useRef } from 'react'
import uPlot, { type Options } from 'uplot'
import 'uplot/dist/uPlot.min.css'

/* ── Public types ─────────────────────────────────────────── */

export type SeriesDef = {
  key: string
  label: string
  color: string
  width?: number
  dash?: number[]
}

export type ThresholdLine = {
  value: number
  label: string
  color: string
}

export type RealtimeScoreChartProps = {
  /** First row is X-axis, rest match `series` order. */
  data: (Float64Array | number[])[]
  series: SeriesDef[]
  xLabel?: string
  yLabel?: string
  thresholdLines?: ThresholdLine[]
  height?: number
  yMin?: number
  yMax?: number
}

/* ── Component ─────────────────────────────────────────────── */

export function RealtimeScoreChart({
  data,
  series,
  xLabel = 'Sample',
  yLabel = 'Score',
  thresholdLines = [],
  height = 200,
  yMin = 0,
  yMax = 1,
}: RealtimeScoreChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotRef = useRef<uPlot | null>(null)

  // Create chart on mount and when series definition changes
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    if (plotRef.current) {
      plotRef.current.destroy()
      plotRef.current = null
    }

    const opts: Options = {
      width: el.clientWidth || 800,
      height,
      cursor: { drag: { setScale: false } },
      scales: {
        x: { time: false },
        y: { min: yMin, max: yMax },
      },
      series: [
        { label: xLabel },
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.5,
          dash: s.dash,
          points: { show: false },
        })),
      ],
      axes: [
        {
          label: xLabel,
          stroke: '#a9b1d6',
          grid: { stroke: 'rgba(255,255,255,0.08)' },
        },
        {
          label: yLabel,
          stroke: '#a9b1d6',
          grid: { stroke: 'rgba(255,255,255,0.08)' },
        },
      ],
      hooks: thresholdLines.length > 0 ? {
        draw: [
          (u) => {
            const ctx = u.ctx
            ctx.save()
            for (const tl of thresholdLines) {
              const yPx = u.valToPos(tl.value, 'y', true)
              const left = u.bbox.left
              const w = u.bbox.width
              ctx.strokeStyle = tl.color
              ctx.lineWidth = 1
              ctx.setLineDash([6, 4])
              ctx.beginPath()
              ctx.moveTo(left, yPx)
              ctx.lineTo(left + w, yPx)
              ctx.stroke()
              ctx.setLineDash([])
              ctx.fillStyle = tl.color
              ctx.font = '10px sans-serif'
              ctx.fillText(tl.label, left + 4, yPx - 4)
            }
            ctx.restore()
          },
        ],
      } : undefined,
    }

    const emptyData: (number[])[] = Array.from({ length: series.length + 1 }, () => [])
    const u = new uPlot(opts, emptyData as any, el)
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, series.length, yMin, yMax])

  // Efficient data update
  useEffect(() => {
    if (plotRef.current && data.length > 0 && data[0].length > 0) {
      plotRef.current.setData(data as any)
    }
  }, [data])

  return (
    <div style={{ position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%' }} />
      {data.length === 0 || data[0].length === 0 ? (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--muted)', fontSize: 13,
        }}>
          Waiting for score data…
        </div>
      ) : null}
      {/* Legend */}
      {data.length > 0 && data[0].length > 0 && (
        <div style={{ display: 'flex', gap: 14, padding: '4px 8px', fontSize: 11, flexWrap: 'wrap' }}>
          {series.map((s) => (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                display: 'inline-block', width: 12, height: 3, borderRadius: 2,
                background: s.color,
              }} />
              <span style={{ color: 'var(--muted)' }}>{s.label}</span>
            </div>
          ))}
          {thresholdLines.map((tl) => (
            <div key={tl.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                display: 'inline-block', width: 12, height: 1, borderRadius: 1,
                borderTop: `1px dashed ${tl.color}`,
              }} />
              <span style={{ color: 'var(--muted)' }}>{tl.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

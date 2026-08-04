import React, { useEffect, useMemo, useRef } from 'react'
import uPlot, { Options } from 'uplot'
import 'uplot/dist/uPlot.min.css'

import { useStreamStore } from '../state/streamStore'
import { useAlertsStore } from '../state/alertsStore'
import { resolveVisiblePlotChannels } from '../utils/plotChannels'

export type AlertMarker = {
  index: number      // stream sample index
  severity?: string  // CRITICAL | WARNING | INFO
}

export type StreamPlotProps = {
  height?: number
  onSelect?: (i0: number, i1: number) => void
}

function seriesColor(idx: number): string {
  const palette = ['#7aa2f7', '#9ece6a', '#f7768e', '#bb9af7', '#7dcfff', '#ff9e64']
  return palette[idx % palette.length]
}

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

export function StreamPlot(props: StreamPlotProps) {
  const height = props.height ?? 320
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotRef = useRef<uPlot | null>(null)

  const channels = useStreamStore((s) => s.channels)
  const series = useStreamStore((s) => s.series)
  const seriesVersion = useStreamStore((s) => s.seriesVersion)
  const followTail = useStreamStore((s) => s.followTail)
  const windowSeconds = useStreamStore((s) => s.windowSeconds)

  // Derive alert markers from the alerts store
  const alerts = useAlertsStore((s) => s.alerts)
  const markersRef = useRef<AlertMarker[]>([])
  markersRef.current = useMemo(() => {
    return alerts
      .filter((a) => typeof a._streamIndex === 'number')
      .map((a) => ({ index: a._streamIndex!, severity: a.severity }))
  }, [alerts])

  const plotChannels = useMemo(() => {
    const chosen = resolveVisiblePlotChannels(series.yByChannel, channels)
    // uPlot expects x + at least one y series; provide a placeholder until data arrives.
    return chosen.length ? chosen : ['y']
  }, [channels, series.yByChannel])

  const plotChannelsKey = useMemo(() => plotChannels.join('|'), [plotChannels])

  const plotState = useMemo(() => {
    const indexAll = series.x
    const timeAll = series.xTime
    const hasTimeValues = timeAll.length === indexAll.length && timeAll.length > 0
    const xAll = hasTimeValues ? timeAll : indexAll
    const ysAll = plotChannels.map((ch) => series.yByChannel[ch] || [])
    const useAbsoluteTime = hasTimeValues && xAll.some((value) => isAbsoluteTimestamp(value))
    const xAxisLabel = useAbsoluteTime ? 'timestamp' : hasTimeValues ? 'elapsed s' : 'sample i'

    const finalizeState = (x: number[], indexData: number[], ys: number[][]) => ({
      data: [x, ...ys],
      indexData,
      useAbsoluteTime,
      xAxisLabel,
    })

    if (!followTail) {
      return finalizeState(xAll, indexAll, ysAll)
    }

    const fs = Number(series.fs) || 1000
    const winSec = Number(windowSeconds) || 0
    const winPts = Math.max(10, Math.floor(winSec * fs))
    if (!Number.isFinite(winPts) || winPts <= 0 || xAll.length <= winPts) {
      return finalizeState(xAll, indexAll, ysAll)
    }

    const start = Math.max(0, xAll.length - winPts)
    const x = xAll.slice(start)
    const indexData = indexAll.slice(start)
    const ys = ysAll.map((arr) => arr.slice(start))
    return finalizeState(x, indexData, ys)
  }, [seriesVersion, series.yByChannel, series.fs, plotChannels, followTail, windowSeconds])
  const plotStateRef = useRef(plotState)
  plotStateRef.current = plotState

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const selectedChannels = plotChannels

    const opts: Options = {
      width: el.clientWidth || 800,
      height,
      cursor: { drag: { setScale: false } },
      select: { show: true } as any,
      scales: { x: { time: plotState.useAbsoluteTime } },
      series: [
        { label: plotState.xAxisLabel },
        ...selectedChannels.map((ch, idx) => ({
          label: ch,
          stroke: seriesColor(idx),
          width: 1,
          points: { show: false },
        })),
      ],
      axes: [
        { label: plotState.xAxisLabel, stroke: '#a9b1d6', grid: { stroke: 'rgba(255,255,255,0.08)' } },
        { stroke: '#a9b1d6', grid: { stroke: 'rgba(255,255,255,0.08)' } },
      ],
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx
            const markers = markersRef.current
            if (!markers.length) return
            const plotLeft = u.bbox.left / devicePixelRatio
            const plotTop = u.bbox.top / devicePixelRatio
            const plotHeight = u.bbox.height / devicePixelRatio
            const currentPlotState = plotStateRef.current
            const xVals = currentPlotState.data[0] as number[]
            const indexVals = currentPlotState.indexData

            for (const m of markers) {
              const idx = lowerBound(indexVals, m.index)
              if (idx >= indexVals.length) continue
              const xVal = xVals[idx]
              if (typeof xVal !== 'number' || !Number.isFinite(xVal)) continue
              const px = u.valToPos(xVal, 'x', true) / devicePixelRatio
              // Skip markers outside the visible area
              if (px < plotLeft || px > plotLeft + u.bbox.width / devicePixelRatio) continue

              const color =
                m.severity === 'CRITICAL' ? 'rgba(247, 118, 142, 0.7)' :
                m.severity === 'WARNING' ? 'rgba(224, 175, 104, 0.7)' :
                'rgba(122, 162, 247, 0.5)'

              ctx.save()
              ctx.strokeStyle = color
              ctx.lineWidth = 1.5
              ctx.setLineDash([4, 3])
              ctx.beginPath()
              ctx.moveTo(px, plotTop)
              ctx.lineTo(px, plotTop + plotHeight)
              ctx.stroke()

              // Small triangle marker at the top
              ctx.fillStyle = color
              ctx.setLineDash([])
              ctx.beginPath()
              ctx.moveTo(px - 4, plotTop)
              ctx.lineTo(px + 4, plotTop)
              ctx.lineTo(px, plotTop + 7)
              ctx.closePath()
              ctx.fill()

              ctx.restore()
            }
          },
        ],
        setSelect: [
          (u) => {
            if (!props.onSelect) return
            const sel = u.select
            if (!sel || sel.width <= 1) return
            const x0 = u.posToVal(sel.left, 'x')
            const x1 = u.posToVal(sel.left + sel.width, 'x')
            const currentPlotState = plotStateRef.current
            if (currentPlotState.useAbsoluteTime) {
              const lo = Math.min(x0, x1)
              const hi = Math.max(x0, x1)
              const xVals = currentPlotState.data[0] as number[]
              const indexVals = currentPlotState.indexData
              let first: number | null = null
              let last: number | null = null
              for (let idx = 0; idx < xVals.length; idx++) {
                const xVal = xVals[idx]
                if (xVal < lo || xVal > hi) continue
                const sampleIndex = indexVals[idx]
                if (first === null) first = sampleIndex
                last = sampleIndex
              }
              if (first !== null && last !== null && last >= first) props.onSelect(first, last + 1)
              return
            }
            const i0 = Math.floor(Math.min(x0, x1))
            const i1 = Math.ceil(Math.max(x0, x1))
            if (Number.isFinite(i0) && Number.isFinite(i1) && i1 > i0) props.onSelect(i0, i1)
          },
        ],
      },
    }

    const plot = new uPlot(opts, plotState.data as any, el)
    plotRef.current = plot

    const resize = () => {
      try {
        plot.setSize({ width: el.clientWidth || 800, height })
      } catch {
        // ignore
      }
    }

    const ro = new ResizeObserver(resize)
    ro.observe(el)

    return () => {
      ro.disconnect()
      plot.destroy()
      plotRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, plotChannelsKey, plotState.useAbsoluteTime, plotState.xAxisLabel])

  useEffect(() => {
    const plot = plotRef.current
    if (!plot) return

    try {
      plot.setData(plotState.data as any)
    } catch {
      // ignore
    }
  }, [plotState])

  // Redraw when alert markers change (even if data hasn't changed)
  useEffect(() => {
    const plot = plotRef.current
    if (!plot) return
    try {
      plot.redraw(false, true)
    } catch {
      // ignore
    }
  }, [alerts])

  return <div ref={containerRef} />
}

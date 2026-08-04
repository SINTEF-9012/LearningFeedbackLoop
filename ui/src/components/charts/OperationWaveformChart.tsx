/**
 * OperationWaveformChart — Full-operation continuous waveform with
 * highlighted event regions, interactive zoom, and pan.
 *
 * Zoom controls:
 *   • Click-drag on the chart to zoom into a time range
 *   • Mouse-wheel scrolls the zoom level at the cursor position
 *   • Double-click to reset zoom
 *   • "Reset Zoom" button in the toolbar
 *
 * Scattering fix: points are now drawn with monotonically-sorted
 * timestamps and gap detection so disjoint time ranges don't
 * produce visual noise.
 */
import React, { useMemo, useState, useRef, useCallback } from 'react'
import { PAL, clamp } from './chartUtils'

export interface WaveformChannel {
  name: string
  timestamps: number[]  // seconds from operation start
  values: number[]
}

export interface EventRegion {
  start_s: number
  end_s: number
  label: string
  sample_id: string
  severity: string
  event_timestamp: string
  source?: string
  feedback_given?: boolean
  feedback_action?: string
  predicted_positive?: boolean
  significance_score?: number
  detected_patterns?: string[]
}

interface Props {
  channels: WaveformChannel[]
  regions: EventRegion[]
  durationSeconds: number
  durationHours: number
  operationId: string
  width?: number
  height?: number
}

const M = { top: 28, right: 20, bottom: 44, left: 60 }

/* ── Monotonic timestamp sort for a channel ─────────────────────────── */
function sortedChannel(ch: WaveformChannel): WaveformChannel {
  if (ch.timestamps.length <= 1) return ch
  const indices = ch.timestamps.map((_, i) => i)
  indices.sort((a, b) => ch.timestamps[a] - ch.timestamps[b])
  return {
    name: ch.name,
    timestamps: indices.map(i => ch.timestamps[i]),
    values: indices.map(i => ch.values[i]),
  }
}

export function OperationWaveformChart({
  channels, regions, durationSeconds, durationHours, operationId,
  width = 900, height = 400,
}: Props) {
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [hoveredRegion, setHoveredRegion] = useState<EventRegion | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number } | null>(null)

  // Zoom state: [startSeconds, endSeconds] or null = full view
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null)
  // Drag-select state
  const [dragStart, setDragStart] = useState<number | null>(null)
  const [dragCurrent, setDragCurrent] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const W = width - M.left - M.right
  const H = height - M.top - M.bottom

  // Effective time range (zoom or full)
  const tMin = zoomRange ? zoomRange[0] : 0
  const tMax = zoomRange ? zoomRange[1] : durationSeconds

  // Visible channels (sorted + filtered)
  const visible = useMemo(
    () => channels.filter(c => !hidden.has(c.name)).map(sortedChannel),
    [channels, hidden],
  )

  // Per-channel Y ranges computed from visible time range only
  const channelRanges = useMemo(() => {
    const out: Record<string, { min: number; max: number }> = {}
    for (const ch of channels) {
      const vals: number[] = []
      for (let i = 0; i < ch.timestamps.length; i++) {
        if (ch.timestamps[i] >= tMin && ch.timestamps[i] <= tMax && Number.isFinite(ch.values[i])) {
          vals.push(ch.values[i])
        }
      }
      if (vals.length === 0) {
        out[ch.name] = { min: -0.5, max: 0.5 }
      } else {
        let mn = vals[0], mx = vals[0]
        for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v }
        if (mx === mn) { mn -= 0.5; mx += 0.5 }
        out[ch.name] = { min: mn, max: mx }
      }
    }
    return out
  }, [channels, tMin, tMax])

  // X scale: seconds → pixels
  const xScale = useCallback((s: number) =>
    M.left + ((s - tMin) / (tMax - tMin)) * W,
    [tMin, tMax, W],
  )

  // Inverse X: pixels → seconds
  const xInverse = useCallback((px: number) =>
    tMin + ((px - M.left) / W) * (tMax - tMin),
    [tMin, tMax, W],
  )

  // Y scale for a channel
  const yScale = useCallback((name: string, v: number) => {
    const r = channelRanges[name]
    if (!r) return M.top + H / 2
    const frac = clamp((v - r.min) / (r.max - r.min), 0, 1)
    return M.top + H * (1 - frac)
  }, [channelRanges, H])

  // Build SVG paths — break segments at large time gaps to avoid scatter lines
  const paths = useMemo(() => {
    return visible.map((ch, ci) => {
      const segments: string[] = []
      let currentPath = ''
      for (let i = 0; i < ch.timestamps.length; i++) {
        const t = ch.timestamps[i]
        if (t < tMin || t > tMax) continue
        if (!Number.isFinite(ch.values[i])) { // skip NaN / Inf
          if (currentPath) { segments.push(currentPath); currentPath = '' }
          continue
        }
        const x = xScale(t)
        const y = yScale(ch.name, ch.values[i])
        if (!currentPath) {
          currentPath = `M${x.toFixed(1)},${y.toFixed(1)}`
        } else {
          // Break path at gaps > 5% of visible range
          const prevT = ch.timestamps[i - 1]
          if ((t - prevT) / (tMax - tMin) > 0.05) {
            segments.push(currentPath)
            currentPath = `M${x.toFixed(1)},${y.toFixed(1)}`
          } else {
            currentPath += ` L${x.toFixed(1)},${y.toFixed(1)}`
          }
        }
      }
      if (currentPath) segments.push(currentPath)
      return {
        name: ch.name,
        d: segments.join(' '),
        color: PAL[ci % PAL.length],
      }
    })
  }, [visible, channelRanges, tMin, tMax, xScale, yScale])

  // X-axis ticks — adaptive to zoom level
  const xTicks = useMemo(() => {
    const ticks: { s: number; label: string }[] = []
    const range = tMax - tMin
    const rangeH = range / 3600
    const intervalH = rangeH > 100 ? 12
      : rangeH > 48 ? 6
      : rangeH > 12 ? 2
      : rangeH > 4 ? 1
      : rangeH > 1 ? 0.5
      : rangeH > 0.25 ? 0.1
      : rangeH > 0.05 ? 0.02
      : 0.005
    const intervalS = intervalH * 3600
    const startS = Math.ceil(tMin / intervalS) * intervalS
    for (let s = startS; s <= tMax; s += intervalS) {
      const label = range > 7200
        ? `${(s / 3600).toFixed(1)}h`
        : range > 120
          ? `${(s / 60).toFixed(1)}m`
          : `${s.toFixed(0)}s`
      ticks.push({ s, label })
    }
    return ticks
  }, [tMin, tMax])

  const toggle = (name: string) => {
    setHidden(prev => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })
  }

  // ── Zoom / drag handlers ──
  const getSvgX = (e: React.MouseEvent) => {
    if (!svgRef.current) return 0
    const rect = svgRef.current.getBoundingClientRect()
    return e.clientX - rect.left
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    const px = getSvgX(e)
    if (px < M.left || px > M.left + W) return
    setDragStart(px)
    setDragCurrent(px)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragStart != null) {
      setDragCurrent(getSvgX(e))
    }
    if (hoveredRegion) {
      setTooltip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
    }
  }

  const handleMouseUp = () => {
    if (dragStart != null && dragCurrent != null) {
      const s1 = xInverse(Math.min(dragStart, dragCurrent))
      const s2 = xInverse(Math.max(dragStart, dragCurrent))
      const pixDiff = Math.abs(dragStart - dragCurrent)
      if (pixDiff > 10 && s2 - s1 > 0.1) {
        setZoomRange([Math.max(0, s1), Math.min(durationSeconds, s2)])
      }
    }
    setDragStart(null)
    setDragCurrent(null)
  }

  const handleDoubleClick = () => setZoomRange(null)

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const px = getSvgX(e as unknown as React.MouseEvent)
    if (px < M.left || px > M.left + W) return
    const cursorT = xInverse(px)
    const range = tMax - tMin
    const factor = e.deltaY > 0 ? 1.3 : 0.7
    const newRange = Math.min(durationSeconds, Math.max(0.5, range * factor))
    const ratio = (cursorT - tMin) / range
    const newMin = cursorT - ratio * newRange
    const newMax = cursorT + (1 - ratio) * newRange
    if (newRange >= durationSeconds * 0.99) {
      setZoomRange(null)
    } else {
      setZoomRange([Math.max(0, newMin), Math.min(durationSeconds, newMax)])
    }
  }

  const zoomPercent = ((tMax - tMin) / durationSeconds * 100).toFixed(0)

  return (
    <div>
      {/* Legend + zoom toolbar */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6, alignItems: 'center' }}>
        {channels.map((ch, ci) => {
          const isHidden = hidden.has(ch.name)
          return (
            <button
              key={ch.name}
              onClick={() => toggle(ch.name)}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '2px 8px', fontSize: 10, cursor: 'pointer',
                borderRadius: 3, border: '1px solid var(--border)',
                background: isHidden ? 'transparent' : 'rgba(122,162,247,0.08)',
                color: isHidden ? 'var(--muted)' : 'var(--fg)',
                opacity: isHidden ? 0.5 : 1,
                textDecoration: isHidden ? 'line-through' : 'none',
              }}
            >
              <span style={{
                width: 10, height: 3, borderRadius: 1,
                background: PAL[ci % PAL.length],
                display: 'inline-block',
              }} />
              {ch.name.replace(/^(Machine_State_|Axis_Power_|Vibration_|Energy_)/, '')}
            </button>
          )
        })}
        {/* Zoom controls */}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          {zoomRange && (
            <>
              <span style={{ fontSize: 10, color: 'var(--muted)' }}>
                🔍 {zoomPercent}% · {((tMax - tMin) / 60).toFixed(1)}min view
              </span>
              <button
                onClick={() => setZoomRange(null)}
                style={{
                  padding: '2px 8px', fontSize: 10, cursor: 'pointer',
                  borderRadius: 3, border: '1px solid var(--border)',
                  background: 'rgba(247,118,142,0.1)', color: '#f7768e',
                }}
                title="Reset zoom to full operation view"
              >
                Reset Zoom
              </button>
            </>
          )}
          {!zoomRange && (
            <span style={{ fontSize: 10, color: 'var(--muted)' }}>
              Drag to zoom · Scroll to zoom · Double-click to reset
            </span>
          )}
        </span>
      </div>

      {/* Region legend */}
      {regions.length > 0 && (
        <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4 }}>
          <span style={{ background: 'rgba(247,118,142,0.15)', padding: '1px 6px', borderRadius: 3, marginRight: 6 }}>
            ■ pre-stoppage regions
          </span>
          <span>{regions.length} event window{regions.length !== 1 ? 's' : ''} detected in this operation</span>
        </div>
      )}

      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ display: 'block', background: 'var(--bg-card, #1a1b26)', borderRadius: 6, userSelect: 'none' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setHoveredRegion(null); setTooltip(null); setDragStart(null); setDragCurrent(null) }}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
      >
        {/* Grid lines */}
        {xTicks.map(t => (
          <line
            key={t.s}
            x1={xScale(t.s)} y1={M.top} x2={xScale(t.s)} y2={M.top + H}
            stroke="var(--border, #333)" strokeWidth={0.5} opacity={0.4}
          />
        ))}

        {/* Event regions (background rectangles) */}
        {regions.map((r, i) => {
          if (r.end_s < tMin || r.start_s > tMax) return null
          const x1 = xScale(Math.max(tMin, r.start_s))
          const x2 = xScale(Math.min(tMax, r.end_s))
          const isHovered = hoveredRegion === r
          return (
            <rect
              key={i}
              x={x1}
              y={M.top}
              width={Math.max(2, x2 - x1)}
              height={H}
              fill={
                r.feedback_given && r.feedback_action?.toUpperCase() === 'CONFIRM' ? '#9ece6a'   // confirmed — green
                : r.feedback_given && r.feedback_action?.toUpperCase() === 'DISMISS' ? '#7aa2f7' // dismissed — blue
                : (r.label === 'pre_stoppage' || r.label === 'pre_break') ? '#f7768e'            // pre-stoppage — red
                : '#e0af68'                                                                       // other — amber
              }
              opacity={isHovered ? 0.35 : 0.12}
              rx={1}
              style={{ cursor: 'crosshair', pointerEvents: 'all' }}
              onMouseEnter={(e) => {
                setHoveredRegion(r)
                setTooltip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
              }}
              onMouseMove={(e) => {
                setTooltip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
              }}
              onMouseLeave={() => {
                setHoveredRegion(null)
                setTooltip(null)
              }}
            />
          )
        })}

        {/* Signal paths */}
        {paths.map(p => (
          <path
            key={p.name}
            d={p.d}
            fill="none"
            stroke={p.color}
            strokeWidth={1.2}
            opacity={0.85}
            style={{ pointerEvents: 'none' }}
          />
        ))}

        {/* Drag selection overlay */}
        {dragStart != null && dragCurrent != null && Math.abs(dragStart - dragCurrent) > 3 && (
          <rect
            x={Math.min(dragStart, dragCurrent)}
            y={M.top}
            width={Math.abs(dragCurrent - dragStart)}
            height={H}
            fill="rgba(122,162,247,0.15)"
            stroke="rgba(122,162,247,0.6)"
            strokeWidth={1}
            strokeDasharray="4 2"
            rx={2}
            style={{ pointerEvents: 'none' }}
          />
        )}

        {/* X-axis */}
        <line
          x1={M.left} y1={M.top + H} x2={M.left + W} y2={M.top + H}
          stroke="var(--border, #444)" strokeWidth={1}
        />
        {xTicks.map(t => (
          <text
            key={t.s}
            x={xScale(t.s)}
            y={M.top + H + 16}
            textAnchor="middle"
            fontSize={9}
            fill="var(--muted, #888)"
          >
            {t.label}
          </text>
        ))}
        <text
          x={M.left + W / 2}
          y={height - 4}
          textAnchor="middle"
          fontSize={10}
          fill="var(--muted, #888)"
        >
          {zoomRange
            ? `${(tMin / 3600).toFixed(2)}h – ${(tMax / 3600).toFixed(2)}h (${((tMax - tMin) / 60).toFixed(1)} min)`
            : `Time (hours) — ${operationId} — ${durationHours.toFixed(1)}h total`}
        </text>

        {/* Title */}
        <text
          x={M.left}
          y={16}
          fontSize={12}
          fontWeight={700}
          fill="var(--fg, #ccc)"
        >
          Full Operation Waveform: {operationId}
          {zoomRange ? ` (zoomed)` : ''}
        </text>

        {/* Tooltip */}
        {hoveredRegion && tooltip && (() => {
          const fbLabel = hoveredRegion.feedback_given
            ? (hoveredRegion.feedback_action?.toUpperCase() === 'CONFIRM' ? '✓ Confirmed'
              : hoveredRegion.feedback_action?.toUpperCase() === 'DISMISS' ? '✗ Dismissed'
              : '')
            : ''
          const boxH = fbLabel ? 62 : 48
          return (
          <g style={{ pointerEvents: 'none' }}>
            <rect
              x={clamp(tooltip.x + 10, M.left, M.left + W - 200)}
              y={clamp(tooltip.y - 60, M.top, M.top + H - 50)}
              width={190}
              height={boxH}
              rx={4}
              fill="rgba(0,0,0,0.85)"
              stroke="var(--border, #444)"
            />
            <text
              x={clamp(tooltip.x + 16, M.left + 6, M.left + W - 194)}
              y={clamp(tooltip.y - 42, M.top + 18, M.top + H - 32)}
              fontSize={10}
              fill={hoveredRegion.feedback_given && hoveredRegion.feedback_action?.toUpperCase() === 'CONFIRM' ? '#9ece6a'
                : hoveredRegion.feedback_given && hoveredRegion.feedback_action?.toUpperCase() === 'DISMISS' ? '#7aa2f7'
                : '#f7768e'}
              fontWeight={600}
            >
              {(hoveredRegion.label === 'pre_stoppage' || hoveredRegion.label === 'pre_break') ? 'Pre-stoppage' : hoveredRegion.label}
              {hoveredRegion.severity ? ` (${hoveredRegion.severity})` : ''}
              {fbLabel ? ` — ${fbLabel}` : ''}
            </text>
            <text
              x={clamp(tooltip.x + 16, M.left + 6, M.left + W - 194)}
              y={clamp(tooltip.y - 28, M.top + 32, M.top + H - 18)}
              fontSize={9}
              fill="var(--muted, #888)"
            >
              {hoveredRegion.sample_id} · {((hoveredRegion.end_s - hoveredRegion.start_s) / 60).toFixed(0)}min window
            </text>
            <text
              x={clamp(tooltip.x + 16, M.left + 6, M.left + W - 194)}
              y={clamp(tooltip.y - 14, M.top + 46, M.top + H - 4)}
              fontSize={9}
              fill="var(--muted, #888)"
            >
              at {(hoveredRegion.start_s / 3600).toFixed(2)}h–{(hoveredRegion.end_s / 3600).toFixed(2)}h
            </text>
            {fbLabel && hoveredRegion.detected_patterns && hoveredRegion.detected_patterns.length > 0 && (
              <text
                x={clamp(tooltip.x + 16, M.left + 6, M.left + W - 194)}
                y={clamp(tooltip.y, M.top + 60, M.top + H + 10)}
                fontSize={8}
                fill="var(--muted, #888)"
              >
                Patterns: {hoveredRegion.detected_patterns.join(', ')}
              </text>
            )}
          </g>
          )
        })()}
      </svg>
    </div>
  )
}

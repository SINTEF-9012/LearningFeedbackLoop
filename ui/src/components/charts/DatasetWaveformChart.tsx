/**
 * DatasetWaveformChart — Interactive timeseries waveform for the Dataset
 * Explorer, with colored process-annotation regions and rich hover tooltips.
 *
 * Built on the same zoom/pan/LTTB foundation as OperationWaveformChart but
 * with a tooltip tailored for CNC process context (process, sub-process,
 * tool, PGM line, OF annotation, breakage status).
 */
import React, { useMemo, useState, useRef, useCallback, useEffect } from 'react'
import { PAL, clamp } from './chartUtils'

/* ── Region hit-test: binary-search visible regions by timestamp ───── */
function findRegionAtTime(regions: ProcessRegion[], t: number): ProcessRegion | null {
  for (const r of regions) {
    if (t >= r.start_s && t <= r.end_s) return r
  }
  return null
}

export interface WaveformChannel {
  name: string
  timestamps: number[]
  values: number[]
}

export interface ProcessRegion {
  start_s: number
  end_s: number
  label: string          // PROCESS name
  sub_process?: string | null
  tool_id?: number | string | null
  tool_name?: string | null
  pgm_line?: number | null
  of_id?: string | null
  of_annotation?: string | null
  is_breakage?: boolean
  severity?: string
  block_prev?: number | null   // CNC block number at start (col J)
  block_next?: number | null   // CNC block number at end   (col K)
  side?: string | null         // A or B side of the part
}

export interface AnnotationMarker {
  time_s: number
  label: string            // tool/process name
  text: string             // full annotation text from OF column
  pictures: string[]       // e.g. ["P1","P2"]
  category: string         // "breakage" | "wear" | "ok" | "picture" | "note" | "not_measured"
  is_breakage: boolean
  side?: string | null
  block_prev?: number | null
  pgm_line?: number | null
  tool_id?: number | string | null
  tool_name?: string | null
}

// Visual config for annotation categories (SVG-safe characters, no emoji)
const ANNOTATION_STYLES: Record<string, { icon: string; color: string }> = {
  breakage:     { icon: '!',  color: '#f7768e' },
  wear:         { icon: '~',  color: '#e0af68' },
  ok:           { icon: '✓',  color: '#9ece6a' },
  picture:      { icon: 'P',  color: '#7aa2f7' },
  note:         { icon: '?',  color: '#bb9af7' },
  not_measured: { icon: '–',  color: '#888888' },
}

interface Props {
  channels: WaveformChannel[]
  regions: ProcessRegion[]
  annotations?: AnnotationMarker[]
  durationSeconds: number
  durationHours: number
  title?: string
  width?: number
  height?: number
  /** Fires when the user zooms/pans — `null` means full extent. */
  onZoomChange?: (range: [number, number] | null) => void
}

const M = { top: 28, right: 20, bottom: 44, left: 60 }

// Distinct background colors for alternating process regions
const REGION_COLORS = [
  'rgba(122,162,247,0.10)',  // blue
  'rgba(158,206,106,0.10)',  // green
  'rgba(224,175,104,0.10)',  // amber
  'rgba(115,218,202,0.10)',  // teal
  'rgba(187,154,247,0.10)',  // purple
  'rgba(42,195,222,0.10)',   // cyan
]
const BREAKAGE_COLOR = 'rgba(247,118,142,0.25)'

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

export function DatasetWaveformChart({
  channels, regions, annotations = [], durationSeconds, durationHours,
  title, width = 900, height = 420, onZoomChange,
}: Props) {
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [hoveredRegion, setHoveredRegion] = useState<ProcessRegion | null>(null)
  const [hoveredAnnotation, setHoveredAnnotation] = useState<AnnotationMarker | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number } | null>(null)
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null)
  const [dragStart, setDragStart] = useState<number | null>(null)
  const [dragCurrent, setDragCurrent] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const W = width - M.left - M.right
  const H = height - M.top - M.bottom

  const tMin = zoomRange ? zoomRange[0] : 0
  const tMax = zoomRange ? zoomRange[1] : durationSeconds

  const visible = useMemo(
    () => channels.filter(c => !hidden.has(c.name)).map(sortedChannel),
    [channels, hidden],
  )

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

  const xScale = useCallback((s: number) =>
    M.left + ((s - tMin) / (tMax - tMin)) * W,
    [tMin, tMax, W],
  )

  const xInverse = useCallback((px: number) =>
    tMin + ((px - M.left) / W) * (tMax - tMin),
    [tMin, tMax, W],
  )

  const yScale = useCallback((name: string, v: number) => {
    const r = channelRanges[name]
    if (!r) return M.top + H / 2
    const frac = clamp((v - r.min) / (r.max - r.min), 0, 1)
    return M.top + H * (1 - frac)
  }, [channelRanges, H])

  const paths = useMemo(() => {
    return visible.map((ch, ci) => {
      const segments: string[] = []
      let currentPath = ''
      for (let i = 0; i < ch.timestamps.length; i++) {
        const t = ch.timestamps[i]
        if (t < tMin || t > tMax) continue
        if (!Number.isFinite(ch.values[i])) {
          if (currentPath) { segments.push(currentPath); currentPath = '' }
          continue
        }
        const x = xScale(t)
        const y = yScale(ch.name, ch.values[i])
        if (!currentPath) {
          currentPath = `M${x.toFixed(1)},${y.toFixed(1)}`
        } else {
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
      return { name: ch.name, d: segments.join(' '), color: PAL[ci % PAL.length] }
    })
  }, [visible, channelRanges, tMin, tMax, xScale, yScale])

  const xTicks = useMemo(() => {
    const ticks: { s: number; label: string }[] = []
    const range = tMax - tMin

    // Pick a "nice" interval that yields ~6-12 ticks across the view.
    // Candidate intervals in seconds — spans from 1s up to 12h.
    const NICE = [
      1, 2, 5, 10, 15, 30,                          // seconds
      60, 120, 300, 600, 900, 1800,                  // minutes
      3600, 7200, 14400, 21600, 43200,               // hours
    ]
    const targetTicks = 8
    let intervalS = NICE[NICE.length - 1]
    for (const n of NICE) {
      if (range / n <= targetTicks * 1.6) { intervalS = n; break }
    }

    const startS = Math.ceil(tMin / intervalS) * intervalS
    for (let s = startS; s <= tMax; s += intervalS) {
      let label: string
      if (intervalS >= 3600) {
        label = `${(s / 3600).toFixed(1)}h`
      } else if (intervalS >= 60) {
        label = `${(s / 60).toFixed(1)}m`
      } else {
        // Show seconds; include minute prefix when total time > 2 min
        const totalMin = Math.floor(s / 60)
        const sec = (s % 60)
        label = range > 120
          ? `${totalMin}:${sec.toFixed(0).padStart(2, '0')}`
          : `${s.toFixed(0)}s`
      }
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
    const px = getSvgX(e)
    if (dragStart != null) {
      setDragCurrent(px)
      return // don't update tooltip while dragging
    }
    // Hit-test: convert cursor px → time → region or annotation
    if (px >= M.left && px <= M.left + W) {
      const t = xInverse(px)

      // Check annotation markers first (they're narrow — ±6px hit zone)
      let hitAnnotation: AnnotationMarker | null = null
      for (const a of annotations) {
        if (a.time_s < tMin || a.time_s > tMax) continue
        const ax = xScale(a.time_s)
        if (Math.abs(px - ax) <= 6) { hitAnnotation = a; break }
      }

      if (hitAnnotation) {
        setHoveredAnnotation(hitAnnotation)
        setHoveredRegion(null)
        setTooltip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
      } else {
        const hit = findRegionAtTime(regions, t)
        setHoveredAnnotation(null)
        if (hit) {
          setHoveredRegion(hit)
          setTooltip({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
        } else {
          setHoveredRegion(null)
          setTooltip(null)
        }
      }
    } else {
      setHoveredRegion(null)
      setHoveredAnnotation(null)
      setTooltip(null)
    }
  }

  const handleMouseUp = () => {
    if (dragStart != null && dragCurrent != null) {
      const s1 = xInverse(Math.min(dragStart, dragCurrent))
      const s2 = xInverse(Math.max(dragStart, dragCurrent))
      if (Math.abs(dragStart - dragCurrent) > 10 && s2 - s1 > 0.1) {
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

  // Notify parent of zoom changes
  useEffect(() => {
    onZoomChange?.(zoomRange)
  }, [zoomRange]) // eslint-disable-line react-hooks/exhaustive-deps

  const zoomPercent = ((tMax - tMin) / durationSeconds * 100).toFixed(0)

  // Tooltip sizing
  const tooltipLines = useMemo(() => {
    // Annotation tooltip
    if (hoveredAnnotation) {
      const a = hoveredAnnotation
      const style = ANNOTATION_STYLES[a.category] || ANNOTATION_STYLES.note
      const lines: { text: string; color?: string }[] = []
      lines.push({ text: `${style.icon} ${a.label}`, color: style.color })
      if (a.text && a.text !== a.label) lines.push({ text: a.text, color: 'rgba(255,255,255,0.9)' })
      if (a.pictures.length > 0) lines.push({ text: `Pictures: ${a.pictures.join(', ')}`, color: '#7aa2f7' })
      lines.push({ text: `Category: ${a.category}`, color: style.color })
      if (a.tool_id) lines.push({ text: `Tool: ${a.tool_name || '—'} (ID ${a.tool_id})` })
      if (a.pgm_line != null) lines.push({ text: `PGM Line: ${a.pgm_line}` })
      if (a.side) lines.push({ text: `Side: ${a.side}` })
      lines.push({ text: `Time: ${(a.time_s / 3600).toFixed(2)}h` })
      return lines
    }
    // Region tooltip
    if (!hoveredRegion) return []
    const lines: { text: string; color?: string }[] = []
    const sideTag = hoveredRegion.side ? `[${hoveredRegion.side}-side] ` : ''
    lines.push({ text: `${sideTag}${hoveredRegion.label || '(unnamed process)'}`, color: hoveredRegion.is_breakage ? '#f7768e' : '#7aa2f7' })
    if (hoveredRegion.sub_process) lines.push({ text: `Sub-process: ${hoveredRegion.sub_process}` })
    if (hoveredRegion.tool_id) lines.push({ text: `Tool: ${hoveredRegion.tool_name || '—'} (ID ${hoveredRegion.tool_id})` })
    if (hoveredRegion.pgm_line != null) lines.push({ text: `PGM Line: ${hoveredRegion.pgm_line}` })
    if (hoveredRegion.block_prev != null || hoveredRegion.block_next != null) {
      lines.push({ text: `Block range: ${hoveredRegion.block_prev ?? '?'} → ${hoveredRegion.block_next ?? '?'}`, color: '#9ece6a' })
    }
    if (hoveredRegion.of_id) lines.push({ text: `OF: ${hoveredRegion.of_id}` })
    if (hoveredRegion.of_annotation) lines.push({ text: `Annotation: ${hoveredRegion.of_annotation}`, color: '#e0af68' })
    const dur = hoveredRegion.end_s - hoveredRegion.start_s
    lines.push({ text: `Duration: ${dur > 3600 ? `${(dur / 3600).toFixed(1)}h` : dur > 60 ? `${(dur / 60).toFixed(1)}m` : `${dur.toFixed(0)}s`}` })
    lines.push({ text: `${(hoveredRegion.start_s / 3600).toFixed(2)}h – ${(hoveredRegion.end_s / 3600).toFixed(2)}h` })
    return lines
  }, [hoveredRegion, hoveredAnnotation])

  const ttWidth = 290
  const ttLineH = 16
  const ttPadY = 10
  const ttH = tooltipLines.length * ttLineH + ttPadY * 2

  return (
    <div style={{ position: 'relative' }}>
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
              {ch.name.replace(/^(Monit_chatter_detection_|Cnc_Override_|Axis_FeedRate_)/, '').replace(/_/g, ' ')}
            </button>
          )
        })}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          {zoomRange ? (
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
              >
                Reset Zoom
              </button>
            </>
          ) : (
            <span style={{ fontSize: 10, color: 'var(--muted)' }}>
              Drag to zoom · Scroll to zoom · Double-click to reset
            </span>
          )}
        </span>
      </div>

      {/* Region legend */}
      {regions.length > 0 && (
        <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 4, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ background: 'rgba(122,162,247,0.15)', padding: '1px 6px', borderRadius: 3 }}>
            ■ Process regions
          </span>
          <span style={{ background: 'rgba(247,118,142,0.25)', padding: '1px 6px', borderRadius: 3 }}>
            ■ Breakage annotation
          </span>
          {annotations.length > 0 && (
            <>
              <span style={{ color: '#f7768e', padding: '1px 6px' }}>! Breakage</span>
              <span style={{ color: '#e0af68', padding: '1px 6px' }}>~ Wear</span>
              <span style={{ color: '#9ece6a', padding: '1px 6px' }}>✓ OK</span>
              <span style={{ color: '#7aa2f7', padding: '1px 6px' }}>P Picture</span>
              <span style={{ color: '#888888', padding: '1px 6px' }}>– Not measured</span>
            </>
          )}
          <span>{regions.length} region{regions.length !== 1 ? 's' : ''} · {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}</span>
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
        onMouseLeave={() => { setHoveredRegion(null); setHoveredAnnotation(null); setTooltip(null); setDragStart(null); setDragCurrent(null) }}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
      >
        {/* Grid */}
        {xTicks.map(t => (
          <line key={t.s} x1={xScale(t.s)} y1={M.top} x2={xScale(t.s)} y2={M.top + H}
            stroke="var(--border, #333)" strokeWidth={0.5} opacity={0.4} />
        ))}

        {/* Process regions (pointer events handled at SVG level via hit-test) */}
        {regions.map((r, i) => {
          if (r.end_s < tMin || r.start_s > tMax) return null
          const x1 = xScale(Math.max(tMin, r.start_s))
          const x2 = xScale(Math.min(tMax, r.end_s))
          const isHovered = hoveredRegion === r
          const fill = r.is_breakage ? BREAKAGE_COLOR : REGION_COLORS[i % REGION_COLORS.length]
          return (
            <rect
              key={i}
              x={x1} y={M.top}
              width={Math.max(2, x2 - x1)} height={H}
              fill={fill}
              opacity={isHovered ? 0.8 : 1}
              rx={1}
              style={{ pointerEvents: 'none' }}
            />
          )
        })}

        {/* Process boundary labels (top of chart, when zoomed enough) */}
        {regions.map((r, i) => {
          if (r.end_s < tMin || r.start_s > tMax) return null
          const x1 = xScale(Math.max(tMin, r.start_s))
          const x2 = xScale(Math.min(tMax, r.end_s))
          const regionW = x2 - x1
          if (regionW < 30) return null
          const label = r.label.length > 25 ? r.label.substring(0, 25) + '…' : r.label
          return (
            <text key={`lbl-${i}`} x={x1 + 3} y={M.top + 12} fontSize={8}
              fill={r.is_breakage ? '#f7768e' : 'var(--muted, #666)'} opacity={0.8}
              style={{ pointerEvents: 'none' }}>
              {label}
            </text>
          )
        })}

        {/* Signal paths */}
        {paths.map(p => (
          <path key={p.name} d={p.d} fill="none" stroke={p.color}
            strokeWidth={1.2} opacity={0.85} style={{ pointerEvents: 'none' }} />
        ))}

        {/* Annotation markers — vertical lines + category icons */}
        {annotations.map((a, i) => {
          if (a.time_s < tMin || a.time_s > tMax) return null
          const ax = xScale(a.time_s)
          const style = ANNOTATION_STYLES[a.category] || ANNOTATION_STYLES.note
          const isHovered = hoveredAnnotation === a
          return (
            <g key={`ann-${i}`} style={{ pointerEvents: 'none' }}>
              {/* Vertical dashed line */}
              <line
                x1={ax} y1={M.top} x2={ax} y2={M.top + H}
                stroke={style.color}
                strokeWidth={isHovered ? 2 : 1.2}
                strokeDasharray="4 3"
                opacity={isHovered ? 1 : 0.7}
              />
              {/* Icon circle at top */}
              <circle
                cx={ax} cy={M.top - 2}
                r={isHovered ? 9 : 7}
                fill="rgba(0,0,0,0.85)"
                stroke={style.color}
                strokeWidth={1.5}
              />
              <text
                x={ax} y={M.top + 2}
                textAnchor="middle" fontSize={isHovered ? 11 : 9}
                fill={style.color}
                style={{ pointerEvents: 'none' }}
              >
                {style.icon}
              </text>
              {/* Picture label below icon when zoomed enough */}
              {a.pictures.length > 0 && (
                <text
                  x={ax} y={M.top + H + 28}
                  textAnchor="middle" fontSize={8}
                  fill={style.color} fontWeight={600}
                  style={{ pointerEvents: 'none' }}
                >
                  {a.pictures.join(',')}
                </text>
              )}
            </g>
          )
        })}

        {/* Drag selection overlay */}
        {dragStart != null && dragCurrent != null && Math.abs(dragStart - dragCurrent) > 3 && (
          <rect
            x={Math.min(dragStart, dragCurrent)} y={M.top}
            width={Math.abs(dragCurrent - dragStart)} height={H}
            fill="rgba(122,162,247,0.15)" stroke="rgba(122,162,247,0.6)"
            strokeWidth={1} strokeDasharray="4 2" rx={2}
            style={{ pointerEvents: 'none' }} />
        )}

        {/* X-axis */}
        <line x1={M.left} y1={M.top + H} x2={M.left + W} y2={M.top + H}
          stroke="var(--border, #444)" strokeWidth={1} />
        {xTicks.map(t => (
          <text key={t.s} x={xScale(t.s)} y={M.top + H + 16}
            textAnchor="middle" fontSize={9} fill="var(--muted, #888)">
            {t.label}
          </text>
        ))}
        <text x={M.left + W / 2} y={height - 4} textAnchor="middle"
          fontSize={10} fill="var(--muted, #888)">
          {zoomRange
            ? `${(tMin / 3600).toFixed(2)}h – ${(tMax / 3600).toFixed(2)}h (${((tMax - tMin) / 60).toFixed(1)} min)`
            : `Time (hours) — ${durationHours.toFixed(1)}h total`}
        </text>

        {/* Title */}
        <text x={M.left} y={16} fontSize={12} fontWeight={700} fill="var(--fg, #ccc)">
          {title || 'Dataset Waveform'}
          {zoomRange ? ' (zoomed)' : ''}
        </text>

      </svg>

      {/* Tooltip — rendered as HTML overlay for reliable positioning */}
      {(hoveredRegion || hoveredAnnotation) && tooltip && (
        <div
          style={{
            position: 'absolute',
            left: clamp(tooltip.x + 16, 0, width - ttWidth - 20),
            top: clamp(tooltip.y - ttH / 2, 0, height - ttH - 4),
            width: ttWidth,
            background: 'rgba(0,0,0,0.94)',
            border: `1px solid ${hoveredAnnotation
              ? (ANNOTATION_STYLES[hoveredAnnotation.category] || ANNOTATION_STYLES.note).color
              : hoveredRegion?.is_breakage ? '#f7768e' : 'rgba(255,255,255,0.12)'}`,
            borderRadius: 5,
            padding: '8px 10px',
            pointerEvents: 'none',
            zIndex: 10,
            boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
          }}
        >
          {tooltipLines.map((line, li) => (
            <div
              key={li}
              style={{
                fontSize: li === 0 ? 11 : 10,
                fontWeight: li === 0 ? 700 : 400,
                lineHeight: '16px',
                color: line.color || 'rgba(255,255,255,0.7)',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {line.text}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

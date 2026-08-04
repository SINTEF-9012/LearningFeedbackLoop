/**
 * FaultIndicatorPanel – Live fault-type likelihood gauges.
 *
 * Shows the four machining fault types with:
 * - A main gauge bar (0–1) colour-coded by severity
 * - Contributing signal breakdown (expandable)
 * - Dominant fault highlight
 * - Time-series mini-chart of fault scores (last N windows)
 *
 * Data source: `fault_indicators` field on each InferencePoint from the WS.
 */
import React, { useMemo, useState, useRef, useEffect } from 'react'
import { useInferenceStore, type FaultIndicators, type FaultIndicator } from '../state/inferenceStore'

/* ── Fault type metadata ─────────────────────────────────── */

type FaultMeta = {
  key: keyof Omit<FaultIndicators, 'dominant_fault'>
  label: string
  icon: string
  color: string
  signalLabels: Record<string, string>
  description: string
}

const FAULT_TYPES: FaultMeta[] = [
  {
    key: 'tool_breakage',
    label: 'High-Frequency Burst',
    icon: 'TB',
    color: '#f7768e',
    description: 'Sudden burst with periodicity loss',
    signalLabels: {
      hf_energy_burst: 'HF Energy Burst',
      impulse_severity: 'Impulse Severity',
      kurtosis_excess: 'Kurtosis Excess',
      periodicity_loss: 'Periodicity Loss',
    },
  },
  {
    key: 'chatter',
    label: 'Vibration Modulation',
    icon: 'CH',
    color: '#ff9e64',
    description: 'Modulated vibration with rising amplitude',
    signalLabels: {
      modulation: 'Modulation Depth',
      amplitude_growth: 'Amplitude Growth',
      harmonic_energy: 'Harmonic Energy',
    },
  },
  {
    key: 'chip_adhesion',
    label: 'Tooth-Passing Irregularity',
    icon: 'CA',
    color: '#bb9af7',
    description: 'Irregular tooth-passing pattern',
    signalLabels: {
      harmonic_irregularity: 'Harmonic Irregularity',
      tooth_passing_variance: 'Tooth-Passing Variance',
    },
  },
  {
    key: 'workpiece_slip',
    label: 'Spindle-Order Shift',
    icon: 'WS',
    color: '#7dcfff',
    description: 'Shift near spindle-order energy',
    signalLabels: {
      spindle_order_shift: 'Spindle Order Shift',
      phase_shift: 'Phase Shift',
    },
  },
]

/* ── Score → severity ─────────────────────────────────────── */

function scoreSeverity(s: number): { label: string; bg: string } {
  if (s >= 0.7) return { label: 'HIGH', bg: 'rgba(247, 118, 142, 0.15)' }
  if (s >= 0.4) return { label: 'MODERATE', bg: 'rgba(255, 158, 100, 0.10)' }
  return { label: 'LOW', bg: 'transparent' }
}

function scoreColor(s: number, baseColor: string): string {
  if (s >= 0.7) return '#f7768e'
  if (s >= 0.4) return baseColor
  return 'var(--muted)'
}

/* ── Mini sparkline for fault score history ────────────────── */

function FaultSparkline({ data, color, height = 32 }: { data: number[]; color: string; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvasRef.current
    if (!c || data.length < 2) return
    const ctx = c.getContext('2d')
    if (!ctx) return

    const w = c.clientWidth
    const dpr = window.devicePixelRatio || 1
    c.width = w * dpr
    c.height = height * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, w, height)

    // Threshold lines
    for (const thresh of [0.4, 0.7]) {
      const y = height - thresh * (height - 4) - 2
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.lineWidth = 0.5
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(w, y)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Score line
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w
      const y = height - (Math.min(1, Math.max(0, data[i]))) * (height - 4) - 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Fill under area
    const lastX = w
    const lastY = height - (Math.min(1, Math.max(0, data[data.length - 1]))) * (height - 4) - 2
    ctx.lineTo(lastX, height)
    ctx.lineTo(0, height)
    ctx.closePath()
    ctx.fillStyle = color.replace(')', ', 0.08)').replace('rgb', 'rgba')
    ctx.fill()
  }, [data, color, height])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height,
        borderRadius: 4,
        background: 'rgba(0,0,0,0.15)',
      }}
    />
  )
}

/* ── Signal bar row ───────────────────────────────────────── */

function SignalBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100))
  return (
    <div className="contextBarRow" style={{ marginBottom: 2 }}>
      <span className="contextBarLabel" style={{ width: 130, fontSize: 10 }}>{label}</span>
      <div className="contextBarTrack" style={{ height: 6 }}>
        <div
          className="contextBarFill"
          style={{
            width: `${pct}%`,
            background: value > 0.7 ? '#f7768e' : value > 0.4 ? color : 'var(--muted)',
            height: 6,
            borderRadius: 3,
          }}
        />
      </div>
      <span className="contextBarValue" style={{ width: 40, fontSize: 9 }}>
        {value.toFixed(2)}
      </span>
    </div>
  )
}

/* ── Fault card ───────────────────────────────────────────── */

function FaultCard({
  meta,
  indicator,
  history,
  isDominant,
}: {
  meta: FaultMeta
  indicator: FaultIndicator | undefined
  history: number[]
  isDominant: boolean
}) {
  const [expanded, setExpanded] = useState(false)
  const score = indicator?.score ?? 0
  const sev = scoreSeverity(score)
  const barColor = scoreColor(score, meta.color)
  const pct = Math.min(100, Math.max(0, score * 100))

  return (
    <div
      className="faultCard"
      style={{
        background: isDominant ? sev.bg : 'transparent',
        borderColor: isDominant ? barColor : 'var(--border)',
      }}
    >
      {/* Header row */}
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
        onClick={() => setExpanded((v) => !v)}
      >
        <span style={{ fontSize: 10, fontWeight: 700, width: 22, height: 22, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', borderRadius: 4, background: 'rgba(255,255,255,0.06)', color: 'var(--muted)', letterSpacing: 0.3 }}>{meta.icon}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 650, fontSize: 13, color: isDominant ? barColor : 'var(--text)' }}>
              {meta.label}
            </span>
            {isDominant && (
              <span
                style={{
                  fontSize: 9,
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  padding: '1px 6px',
                  borderRadius: 999,
                  background: barColor + '22',
                  color: barColor,
                  letterSpacing: 0.5,
                }}
              >
                strongest
              </span>
            )}
          </div>
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 1 }}>{meta.description}</div>
        </div>
        <div style={{ textAlign: 'right', minWidth: 55 }}>
          <div style={{ fontFamily: 'monospace', fontSize: 16, fontWeight: 700, color: barColor }}>
            {(score * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)' }}>{sev.label}</div>
        </div>
      </div>

      {/* Main gauge */}
      <div style={{ marginTop: 6 }}>
        <div
          style={{
            height: 6,
            borderRadius: 3,
            background: 'rgba(255,255,255,0.06)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: '100%',
              borderRadius: 3,
              background: barColor,
              transition: 'width 0.3s ease, background 0.3s ease',
            }}
          />
        </div>
      </div>

      {/* Sparkline history */}
      {history.length > 1 && (
        <div style={{ marginTop: 6 }}>
          <FaultSparkline data={history} color={meta.color} height={28} />
        </div>
      )}

      {/* Expanded signals detail */}
      {expanded && indicator?.signals && (
        <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Contributing Signals
          </div>
          {Object.entries(indicator.signals).map(([key, value]) => (
            <SignalBar
              key={key}
              label={meta.signalLabels[key] || key}
              value={value}
              color={meta.color}
            />
          ))}
        </div>
      )}

      {!expanded && (
        <div
          style={{ fontSize: 10, color: 'var(--accent)', cursor: 'pointer', marginTop: 4, textAlign: 'center' }}
          onClick={() => setExpanded(true)}
        >
          Show signals
        </div>
      )}
    </div>
  )
}

/* ── Main panel ───────────────────────────────────────────── */

export function FaultIndicatorPanel() {
  const points = useInferenceStore((s) => s.points)

  // Latest indicators
  const latest = points.length > 0 ? points[points.length - 1].fault_indicators : undefined

  // History per fault type (last 60 points)
  const histories = useMemo(() => {
    const tail = points.slice(-60)
    const out: Record<string, number[]> = {}
    for (const ft of FAULT_TYPES) {
      out[ft.key] = tail.map((p) => {
        const fi = p.fault_indicators
        if (!fi) return 0
        const ind = fi[ft.key] as FaultIndicator | undefined
        return ind?.score ?? 0
      })
    }
    return out
  }, [points])

  const dominant = latest?.dominant_fault ?? null

  // Overall status
  const maxScore = FAULT_TYPES.reduce(
    (mx, ft) => Math.max(mx, (latest?.[ft.key] as FaultIndicator | undefined)?.score ?? 0),
    0,
  )
  const statusColor = maxScore >= 0.7 ? 'var(--danger)' : maxScore >= 0.4 ? '#f0a050' : 'var(--ok)'
  const statusText = maxScore >= 0.7 ? 'CHECK OBSERVATION' : maxScore >= 0.4 ? 'ELEVATED' : 'NORMAL'

  return (
    <div className="panelCard" style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div>
          <h4 style={{ margin: 0, fontSize: 13, color: 'var(--accent)' }}>
            Observation Indicators — Classical Model
          </h4>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: statusColor,
              display: 'inline-block',
              boxShadow: maxScore >= 0.4 ? `0 0 6px ${statusColor}` : 'none',
            }}
          />
          <span style={{ fontSize: 11, fontWeight: 700, color: statusColor, letterSpacing: 0.5 }}>
            {statusText}
          </span>
          <span style={{ fontSize: 10, color: 'var(--muted)' }}>
            {points.length > 0 ? `${points.length} windows` : 'waiting…'}
          </span>
        </div>
      </div>

      {!latest ? (
        <div style={{ color: 'var(--muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
          Waiting for inference data to compute observation indicators…
        </div>
      ) : (
        <div className="faultGrid">
          {FAULT_TYPES.map((ft) => (
            <FaultCard
              key={ft.key}
              meta={ft}
              indicator={latest[ft.key] as FaultIndicator | undefined}
              history={histories[ft.key] || []}
              isDominant={dominant === ft.key}
            />
          ))}
        </div>
      )}
    </div>
  )
}

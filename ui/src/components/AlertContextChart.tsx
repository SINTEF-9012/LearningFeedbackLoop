/**
 * AlertContextChart – Visualises the sensor data context that triggered an alert.
 *
 * Two sections:
 * 1. Feature bar chart – horizontal bars for each of the 17 CNC features,
 *    colour-coded by anomaly severity. Model signals shown at the top.
 * 2. Stream sparklines – mini line charts of the raw stream channels around
 *    the alert's _streamIndex (±500 samples).
 */
import React, { useMemo, useRef, useEffect, useState } from 'react'
import type { SignificantEventAlert } from '../state/alertsStore'
import { useInferenceStore } from '../state/inferenceStore'
import { useStreamStore } from '../state/streamStore'
import { HarmonicContextSnapshot } from './HarmonicContextSnapshot'
import { alertModelSourceExplanation } from '../utils/alerts'

/* ── Feature metadata ─────────────────────────────────── */

const FEATURE_LABELS: Record<string, string> = {
  power_spindle_mean: 'Spindle power (mean)',
  power_spindle_max: 'Spindle power (max)',
  power_spindle_std: 'Spindle power (σ)',
  power_x_mean: 'X-axis power',
  power_y_mean: 'Y-axis power',
  power_y_max: 'Y-axis power (max)',
  power_z_mean: 'Z-axis power',
  chatter_ratio: 'Cross-axis vibration ratio',
  vib_severity_mean: 'Vibration severity (mean)',
  vib_severity_max: 'Vibration severity (max)',
  vib_severity_x_mean: 'Vibration severity X (mean)',
  vib_severity_x_max: 'Vibration severity X (max)',
  vib_severity_y_mean: 'Vibration severity Y (mean)',
  vib_severity_y_max: 'Vibration severity Y (max)',
  chatter_amp_x_max: 'Modulation amplitude X',
  chatter_amp_y_max: 'Modulation amplitude Y',
  chatter_freq_max: 'Modulation frequency (max)',
  power_active_mean: 'Active power (mean)',
  power_active_std: 'Active power (σ)',
  power_factor_mean: 'Power factor (mean)',
  feed_rate_mean: 'Feed rate (mean)',
  spindle_speed_mean: 'Spindle speed (mean)',
  temp_mean: 'Temperature (mean)',
  temp_head_mean: 'Temperature head (mean)',
  tool_changes: 'Tool changes',
  // Physics-based fault features
  hf_energy_ratio: 'HF energy ratio',
  impulse_crest_factor: 'Impulse crest factor',
  kurtosis_max: 'Kurtosis (max)',
  periodicity_strength: 'Periodicity strength',
  modulation_depth: 'Modulation depth',
  vib_amplitude_growth: 'Vib amplitude growth',
  tp_harmonic_energy: 'Tooth-passing harmonic energy',
  harmonic_amplitude_cv: 'Harmonic amplitude CV',
  tp_amplitude_variance: 'Tooth-passing amplitude var',
  spindle_order_amplitude: 'Spindle order amplitude',
  spindle_phase_shift: 'Spindle phase shift',
}

const MODEL_KEYS: Record<string, string> = {
  anomaly_detector_score: 'Anomaly score',
  model_confidence: 'Model confidence',
  breakage_prediction: 'Heuristic breakage risk',
  tool_wear_estimate: 'Heuristic tool wear estimate',
  harmonic_context_score: 'Harmonic context',
}

/** Typical normal-range maxima for feature normalisation (rough heuristic). */
const NORMAL_MAX: Record<string, number> = {
  power_spindle_mean: 50,
  power_spindle_max: 80,
  power_spindle_std: 15,
  power_x_mean: 30,
  power_y_mean: 30,
  power_y_max: 50,
  power_z_mean: 30,
  chatter_ratio: 0.05,
  vib_severity_mean: 2,
  vib_severity_max: 5,
  vib_severity_x_mean: 2,
  vib_severity_x_max: 5,
  vib_severity_y_mean: 2,
  vib_severity_y_max: 5,
  chatter_amp_x_max: 0.3,
  chatter_amp_y_max: 0.3,
  chatter_freq_max: 500,
  power_active_mean: 3000,
  power_active_std: 400,
  power_factor_mean: 0.95,
  feed_rate_mean: 800,
  spindle_speed_mean: 8000,
  temp_mean: 40,
  temp_head_mean: 40,
  tool_changes: 2,
  // Physics-based fault features
  hf_energy_ratio: 0.1,
  impulse_crest_factor: 4.0,
  kurtosis_max: 3.0,
  periodicity_strength: 0.8,
  modulation_depth: 0.3,
  vib_amplitude_growth: 1.5,
  tp_harmonic_energy: 0.03,
  harmonic_amplitude_cv: 0.3,
  tp_amplitude_variance: 0.2,
  spindle_order_amplitude: 0.2,
  spindle_phase_shift: 0.5,
}

/** Thresholds for anomalous colour (ratio above normal max). */
const WARN_RATIO = 1.3
const CRIT_RATIO = 2.0

function featureColor(key: string, value: number): string {
  const max = NORMAL_MAX[key]
  if (max == null || max === 0) return 'var(--accent)'
  // Power factor is inverted: LOW is bad
  if (key === 'power_factor_mean') {
    if (value < 0.7) return 'var(--danger)'
    if (value < 0.85) return 'var(--accent)'
    return 'var(--ok)'
  }
  const ratio = value / max
  if (ratio > CRIT_RATIO) return 'var(--danger)'
  if (ratio > WARN_RATIO) return '#f0a050'
  return 'var(--ok)'
}

function barWidth(key: string, value: number): number {
  const max = NORMAL_MAX[key]
  if (max == null || max === 0) return Math.min(value * 100, 100)
  return Math.min((value / (max * CRIT_RATIO)) * 100, 100)
}

function asMetricsRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  return value as Record<string, unknown>
}

/* ── Stream sparkline ─────────────────────────────────── */
// Default to 500 samples each side (~0.5 s at 1 kHz, ~500 s at 1 Hz).
// The component derives a sensible window from the alert's fs when available.
const DEFAULT_SPARKLINE_WINDOW = 500

const SPARK_H = 40
const SPARK_W = 280

function Sparkline({ data, label }: { data: number[]; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvasRef.current
    if (!c || data.length === 0) return
    const ctx = c.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    c.width = SPARK_W * dpr
    c.height = SPARK_H * dpr
    ctx.scale(dpr, dpr)
    c.style.width = `${SPARK_W}px`
    c.style.height = `${SPARK_H}px`

    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1

    ctx.clearRect(0, 0, SPARK_W, SPARK_H)
    ctx.strokeStyle = 'var(--accent)'
    ctx.lineWidth = 1.2
    ctx.beginPath()
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * SPARK_W
      const y = SPARK_H - ((data[i] - min) / range) * (SPARK_H - 4) - 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Centre mark (alert position)
    const cx = SPARK_W / 2
    ctx.strokeStyle = 'var(--danger)'
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(cx, 0)
    ctx.lineTo(cx, SPARK_H)
    ctx.stroke()
    ctx.setLineDash([])
  }, [data])

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 2 }}>{label}</div>
      <canvas ref={canvasRef} style={{ background: 'var(--bg)', borderRadius: 4, border: '1px solid var(--border)' }} />
    </div>
  )
}

/* ── Main component ───────────────────────────────────── */

interface AlertContextChartProps {
  alert: SignificantEventAlert
  showStreamContext?: boolean
  compact?: boolean
}

export function AlertContextChart({ alert, showStreamContext = true, compact = false }: AlertContextChartProps) {
  const [showFeatures, setShowFeatures] = useState(!compact)
  const [showStream, setShowStream] = useState(false)

  const inferencePoints = useInferenceStore((s) => s.points)

  const nearestInferenceMetrics = useMemo(() => {
    if (inferencePoints.length === 0) return undefined

    const targetIndex = alert._streamIndex
    const latest = inferencePoints[inferencePoints.length - 1]

    if (typeof targetIndex !== 'number') {
      return {
        ...latest.features,
        anomaly_detector_score: latest.scores.ensemble,
        harmonic_context_score: latest.scores.harmonic_context_score,
        fs: latest.fs,
        harmonic_context_weights: latest.harmonic_context_weights,
        harmonic_feature_labels: latest.harmonic_feature_labels,
        harmonic_values: latest.harmonic_values,
      }
    }

    let best = latest
    let bestDelta = Number.POSITIVE_INFINITY
    for (const point of inferencePoints) {
      const delta = Math.abs(point.i_center - targetIndex)
      if (delta < bestDelta) {
        best = point
        bestDelta = delta
      }
    }

    const maxAcceptableDelta = Math.max(64, Math.round(best.fs * 30))
    if (bestDelta > maxAcceptableDelta) return undefined

    return {
      ...best.features,
      anomaly_detector_score: best.scores.ensemble,
      harmonic_context_score: best.scores.harmonic_context_score,
      fs: best.fs,
      harmonic_context_weights: best.harmonic_context_weights,
      harmonic_feature_labels: best.harmonic_feature_labels,
      harmonic_values: best.harmonic_values,
    }
  }, [alert._streamIndex, inferencePoints])

  const metrics = useMemo<Record<string, unknown> | undefined>(() => {
    const alertMetrics = asMetricsRecord(alert.metrics)
    if (!nearestInferenceMetrics) return alertMetrics
    return {
      ...nearestInferenceMetrics,
      ...(alertMetrics || {}),
    }
  }, [alert.metrics, nearestInferenceMetrics])
  const modelSourceExplanation = useMemo(() => alertModelSourceExplanation(alert), [alert])

  /* ── Parse model signals ── */
  const modelSignals = useMemo(() => {
    if (!metrics) return []
    return Object.entries(MODEL_KEYS)
      .map(([key, label]) => {
        const v = metrics[key]
        return typeof v === 'number' ? { key, label, value: v } : null
      })
      .filter(Boolean) as { key: string; label: string; value: number }[]
  }, [metrics])

  /* ── Parse CNC features ── */
  const features = useMemo(() => {
    if (!metrics) return []
    return Object.entries(FEATURE_LABELS)
      .map(([key, label]) => {
        const v = metrics[key]
        return typeof v === 'number' ? { key, label, value: v } : null
      })
      .filter(Boolean) as { key: string; label: string; value: number }[]
  }, [metrics])

  /* ── Hz-aware sparkline window ── */
  // Aim for ~1 second of context each side; fall back to DEFAULT_SPARKLINE_WINDOW
  const sparklineWindow = useMemo(() => {
    const fs = (metrics as any)?.sample_rate_hz ?? (metrics as any)?.fs
    if (typeof fs === 'number' && fs > 0) return Math.max(32, Math.round(fs * 1.0))
    return DEFAULT_SPARKLINE_WINDOW
  }, [metrics])

  /* ── Stream window data ── */
  const streamData = useMemo(() => {
    const idx = alert._streamIndex
    if (idx == null || !showStream) return null
    const store = useStreamStore.getState()
    if (!store.getWindowSamples) return null
    try {
      const i0 = Math.max(0, idx - sparklineWindow)
      const i1 = idx + sparklineWindow
      return store.getWindowSamples(i0, i1)
    } catch {
      return null
    }
  }, [alert._streamIndex, showStream, sparklineWindow])

  if (!metrics || (modelSignals.length === 0 && features.length === 0)) {
    return <div style={{ color: 'var(--muted)', fontSize: 12, padding: 8 }}>No feature data available for this alert.</div>
  }

  return (
    <div className="contextChart">
      {/* ── Model signals ── */}
      {modelSignals.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 2 }}>
            Model And Heuristic Signals (indicative)
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 6 }}>
            Statistical estimates — verify before acting
          </div>
          {modelSourceExplanation && (
            <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 6 }}>
              {modelSourceExplanation}
            </div>
          )}
          {modelSignals.map(s => (
            <div key={s.key} className="contextBarRow">
              <span className="contextBarLabel">{s.label}</span>
              <div className="contextBarTrack">
                <div
                  className="contextBarFill"
                  style={{
                    width: `${Math.min(s.value * 100, 100)}%`,
                    background:
                      s.key === 'model_confidence'
                        ? 'var(--accent)'
                        : s.value > 0.7
                        ? 'var(--danger)'
                        : s.value > 0.4
                        ? '#f0a050'
                        : 'var(--ok)',
                  }}
                />
              </div>
              <span className="contextBarValue">{s.value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Harmonic context weights ── */}
      {(() => {
        const hcWeights = metrics?.harmonic_context_weights as number[] | undefined
        const hcLabels = metrics?.harmonic_feature_labels as string[] | undefined
        const hcValues = metrics?.harmonic_values as number[] | undefined
        const hcScore = metrics?.harmonic_context_score as number | undefined
        const hasWeights = Array.isArray(hcWeights) && hcWeights.length > 0
        const hasValues = Array.isArray(hcValues) && hcValues.length > 0
        if (!hasWeights && !hasValues) return null
        return (
          <div style={{ marginTop: 8, marginBottom: 8 }}>
            <HarmonicContextSnapshot
              weights={hcWeights}
              labels={hcLabels}
              values={hcValues}
              score={typeof hcScore === 'number' ? hcScore : undefined}
              compact={compact}
              title="Harmonic context"
              subtitle="Nearest live inference window aligned to this alert."
            />
          </div>
        )
      })()}

      {/* ── CNC features toggle ── */}
      <button
        className="contextToggle"
        onClick={() => setShowFeatures(v => !v)}
      >
        {showFeatures ? '▾' : '▸'} CNC Features ({features.length})
      </button>

      {showFeatures && features.length > 0 && (
        <div style={{ marginTop: 4 }}>
          {features.map(f => (
            <div key={f.key} className="contextBarRow">
              <span className="contextBarLabel">{f.label}</span>
              <div className="contextBarTrack">
                <div
                  className="contextBarFill"
                  style={{
                    width: `${barWidth(f.key, f.value)}%`,
                    background: featureColor(f.key, f.value),
                  }}
                />
              </div>
              <span className="contextBarValue">
                {f.value < 0.01 && f.value !== 0
                  ? f.value.toExponential(1)
                  : f.value >= 1000
                  ? f.value.toFixed(0)
                  : f.value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ── Stream context ── */}
      {showStreamContext && alert._streamIndex != null && (
        <>
          <button
            className="contextToggle"
            onClick={() => setShowStream(v => !v)}
            style={{ marginTop: 10 }}
          >
            {showStream ? '▾' : '▸'} Stream Context (±{sparklineWindow} samples around alert)
          </button>

          {showStream && streamData && streamData.channels.length > 0 && (
            <div style={{ marginTop: 6 }}>
              {streamData.channels.map(ch => {
                const samples = streamData.samples[ch]
                if (!samples || samples.length === 0) return null
                return <Sparkline key={ch} data={samples} label={ch} />
              })}
            </div>
          )}

          {showStream && (!streamData || streamData.channels.length === 0) && (
            <div style={{ fontSize: 11, color: 'var(--muted)', padding: '4px 0' }}>
              Stream data not available for this window.
            </div>
          )}
        </>
      )}
    </div>
  )
}

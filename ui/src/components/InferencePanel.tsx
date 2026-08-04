import React, { useState } from 'react'
import { useAlertsStore, type SignificantEventAlert } from '../state/alertsStore'
import { useInferenceStore } from '../state/inferenceStore'
import { humanPatterns } from '../utils/patternNames'
import { AlertContextChart } from './AlertContextChart'
import { HarmonicContextSnapshot } from './HarmonicContextSnapshot'
import { HarmonicWeightsChart } from './HarmonicWeightsChart'
import { InferenceChart } from './InferenceChart'

/* ── Feature display names ───────────────────────────────── */
const FEATURE_LABELS: Record<string, string> = {
  power_spindle_mean: 'Spindle Power (mean)',
  power_spindle_std: 'Spindle Power (σ)',
  power_x_mean: 'X-Axis Power (mean)',
  power_y_mean: 'Y-Axis Power (mean)',
  power_z_mean: 'Z-Axis Power (mean)',
  chatter_ratio: 'Cross-Axis Vibration Ratio',
  vib_severity_mean: 'Vibration Severity (mean)',
  vib_severity_max: 'Vibration Severity (max)',
  chatter_amp_x_max: 'Modulation Amplitude X (max)',
  chatter_amp_y_max: 'Modulation Amplitude Y (max)',
  chatter_freq_max: 'Modulation Frequency (max)',
  power_active_mean: 'Active Power (mean)',
  power_factor_mean: 'Power Factor (mean)',
  feed_rate_mean: 'Feed Rate (mean)',
  spindle_speed_mean: 'Spindle Speed (mean)',
  temp_mean: 'Temperature (mean)',
  tool_changes: 'Tool Changes',
  // Physics-based fault features
  hf_energy_ratio: 'HF Energy Ratio',
  impulse_crest_factor: 'Impulse Crest Factor',
  kurtosis_max: 'Kurtosis (max)',
  periodicity_strength: 'Periodicity Strength',
  modulation_depth: 'Modulation Depth',
  vib_amplitude_growth: 'Vib Amplitude Growth',
  tp_harmonic_energy: 'Tooth-Passing Harmonic Energy',
  harmonic_amplitude_cv: 'Harmonic Amplitude CV',
  tp_amplitude_variance: 'Tooth-Passing Amplitude Var',
  spindle_order_amplitude: 'Spindle Order Amplitude',
  spindle_phase_shift: 'Spindle Phase Shift',
  breakage_prediction: 'Heuristic Breakage Risk',
}

const MODEL_KEYS = ['anomaly_detector_score', 'model_confidence', 'breakage_prediction', 'tool_wear_estimate', 'harmonic_context_score']

/* ── Score bar ───────────────────────────────────────────── */
function ScoreBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 100 }}>
      <div style={{
        flex: 1, height: 8, borderRadius: 4,
        background: 'rgba(255,255,255,0.06)',
      }}>
        <div style={{
          width: `${pct}%`, height: '100%', borderRadius: 4,
          background: color,
          transition: 'width 0.3s ease',
        }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: 'monospace', minWidth: 36, textAlign: 'right' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

/* ── Severity color helper ───────────────────────────────── */
function sevColor(sev?: string) {
  if (sev === 'CRITICAL') return 'var(--danger)'
  if (sev === 'WARNING') return 'var(--accent)'
  return 'var(--muted)'
}

function scoreColor(score?: number | null, dangerThreshold = 0.7) {
  if (typeof score !== 'number' || !Number.isFinite(score)) return 'var(--muted)'
  if (score >= dangerThreshold) return 'var(--danger)'
  if (score > 0.4) return '#f0a050'
  return 'var(--ok)'
}

/* ── Main Panel ──────────────────────────────────────────── */
export function InferencePanel() {
  const events = useAlertsStore((s) => s.scoredEvents)
  const inferencePoints = useInferenceStore((s) => s.points)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  // Reverse so newest is first
  const sorted = [...events].reverse()

  const isDispatchedAlert = (a: SignificantEventAlert) => a.type === 'significant_event'
  const dispatchedAlertCount = sorted.filter(isDispatchedAlert).length
  const scoredOnlyCount = sorted.length - dispatchedAlertCount

  const modelScore = (a: SignificantEventAlert) => {
    const m = a.metrics as Record<string, unknown> | undefined
    const v = m?.anomaly_detector_score
    return typeof v === 'number' ? v : null
  }
  const sigScore = (a: SignificantEventAlert) => {
    const v = a.significance?.score
    return typeof v === 'number' ? v : null
  }
  const confidence = (a: SignificantEventAlert) => {
    const m = a.metrics as Record<string, unknown> | undefined
    const v = m?.model_confidence
    return typeof v === 'number' ? v : null
  }

  // Stats
  const withModel = sorted.filter(a => modelScore(a) !== null)
  const avgModel = withModel.length
    ? withModel.reduce((s, a) => s + (modelScore(a) ?? 0), 0) / withModel.length
    : null

  // Latest inference scores
  const latest = inferencePoints.length > 0 ? inferencePoints[inferencePoints.length - 1] : null
  // Context model
  const contextWeights = latest?.harmonic_context_weights ?? []
  const contextLabels =
    latest?.harmonic_context_feature_labels ?? latest?.harmonic_feature_labels ?? []
  const contextValues = latest?.harmonic_context_values ?? latest?.harmonic_values ?? []
  const hasContextWeights = contextWeights.length > 0
  const hasContextValues = contextValues.length > 0
  // Pair model
  const pairWeights = latest?.harmonic_pair_weights ?? []
  const pairLabels = latest?.harmonic_pair_feature_labels ?? []
  const pairValues = latest?.harmonic_pair_values ?? []
  const hasPairWeights = pairWeights.length > 0
  const hasPairValues = pairValues.length > 0
  const pairThreshold = latest?.harmonic_thresholds?.pair

  return (
    <div style={{ padding: 12 }}>
      {/* Header */}
      <div className="panelCard" style={{ marginBottom: 12 }}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: 'var(--accent)' }}>
          Live Inference Monitor
        </h3>
        <div style={{ display: 'flex', gap: 24, fontSize: 12, flexWrap: 'wrap' }}>
          <div>
            <span style={{ color: 'var(--muted)' }}>Windows scored: </span>
            <strong>{inferencePoints.length}</strong>
          </div>
          {latest && (
            <>
              <div>
                <span style={{ color: 'var(--muted)' }}>Ensemble: </span>
                <strong style={{ color: latest.scores.ensemble > 0.7 ? 'var(--danger)' : latest.scores.ensemble > 0.4 ? '#f0a050' : 'var(--ok)' }}>
                  {latest.scores.ensemble.toFixed(3)}
                </strong>
              </div>
              <div>
                <span style={{ color: 'var(--muted)' }}>IF: </span>
                <strong>{latest.scores.isolation_forest.toFixed(3)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--muted)' }}>LOF: </span>
                <strong>{latest.scores.lof.toFixed(3)}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--muted)' }}>Z-score: </span>
                <strong>{latest.scores.z_score.toFixed(3)}</strong>
              </div>
              {latest.scores.harmonic_context_score != null && (
                <div>
                  <span style={{ color: 'var(--muted)' }}>Harmonic: </span>
                  <strong style={{
                    color: scoreColor(latest.scores.harmonic_context_score),
                  }}>
                    {latest.scores.harmonic_context_score.toFixed(3)}
                  </strong>
                </div>
              )}
              {latest.scores.harmonic_pair_score != null && (
                <div>
                  <span style={{ color: 'var(--muted)' }}>Pair: </span>
                  <strong style={{
                    color: scoreColor(latest.scores.harmonic_pair_score, typeof pairThreshold === 'number' ? pairThreshold : 0.7),
                  }} title={typeof pairThreshold === 'number' ? `Threshold ${pairThreshold.toFixed(3)}` : undefined}>
                    {latest.scores.harmonic_pair_score.toFixed(3)}
                  </strong>
                </div>
              )}
            </>
          )}
          <div>
            <span style={{ color: 'var(--muted)' }}>Events: </span>
            <strong>{sorted.length}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--muted)' }}>Dispatched alerts: </span>
            <strong style={{ color: 'var(--danger)' }}>{dispatchedAlertCount}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--muted)' }}>Scored only: </span>
            <strong>{scoredOnlyCount}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--muted)' }}>Model: </span>
            <span style={{
              color: inferencePoints.length > 0 ? 'var(--ok)' : 'var(--muted)',
              fontWeight: 600,
            }}>
              {inferencePoints.length > 0 ? 'Streaming' : 'Waiting'}
            </span>
          </div>
        </div>
      </div>

      {/* Time-series anomaly score chart */}
      <div className="panelCard" style={{ marginBottom: 12 }}>
        <h4 style={{ margin: '0 0 6px', fontSize: 12, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          Anomaly Score Time-Series
        </h4>
        <InferenceChart height={220} />
      </div>

      {/* Harmonic model detail — Context model */}
      {latest && hasContextWeights && (
        <div className="panelCard" style={{ marginBottom: 12 }}>
          <HarmonicWeightsChart
            weights={contextWeights}
            labels={contextLabels}
            score={latest.scores.harmonic_context_score}
            height={160}
          />
        </div>
      )}
      {latest && !hasContextWeights && hasContextValues && (
        <div className="panelCard" style={{ marginBottom: 12 }}>
          <HarmonicContextSnapshot
            score={latest.scores.harmonic_context_score}
            labels={contextLabels}
            values={contextValues}
            compact
            title="Harmonic context outputs"
          />
        </div>
      )}
      {/* Harmonic model detail — Pair model */}
      {latest && hasPairWeights && (
        <div className="panelCard" style={{ marginBottom: 12 }}>
          <HarmonicWeightsChart
            weights={pairWeights}
            labels={pairLabels}
            score={latest.scores.harmonic_pair_score}
            height={160}
          />
        </div>
      )}
      {latest && !hasPairWeights && hasPairValues && (
        <div className="panelCard" style={{ marginBottom: 12 }}>
          <HarmonicContextSnapshot
            score={latest.scores.harmonic_pair_score}
            labels={pairLabels}
            values={pairValues}
            compact
            title="Harmonic pair outputs"
          />
        </div>
      )}

      {/* Table */}
      {sorted.length === 0 ? (
        <div style={{ color: 'var(--muted)', fontSize: 13, textAlign: 'center', padding: 40 }}>
          No events scored yet. Start a session to see live inference results.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                <th style={thStyle}>Time</th>
                <th style={thStyle}>Type</th>
                <th style={thStyle}>Sev</th>
                <th style={{ ...thStyle, minWidth: 120 }}>Model Score</th>
                <th style={{ ...thStyle, minWidth: 120 }}>Significance</th>
                <th style={{ ...thStyle, minWidth: 80 }}>Confidence</th>
                <th style={thStyle}>Patterns</th>
                <th style={thStyle}></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(a => {
                const ms = modelScore(a)
                const ss = sigScore(a)
                const conf = confidence(a)
                const isExpanded = expandedId === a.event_id
                const hasMetrics = a.metrics && Object.keys(a.metrics).length > 0
                const dispatched = isDispatchedAlert(a)
                // Flag disagreement: model says anomalous but significance low, or vice versa
                const disagree = ms !== null && ss !== null &&
                  ((ms > 0.7 && ss < 0.4) || (ms < 0.3 && ss > 0.6))

                return (
                  <React.Fragment key={a.event_id}>
                    <tr
                      style={{
                        borderBottom: isExpanded ? 'none' : '1px solid var(--border)',
                        background: disagree ? 'rgba(247, 118, 142, 0.06)' : undefined,
                        cursor: hasMetrics ? 'pointer' : 'default',
                      }}
                      onClick={() => hasMetrics && setExpandedId(isExpanded ? null : a.event_id)}
                    >
                      <td style={tdStyle}>
                        {a.timestamp
                          ? new Date(a.timestamp).toLocaleTimeString()
                          : a._received_at
                          ? new Date(a._received_at).toLocaleTimeString()
                          : '—'}
                      </td>
                      <td style={tdStyle}>
                        <span
                          title={dispatched
                            ? 'Dispatched live alert — shown in the Alerts panel and eligible for freeze-on-alert.'
                            : 'Scored telemetry event — shown for inference visibility, but not dispatched as a live alert.'}
                          style={{
                            display: 'inline-block',
                            padding: '1px 6px',
                            borderRadius: 3,
                            fontSize: 10,
                            fontWeight: 600,
                            textTransform: 'uppercase',
                            background: dispatched ? 'rgba(247, 118, 142, 0.12)' : 'rgba(255,255,255,0.06)',
                            color: dispatched ? 'var(--danger)' : 'var(--muted)',
                          }}
                        >
                          {dispatched ? 'alert' : 'scored'}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        <span style={{
                          display: 'inline-block', padding: '1px 6px', borderRadius: 3,
                          fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                          background: sevColor(a.severity) + '22',
                          color: sevColor(a.severity),
                        }}>
                          {a.severity || 'info'}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {ms !== null
                          ? <ScoreBar value={ms} color={ms > 0.7 ? 'var(--danger)' : ms > 0.4 ? '#f0a050' : 'var(--ok)'} />
                          : <span style={{ color: 'var(--muted)', fontSize: 11 }}>—</span>}
                      </td>
                      <td style={tdStyle}>
                        {ss !== null
                          ? <ScoreBar value={ss} color={ss > 0.6 ? 'var(--danger)' : ss > 0.3 ? '#f0a050' : 'var(--ok)'} />
                          : <span style={{ color: 'var(--muted)', fontSize: 11 }}>—</span>}
                      </td>
                      <td style={tdStyle}>
                        {conf !== null
                          ? <span style={{ fontFamily: 'monospace' }}>{conf.toFixed(2)}</span>
                          : <span style={{ color: 'var(--muted)' }}>—</span>}
                      </td>
                      <td style={{ ...tdStyle, maxWidth: 200 }}>
                        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                          {humanPatterns(a.patterns || []).slice(0, 3).join(', ')}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {hasMetrics && (
                          <span style={{ fontSize: 10, color: 'var(--accent)' }}>
                            {isExpanded ? '▲' : '▼'}
                          </span>
                        )}
                        {disagree && (
                          <span title="Model and significance scores disagree" style={{ marginLeft: 4, fontSize: 11, fontWeight: 700, color: 'var(--accent)' }}>
                            (!)
                          </span>
                        )}
                      </td>
                    </tr>
                    {isExpanded && hasMetrics && (
                      <tr>
                        <td colSpan={8} style={{ padding: '4px 8px 12px' }}>
                          <AlertContextChart alert={a} showStreamContext compact={false} />
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const thStyle: React.CSSProperties = {
  padding: '6px 8px', fontSize: 11, fontWeight: 600,
  color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5,
}

const tdStyle: React.CSSProperties = {
  padding: '6px 8px', verticalAlign: 'middle',
}

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useAppContext } from '../contexts/AppContext'
import { useAlertsStore, type SignificantEventAlert } from '../state/alertsStore'

type AlertDraft = {
  targetSessionId: string
  severity: 'CRITICAL' | 'WARNING' | 'INFO'
  category: string
  score: number
  action: 'CRITICAL' | 'ALERT' | 'STORE'
  priorBoost: number
  anomalyDetectorScore: number
  modelConfidence: number
  breakagePrediction: number
  summary: string
  explanation: string
  patterns: string
  reasons: string
  similarMemoryIds: string
  toolType: string
  workpieceMaterial: string
  operatingRegime: string
  spindleSpeed: number
  feedRate: number
  axialDepth: number
  radialDepth: number
}

type AlertPreset = {
  name: string
  description: string
  draft: Omit<AlertDraft, 'targetSessionId'>
}

const PRESETS: AlertPreset[] = [
  {
    name: 'High-Frequency Burst',
    description: 'Critical operator-facing alert for a sudden burst observation.',
    draft: {
      severity: 'CRITICAL',
      category: 'High-Frequency Burst',
      score: 0.94,
      action: 'CRITICAL',
      priorBoost: 0.12,
      anomalyDetectorScore: 0.88,
      modelConfidence: 0.81,
      breakagePrediction: 0.93,
      summary:
        'High-frequency burst with periodicity loss observed — pause at the next safe opportunity and inspect the tool edge before continuing.',
      explanation:
        'Force peaks and spindle current spikes are both elevated. This observation pattern warrants immediate inspection before resuming the cut.',
      patterns: 'signature:hf_burst_periodicity_loss, POWER_SPIKE, VIBRATION_REGIME_SHIFT',
      reasons: 'High-frequency burst with periodicity loss observed\nForce peaks increased across recent windows\nHistorical labels on similar events include "tool break"',
      similarMemoryIds: 'mem-burst-017\nmem-burst-041\nmem-burst-052',
      toolType: '12 mm carbide end mill',
      workpieceMaterial: 'Ti-6Al-4V',
      operatingRegime: 'roughing',
      spindleSpeed: 9500,
      feedRate: 1350,
      axialDepth: 2.4,
      radialDepth: 0.8,
    },
  },
  {
    name: 'Vibration Modulation',
    description: 'Mid-priority alert for a sustained vibration modulation pattern.',
    draft: {
      severity: 'WARNING',
      category: 'Vibration Modulation',
      score: 0.78,
      action: 'ALERT',
      priorBoost: 0.05,
      anomalyDetectorScore: 0.71,
      modelConfidence: 0.69,
      breakagePrediction: 0.28,
      summary:
        'Vibration modulation observed near tooth-passing harmonics — reduce engagement and verify tool clamping.',
      explanation:
        'High-frequency vibration energy is climbing while the spindle load remains stable. This usually indicates unstable cutting rather than overload.',
      patterns: 'signature:modulated_tooth_passing_vibration, VIBRATION_REGIME_SHIFT, SPECTRAL_PEAK',
      reasons: 'Vibration modulation severity: 0.78\nHigh-frequency vibration energy is rising\nUnusual spectral peak around tooth-passing harmonic',
      similarMemoryIds: 'mem-vibration-004\nmem-vibration-019',
      toolType: '10 mm end mill',
      workpieceMaterial: 'Aluminium 7075',
      operatingRegime: 'semi_finishing',
      spindleSpeed: 14000,
      feedRate: 2400,
      axialDepth: 1.1,
      radialDepth: 0.35,
    },
  },
  {
    name: 'Watchlist Anomaly',
    description: 'Low-friction warning for tuning spacing and text density.',
    draft: {
      severity: 'INFO',
      category: 'Anomaly',
      score: 0.63,
      action: 'STORE',
      priorBoost: 0.0,
      anomalyDetectorScore: 0.66,
      modelConfidence: 0.58,
      breakagePrediction: 0.18,
      summary:
        'Anomaly detected by the ensemble model — monitor the next few cutting passes for drift.',
      explanation:
        'The current window departs from the learned baseline, but the deviation is not yet strong enough for a stop recommendation.',
      patterns: 'ANOMALY_FORCE_RATIO, ENERGY_ACCUMULATION',
      reasons: 'Ensemble anomaly score exceeded watch threshold\nForce ratio shifted from the learned baseline',
      similarMemoryIds: 'mem-watch-002',
      toolType: '8 mm ball mill',
      workpieceMaterial: 'Stainless steel',
      operatingRegime: 'finishing',
      spindleSpeed: 12000,
      feedRate: 900,
      axialDepth: 0.4,
      radialDepth: 0.15,
    },
  },
]

function defaultDraft(targetSessionId: string): AlertDraft {
  return {
    targetSessionId,
    ...PRESETS[0].draft,
  }
}

function toLines(value: string): string[] {
  return value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
}

function toPatterns(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function buildFakeAlert(draft: AlertDraft): SignificantEventAlert {
  const timestamp = new Date().toISOString()
  const eventId = `dev-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  return {
    type: 'significant_event',
    event_id: eventId,
    session_id: draft.targetSessionId.trim() || 'design-session',
    timestamp,
    severity: draft.severity,
    category: draft.category.trim(),
    significance: {
      score: draft.score,
      action: draft.action,
      reasons: toLines(draft.reasons),
      prior_boost: draft.priorBoost,
      pattern_priors: {},
    },
    patterns: toPatterns(draft.patterns),
    summary: draft.summary.trim(),
    summary_source: 'dev',
    explanation: draft.explanation.trim() || null,
    explanation_source: draft.explanation.trim() ? 'dev' : null,
    similar_memories: toLines(draft.similarMemoryIds),
    context: {
      source: 'development-testbed',
      operator_preview: true,
      tool_type: draft.toolType.trim(),
      workpiece_material: draft.workpieceMaterial.trim(),
      operating_regime: draft.operatingRegime.trim(),
      spindle_speed: draft.spindleSpeed,
      feed_rate: draft.feedRate,
      axial_depth: draft.axialDepth,
      radial_depth: draft.radialDepth,
    },
    metrics: {
      anomaly_detector_score: draft.anomalyDetectorScore,
      model_confidence: draft.modelConfidence,
      breakage_prediction: draft.breakagePrediction,
      prior_boost: draft.priorBoost,
      n_rules_triggered: toPatterns(draft.patterns).length,
    },
  }
}

export default function DevelopmentPage() {
  const navigate = useNavigate()
  const { streamSessionId } = useAppContext()
  const pushAlert = useAlertsStore((state) => state.pushAlert)
  const clearAlerts = useAlertsStore((state) => state.clear)
  const alerts = useAlertsStore((state) => state.alerts)

  const [draft, setDraft] = useState<AlertDraft>(() => defaultDraft(streamSessionId || ''))
  const [lastEventId, setLastEventId] = useState<string | null>(null)

  useEffect(() => {
    setDraft((current) => {
      if (current.targetSessionId.trim()) return current
      return { ...current, targetSessionId: streamSessionId || '' }
    })
  }, [streamSessionId])

  const preview = useMemo(() => buildFakeAlert(draft), [draft])

  const applyPreset = (preset: AlertPreset) => {
    setDraft((current) => ({
      ...current,
      ...preset.draft,
    }))
  }

  const updateDraft = <K extends keyof AlertDraft>(key: K, value: AlertDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
  }

  const triggerAlert = () => {
    const alert = buildFakeAlert(draft)
    pushAlert(alert)
    setLastEventId(alert.event_id)
  }

  return (
    <div style={{ padding: '12px 20px', display: 'grid', gap: 16, maxWidth: 1080 }}>
      <div>
        <h2 style={{ margin: 0 }}>Development Testbed</h2>
        <p className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
          Trigger fake alerts through the real alert store to tune the operator-facing look and the live-style popup before wiring more components.
        </p>
      </div>

      <div className="card" style={{ padding: 16, display: 'grid', gap: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontWeight: 700 }}>Target Session</div>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
              Use the active session if you want the global toast and monitoring tab to react exactly as they do in production.
            </div>
          </div>
          <button className="small" onClick={() => navigate('/operator')} style={{ padding: '4px 10px' }}>
            Open Monitoring Tab
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 1fr) minmax(200px, 220px)', gap: 12 }}>
          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Session ID</span>
            <input
              value={draft.targetSessionId}
              onChange={(e) => updateDraft('targetSessionId', e.target.value)}
              placeholder={streamSessionId || 'design-session'}
            />
          </label>
          <div className="small" style={{ color: 'var(--muted)', alignSelf: 'end' }}>
            Active app session: {streamSessionId || 'none selected'}
          </div>
        </div>

        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ fontWeight: 700 }}>Presets</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {PRESETS.map((preset) => (
              <button
                key={preset.name}
                className="small"
                onClick={() => applyPreset(preset)}
                title={preset.description}
                style={{ padding: '6px 10px' }}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 16, display: 'grid', gap: 12 }}>
        <div style={{ fontWeight: 700 }}>Alert Draft</div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Severity</span>
            <select value={draft.severity} onChange={(e) => updateDraft('severity', e.target.value as AlertDraft['severity'])}>
              <option value="CRITICAL">CRITICAL</option>
              <option value="WARNING">WARNING</option>
              <option value="INFO">INFO</option>
            </select>
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Category</span>
            <input value={draft.category} onChange={(e) => updateDraft('category', e.target.value)} />
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Action</span>
            <select value={draft.action} onChange={(e) => updateDraft('action', e.target.value as AlertDraft['action'])}>
              <option value="CRITICAL">CRITICAL</option>
              <option value="ALERT">ALERT</option>
              <option value="STORE">STORE</option>
            </select>
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Score</span>
            <input type="number" min="0" max="1" step="0.01" value={draft.score} onChange={(e) => updateDraft('score', Number(e.target.value))} />
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Prior Boost</span>
            <input type="number" min="0" max="1" step="0.01" value={draft.priorBoost} onChange={(e) => updateDraft('priorBoost', Number(e.target.value))} />
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Model Score</span>
            <input type="number" min="0" max="1" step="0.01" value={draft.anomalyDetectorScore} onChange={(e) => updateDraft('anomalyDetectorScore', Number(e.target.value))} />
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Model Confidence</span>
            <input type="number" min="0" max="1" step="0.01" value={draft.modelConfidence} onChange={(e) => updateDraft('modelConfidence', Number(e.target.value))} />
          </label>

          <label style={{ display: 'grid', gap: 6 }}>
            <span className="small">Risk Model</span>
            <input type="number" min="0" max="1" step="0.01" value={draft.breakagePrediction} onChange={(e) => updateDraft('breakagePrediction', Number(e.target.value))} />
          </label>
        </div>

        <label style={{ display: 'grid', gap: 6 }}>
          <span className="small">Short Summary</span>
          <textarea value={draft.summary} onChange={(e) => updateDraft('summary', e.target.value)} rows={3} />
        </label>

        <label style={{ display: 'grid', gap: 6 }}>
          <span className="small">Explanation</span>
          <textarea value={draft.explanation} onChange={(e) => updateDraft('explanation', e.target.value)} rows={3} />
        </label>

        <label style={{ display: 'grid', gap: 6 }}>
          <span className="small">Patterns (comma-separated)</span>
          <input value={draft.patterns} onChange={(e) => updateDraft('patterns', e.target.value)} />
        </label>

        <label style={{ display: 'grid', gap: 6 }}>
          <span className="small">Reasons (one per line)</span>
          <textarea value={draft.reasons} onChange={(e) => updateDraft('reasons', e.target.value)} rows={4} />
        </label>

        <label style={{ display: 'grid', gap: 6 }}>
          <span className="small">Similar History IDs (one per line)</span>
          <textarea value={draft.similarMemoryIds} onChange={(e) => updateDraft('similarMemoryIds', e.target.value)} rows={3} />
        </label>

        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ fontWeight: 700 }}>Cutting Context</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Tool</span>
              <input value={draft.toolType} onChange={(e) => updateDraft('toolType', e.target.value)} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Material</span>
              <input value={draft.workpieceMaterial} onChange={(e) => updateDraft('workpieceMaterial', e.target.value)} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Regime</span>
              <input value={draft.operatingRegime} onChange={(e) => updateDraft('operatingRegime', e.target.value)} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Spindle Speed</span>
              <input type="number" min="0" step="1" value={draft.spindleSpeed} onChange={(e) => updateDraft('spindleSpeed', Number(e.target.value))} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Feed Rate</span>
              <input type="number" min="0" step="1" value={draft.feedRate} onChange={(e) => updateDraft('feedRate', Number(e.target.value))} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Axial Depth</span>
              <input type="number" min="0" step="0.01" value={draft.axialDepth} onChange={(e) => updateDraft('axialDepth', Number(e.target.value))} />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span className="small">Radial Depth</span>
              <input type="number" min="0" step="0.01" value={draft.radialDepth} onChange={(e) => updateDraft('radialDepth', Number(e.target.value))} />
            </label>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="primary" onClick={triggerAlert} style={{ padding: '6px 16px' }}>
            Trigger Fake Alert Popup
          </button>
          <button onClick={clearAlerts} style={{ padding: '6px 16px' }}>
            Clear Alert Store
          </button>
          {lastEventId && (
            <span className="small" style={{ color: 'var(--muted)', alignSelf: 'center', fontFamily: 'monospace' }}>
              Last event: {lastEventId}
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) minmax(320px, 1.2fr)', gap: 16 }}>
        <div className="card" style={{ padding: 16, display: 'grid', gap: 8 }}>
          <div style={{ fontWeight: 700 }}>What This Will Exercise</div>
          <div className="small" style={{ color: 'var(--muted)' }}>
            Triggering here updates the same alert store used by the global toast, the operator alert strip, the alerts list, and the inference panel.
          </div>
          <div className="small" style={{ color: 'var(--muted)' }}>
            Current store size: {alerts.length} alert{alerts.length === 1 ? '' : 's'}
          </div>
          <div className="small" style={{ color: 'var(--muted)' }}>
            Target session for the next trigger: {preview.session_id}
          </div>
        </div>

        <div className="card" style={{ padding: 16, display: 'grid', gap: 8 }}>
          <div style={{ fontWeight: 700 }}>Payload Preview</div>
          <pre style={{ margin: 0, fontSize: 11, lineHeight: 1.45, overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(preview, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}
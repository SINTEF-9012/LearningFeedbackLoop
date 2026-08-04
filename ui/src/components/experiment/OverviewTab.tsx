/**
 * OverviewTab — Tab 0: configuration, metrics table, confusion matrices,
 *               feedback summary, live priors toggle.
 */
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/http'
import { PriorsChart, type PriorRow } from '../PriorsChart'
import { ConfusionHeatmap, pct, f3, num } from '../charts'
import type { ExperimentTabProps } from './types'
import { HelpIcon } from '../Tooltip'

type PatternPolarity = 'fault_supporting' | 'protective' | 'uninformative' | 'mixed'

function extractPolaritySummary(raw: unknown): Record<PatternPolarity, number> | null {
  const data = raw as Record<string, unknown> | null | undefined
  if (!data) return null

  const explicitCounts = data.polarity_counts
  if (explicitCounts && typeof explicitCounts === 'object' && !Array.isArray(explicitCounts)) {
    const counts = explicitCounts as Record<string, unknown>
    const normalized: Record<PatternPolarity, number> = {
      fault_supporting: Number(counts.fault_supporting || 0),
      protective: Number(counts.protective || 0),
      uninformative: Number(counts.uninformative || 0),
      mixed: Number(counts.mixed || 0),
    }
    const total = Object.values(normalized).reduce((sum, value) => sum + value, 0)
    if (total > 0) return normalized
  }

  const candidates = [
    data.calibrated_pattern_thresholds,
    (data.train_meta as Record<string, unknown> | undefined)?.calibrated_pattern_thresholds,
    (data.train_phase as Record<string, unknown> | undefined)?.calibrated_pattern_thresholds,
  ]
  const calibrated = candidates.find(
    candidate => candidate && typeof candidate === 'object' && !Array.isArray(candidate),
  ) as Record<string, unknown> | undefined
  if (!calibrated) return null

  const counts: Record<PatternPolarity, number> = {
    fault_supporting: 0,
    protective: 0,
    uninformative: 0,
    mixed: 0,
  }

  Object.values(calibrated).forEach((entry) => {
    if (!entry || typeof entry !== 'object') return
    const info = entry as Record<string, unknown>
    let polarity = typeof info.polarity === 'string' ? info.polarity : ''

    if (!polarity) {
      const thresholds = info.thresholds as Record<string, unknown> | undefined
      const thresholdPolarities = new Set(
        Object.values(thresholds || {})
          .map(v => (v && typeof v === 'object' ? (v as Record<string, unknown>).polarity : undefined))
          .filter((v): v is string => typeof v === 'string'),
      )
      if (thresholdPolarities.size === 1) {
        polarity = Array.from(thresholdPolarities)[0]
      } else if (thresholdPolarities.size > 1) {
        polarity = 'mixed'
      }
    }

    if (polarity === 'fault_supporting' || polarity === 'protective' || polarity === 'uninformative' || polarity === 'mixed') {
      counts[polarity] += 1
    }
  })

  const total = Object.values(counts).reduce((sum, value) => sum + value, 0)
  return total > 0 ? counts : null
}

function PolarityPill({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ padding: '8px 10px', borderRadius: 8, background: `${color}22`, border: `1px solid ${color}44` }}>
      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.4, color: 'var(--muted)' }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color }}>{value}</div>
    </div>
  )
}

/** Format a signed delta: green ↑ if positive, red ↓ if negative, grey – if zero/null. */
function delta(a: number | undefined | null, b: number | undefined | null, asPct = false): React.ReactNode {
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return '–'
  const d = b - a
  if (Math.abs(d) < 1e-6) return <span style={{ color: 'var(--muted)' }}>0</span>
  const arrow = d > 0 ? '▲' : '▼'
  const color = d > 0 ? 'var(--ok, #4caf50)' : 'var(--danger, #e74c3c)'
  const formatted = asPct
    ? `${(d * 100).toFixed(1)}pp`
    : d.toFixed(3)
  return <span style={{ color, fontWeight: 600 }}>{arrow} {d > 0 ? '+' : ''}{formatted}</span>
}

/* Internal live-priors widget */
function LivePriors() {
  const priorsQ = useQuery<{ priors: PriorRow[] }>({
    queryKey: ['breakage-live-priors'],
    queryFn: () => api('/agent/memory/scorer/priors'),
    refetchInterval: 3000,
    retry: 1,
  })
  const priors = priorsQ.data?.priors || []
  return (
    <div>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Live Pattern Priors</div>
      {priorsQ.isLoading && <div className="small">Loading…</div>}
      {priors.length > 0 ? (
        <PriorsChart priors={priors} maxRows={12} />
      ) : (
        <div className="small" style={{ color: 'var(--muted)' }}>
          No live priors. Start API and confirm events first.
        </div>
      )}
    </div>
  )
}

export function OverviewTab({ effectiveRunId, selectedRun, fullResultsQ }: ExperimentTabProps) {
  const [showLive, setShowLive] = useState(false)
  const polaritySummary = extractPolaritySummary(fullResultsQ.data)

  if (!effectiveRunId) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <p>No experiment runs found. Use the <strong>🚀 Run</strong> tab to run an experiment.</p>
      </div>
    )
  }

  if (!selectedRun || (selectedRun.error && !selectedRun.error_message)) return null

  if (selectedRun.error) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8, color: 'var(--danger)' }}>✗ Experiment Failed</div>
        <div style={{ padding: '8px 12px', background: 'rgba(231,76,60,0.08)', borderRadius: 6, marginBottom: 12 }}>
          <div className="small" style={{ fontWeight: 600, color: 'var(--danger)' }}>
            {selectedRun.error_message || 'Unknown error'}
          </div>
        </div>
        <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
          <strong>Troubleshooting:</strong><br/>
          • Check the backend terminal for Python tracebacks<br/>
          • If using API mode, ensure the backend is reachable at localhost:8000<br/>
          • Verify that feature CSVs exist (run Feature Extraction first)<br/>
          • Check that the dataset has labelled positive samples
        </div>
      </div>
    )
  }

  return (
    <>
      {/* Configuration */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Configuration <HelpIcon text="Experiment setup parameters: which operations were used for training, testing, and evaluation, plus the prediction gap and variant (cold = no prior state, warm = seeded with live priors)." /></div>
        <div className="small" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 4 }}>
          <span>Train: {selectedRun.config.train_ops?.join(', ') || '–'}</span>
          <span>Test: {selectedRun.config.test_op || '–'}</span>
          <span>Eval: {selectedRun.config.eval_op || '–'}</span>
          <span>Gap: {selectedRun.gap_s ?? '?'}s</span>
          <span>Variant: {selectedRun.config.eval_variant || 'cold'}</span>
        </div>
      </div>

      {polaritySummary && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>
            Pattern Taxonomy <HelpIcon text="Patterns are now classified from experiment calibration data: fault-supporting = fires more on pre-stoppage, protective = fires more on normal operation and suppresses score, uninformative = weak/no separation, mixed = sub-thresholds disagree." />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
            <PolarityPill label="Fault" value={polaritySummary.fault_supporting} color="#e0af68" />
            <PolarityPill label="Protective" value={polaritySummary.protective} color="#7dcfff" />
            <PolarityPill label="Uninformative" value={polaritySummary.uninformative} color="#9aa5ce" />
            <PolarityPill label="Mixed" value={polaritySummary.mixed} color="#c0caf5" />
          </div>
          {polaritySummary.mixed > 0 && (
            <div className="small" style={{ color: 'var(--muted)', marginTop: 8 }}>
              Mixed means individual sub-thresholds within the same pattern do not agree on one polarity yet.
            </div>
          )}
        </div>
      )}

      {/* Classification Metrics */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Classification Metrics <HelpIcon text="Precision = TP/(TP+FP), how many flagged samples are real. Recall = TP/(TP+FN), how many real events were caught. F1 = harmonic mean of precision & recall. AUC = area under ROC curve. Compare Test (no feedback) vs Eval (with feedback) to see how much the feedback loop improved performance." /></div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '4px 8px' }}>Phase</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>N</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Precision</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Recall</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>F1</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>AUC</th>
            </tr>
          </thead>
          <tbody>
            {[
              { label: `Test (${selectedRun.config.test_op})`, m: selectedRun.test_metrics },
              { label: `Eval (${selectedRun.config.eval_op})`, m: selectedRun.eval_metrics },
            ].map((row, i) => (
              <tr
                key={i}
                style={{
                  borderBottom: '1px solid var(--border)',
                  fontWeight: i === 1 ? 700 : 400,
                  background: i === 1 ? 'rgba(122,162,247,0.08)' : undefined,
                }}
              >
                <td style={{ padding: '4px 8px' }}>{row.label}</td>
                <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{row.m?.n_samples ?? '–'}</td>
                <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{pct(row.m?.precision)}</td>
                <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{pct(row.m?.recall)}</td>
                <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{f3(row.m?.f1)}</td>
                <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{f3(row.m?.auc_roc)}</td>
              </tr>
            ))}
            {/* Delta row: Eval − Test */}
            <tr style={{ borderTop: '2px solid var(--border)', background: 'rgba(158,206,106,0.06)' }}>
              <td style={{ padding: '4px 8px', fontStyle: 'italic', color: 'var(--muted)' }}>Δ (Eval − Test)</td>
              <td style={{ textAlign: 'right', padding: '4px 8px' }}>–</td>
              <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{delta(selectedRun.test_metrics?.precision, selectedRun.eval_metrics?.precision, true)}</td>
              <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{delta(selectedRun.test_metrics?.recall, selectedRun.eval_metrics?.recall, true)}</td>
              <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{delta(selectedRun.test_metrics?.f1, selectedRun.eval_metrics?.f1)}</td>
              <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{delta(selectedRun.test_metrics?.auc_roc, selectedRun.eval_metrics?.auc_roc)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Confusion Matrices */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Confusion Matrices <HelpIcon text="Visual confusion matrix heatmaps. TP = correctly detected pre-stoppage, TN = correctly passed normal, FP = false alarm (normal flagged), FN = missed event (pre-stoppage not flagged). Darker colour = higher count." /></div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20 }}>
          <ConfusionHeatmap
            cm={{ tp: num(selectedRun.test_metrics?.tp), fp: num(selectedRun.test_metrics?.fp), tn: num(selectedRun.test_metrics?.tn), fn: num(selectedRun.test_metrics?.fn) }}
            label={`Test: ${selectedRun.config.test_op}`}
          />
          <ConfusionHeatmap
            cm={{ tp: num(selectedRun.eval_metrics?.tp), fp: num(selectedRun.eval_metrics?.fp), tn: num(selectedRun.eval_metrics?.tn), fn: num(selectedRun.eval_metrics?.fn) }}
            label={`Eval: ${selectedRun.config.eval_op}`}
          />
        </div>
      </div>

      {/* Feedback Impact — before / after bar chart */}
      {(() => {
        const tm = selectedRun.test_metrics ?? {}
        const em = selectedRun.eval_metrics ?? {}
        const metrics = [
          { label: 'Precision', before: tm.precision ?? 0, after: em.precision ?? 0 },
          { label: 'Recall',    before: tm.recall ?? 0,    after: em.recall ?? 0 },
          { label: 'F1 Score',  before: tm.f1 ?? 0,        after: em.f1 ?? 0 },
          { label: 'AUC-ROC',   before: tm.auc_roc ?? 0,   after: em.auc_roc ?? 0 },
        ]
        const hasData = metrics.some(m => m.before > 0 || m.after > 0)
        if (!hasData) return null

        const W = 520, barH = 14, gapInGroup = 3, groupGap = 16
        const groupH = barH * 2 + gapInGroup
        const PAD = { t: 12, r: 70, b: 24, l: 72 }
        const plotW = W - PAD.l - PAD.r
        const plotH = metrics.length * (groupH + groupGap) - groupGap
        const H = PAD.t + plotH + PAD.b

        const maxVal = Math.max(1, ...metrics.flatMap(m => [m.before, m.after]))
        const x = (v: number) => PAD.l + Math.max(0, Math.min(v / maxVal, 1)) * plotW

        const testColor = '#7aa2f7'
        const evalColor = '#9ece6a'

        return (
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Feedback Impact <HelpIcon text="Grouped bar chart comparing model performance before (Test, blue) and after (Eval, green) the feedback loop. Deltas on the right show the absolute change — green ▲ = improvement, red ▼ = degradation." /></div>
            <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.12)', borderRadius: 6, display: 'block' }}>
              {/* Grid lines */}
              {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
                const xPos = PAD.l + frac * plotW
                return (
                  <g key={frac}>
                    <line x1={xPos} y1={PAD.t - 4} x2={xPos} y2={PAD.t + plotH} stroke="rgba(255,255,255,0.06)" />
                    <text x={xPos} y={PAD.t + plotH + 14} textAnchor="middle" fill="var(--muted)" fontSize={9}>
                      {(frac * maxVal).toFixed(2)}
                    </text>
                  </g>
                )
              })}

              {metrics.map((m, i) => {
                const yBase = PAD.t + i * (groupH + groupGap)
                const d = m.after - m.before
                const dSign = d > 0 ? '+' : ''
                const dColor = Math.abs(d) < 0.001 ? 'var(--muted)' : d > 0 ? 'var(--ok, #4caf50)' : 'var(--danger, #e74c3c)'
                const arrow = Math.abs(d) < 0.001 ? '' : d > 0 ? '▲ ' : '▼ '

                return (
                  <g key={m.label}>
                    {/* Metric label */}
                    <text x={PAD.l - 8} y={yBase + groupH / 2 + 4} textAnchor="end" fill="#cdd6f4" fontSize={11} fontWeight={500}>
                      {m.label}
                    </text>
                    {/* Test bar (before) */}
                    <rect x={PAD.l} y={yBase} width={Math.max(2, x(m.before) - PAD.l)} height={barH} rx={3}
                      fill={testColor} opacity={0.75} />
                    <text x={x(m.before) + 4} y={yBase + barH - 2} fill={testColor} fontSize={9} fontWeight={600}>
                      {(m.before ?? 0).toFixed(3)}
                    </text>
                    {/* Eval bar (after) */}
                    <rect x={PAD.l} y={yBase + barH + gapInGroup} width={Math.max(2, x(m.after) - PAD.l)} height={barH} rx={3}
                      fill={evalColor} opacity={0.85} />
                    <text x={x(m.after) + 4} y={yBase + barH + gapInGroup + barH - 2} fill={evalColor} fontSize={9} fontWeight={600}>
                      {(m.after ?? 0).toFixed(3)}
                    </text>
                    {/* Delta annotation on right */}
                    <text x={W - PAD.r + 8} y={yBase + groupH / 2 + 4} textAnchor="start"
                      fill={dColor} fontSize={10} fontWeight={700}>
                      {arrow}{dSign}{(d ?? 0).toFixed(3)}
                    </text>
                  </g>
                )
              })}
            </svg>
            {/* Legend */}
            <div style={{ display: 'flex', gap: 20, marginTop: 6, fontSize: 11 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ display: 'inline-block', width: 14, height: 10, background: testColor, borderRadius: 3, opacity: 0.75 }} />
                Test (before feedback)
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ display: 'inline-block', width: 14, height: 10, background: evalColor, borderRadius: 3, opacity: 0.85 }} />
                Eval (after feedback)
              </span>
            </div>
          </div>
        )
      })()}

      {/* Feedback Summary */}
      {selectedRun.feedback_stats && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Feedback Summary <HelpIcon text="Counts of operator feedback events recorded during the experiment. Confirms strengthen the pattern that triggered the detection. Dismissals weaken it. Accuracy = proportion of feedback-reviewed events that were correct." /></div>
          <div className="small" style={{ display: 'flex', gap: 16 }}>
            <span>Events: {selectedRun.feedback_stats.n_events ?? 0}</span>
            <span style={{ color: 'var(--ok)' }}>Confirms: {selectedRun.feedback_stats.n_confirms ?? 0}</span>
            <span style={{ color: 'var(--danger)' }}>Dismissals: {selectedRun.feedback_stats.n_dismissals ?? 0}</span>
            <span>Accuracy: {pct(selectedRun.feedback_stats.accuracy)}</span>
          </div>
        </div>
      )}

      {/* Live Feedback Tracking */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <div style={{ fontWeight: 700 }}>Live Feedback Tracking <HelpIcon text="When ON, polls the running backend API for live pattern priors. Shows how pattern priors evolve in real time as operators confirm or dismiss alerts in the live monitoring system." /></div>
          <button
            className="small"
            style={{
              padding: '2px 10px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              background: showLive ? 'var(--accent)' : 'transparent',
              color: showLive ? '#fff' : 'var(--fg)',
              cursor: 'pointer',
            }}
            onClick={() => setShowLive(!showLive)}
          >
            {showLive ? 'ON' : 'OFF'}
          </button>
        </div>
        {showLive && <LivePriors />}
      </div>
    </>
  )
}

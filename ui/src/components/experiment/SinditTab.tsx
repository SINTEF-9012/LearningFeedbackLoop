/**
 * SinditTab — Tab 9: SINDIT digital twin context per-phase.
 */
import React from 'react'
import { PAL } from '../charts'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'
import type { PhaseDetail, SinditContext, SinditContextSummary } from '../../state/experimentStore'

export function SinditTab({ evalPhase, testPhase, effectiveRunId, evaluateQ }: ExperimentTabProps) {
  const detail = evalPhase || testPhase  // just need to know if detail is loaded

  if (!detail) {
    return (
      <div className="card" style={{ padding: 16, textAlign: 'center' }}>
        <p className="small" style={{ marginBottom: 8 }}>
          SINDIT context is extracted during evaluation. Click below to load evaluation data.
        </p>
        <button
          className="primary"
          onClick={() => evaluateQ.refetch()}
          disabled={evaluateQ.isFetching || !effectiveRunId}
          style={{ padding: '6px 20px' }}
        >
          {evaluateQ.isFetching ? 'Evaluating…' : '🔬 Load Evaluation + SINDIT Context'}
        </button>
        {evaluateQ.isError && (
          <div className="small" style={{ color: 'var(--danger)', marginTop: 8 }}>
            {String((evaluateQ.error as Error)?.message || evaluateQ.error)}
          </div>
        )}
      </div>
    )
  }

  const phases = [evalPhase, testPhase].filter(Boolean) as PhaseDetail[]

  return (
    <>
      {/* SINDIT status badge */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ fontWeight: 700, fontSize: 15 }}>🌐 SINDIT Digital Twin Context <HelpIcon text="SINDIT is a digital twin platform that provides real-time machine state information. When enabled, each sample is enriched with spindle speed, feed rate, tool ID, and power level from the digital twin. This helps correlate detected anomalies with known machine conditions. When SINDIT is not running, context is simulated from raw CNC sensor channels." /></div>
          <span style={{ padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: 'rgba(46,204,113,0.15)', color: 'var(--ok)', border: '1px solid rgba(46,204,113,0.3)' }}>Simulated</span>
        </div>
        <p className="small" style={{ color: 'var(--muted)', marginBottom: 0 }}>
          SINDIT enriches each sample with machine-state context (spindle speed, feed rate, tool info, power level).
          When SINDIT is live (<code>SINDIT_ENABLED=true</code>), data comes from the digital twin API. Otherwise, context is simulated from CNC sensor columns.
        </p>
      </div>

      {/* Phase summary cards */}
      {phases.map(phase => {
        const summary = phase.sindit_context_summary || {} as SinditContextSummary
        const total = summary.total ?? 0
        const nNormal = summary.n_normal ?? 0
        const nDegraded = summary.n_degraded ?? 0
        const degradedPct = total > 0 ? ((nDegraded / total) * 100).toFixed(1) : '–'
        const samplesWithContext = phase.samples.filter(s => s.sindit_context && Object.keys(s.sindit_context).length > 0)

        return (
          <div key={phase.phase} className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>
              {phase.phase === 'eval' ? 'Evaluation' : 'Test'}: {phase.operation}
            </div>

            {/* Metric cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
              <div style={{ background: 'rgba(122,162,247,0.1)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700 }}>{total}</div>
                <div className="small" style={{ color: 'var(--muted)' }}>Total Samples</div>
              </div>
              <div style={{ background: 'rgba(158,206,106,0.1)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--ok)' }}>{nNormal}</div>
                <div className="small" style={{ color: 'var(--muted)' }}>Normal State</div>
              </div>
              <div style={{ background: 'rgba(247,118,142,0.1)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--danger)' }}>{nDegraded}</div>
                <div className="small" style={{ color: 'var(--muted)' }}>Degraded State</div>
              </div>
              <div style={{ background: 'rgba(224,175,104,0.1)', borderRadius: 8, padding: '12px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: PAL[3] }}>{degradedPct}%</div>
                <div className="small" style={{ color: 'var(--muted)' }}>Degraded Rate</div>
              </div>
            </div>

            {/* Machine state bar */}
            {total > 0 && (
              <div style={{ marginBottom: 16 }}>
                <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>Machine State Distribution</div>
                <div style={{ display: 'flex', height: 20, borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${(nNormal / total) * 100}%`, background: 'rgba(158,206,106,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 600 }}>
                    {nNormal > 0 && 'Normal'}
                  </div>
                  <div style={{ width: `${(nDegraded / total) * 100}%`, background: 'rgba(247,118,142,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 600 }}>
                    {nDegraded > 0 && 'Degraded'}
                  </div>
                </div>
              </div>
            )}

            {/* Per-sample context table */}
            {samplesWithContext.length > 0 ? (
              <div style={{ overflowX: 'auto', maxHeight: 400 }}>
                <div className="small" style={{ fontWeight: 600, marginBottom: 6 }}>Per-Sample Context ({samplesWithContext.length} samples)</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border)', position: 'sticky', top: 0, background: 'var(--bg, #1a1b26)' }}>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>Sample</th>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>Label</th>
                      <th style={{ textAlign: 'center', padding: '4px 6px' }}>Machine State</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Spindle Speed</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Feed Rate</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Feed Override</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Power Level</th>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>Tool ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {samplesWithContext.slice(0, 200).map(s => {
                      const ctx = s.sindit_context as SinditContext
                      const isDegraded = ctx.machine_state === 'degraded'
                      return (
                        <tr key={s.sample_id} style={{ borderBottom: '1px solid var(--border)', background: isDegraded ? 'rgba(247,118,142,0.06)' : undefined }}>
                          <td style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{s.sample_id}</td>
                          <td style={{ padding: '3px 6px', color: (s.label === 'pre_stoppage' || s.label === 'pre_break') ? 'var(--danger)' : 'var(--ok)' }}>{s.label}</td>
                          <td style={{ textAlign: 'center', padding: '3px 6px' }}>
                            <span style={{ padding: '1px 8px', borderRadius: 8, fontSize: 10, fontWeight: 600, background: isDegraded ? 'rgba(247,118,142,0.2)' : 'rgba(158,206,106,0.2)', color: isDegraded ? 'var(--danger)' : 'var(--ok)' }}>
                              {ctx.machine_state ?? '–'}
                            </span>
                          </td>
                          <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{ctx.spindle_speed != null && Number.isFinite(ctx.spindle_speed) ? ctx.spindle_speed.toFixed(1) : '–'}</td>
                          <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{ctx.feed_rate != null && Number.isFinite(ctx.feed_rate) ? ctx.feed_rate.toFixed(1) : '–'}</td>
                          <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{ctx.feed_override != null && Number.isFinite(ctx.feed_override) ? ctx.feed_override.toFixed(1) : '–'}</td>
                          <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{ctx.power_level != null && Number.isFinite(ctx.power_level) ? ctx.power_level.toFixed(1) : '–'}</td>
                          <td style={{ padding: '3px 6px' }}>{ctx.tool_id ?? '–'}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="small" style={{ color: 'var(--muted)', fontStyle: 'italic' }}>No per-sample SINDIT context available for this phase.</div>
            )}
          </div>
        )
      })}
    </>
  )
}

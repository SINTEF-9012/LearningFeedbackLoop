import React from 'react'
import { HelpIcon } from '../Tooltip'
import { humanPattern } from '../../utils/patternNames'
import type { FeedbackEvent, PatternFeedbackSummaryEntry, PhaseDetail } from '../../state/experimentStore'
import type { ExperimentTabProps } from './types'

function fmt(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return '0'
  return value.toFixed(digits)
}

function polarityColor(polarity?: string | null): string {
  if (polarity === 'fault_supporting') return 'var(--ok, #4caf50)'
  if (polarity === 'protective') return '#7aa2f7'
  if (polarity === 'uninformative') return 'var(--muted)'
  return '#c6c6c6'
}

function summarizeCoverage(feedbackEvents: FeedbackEvent[], summary: Record<string, PatternFeedbackSummaryEntry>) {
  const detected = new Set<string>()
  let propagatedEventCount = 0
  for (const event of feedbackEvents) {
    for (const patternKey of event.detected_patterns || []) detected.add(patternKey)
    if (event.propagated_prior_deltas && Object.keys(event.propagated_prior_deltas).length > 0) {
      propagatedEventCount += 1
    }
  }
  const directKeys = new Set(Object.keys(summary))
  let indirectOnly = 0
  for (const patternKey of detected) {
    if (!directKeys.has(patternKey)) indirectOnly += 1
  }
  return {
    uniqueDetected: detected.size,
    directPatterns: directKeys.size,
    indirectOnly,
    propagatedEventCount,
  }
}

export function PatternFeedbackTab({ evalPhase, fullResultsQ, effectiveRunId }: ExperimentTabProps) {
  const phase = evalPhase

  if (!effectiveRunId) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Pattern Feedback Summary</div>
        <p className="small" style={{ color: 'var(--muted)', margin: 0 }}>
          Select an experiment run to inspect pattern-level feedback updates.
        </p>
      </div>
    )
  }

  if ((fullResultsQ.isLoading || fullResultsQ.isFetching) && !phase) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <p className="small" style={{ margin: 0 }}>Loading pattern feedback audit…</p>
      </div>
    )
  }

  if (fullResultsQ.isError && !phase) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <p className="small" style={{ color: 'var(--danger)', margin: 0 }}>
          Failed to load pattern feedback audit: {String((fullResultsQ.error as Error)?.message || fullResultsQ.error)}
        </p>
      </div>
    )
  }

  const summary = phase?.pattern_feedback_summary || {}
  const feedbackEvents = phase?.feedback_events || []
  const summaryRows = Object.entries(summary).sort((a, b) => {
    const bEvents = b[1].n_feedback_events || 0
    const aEvents = a[1].n_feedback_events || 0
    if (bEvents !== aEvents) return bEvents - aEvents
    return Math.abs((b[1].total_prior_delta || 0)) - Math.abs((a[1].total_prior_delta || 0))
  })
  const coverage = summarizeCoverage(feedbackEvents, summary)
  const recentEvents = [...feedbackEvents].slice(-10).reverse()

  if (!summaryRows.length && !feedbackEvents.length) {
    return (
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>
          Pattern Feedback Summary
          <HelpIcon text="This tab reads the normalized evaluation payload. Older runs saved before the feedback-audit patch may not contain pattern_feedback_summary or feedback_events." />
        </div>
        <p className="small" style={{ color: 'var(--muted)', margin: 0 }}>
          No persisted feedback audit was found for this run.
        </p>
      </div>
    )
  }

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>
          Pattern Feedback Summary
          <HelpIcon text="Direct updates are the patterns whose priors were explicitly changed by operator feedback. Indirect-only patterns were detected during feedback events but only moved through co-occurrence propagation or discovery, not a direct Beta-Binomial prior update." />
        </div>
        <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
          Run <strong>{effectiveRunId}</strong> recorded {feedbackEvents.length} feedback events. The table below shows which patterns actually received direct prior updates and how far they moved.
        </p>
        <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}>
          <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,255,255,0.04)' }}>
            <div className="small" style={{ color: 'var(--muted)' }}>Direct-updated patterns</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{coverage.directPatterns}</div>
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,255,255,0.04)' }}>
            <div className="small" style={{ color: 'var(--muted)' }}>Unique detected in feedback</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{coverage.uniqueDetected}</div>
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,255,255,0.04)' }}>
            <div className="small" style={{ color: 'var(--muted)' }}>Indirect-only patterns</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{coverage.indirectOnly}</div>
          </div>
          <div style={{ padding: 12, borderRadius: 8, background: 'rgba(255,255,255,0.04)' }}>
            <div className="small" style={{ color: 'var(--muted)' }}>Events with propagation</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{phase?.n_propagation_events ?? coverage.propagatedEventCount}</div>
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 16, overflowX: 'auto' }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Direct Prior Updates</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th style={{ padding: '8px 6px' }}>Pattern</th>
              <th style={{ padding: '8px 6px' }}>Polarity</th>
              <th style={{ padding: '8px 6px' }}>Events</th>
              <th style={{ padding: '8px 6px' }}>Confirms</th>
              <th style={{ padding: '8px 6px' }}>Dismissals</th>
              <th style={{ padding: '8px 6px' }}>Total Δ prior</th>
              <th style={{ padding: '8px 6px' }}>Mean Δ</th>
              <th style={{ padding: '8px 6px' }}>Max |Δ|</th>
              <th style={{ padding: '8px 6px' }}>Last prior</th>
            </tr>
          </thead>
          <tbody>
            {summaryRows.map(([patternKey, entry]) => (
              <tr key={patternKey} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                <td style={{ padding: '8px 6px', minWidth: 240 }}>{humanPattern(patternKey)}</td>
                <td style={{ padding: '8px 6px' }}>
                  <span
                    style={{
                      display: 'inline-block',
                      borderRadius: 999,
                      padding: '2px 8px',
                      fontSize: 11,
                      color: '#fff',
                      background: polarityColor(entry.polarity),
                    }}
                  >
                    {entry.polarity || 'unknown'}
                  </span>
                </td>
                <td style={{ padding: '8px 6px' }}>{entry.n_feedback_events || 0}</td>
                <td style={{ padding: '8px 6px' }}>{entry.n_confirms || 0}</td>
                <td style={{ padding: '8px 6px' }}>{entry.n_dismissals || 0}</td>
                <td style={{ padding: '8px 6px', color: (entry.total_prior_delta || 0) >= 0 ? 'var(--ok, #4caf50)' : 'var(--danger, #e74c3c)' }}>
                  {fmt(entry.total_prior_delta)}
                </td>
                <td style={{ padding: '8px 6px' }}>{fmt(entry.mean_prior_delta)}</td>
                <td style={{ padding: '8px 6px' }}>{fmt(entry.max_abs_prior_delta)}</td>
                <td style={{ padding: '8px 6px' }}>{fmt(entry.last_prior)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ padding: 16, overflowX: 'auto' }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Recent Feedback Events</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th style={{ padding: '8px 6px' }}>Sample</th>
              <th style={{ padding: '8px 6px' }}>Action</th>
              <th style={{ padding: '8px 6px' }}>Updated Patterns</th>
              <th style={{ padding: '8px 6px' }}>Detected</th>
              <th style={{ padding: '8px 6px' }}>Propagated</th>
              <th style={{ padding: '8px 6px' }}>Threshold Shift</th>
              <th style={{ padding: '8px 6px' }}>Retrained</th>
            </tr>
          </thead>
          <tbody>
            {recentEvents.map((event) => {
              const updated = (event.pattern_updates || []).map((update) => humanPattern(update.pattern_key))
              const detected = (event.detected_patterns || []).length
              const propagated = Object.keys(event.propagated_prior_deltas || {}).length
              const thresholdDelta = (event.threshold_after ?? 0) - (event.threshold_before ?? 0)
              return (
                <tr key={`${event.source_sample_id}-${event.applied_at_index}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  <td style={{ padding: '8px 6px' }}>{event.source_sample_id}</td>
                  <td style={{ padding: '8px 6px' }}>
                    <span style={{ color: event.feedback_action === 'CONFIRM' ? 'var(--ok, #4caf50)' : 'var(--danger, #e74c3c)', fontWeight: 700 }}>
                      {event.feedback_action}
                    </span>
                  </td>
                  <td style={{ padding: '8px 6px', minWidth: 220 }}>{updated.length ? updated.join(', ') : 'None'}</td>
                  <td style={{ padding: '8px 6px' }}>{detected}</td>
                  <td style={{ padding: '8px 6px' }}>{propagated}</td>
                  <td style={{ padding: '8px 6px' }}>{thresholdDelta >= 0 ? '+' : ''}{fmt(thresholdDelta)}</td>
                  <td style={{ padding: '8px 6px' }}>{event.model_retrained ? 'Yes' : 'No'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
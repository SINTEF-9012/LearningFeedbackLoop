/**
 * SampleInspectorTab — Tab 4: Sortable, filterable sample table.
 */
import React, { useState, useMemo, useCallback } from 'react'
import { num } from '../charts'
import { HelpIcon } from '../Tooltip'
import { humanPattern, patternDescription } from '../../utils/patternNames'
import { SampleEvidencePanel } from './SampleEvidencePanel'
import type { ExperimentTabProps } from './types'
import type { SampleResult } from '../../state/experimentStore'

function protectivePenalty(sample: SampleResult): number | null {
  const entry = sample.score_trace?.find(
    (item) => item.component === 'protective_pattern_match' && typeof item.value === 'number' && item.value < 0,
  )
  return entry ? Math.abs(entry.value) : null
}

export function SampleInspectorTab({ evalPhase }: ExperimentTabProps) {
  const [sampleSort, setSampleSort] = useState<{ key: string; asc: boolean }>({ key: 'significance_score', asc: false })
  const [sampleFilter, setSampleFilter] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const sortedSamples = useMemo(() => {
    if (!evalPhase?.samples) return []
    let s = [...evalPhase.samples]
    if (sampleFilter) {
      const f = sampleFilter.toLowerCase()
      s = s.filter(r =>
        r.sample_id.toLowerCase().includes(f) ||
        r.label.toLowerCase().includes(f) ||
        r.action.toLowerCase().includes(f) ||
        (r.detected_patterns || []).some(p => p.toLowerCase().includes(f)),
      )
    }
    const { key, asc } = sampleSort
    s.sort((a, b) => {
      const av = key === 'protectivePenalty'
        ? protectivePenalty(a)
        : (a as unknown as Record<string, unknown>)[key]
      const bv = key === 'protectivePenalty'
        ? protectivePenalty(b)
        : (b as unknown as Record<string, unknown>)[key]
      if (typeof av === 'number' && typeof bv === 'number') return asc ? av - bv : bv - av
      return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })
    return s
  }, [evalPhase?.samples, sampleSort, sampleFilter])

  const handleSortClick = useCallback((key: string) => {
    setSampleSort(prev => ({ key, asc: prev.key === key ? !prev.asc : false }))
  }, [])

  if (!evalPhase) return null

  const hasAnyExplanation = evalPhase.samples.some((s: SampleResult) => !!s.explanation)
  const nExplained = evalPhase.samples.filter((s: SampleResult) => !!s.explanation).length

  const columns = [
    { key: 'sample_id', label: 'Sample' },
    { key: 'label', label: 'Label' },
    { key: 'significance_score', label: 'Score' },
    { key: 'predicted_positive', label: 'Pred' },
    { key: 'action', label: 'Action' },
    { key: 'supervised_score', label: 'RF' },
    { key: 'pattern_rule_score', label: 'Pattern' },
    { key: 'prior_boost', label: 'Prior↑' },
    { key: 'protectivePenalty', label: 'Protect↓' },
    { key: 'n_rules_triggered', label: '#Rules' },
    { key: 'detected_patterns', label: 'Patterns' },
    { key: 'feedback_action', label: 'Feedback' },
    { key: 'prediction_flipped', label: 'Flip' },
    { key: 'details', label: 'Details', sortable: false },
  ]

  const columnTips: Record<string, string> = {
    significance_score: 'Combined anomaly score from all scoring layers (model + pattern + prior)',
    predicted_positive: 'Whether this sample was flagged as anomalous (⬆)',
    supervised_score: 'Random Forest classifier score from supervised model',
    pattern_rule_score: 'Score from pattern rule matching (detected patterns boost this)',
    prior_boost: 'Bayesian prior boost from accumulated feedback',
    protectivePenalty: 'Penalty applied when protective (normality-supporting) patterns fired for this sample',
    n_rules_triggered: 'Number of pattern rules that matched this sample',
    feedback_action: 'Feedback applied: confirm = true event, dismiss = false alarm',
    prediction_flipped: 'Whether feedback caused this prediction to change (⟳)',
    details: 'Open the per-sample evidence view with narrative explanation, model fusion, and harmonic weights when available',
  }

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>
        Eval Samples ({evalPhase.samples.length}){hasAnyExplanation ? <span className="small" style={{ fontWeight: 400, color: 'var(--accent)', marginLeft: 8 }}>{nExplained} with LLM explanation</span> : null} <HelpIcon text="Individual sample results from the evaluation phase. Click column headers to sort. Filter by sample ID, label, action, or pattern name. Colour coding: green = true positive (TP), red = false positive (FP), light red = false negative (FN). Use Details to inspect narrative explanations, supervised vs unsupervised model evidence, and harmonic weights when available." />
      </div>
      <input
        type="text"
        placeholder="Filter by sample ID, label, action, pattern…"
        value={sampleFilter}
        onChange={e => setSampleFilter(e.target.value)}
        style={{ fontSize: 12, padding: '4px 8px', width: 320, marginBottom: 8 }}
      />
      <div style={{ overflowX: 'auto', maxHeight: 600 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)', position: 'sticky', top: 0, background: 'var(--bg, #1a1b26)' }}>
              {columns.map(col => (
                <th
                  key={col.key}
                  style={{
                    textAlign: col.key === 'sample_id' || col.key === 'label' ? 'left' : 'right',
                    padding: '4px 6px',
                    cursor: col.sortable === false ? 'default' : 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                  onClick={() => col.sortable === false ? undefined : handleSortClick(col.key)}
                  title={columnTips[col.key] || ''}
                >
                  {col.label} {sampleSort.key === col.key ? (sampleSort.asc ? '▲' : '▼') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedSamples.slice(0, 200).map((s: SampleResult) => {
              const isTP = s.label === 'pre_stoppage' && s.predicted_positive
              const isFP = s.label === 'normal' && s.predicted_positive
              const isFN = s.label === 'pre_stoppage' && !s.predicted_positive
              const bg = isTP
                ? 'rgba(158,206,106,0.08)'
                : isFP
                  ? 'rgba(247,118,142,0.08)'
                  : isFN
                    ? 'rgba(247,118,142,0.05)'
                    : undefined
              const protective = protectivePenalty(s)

              return (
                <React.Fragment key={s.sample_id}>
                <tr key={s.sample_id} style={{ borderBottom: '1px solid var(--border)', background: bg }}>
                  <td style={{ padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{s.sample_id}</td>
                  <td style={{ padding: '3px 6px', color: s.label === 'pre_stoppage' ? 'var(--danger)' : 'var(--ok)' }}>{s.label}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{num(s.significance_score).toFixed(3)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{s.predicted_positive ? '⬆' : '–'}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{s.action}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{num(s.supervised_score).toFixed(2)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{num(s.pattern_rule_score).toFixed(2)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums' }}>{num(s.prior_boost).toFixed(2)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums', color: protective ? '#7dcfff' : 'var(--muted)' }}>
                    {protective ? `-${protective.toFixed(2)}` : '–'}
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{s.n_rules_triggered}</td>
                  <td style={{ padding: '3px 6px', fontSize: 10, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={(s.detected_patterns || []).map(p => `${humanPattern(p)}: ${patternDescription(p)}`).join('\n')}>{(s.detected_patterns || []).map(p => humanPattern(p)).join(', ') || '–'}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px', color: s.feedback_action === 'confirm' ? 'var(--ok)' : s.feedback_action === 'dismiss' ? 'var(--danger)' : 'var(--muted)' }}>{s.feedback_action || '–'}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{s.prediction_flipped ? '⟳' : ''}</td>
                  <td style={{ textAlign: 'center', padding: '3px 6px' }}>
                    <button
                      onClick={() => setExpandedId(expandedId === s.sample_id ? null : s.sample_id)}
                      style={{
                        background: expandedId === s.sample_id ? 'rgba(122,162,247,0.18)' : 'transparent',
                        border: '1px solid var(--border)',
                        borderRadius: 4,
                        cursor: 'pointer',
                        fontSize: 11,
                        padding: '2px 8px',
                        color: expandedId === s.sample_id ? 'var(--accent)' : 'var(--fg)',
                      }}
                      title={`${expandedId === s.sample_id ? 'Hide' : 'View'} sample evidence`}
                    >
                      {expandedId === s.sample_id ? 'Hide' : 'View'}
                    </button>
                  </td>
                </tr>
                {expandedId === s.sample_id && (
                  <tr key={`${s.sample_id}-llm`} style={{ background: 'rgba(122,162,247,0.06)' }}>
                    <td colSpan={columns.length} style={{ padding: '8px 12px', fontSize: 12, lineHeight: 1.5 }}>
                      <SampleEvidencePanel sample={s} />
                    </td>
                  </tr>
                )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

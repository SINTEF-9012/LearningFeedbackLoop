import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/http'
import { f3 } from './charts'
import { HelpIcon } from './Tooltip'
import type { RunSummary, EvaluationDetail } from '../state/experimentStore'
import {
  OverviewTab,
  ScoreAnalysisTab,
  PriorEvolutionTab,
  FeedbackImpactTab,
  PatternFeedbackTab,
  SampleInspectorTab,
  KnowledgeGraphTab,
  GapComparisonTab,
  SignalTimelineTab,
  RunTab,
  SinditTab,
  DiscoveredPatternsTab,
  HarmonicsTab,
} from './experiment'
import type { ExperimentTabProps } from './experiment'

/** Format a run for the dropdown selector. */
function formatRunLabel(r: RunSummary): string {
  const type = r.experiment_type === 'breakage' ? '🔧 Breakage' : '📊 Stoppage'
  // Try to extract a human-readable date from the run_id or timestamp
  let dateStr = ''
  if (r.timestamp) {
    const d = new Date(typeof r.timestamp === 'number' ? r.timestamp * 1000 : r.timestamp)
    if (!isNaN(d.getTime())) {
      dateStr = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) +
                ' ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
    }
  }
  const gap = r.gap_s != null && r.gap_s > 0 ? `gap=${r.gap_s}s` : ''
  const f1 = r.eval_metrics?.f1 != null && Number.isFinite(r.eval_metrics.f1) ? `F1=${f3(r.eval_metrics.f1)}` : ''
  const errTag = r.error ? '✗ FAILED' : ''
  const parts = [type, dateStr, gap, errTag || f1].filter(Boolean)
  return parts.join(' · ')
}

const TAB_DEFS: { name: string; tip: string }[] = [
  { name: 'Overview', tip: 'Experiment configuration, metrics summary (F1, Precision, Recall, AUC), confusion matrices, feedback counts, and live pattern priors.' },
  { name: '🎵 Harmonics', tip: 'Harmonic context-weighted CNN model — status, configuration, training metrics, and retraining controls. Offline model view; not tied to the live demo stream.' },
  { name: 'Score Analysis', tip: 'Score distribution histograms showing how pre-stoppage and normal samples are scored in each phase. Helps assess threshold placement.' },
  { name: 'Prior Evolution', tip: 'How pattern priors (belief strengths) change over time through feedback. Shows learning effect per pattern key.' },
  { name: 'Feedback Impact', tip: 'Feedback statistics: confirms vs dismissals, weight history for scoring components, and negative sampling metrics.' },
  { name: 'Pattern Feedback', tip: 'Direct pattern-level feedback audit. Shows which priors were updated, how much they moved, and which events only propagated changes indirectly.' },
  { name: 'Sample Inspector', tip: 'Sortable, filterable table of every sample with scores, labels, predictions, detected patterns. Click column headers to sort.' },
  { name: 'Knowledge Graph', tip: 'Pattern co-occurrence graph from Neo4j — shows which patterns fire together and their relationship strengths.' },
  { name: 'Gap Comparison', tip: 'Compare experiment results across different prediction gap values (e.g. 0s, 5s, 10s, 30s) to find the optimal look-ahead window.' },
  { name: 'Signal Timeline', tip: 'Raw multi-channel waveform viewer. Browse individual 60s sample windows, view full operation waveforms with highlighted event regions, or scatter-plot extracted features.' },
  { name: '🚀 Run', tip: 'Configure and launch experiments. Supports stoppage prediction (3-phase with LOOCV) and breakage detection (Site_a_line2/casedata). Extract features, run subprocess or live experiments, bridge results to live system.' },
  { name: '🌐 Digital Twin', tip: 'Digital twin integration — view twin state, query the twin API, and browse the connected Neo4j graph database.' },
  { name: '🔍 Discovered', tip: 'Auto-discovered patterns found by the pattern discovery engine. Shows provenance (source events), confidence scores, and Neo4j persistence status.' },
]

/* ══════════════════════════════════════════════════════════
   MAIN COMPONENT  (slimmed — each tab is its own file)
   ══════════════════════════════════════════════════════════ */

export function ExperimentDashboard() {
  const [activeTab, setActiveTab] = useState(0)
  const [selectedRunId, setSelectedRunId] = useState('')

  // ── Fetch run list
  const runsQ = useQuery<{ runs: RunSummary[] }>({ queryKey: ['experiment-runs'], queryFn: () => api('/agent/memory/experiment/runs'), retry: 1, staleTime: 30_000 })
  const runs = runsQ.data?.runs || []
  const effectiveRunId = selectedRunId || (runs.length ? runs[0].run_id : '')
  const selectedRun = runs.find(r => r.run_id === effectiveRunId)

  // ── Full results for selected run
  const fullResultsQ = useQuery<unknown>({ queryKey: ['experiment-run-full', effectiveRunId], queryFn: () => api(`/agent/memory/experiment/runs/${encodeURIComponent(effectiveRunId)}`), enabled: Boolean(effectiveRunId), staleTime: 60_000 })

  // ── Re-evaluate for sample detail (cached per run_id)
  // Tabs backed by the normalized EvaluationDetail payload.
  const needsDetail = [2, 3, 4, 5, 6, 11].includes(activeTab)
  const evaluateQ = useQuery<EvaluationDetail>({
    queryKey: ['experiment-evaluate', effectiveRunId],
    queryFn: () => api(`/agent/memory/experiment/runs/${encodeURIComponent(effectiveRunId)}/evaluate`, 'POST'),
    enabled: needsDetail && Boolean(effectiveRunId),
    staleTime: 5 * 60_000,  // cache for 5 minutes
    retry: 1,
  })
  const detail = evaluateQ.data || null
  const evalPhase = detail?.eval || null
  const testPhase = detail?.test || null

  // ── Shared props for every tab
  const tabProps: ExperimentTabProps = {
    effectiveRunId, selectedRun, runs, setSelectedRunId,
    detail, evalPhase, testPhase, fullResultsQ, evaluateQ,
  }

  return (
    <div style={{ padding: '12px 20px', display: 'grid', gap: 16, maxWidth: 1060 }}>
      <div>
        <h2 style={{ margin: 0 }}>Experiment Dashboard</h2>
        <p className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
          Stoppage prediction &amp; breakage detection experiments &mdash; configure, run, analyse, and bridge results to the live system.
          <HelpIcon text="This dashboard supports two experiment types: Stoppage Prediction (3-phase train/test/eval with LOOCV rotations and gap sweeps) and Breakage Detection (LOOCV using Site_a_line2 or casedata). Select a run from the dropdown, then navigate tabs to explore results. Use the Run tab to launch new experiments." position="bottom" />
        </p>
      </div>

      {/* Run selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
        <select
          value={effectiveRunId}
          onChange={e => setSelectedRunId(e.target.value)}
          title="Select an experiment run to view its results. Runs are sorted by timestamp (newest first). Each run shows its type, date, gap value and eval F1 score."
          style={{ fontSize: 12, padding: '4px 8px', maxWidth: 520 }}
        >
          <option value="">&mdash; select run &mdash;</option>
          {runs.map(r => (
            <option key={r.run_id} value={r.run_id}>
              {formatRunLabel(r)}
            </option>
          ))}
        </select>
        {runsQ.isLoading && <span className="small">Loading runs&hellip;</span>}
        <button className="small" onClick={() => runsQ.refetch()} title="Refresh the list of experiment runs from disk" style={{ padding: '2px 8px' }}>↻</button>
        {effectiveRunId && (
          <span className="small" style={{ color: 'var(--muted)', fontSize: 10, fontFamily: 'monospace' }} title="Raw run ID on disk">{effectiveRunId}</span>
        )}
      </div>

      {/* Sub-tabs */}
      <div className="expTabs">
        {TAB_DEFS.map((tab, i) => (
          <button
            key={i}
            className={`expTab${activeTab === i ? ' active' : ''}`}
            onClick={() => setActiveTab(i)}
            title={tab.tip}
          >{tab.name}</button>
        ))}
      </div>

      {/* Detail loader for tabs that need it */}
      {needsDetail && !detail && (
        <div className="card" style={{ padding: 16, textAlign: 'center' }}>
          {evaluateQ.isFetching ? (
            <p className="small">Evaluating&hellip;</p>
          ) : evaluateQ.isError ? (
            <>
              <p className="small" style={{ color: 'var(--danger)', marginBottom: 8 }}>{String((evaluateQ.error as Error)?.message || evaluateQ.error)}</p>
              <button className="primary" onClick={() => evaluateQ.refetch()} style={{ padding: '6px 20px' }}>Retry</button>
            </>
          ) : (
            <p className="small" style={{ color: 'var(--muted)' }}>Loading evaluation data&hellip;</p>
          )}
        </div>
      )}

      {/* ── Tab content ── */}
      {activeTab === 0 && <OverviewTab {...tabProps} />}
      {activeTab === 1 && <HarmonicsTab />}
      {activeTab === 2 && detail && <ScoreAnalysisTab {...tabProps} />}
      {activeTab === 3 && detail && <PriorEvolutionTab {...tabProps} />}
      {activeTab === 4 && detail && <FeedbackImpactTab {...tabProps} />}
      {activeTab === 5 && detail && <PatternFeedbackTab {...tabProps} />}
      {activeTab === 6 && detail && <SampleInspectorTab {...tabProps} />}
      {activeTab === 7 && <KnowledgeGraphTab {...tabProps} />}
      {activeTab === 8 && <GapComparisonTab {...tabProps} />}
      {activeTab === 9 && <SignalTimelineTab {...tabProps} />}
      {activeTab === 10 && <RunTab {...tabProps} runsRefetch={() => runsQ.refetch()} />}
      {activeTab === 11 && <SinditTab {...tabProps} />}
      {activeTab === 12 && <DiscoveredPatternsTab {...tabProps} />}
    </div>
  )
}

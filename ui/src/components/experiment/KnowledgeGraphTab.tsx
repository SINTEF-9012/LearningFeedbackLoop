/**
 * KnowledgeGraphTab — Tab 5: Pattern co-occurrence network.
 *
 * When an experiment run is selected, shows the experiment-scoped graph
 * from Neo4j (per-experiment :Experiment → :Session → :Memory → :Pattern
 * sub-graph).  Falls back to inline experiment data or the global live
 * graph if scoped data is unavailable.
 */
import React from 'react'
import { KnowledgeGraph } from '../KnowledgeGraph'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'

export function KnowledgeGraphTab({ evalPhase, fullResultsQ, effectiveRunId }: ExperimentTabProps) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Pattern Co-occurrence Network <HelpIcon text="Force-directed graph showing which breakage patterns fire together. Nodes = patterns, sized by prior probability. Edges = co-occurrence links, width proportional to how often the two patterns fired on the same sample. Clusters indicate related failure modes." /></div>
      <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
        Patterns that fire together are linked. Node colour encodes prior. Edge width = co-occurrence count.
        {effectiveRunId && (
          <span style={{ marginLeft: 8, color: 'var(--accent)' }}>
            Showing graph for run: <strong>{effectiveRunId}</strong>
          </span>
        )}
      </p>
      <KnowledgeGraph
        coOccurrenceGraph={
          (evalPhase?.co_occurrence_graph ||
          ((fullResultsQ.data as Record<string, unknown>)?.eval_phase &&
            ((fullResultsQ.data as Record<string, unknown>)?.eval_phase as Record<string, unknown>)?.co_occurrence_graph) ||
          {}) as Record<string, number>
        }
        priors={
          (evalPhase?.samples?.[evalPhase.samples.length - 1]?.prior_snapshot || {}) as Record<string, number>
        }
        experimentRunId={effectiveRunId}
        fetchLive
        width={560}
        height={420}
      />
    </div>
  )
}

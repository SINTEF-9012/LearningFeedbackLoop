import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/http'
import { KnowledgeGraph } from './KnowledgeGraph'
import { PriorsChart, type PriorRow } from './PriorsChart'
import { GraphManagement } from './GraphManagement'

const NEO4J_BROWSER_URL = 'http://localhost:7474'

export function KnowledgeGraphView() {
  const [subTab, setSubTab] = useState<'cooccurrence' | 'neo4j' | 'manage'>('cooccurrence')

  const priorsQ = useQuery<{ priors: PriorRow[] }>({
    queryKey: ['kg-live-priors'],
    queryFn: () => api('/agent/memory/scorer/priors?limit=50'),
    refetchInterval: 4000,
    retry: 1,
  })

  const priors = priorsQ.data?.priors || []
  const priorsMap: Record<string, number> = {}
  for (const p of priors) priorsMap[p.pattern] = p.prior
  const totalEffectiveWeight = priors.reduce(
    (sum, item) => sum + (Number(item.effective_weight_total) || 0),
    0,
  )
  const passiveOutcomePatterns = priors.filter((item) => (Number(item.passive_outcome_count) || 0) > 0).length
  const passiveOutcomeTotal = priors.reduce(
    (sum, item) => sum + (Number(item.passive_outcome_count) || 0),
    0,
  )
  const severityCorrectionPatterns = priors.filter((item) => (Number(item.severity_correction_count) || 0) > 0).length

  return (
    <div style={{ padding: '12px 20px', display: 'grid', gap: 16, maxWidth: 1200 }}>
      <div>
        <h2 style={{ margin: 0 }}>🔗 Knowledge Graph</h2>
        <p className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
          Live pattern co-occurrence network from the memory store, plus direct access to the Neo4j Browser.
        </p>
      </div>

      {/* Sub-tabs */}
      <div style={{ display: 'flex', gap: 2 }}>
        {([
          { key: 'cooccurrence' as const, label: '🕸️ Co-occurrence Graph' },
          { key: 'neo4j' as const, label: '🗄️ Neo4j Browser' },
          { key: 'manage' as const, label: '⚙️ Manage' },
        ]).map(({ key, label }) => (
          <button
            key={key}
            className="small"
            style={{
              padding: '6px 16px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              cursor: 'pointer',
              background: subTab === key ? 'var(--accent)' : 'transparent',
              color: subTab === key ? '#fff' : 'var(--fg)',
              fontWeight: subTab === key ? 700 : 400,
            }}
            onClick={() => setSubTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Co-occurrence Graph */}
      {subTab === 'cooccurrence' && (
        <>
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Pattern Co-occurrence Network</div>
            <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
              Patterns that fire together are linked. Node colour encodes prior (blue → yellow → red). Edge width = co-occurrence count.
              Data refreshes live from the backend every 8 seconds.
            </p>
            <KnowledgeGraph
              coOccurrenceGraph={{}}
              priors={priorsMap}
              fetchLive
              width={700}
              height={500}
            />
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Live Breakage Priors</div>
            <p className="small" style={{ color: 'var(--muted)', margin: '0 0 8px' }}>
              Pattern prior probabilities and feedback diagnostics, updated in real time from operator feedback.
            </p>
            {priorsQ.isLoading && <div className="small">Loading…</div>}
            {priors.length > 0 ? (
              <>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                    gap: 8,
                    marginBottom: 12,
                  }}
                >
                  <div className="small" style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
                    <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Effective Weight</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--fg)' }}>{totalEffectiveWeight.toFixed(2)}</div>
                  </div>
                  <div className="small" style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
                    <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Passive Outcomes</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--fg)' }}>{passiveOutcomeTotal}</div>
                    <div style={{ color: 'var(--muted)', marginTop: 2 }}>{passiveOutcomePatterns} patterns affected</div>
                  </div>
                  <div className="small" style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
                    <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Severity Retargeting</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--fg)' }}>{severityCorrectionPatterns}</div>
                    <div style={{ color: 'var(--muted)', marginTop: 2 }}>patterns with calibration history</div>
                  </div>
                </div>
                <PriorsChart priors={priors} maxRows={20} showDiagnostics />
                <div className="small" style={{ color: 'var(--muted)', marginTop: 8 }}>
                  Diagnostic chips show weighted evidence, passive cycle outcomes, and severity calibration targets for each pattern.
                </div>
              </>
            ) : (
              <div className="small" style={{ color: 'var(--muted)' }}>
                No priors yet. Run an experiment or confirm/dismiss alerts to build priors.
              </div>
            )}
          </div>
        </>
      )}

      {/* Neo4j Browser */}
      {subTab === 'neo4j' && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <div style={{ fontWeight: 700 }}>Neo4j Browser</div>
            <a
              href={NEO4J_BROWSER_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="small"
              style={{ color: 'var(--accent)', textDecoration: 'underline' }}
            >
              Open in new tab ↗
            </a>
          </div>
          <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
            Direct access to the Neo4j Browser for Cypher queries, graph exploration, and schema inspection.
            Default credentials: <code>neo4j / changeme</code> • Bolt: <code>bolt://localhost:7687</code>
          </p>
          <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden' }}>
            <iframe
              src={NEO4J_BROWSER_URL}
              title="Neo4j Browser"
              style={{
                width: '100%',
                height: 600,
                border: 'none',
                background: '#1a1b26',
              }}
            />
          </div>
          <div className="small" style={{ color: 'var(--muted)', marginTop: 8 }}>
            💡 Tip: Try <code>MATCH (n) RETURN n LIMIT 50</code> to explore all nodes, or <code>MATCH (m:Memory)-[:CO_OCCURS_WITH]-(m2) RETURN m, m2</code> for co-occurrence edges.
          </div>
        </div>
      )}

      {/* Graph Management */}
      {subTab === 'manage' && <GraphManagement />}
    </div>
  )
}

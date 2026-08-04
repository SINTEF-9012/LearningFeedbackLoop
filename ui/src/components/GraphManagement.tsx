import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/http'

/* ---------- Types ---------- */

interface GraphStats {
  total_nodes: number
  total_relationships: number
  node_counts: Record<string, number>
  relationship_counts: Record<string, number>
  subgraph_integrity?: {
    healthy: boolean
    mixed_label_nodes: number
    disallowed_cross_graph_edges: number
    disallowed_relationship_types: string[]
    memory_labels: string[]
    knowledge_labels: string[]
    allowed_cross_relationships: string[]
  }
  error?: string
}

interface CleanupPreview {
  scope: 'memory_graph'
  total_nodes_to_delete: number
  total_relationships_to_delete: number
  node_counts: Record<string, number>
  bridge_relationship_counts: Record<string, number>
  legacy_candidate_summary?: {
    heuristic: string
    total_memories: number
    candidate_memories: number
    candidate_sessions: number
    oldest_memory_at?: string
    newest_memory_at?: string
    oldest_candidate_created_at?: string
    newest_candidate_created_at?: string
    created_by_counts: Record<string, number>
    usecase_counts: Record<string, number>
    top_sessions: Array<{
      session_id: string
      memory_count: number
      oldest_created_at?: string
      newest_created_at?: string
    }>
  }
  memory_labels: string[]
  knowledge_labels_preserved: string[]
  allowed_cross_relationships: string[]
  error?: string
}

interface Experiment {
  run_id: string
  experiment_type?: string
  created_at?: string
  test_f1?: number
  eval_f1?: number
  delta_f1?: number
  n_sessions?: number
  n_memories?: number
}

interface Snapshot {
  id: string
  run_id?: string
  label?: string
  created_at?: string
  n_priors?: number
  n_co_occurrence_edges?: number
  node_counts_json?: string
}

/* ---------- Helpers ---------- */

function fmtDate(iso?: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso.slice(0, 16) }
}

function fmtNum(n?: number | null, dec = 2): string {
  if (n == null || isNaN(n)) return '—'
  return n.toFixed(dec)
}

const CARD: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  border: '1px solid var(--border)',
  background: 'var(--card-bg, var(--bg))',
}

const DANGER_CARD: React.CSSProperties = {
  ...CARD,
  border: '2px solid #e53e3e',
}

const BTN: React.CSSProperties = {
  padding: '4px 10px',
  borderRadius: 4,
  border: '1px solid var(--border)',
  cursor: 'pointer',
  fontSize: 13,
  background: 'transparent',
  color: 'var(--fg)',
}

const BTN_DANGER: React.CSSProperties = {
  ...BTN,
  background: '#e53e3e',
  color: '#fff',
  border: '1px solid #e53e3e',
}

const BTN_PRIMARY: React.CSSProperties = {
  ...BTN,
  background: 'var(--accent)',
  color: '#fff',
  border: '1px solid var(--accent)',
}

/* ---------- Sub-components ---------- */

function StatsCard({ stats }: { stats?: GraphStats }) {
  if (!stats || stats.error) {
    return (
      <div style={CARD}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Graph Statistics</div>
        <div className="small" style={{ color: 'var(--muted)' }}>
          {stats?.error || 'Neo4j store not available.'}
        </div>
      </div>
    )
  }

  const nodeEntries = Object.entries(stats.node_counts).filter(([, v]) => v > 0)
  const relEntries = Object.entries(stats.relationship_counts).filter(([, v]) => v > 0)
  const integrity = stats.subgraph_integrity
  const integrityIssues = (integrity?.mixed_label_nodes || 0) + (integrity?.disallowed_cross_graph_edges || 0)

  return (
    <div style={CARD}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Graph Statistics</div>
      <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{stats.total_nodes.toLocaleString()}</div>
          <div className="small" style={{ color: 'var(--muted)' }}>Total Nodes</div>
        </div>
        <div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>{stats.total_relationships.toLocaleString()}</div>
          <div className="small" style={{ color: 'var(--muted)' }}>Total Relationships</div>
        </div>
      </div>
      {integrity && (
        <div
          style={{
            marginBottom: 12,
            padding: 12,
            borderRadius: 8,
            border: integrity.healthy ? '1px solid #38a16955' : '1px solid #e53e3e',
            background: integrity.healthy ? '#38a16912' : '#e53e3e12',
            display: 'grid',
            gap: 8,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ fontWeight: 700 }}>
              Subgraph Integrity
            </div>
            <span
              className="small"
              style={{
                padding: '2px 8px',
                borderRadius: 999,
                background: integrity.healthy ? '#38a169' : '#e53e3e',
                color: '#fff',
                fontWeight: 700,
              }}
            >
              {integrity.healthy ? 'Healthy' : `${integrityIssues} issue${integrityIssues === 1 ? '' : 's'}`}
            </span>
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: 8,
            }}
          >
            <div className="small" style={{ border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px', background: 'var(--bg)' }}>
              <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Mixed-Label Nodes</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{integrity.mixed_label_nodes}</div>
            </div>
            <div className="small" style={{ border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px', background: 'var(--bg)' }}>
              <div style={{ color: 'var(--muted)', marginBottom: 4 }}>Disallowed Cross-Graph Edges</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{integrity.disallowed_cross_graph_edges}</div>
            </div>
          </div>
          <div className="small" style={{ color: 'var(--muted)' }}>
            Allowed seam relationships: {integrity.allowed_cross_relationships.join(', ') || 'none'}
          </div>
          <details>
            <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>
              Integrity contract details
            </summary>
            <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
              <div className="small">
                <strong>Memory graph labels:</strong> {integrity.memory_labels.join(', ')}
              </div>
              <div className="small">
                <strong>Knowledge graph labels:</strong> {integrity.knowledge_labels.join(', ')}
              </div>
              {!integrity.healthy && integrity.disallowed_relationship_types.length > 0 && (
                <div className="small" style={{ color: '#e53e3e' }}>
                  <strong>Violating relationship types:</strong> {integrity.disallowed_relationship_types.join(', ')}
                </div>
              )}
            </div>
          </details>
        </div>
      )}
      <details>
        <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>
          Node breakdown ({nodeEntries.length} labels)
        </summary>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
          {nodeEntries.map(([label, count]) => (
            <span
              key={label}
              className="small"
              style={{
                padding: '2px 8px',
                borderRadius: 4,
                background: 'var(--bg)',
                border: '1px solid var(--border)',
              }}
            >
              {label}: <strong>{count}</strong>
            </span>
          ))}
        </div>
      </details>
      {relEntries.length > 0 && (
        <details style={{ marginTop: 8 }}>
          <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>
            Relationship breakdown ({relEntries.length} types)
          </summary>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            {relEntries.map(([type, count]) => (
              <span
                key={type}
                className="small"
                style={{
                  padding: '2px 8px',
                  borderRadius: 4,
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                }}
              >
                {type}: <strong>{count}</strong>
              </span>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

function ExperimentsTable({
  experiments,
  onDelete,
  deleting,
}: {
  experiments: Experiment[]
  onDelete: (runId: string) => void
  deleting?: string
}) {
  const [confirmId, setConfirmId] = useState<string | null>(null)

  if (experiments.length === 0) {
    return (
      <div style={CARD}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Experiments</div>
        <div className="small" style={{ color: 'var(--muted)' }}>No experiments in the graph yet.</div>
      </div>
    )
  }

  return (
    <div style={CARD}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>
        Experiments ({experiments.length})
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px' }}>Run ID</th>
              <th style={{ padding: '6px 8px' }}>Type</th>
              <th style={{ padding: '6px 8px' }}>Date</th>
              <th style={{ padding: '6px 8px', textAlign: 'right' }}>Test F1</th>
              <th style={{ padding: '6px 8px', textAlign: 'right' }}>Eval F1</th>
              <th style={{ padding: '6px 8px', textAlign: 'right' }}>ΔF1</th>
              <th style={{ padding: '6px 8px', textAlign: 'right' }}>Sessions</th>
              <th style={{ padding: '6px 8px', textAlign: 'right' }}>Memories</th>
              <th style={{ padding: '6px 8px' }}></th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.run_id} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 12 }}>
                  {exp.run_id}
                </td>
                <td style={{ padding: '6px 8px' }}>{exp.experiment_type || '—'}</td>
                <td style={{ padding: '6px 8px' }}>{fmtDate(exp.created_at)}</td>
                <td style={{ padding: '6px 8px', textAlign: 'right' }}>{fmtNum(exp.test_f1)}</td>
                <td style={{ padding: '6px 8px', textAlign: 'right' }}>{fmtNum(exp.eval_f1)}</td>
                <td style={{
                  padding: '6px 8px',
                  textAlign: 'right',
                  color: (exp.delta_f1 ?? 0) > 0 ? '#38a169' : (exp.delta_f1 ?? 0) < 0 ? '#e53e3e' : 'var(--fg)',
                }}>
                  {exp.delta_f1 != null ? (exp.delta_f1 > 0 ? '+' : '') + fmtNum(exp.delta_f1) : '—'}
                </td>
                <td style={{ padding: '6px 8px', textAlign: 'right' }}>{exp.n_sessions ?? '—'}</td>
                <td style={{ padding: '6px 8px', textAlign: 'right' }}>{exp.n_memories ?? '—'}</td>
                <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                  {confirmId === exp.run_id ? (
                    <span style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                      <button
                        style={BTN_DANGER}
                        disabled={deleting === exp.run_id}
                        onClick={() => { onDelete(exp.run_id); setConfirmId(null) }}
                      >
                        {deleting === exp.run_id ? '…' : 'Confirm'}
                      </button>
                      <button style={BTN} onClick={() => setConfirmId(null)}>Cancel</button>
                    </span>
                  ) : (
                    <button
                      style={{ ...BTN, color: '#e53e3e' }}
                      title="Delete experiment and its data"
                      onClick={() => setConfirmId(exp.run_id)}
                    >
                      🗑️
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function SnapshotsTable({
  snapshots,
  onRestore,
  onDelete,
  onCreate,
  creating,
  restoring,
}: {
  snapshots: Snapshot[]
  onRestore: (id: string) => void
  onDelete: (id: string) => void
  onCreate: () => void
  creating?: boolean
  restoring?: string
}) {
  const [confirmRestore, setConfirmRestore] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  return (
    <div style={CARD}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontWeight: 700 }}>
          Snapshots ({snapshots.length})
        </div>
        <button style={BTN_PRIMARY} onClick={onCreate} disabled={creating}>
          {creating ? 'Creating…' : '📸 Take Snapshot'}
        </button>
      </div>
      <p className="small" style={{ color: 'var(--muted)', margin: '0 0 8px' }}>
        Snapshots capture pattern priors and co-occurrence edges. Restoring a snapshot rewinds
        these shared states without affecting memory or feedback history.
      </p>

      {snapshots.length === 0 ? (
        <div className="small" style={{ color: 'var(--muted)' }}>
          No snapshots yet. Snapshots are created automatically before and after each experiment.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                <th style={{ padding: '6px 8px' }}>Label</th>
                <th style={{ padding: '6px 8px' }}>Run</th>
                <th style={{ padding: '6px 8px' }}>Date</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Priors</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Co-occ Edges</th>
                <th style={{ padding: '6px 8px' }}></th>
              </tr>
            </thead>
            <tbody>
              {snapshots.map((snap) => (
                <tr key={snap.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 12 }}>
                    {snap.label || snap.id.slice(0, 8)}
                  </td>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 12 }}>
                    {snap.run_id || '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{fmtDate(snap.created_at)}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{snap.n_priors ?? '—'}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                    {snap.n_co_occurrence_edges ?? '—'}
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                      {confirmRestore === snap.id ? (
                        <>
                          <button
                            style={BTN_PRIMARY}
                            disabled={restoring === snap.id}
                            onClick={() => { onRestore(snap.id); setConfirmRestore(null) }}
                          >
                            {restoring === snap.id ? '…' : 'Confirm Restore'}
                          </button>
                          <button style={BTN} onClick={() => setConfirmRestore(null)}>Cancel</button>
                        </>
                      ) : (
                        <button
                          style={BTN}
                          title="Restore priors and co-occurrence from this snapshot"
                          onClick={() => setConfirmRestore(snap.id)}
                        >
                          ↩️
                        </button>
                      )}
                      {confirmDelete === snap.id ? (
                        <>
                          <button
                            style={BTN_DANGER}
                            onClick={() => { onDelete(snap.id); setConfirmDelete(null) }}
                          >
                            Delete
                          </button>
                          <button style={BTN} onClick={() => setConfirmDelete(null)}>Cancel</button>
                        </>
                      ) : (
                        <button
                          style={{ ...BTN, color: '#e53e3e' }}
                          title="Delete this snapshot"
                          onClick={() => setConfirmDelete(snap.id)}
                        >
                          🗑️
                        </button>
                      )}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function DangerZone({
  onClearMemory,
  onClearLegacyCandidates,
  preview,
  clearing,
  clearingLegacy,
}: {
  onClearMemory: () => void
  onClearLegacyCandidates: () => void
  preview?: CleanupPreview
  clearing?: boolean
  clearingLegacy?: boolean
}) {
  const [confirmText, setConfirmText] = useState('')
  const [confirmMode, setConfirmMode] = useState<'legacy' | 'memory' | null>(null)
  const previewNodeEntries = Object.entries(preview?.node_counts || {}).filter(([, count]) => count > 0)
  const previewBridgeEntries = Object.entries(preview?.bridge_relationship_counts || {}).filter(([, count]) => count > 0)
  const legacySummary = preview?.legacy_candidate_summary
  const legacyCreatedByEntries = Object.entries(legacySummary?.created_by_counts || {}).filter(([, count]) => count > 0)
  const legacyUsecaseEntries = Object.entries(legacySummary?.usecase_counts || {}).filter(([, count]) => count > 0)
  const anyClearing = Boolean(clearing || clearingLegacy)

  return (
    <div style={DANGER_CARD}>
      <div style={{ fontWeight: 700, color: '#e53e3e', marginBottom: 8 }}>
        Memory Graph Cleanup
      </div>
      <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
        Delete only the <strong>memory-side</strong> graph from Neo4j: memories, patterns, sessions,
        feedback, traces, experiments, machines, tools, snapshots, and co-occurrence updates.
        The document/entity knowledge graph remains intact.
      </p>
      {preview && !preview.error && (
        <div style={{ display: 'grid', gap: 8, marginBottom: 12 }}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <div className="small">
              <strong>{preview.total_nodes_to_delete.toLocaleString()}</strong> nodes will be removed
            </div>
            <div className="small">
              <strong>{preview.total_relationships_to_delete.toLocaleString()}</strong> relationships will be detached
            </div>
          </div>
          {previewNodeEntries.length > 0 && (
            <details>
              <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>
                Memory-graph node preview ({previewNodeEntries.length} labels)
              </summary>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                {previewNodeEntries.map(([label, count]) => (
                  <span key={label} className="small" style={{ padding: '2px 8px', borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                    {label}: <strong>{count}</strong>
                  </span>
                ))}
              </div>
            </details>
          )}
          <div className="small" style={{ color: 'var(--muted)' }}>
            Preserved knowledge labels: {preview.knowledge_labels_preserved.join(', ')}
          </div>
          {legacySummary && (
            <details>
              <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>
                Legacy candidate memory preview ({legacySummary.candidate_memories}/{legacySummary.total_memories} memories)
              </summary>
              <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                  <div className="small">
                    <strong>{legacySummary.candidate_memories.toLocaleString()}</strong> candidate memories
                  </div>
                  <div className="small">
                    <strong>{legacySummary.candidate_sessions.toLocaleString()}</strong> candidate sessions
                  </div>
                </div>
                <div className="small" style={{ color: 'var(--muted)' }}>
                  Memory window: {fmtDate(legacySummary.oldest_memory_at)} to {fmtDate(legacySummary.newest_memory_at)}
                </div>
                <div className="small" style={{ color: 'var(--muted)' }}>
                  Candidate window: {fmtDate(legacySummary.oldest_candidate_created_at)} to {fmtDate(legacySummary.newest_candidate_created_at)}
                </div>
                {legacyCreatedByEntries.length > 0 && (
                  <div className="small" style={{ color: 'var(--muted)' }}>
                    Created by: {legacyCreatedByEntries.map(([name, count]) => `${name}=${count}`).join(', ')}
                  </div>
                )}
                {legacyUsecaseEntries.length > 0 && (
                  <div className="small" style={{ color: 'var(--muted)' }}>
                    Usecases: {legacyUsecaseEntries.map(([name, count]) => `${name}=${count}`).join(', ')}
                  </div>
                )}
                {legacySummary.top_sessions.length > 0 && (
                  <div style={{ display: 'grid', gap: 6 }}>
                    {legacySummary.top_sessions.map((session) => (
                      <div key={session.session_id} className="small" style={{ padding: '8px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)' }}>
                        <strong>{session.session_id}</strong> · {session.memory_count} memories · {fmtDate(session.oldest_created_at)} to {fmtDate(session.newest_created_at)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </details>
          )}
          {previewBridgeEntries.length > 0 && (
            <div className="small" style={{ color: 'var(--muted)' }}>
              Cross-graph seam removed with cleanup: {previewBridgeEntries.map(([type, count]) => `${type}=${count}`).join(', ')}
            </div>
          )}
        </div>
      )}
      {confirmMode == null ? (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button style={BTN_DANGER} onClick={() => setConfirmMode('legacy')} disabled={anyClearing}>
            {clearingLegacy ? 'Clearing…' : 'Clear Legacy Candidates'}
          </button>
          <button style={BTN_DANGER} onClick={() => setConfirmMode('memory')} disabled={anyClearing}>
            {clearing ? 'Clearing…' : 'Clear Memory Graph'}
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="text"
            placeholder={confirmMode === 'legacy' ? 'Type DELETE LEGACY to confirm' : 'Type DELETE to confirm'}
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            style={{
              padding: '6px 10px',
              borderRadius: 4,
              border: '1px solid #e53e3e',
              background: 'var(--bg)',
              color: 'var(--fg)',
              fontSize: 13,
              width: 200,
            }}
          />
          <button
            style={BTN_DANGER}
            disabled={
              (confirmMode === 'legacy' ? confirmText !== 'DELETE LEGACY' : confirmText !== 'DELETE')
              || anyClearing
            }
            onClick={() => {
              if (confirmMode === 'legacy') {
                onClearLegacyCandidates()
              } else {
                onClearMemory()
              }
              setConfirmMode(null)
              setConfirmText('')
            }}
          >
            {confirmMode === 'legacy'
              ? (clearingLegacy ? 'Clearing…' : 'Confirm Legacy Cleanup')
              : (clearing ? 'Clearing…' : 'Confirm Memory Cleanup')}
          </button>
          <button style={BTN} onClick={() => { setConfirmMode(null); setConfirmText('') }}>
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

/* ---------- Main component ---------- */

export function GraphManagement() {
  const qc = useQueryClient()

  // --- Queries ---
  const statsQ = useQuery<GraphStats>({
    queryKey: ['graph-stats'],
    queryFn: () => api('/agent/memory/graph/stats'),
    retry: 1,
  })

  const cleanupPreviewQ = useQuery<CleanupPreview>({
    queryKey: ['graph-cleanup-preview'],
    queryFn: () => api('/agent/memory/graph/cleanup-preview'),
    retry: 1,
  })

  const experimentsQ = useQuery<{ experiments: Experiment[] }>({
    queryKey: ['neo4j-experiments'],
    queryFn: () => api('/agent/memory/experiments'),
    retry: 1,
  })

  const snapshotsQ = useQuery<{ snapshots: Snapshot[] }>({
    queryKey: ['graph-snapshots'],
    queryFn: () => api('/agent/memory/graph/snapshots'),
    retry: 1,
  })

  // --- Mutations ---
  const invalidateAll = () => {
    qc.invalidateQueries({ queryKey: ['graph-stats'] })
    qc.invalidateQueries({ queryKey: ['graph-cleanup-preview'] })
    qc.invalidateQueries({ queryKey: ['neo4j-experiments'] })
    qc.invalidateQueries({ queryKey: ['graph-snapshots'] })
    qc.invalidateQueries({ queryKey: ['kg-live-priors'] })
  }

  const deleteExpMut = useMutation({
    mutationFn: (runId: string) =>
      api(`/agent/memory/experiments/${encodeURIComponent(runId)}`, 'DELETE'),
    onSuccess: invalidateAll,
  })

  const createSnapMut = useMutation({
    mutationFn: () =>
      api('/agent/memory/graph/snapshot', 'POST', { label: `manual_${new Date().toISOString().slice(0, 19)}` }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graph-snapshots'] })
      qc.invalidateQueries({ queryKey: ['graph-stats'] })
    },
  })

  const restoreSnapMut = useMutation({
    mutationFn: (id: string) =>
      api(`/agent/memory/graph/snapshot/${encodeURIComponent(id)}/restore`, 'POST'),
    onSuccess: invalidateAll,
  })

  const deleteSnapMut = useMutation({
    mutationFn: (id: string) =>
      api(`/agent/memory/graph/snapshot/${encodeURIComponent(id)}`, 'DELETE'),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graph-snapshots'] })
      qc.invalidateQueries({ queryKey: ['graph-stats'] })
    },
  })

  const clearMemoryMut = useMutation({
    mutationFn: () => api('/agent/memory/graph/clear-memory', 'DELETE'),
    onSuccess: invalidateAll,
  })

  const clearLegacyMut = useMutation({
    mutationFn: () => api('/agent/memory/graph/clear-legacy-candidates', 'DELETE'),
    onSuccess: invalidateAll,
  })

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <StatsCard stats={statsQ.data} />

      <ExperimentsTable
        experiments={experimentsQ.data?.experiments || []}
        onDelete={(rid) => deleteExpMut.mutate(rid)}
        deleting={deleteExpMut.isPending ? (deleteExpMut.variables as string) : undefined}
      />

      <SnapshotsTable
        snapshots={snapshotsQ.data?.snapshots || []}
        onRestore={(id) => restoreSnapMut.mutate(id)}
        onDelete={(id) => deleteSnapMut.mutate(id)}
        onCreate={() => createSnapMut.mutate()}
        creating={createSnapMut.isPending}
        restoring={restoreSnapMut.isPending ? (restoreSnapMut.variables as string) : undefined}
      />

      <DangerZone
        onClearMemory={() => clearMemoryMut.mutate()}
        onClearLegacyCandidates={() => clearLegacyMut.mutate()}
        preview={cleanupPreviewQ.data}
        clearing={clearMemoryMut.isPending}
        clearingLegacy={clearLegacyMut.isPending}
      />

      {/* Status messages */}
      {(deleteExpMut.isSuccess || restoreSnapMut.isSuccess || clearMemoryMut.isSuccess || clearLegacyMut.isSuccess) && (
        <div
          className="small"
          style={{
            padding: '8px 12px',
            borderRadius: 4,
            background: '#38a16920',
            color: '#38a169',
            border: '1px solid #38a16940',
          }}
        >
          ✓ Operation completed. Graph stats refreshed.
        </div>
      )}
      {(deleteExpMut.isError || restoreSnapMut.isError || clearMemoryMut.isError || clearLegacyMut.isError) && (
        <div
          className="small"
          style={{
            padding: '8px 12px',
            borderRadius: 4,
            background: '#e53e3e20',
            color: '#e53e3e',
            border: '1px solid #e53e3e40',
          }}
        >
          ✗ Operation failed:{' '}
          {String(
            (deleteExpMut.error || restoreSnapMut.error || clearMemoryMut.error || clearLegacyMut.error)?.message || 'Unknown error',
          )}
        </div>
      )}
    </div>
  )
}

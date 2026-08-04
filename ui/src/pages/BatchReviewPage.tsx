/**
 * BatchReviewPage — end-of-batch reconfiguration review (Phase 2, T2.4).
 *
 * The demo headline: at the end of a batch (one OF / workpiece), aggregate the
 * anomalies + operator feedback into reconfiguration suggestions, then let the
 * operator fine-tune and approve. Nothing is applied without confirmation.
 *
 * Backend: POST /reconfig/compose-batch, GET /reconfig/batch/{session},
 * POST /reconfig/{id}/accept|reject|modify.
 */

import { useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/http'
import { useAppContext } from '../contexts/AppContext'
import { GraphQueryLink } from '../components/GraphQueryLink'
import { memoriesGraphQuery } from '../utils/graphLink'
import { colors, fontSize, radii, shadows, spacing } from '../styles/tokens'

interface SessionSummary { session_id: string; source_label?: string; status_label?: string }
interface SessionsResponse { sessions?: string[]; session_summaries?: SessionSummary[] }

interface ParameterDelta {
  parameter: string
  direction: string
  magnitude_pct: number
  confidence: number
  evidence?: string[]
  rationale?: string
}
interface ToolAction {
  action: string
  tool_number?: number | null
  tool_id?: string | null
  reason_code?: string
  confidence: number
  evidence?: string[]
}
interface Reconfig {
  proposal_id: string
  created_at: string
  triggered_by: string[]
  context: { machine_type?: string | null; tool_type?: string | null; material?: string | null; regime?: string | null }
  parameter_deltas: ParameterDelta[]
  tool_actions: ToolAction[]
  risk: 'low' | 'medium' | 'high'
  requires_operator_confirmation: boolean
  operator_decision?: string | null
  notes: string[]
  reasoning?: string | null
  generator?: string | null
  source_evidence?: {
    confirmed?: number
    dismissed?: number
    anomalies?: { signature: string; count: number }[]
    feedback?: { action: string; reason?: string | null; memory_id?: string }[]
    events?: string[]
  } | null
}
interface BatchListResponse { items: Reconfig[] }
interface ComposeBatchResponse { session_id: string; proposals: Reconfig[]; summary: Record<string, number> }
interface BatchContextSummary {
  context: { machine_type?: string | null; tool_type?: string | null; material?: string | null; regime?: string | null }
  confirmed: number
  dismissed: number
  events: number
}
interface BatchSummaryResponse {
  session_id: string
  n_memories: number
  total_confirmed: number
  total_dismissed: number
  n_contexts: number
  contexts: BatchContextSummary[]
}

interface TotalSummaryResponse {
  scope: 'total'
  n_sessions: number
  n_memories: number
  total_confirmed: number
  total_dismissed: number
  n_contexts: number
  contexts: BatchContextSummary[]
}

const pageStyle: React.CSSProperties = { background: colors.bg, color: colors.text, minHeight: '100%', padding: spacing.xl }
const panelStyle: React.CSSProperties = { background: colors.surface, border: `1px solid ${colors.border}`, borderRadius: radii.lg, boxShadow: shadows.panel, padding: spacing.lg }
const cardStyle: React.CSSProperties = { background: colors.surfaceAlt, border: `1px solid ${colors.border}`, borderRadius: radii.md, padding: spacing.lg }

const riskColor = (risk: string): string => (risk === 'high' ? colors.bad : risk === 'medium' ? colors.warn : colors.good)

function btn(kind: 'accept' | 'reject' | 'neutral'): React.CSSProperties {
  const c = kind === 'accept' ? colors.good : kind === 'reject' ? colors.bad : colors.accent
  return { background: 'transparent', border: `1px solid ${c}`, borderRadius: radii.sm, color: c, cursor: 'pointer', fontSize: fontSize.sm, fontWeight: 600, padding: '6px 14px' }
}

function contextLabel(ctx: Reconfig['context']): string {
  return [ctx.machine_type, ctx.tool_type, ctx.material, ctx.regime].filter(Boolean).join(' · ') || 'unspecified context'
}

export default function BatchReviewPage() {
  const ctx = useAppContext()
  const qc = useQueryClient()
  // The batch to review is chosen here, independent of the live stream session —
  // batch review happens at end-of-batch, when streaming may already be stopped.
  // Remember the chosen batch across tab switches / remounts so a composed
  // batch does not "disappear" when the operator navigates away and back.
  const [selectedSession, setSelectedSession] = useState(
    () => (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('batchReviewSession')) || '',
  )
  useEffect(() => {
    if (selectedSession) { try { sessionStorage.setItem('batchReviewSession', selectedSession) } catch { /* ignore */ } }
  }, [selectedSession])
  const sessionsQ = useQuery<SessionsResponse>({
    queryKey: ['sessions-for-batch'],
    queryFn: () => api<SessionsResponse>('/sessions'),
    refetchInterval: 10000,
  })
  // Completed batches: their live session dict is gone once playback ends, but the
  // reconfiguration persists in the reconfig store — offer those too, so an
  // end-of-batch review works after the stream has stopped.
  const batchSessionsQ = useQuery<{ sessions?: string[] }>({
    queryKey: ['batch-sessions'],
    queryFn: () => api<{ sessions?: string[] }>('/reconfig/batch-sessions'),
    refetchInterval: 10000,
  })
  const sessionSummaries = sessionsQ.data?.session_summaries ?? []
  const activeIds = sessionSummaries.length
    ? sessionSummaries.map((s) => s.session_id)
    : (sessionsQ.data?.sessions ?? [])
  const completedBatchIds = batchSessionsQ.data?.sessions ?? []
  const sessionIds = [...activeIds, ...completedBatchIds.filter((s) => !activeIds.includes(s))]
  // Default to the live stream session if present, else the first available.
  useEffect(() => {
    if (selectedSession && sessionIds.includes(selectedSession)) return
    const fallback = (ctx.streamSessionId && sessionIds.includes(ctx.streamSessionId))
      ? ctx.streamSessionId
      : (sessionIds[0] || '')
    if (fallback) setSelectedSession(fallback)
  }, [ctx.streamSessionId, sessionIds, selectedSession])
  const sessionId = selectedSession
  // Scope: one batch (this operation, default) vs the all-time total.
  const [scope, setScope] = useState<'batch' | 'total'>('batch')
  const totalQ = useQuery<TotalSummaryResponse>({
    queryKey: ['reconfig-total'],
    queryFn: () => api<TotalSummaryResponse>('/reconfig/summary/total'),
    enabled: scope === 'total',
  })
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err' | 'info'; text: string } | null>(null)
  // Per-proposal fine-tune of the feed-reduction magnitude before approving.
  const [feedEdits, setFeedEdits] = useState<Record<string, number>>({})

  const proposalsQ = useQuery<BatchListResponse>({
    queryKey: ['batch-reconfig', sessionId],
    queryFn: () => api<BatchListResponse>(`/reconfig/batch/${encodeURIComponent(sessionId)}`),
    enabled: Boolean(sessionId),
    refetchInterval: 5000,
  })
  const proposals = proposalsQ.data?.items ?? []

  // Fast, read-only feedback tally for the batch (no LLM) — always visible so the
  // operator can see how many confirmed catches the batch holds before composing.
  const summaryQ = useQuery<BatchSummaryResponse>({
    queryKey: ['batch-summary', sessionId],
    queryFn: () => api<BatchSummaryResponse>(`/reconfig/batch/${encodeURIComponent(sessionId)}/summary`),
    enabled: Boolean(sessionId),
    refetchInterval: 5000,
  })
  const summary = summaryQ.data

  const closeBatch = async () => {
    if (!sessionId || busy) return
    setBusy(true)
    setMsg({ kind: 'info', text: 'Aggregating this batch…' })
    try {
      const res = await api<ComposeBatchResponse>('/reconfig/compose-batch', 'POST', { session_id: sessionId, operator_id: 'operator' })
      const n = res.proposals.length
      setMsg({
        kind: 'ok',
        text: n > 0
          ? `${n} reconfiguration suggestion${n === 1 ? '' : 's'} from ${res.summary.total_confirmed ?? 0} confirmed catch(es) across ${res.summary.n_memories ?? 0} events.`
          : `No suggestions — nothing was confirmed in this batch yet (${res.summary.n_memories ?? 0} events).`,
      })
      await qc.invalidateQueries({ queryKey: ['batch-reconfig', sessionId] })
    } catch (e) {
      setMsg({ kind: 'err', text: `Aggregation failed: ${String(e)}` })
    } finally {
      setBusy(false)
    }
  }

  const decide = async (p: Reconfig, decision: 'accept' | 'reject' | 'modify') => {
    if (busy) return
    setBusy(true)
    setMsg({ kind: 'info', text: `${decision === 'accept' ? 'Approving' : decision === 'reject' ? 'Rejecting' : 'Applying adjustment'}…` })
    try {
      if (decision === 'modify') {
        const newPct = feedEdits[p.proposal_id]
        const parameter_deltas = p.parameter_deltas.map((d) => (d.parameter === 'feed_rate' && typeof newPct === 'number' ? { ...d, magnitude_pct: newPct } : d))
        await api(`/reconfig/${encodeURIComponent(p.proposal_id)}/modify`, 'POST', { operator_id: 'operator', reason: 'operator fine-tuned feed reduction', parameter_deltas })
      } else {
        await api(`/reconfig/${encodeURIComponent(p.proposal_id)}/${decision}`, 'POST', { operator_id: 'operator', reason: `operator ${decision}` })
      }
      setMsg({ kind: 'ok', text: `Proposal ${decision === 'reject' ? 'rejected' : decision === 'modify' ? 'adjusted' : 'approved'} ✓` })
      await qc.invalidateQueries({ queryKey: ['batch-reconfig', sessionId] })
    } catch (e) {
      setMsg({ kind: 'err', text: `${decision} failed: ${String(e)}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={pageStyle}>
      <h1 style={{ fontSize: fontSize.xxl, fontWeight: 700, margin: 0 }}>Batch review</h1>
      <p style={{ color: colors.textMuted, fontSize: fontSize.md, margin: `${spacing.sm}px 0 ${spacing.lg}px`, maxWidth: 760 }}>
        At the end of a batch, the system aggregates the anomalies, your feedback and the learned
        updates into reconfiguration suggestions. Review, fine-tune, and approve — nothing is applied
        automatically.
      </p>

      {/* Scope: this operation (batch, default) vs the all-time total. */}
      <div style={{ display: 'flex', gap: spacing.sm, marginBottom: spacing.lg }}>
        {(['batch', 'total'] as const).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setScope(s)}
            style={{
              background: scope === s ? colors.accent : 'transparent',
              border: `1px solid ${scope === s ? colors.accent : colors.border}`,
              borderRadius: radii.sm,
              color: scope === s ? '#fff' : colors.textMuted,
              cursor: 'pointer',
              fontSize: fontSize.sm,
              fontWeight: 600,
              padding: '6px 14px',
            }}
          >
            {s === 'batch' ? 'This batch' : 'Total (all operations)'}
          </button>
        ))}
      </div>

      {scope === 'total' && (
        <div style={{ ...panelStyle, marginBottom: spacing.lg }}>
          <div style={{ color: colors.text, fontSize: fontSize.md, fontWeight: 600 }}>
            All operations: <span style={{ color: colors.good }}>{totalQ.data?.total_confirmed ?? '…'} confirmed</span>
            {' / '}<span style={{ color: colors.bad }}>{totalQ.data?.total_dismissed ?? '…'} dismissed</span>
            <span style={{ color: colors.textMuted, fontWeight: 400 }}>
              {' '}across {totalQ.data?.n_memories ?? '…'} events · {totalQ.data?.n_sessions ?? '…'} operations · {totalQ.data?.n_contexts ?? '…'} contexts
            </span>
          </div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.xs }}>
            Cumulative across all operations. Recommendations are per-batch — see <em>This batch</em> to approve one.
          </div>
          {(totalQ.data?.contexts ?? []).filter((c) => c.confirmed > 0 || c.dismissed > 0).length > 0 && (
            <div style={{ display: 'grid', gap: spacing.xs, marginTop: spacing.md }}>
              {totalQ.data!.contexts.filter((c) => c.confirmed > 0 || c.dismissed > 0).slice(0, 12).map((c, i) => (
                <div key={i} style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', flexWrap: 'wrap', fontSize: fontSize.sm }}>
                  <span style={{ color: colors.text }}>{contextLabel(c.context)}</span>
                  <span style={{ color: colors.textMuted }}>
                    <span style={{ color: colors.good }}>{c.confirmed}✓</span>{' / '}
                    <span style={{ color: colors.bad }}>{c.dismissed}✗</span>{' · '}{c.events} events
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {scope === 'batch' && (
      <>
      <div style={{ ...panelStyle, marginBottom: spacing.lg }}>
        <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.md }}>
          <label style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
            Batch:{' '}
            <select
              value={sessionId}
              onChange={(e) => setSelectedSession(e.target.value)}
              style={{ background: colors.surfaceAlt, border: `1px solid ${colors.border}`, borderRadius: radii.sm, color: colors.text, fontSize: fontSize.sm, padding: '4px 8px', maxWidth: 320 }}
            >
              {sessionIds.length === 0 && <option value="">(no sessions)</option>}
              {sessionSummaries.length
                ? sessionSummaries.map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {s.source_label ? `${s.source_label} · ${s.session_id}` : s.session_id}
                    </option>
                  ))
                : sessionIds.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          </label>
          <button
            type="button"
            onClick={closeBatch}
            disabled={!sessionId || busy}
            style={{
              background: sessionId && !busy ? colors.accent : colors.surfaceAlt,
              border: `1px solid ${sessionId && !busy ? colors.accent : colors.border}`,
              borderRadius: radii.sm,
              color: sessionId && !busy ? '#fff' : colors.textDim,
              cursor: sessionId && !busy ? 'pointer' : 'not-allowed',
              fontSize: fontSize.md,
              fontWeight: 600,
              padding: '8px 16px',
            }}
          >
            {busy ? 'Working…' : 'Close batch & review'}
          </button>
          <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
            {sessionId ? 'Aggregates this batch’s confirmed catches into suggestions.' : 'No sessions available.'}
          </span>
        </div>
        {msg && (
          <div style={{ color: msg.kind === 'ok' ? colors.good : msg.kind === 'err' ? colors.bad : colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.md }}>
            {msg.text}
          </div>
        )}
      </div>

      {/* Always-visible batch tally (read-only, no compose needed) */}
      {sessionId && summary && (
        <div style={{ ...panelStyle, marginBottom: spacing.lg }}>
          <div style={{ alignItems: 'baseline', display: 'flex', flexWrap: 'wrap', gap: spacing.md, justifyContent: 'space-between' }}>
            <div style={{ color: colors.text, fontSize: fontSize.md, fontWeight: 600 }}>
              This batch: <span style={{ color: colors.good }}>{summary.total_confirmed} confirmed</span>
              {' / '}<span style={{ color: colors.bad }}>{summary.total_dismissed} dismissed</span>
              <span style={{ color: colors.textMuted, fontWeight: 400 }}>
                {' '}across {summary.n_memories} events · {summary.n_contexts} context{summary.n_contexts === 1 ? '' : 's'}
              </span>
            </div>
            {summary.total_confirmed > 0 && proposals.length === 0 && (
              <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                Click <em>Close batch &amp; review</em> to turn these into suggestions.
              </span>
            )}
          </div>
          {summary.contexts.length > 0 && (
            <div style={{ display: 'grid', gap: spacing.xs, marginTop: spacing.md }}>
              {summary.contexts.filter((c) => c.confirmed > 0 || c.dismissed > 0).map((c, i) => (
                <div key={i} style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', flexWrap: 'wrap', fontSize: fontSize.sm }}>
                  <span style={{ color: colors.text }}>{contextLabel(c.context)}</span>
                  <span style={{ color: colors.textMuted }}>
                    <span style={{ color: colors.good }}>{c.confirmed}✓</span>{' / '}
                    <span style={{ color: colors.bad }}>{c.dismissed}✗</span>{' · '}{c.events} events
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {proposals.length === 0 ? (
        <div style={{ ...panelStyle, color: colors.textMuted, fontSize: fontSize.sm }}>
          No suggestions yet. Confirm some alerts during the batch, then click <em>Close batch &amp; review</em>.
        </div>
      ) : (
        <div style={{ display: 'grid', gap: spacing.lg }}>
          {proposals.map((p) => {
            const decided = Boolean(p.operator_decision)
            const feedDelta = p.parameter_deltas.find((d) => d.parameter === 'feed_rate')
            const currentFeed = feedEdits[p.proposal_id] ?? feedDelta?.magnitude_pct ?? 0
            return (
              <section key={p.proposal_id} style={cardStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, flexWrap: 'wrap', marginBottom: spacing.md }}>
                  <div>
                    <div style={{ color: colors.text, fontSize: fontSize.md, fontWeight: 600 }}>{contextLabel(p.context)}</div>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>
                      proposed {new Date(p.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                  <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm }}>
                    <span style={{ border: `1px solid ${riskColor(p.risk)}`, borderRadius: 999, color: riskColor(p.risk), fontSize: fontSize.xs, fontWeight: 700, padding: '2px 10px', textTransform: 'uppercase' }}>
                      {p.risk} risk
                    </span>
                    {decided && (
                      <span style={{ color: p.operator_decision === 'reject' ? colors.bad : colors.good, fontSize: fontSize.sm, fontWeight: 700 }}>
                        {p.operator_decision === 'reject' ? 'Rejected' : p.operator_decision === 'modify' ? 'Adjusted' : 'Approved'} ✓
                      </span>
                    )}
                  </div>
                </div>

                {/* Reasoning (LLM-generated) */}
                {p.reasoning && (
                  <div style={{ background: colors.surface, border: `1px solid ${colors.border}`, borderRadius: radii.sm, padding: `${spacing.sm}px ${spacing.md}px`, marginBottom: spacing.md }}>
                    <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, marginBottom: spacing.xs }}>
                      <span style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Why this proposal</span>
                      {p.generator && (
                        <span style={{ border: `1px solid ${p.generator === 'llm' ? colors.accent : colors.border}`, borderRadius: 999, color: p.generator === 'llm' ? colors.accent : colors.textMuted, fontSize: 10, padding: '1px 7px' }}>
                          {p.generator === 'llm' ? 'AI-generated' : 'rule-based'}
                        </span>
                      )}
                    </div>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, lineHeight: 1.5 }}>{p.reasoning}</div>
                  </div>
                )}

                {/* Tool actions */}
                {p.tool_actions.map((t, i) => (
                  <div key={i} style={{ marginBottom: spacing.sm }}>
                    <span style={{ color: colors.text, fontWeight: 600, textTransform: 'capitalize' }}>{t.action} tool</span>
                    {t.tool_id ? <span style={{ color: colors.textMuted }}> ({t.tool_id})</span> : null}
                    <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}> · confidence {t.confidence.toFixed(2)}</span>
                  </div>
                ))}

                {/* Parameter deltas */}
                {p.parameter_deltas.map((d, i) => (
                  <div key={i} style={{ borderLeft: `3px solid ${colors.warn}`, borderRadius: radii.sm, background: 'rgba(224,175,104,0.06)', padding: `${spacing.sm}px ${spacing.md}px`, marginBottom: spacing.sm }}>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600 }}>
                      {d.direction} {d.parameter.replace(/_/g, ' ')} by {d.magnitude_pct}%
                      <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}> · confidence {d.confidence.toFixed(3)} (volume-shrunk)</span>
                    </div>
                    {d.rationale && <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>{d.rationale}</div>}
                  </div>
                ))}

                {/* Traceable evidence — the feedback items and event trace this
                    proposal was built from (on request). */}
                {p.source_evidence && ((p.source_evidence.feedback?.length ?? 0) > 0 || (p.source_evidence.anomalies?.length ?? 0) > 0) && (
                  <details style={{ marginTop: spacing.sm }}>
                    <summary style={{ color: colors.accent, cursor: 'pointer', fontSize: fontSize.xs }}>
                      Evidence — {p.source_evidence.confirmed ?? 0} confirmed / {p.source_evidence.dismissed ?? 0} dismissed · {p.source_evidence.events?.length ?? 0} events
                    </summary>
                    <div style={{ marginTop: spacing.sm, display: 'grid', gap: spacing.sm }}>
                      {(p.source_evidence.anomalies?.length ?? 0) > 0 && (
                        <div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginBottom: 2 }}>Anomalies</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: spacing.xs }}>
                            {p.source_evidence.anomalies!.map((a) => (
                              <span key={a.signature} style={{ background: colors.surface, border: `1px solid ${colors.border}`, borderRadius: 999, color: colors.text, fontSize: fontSize.xs, padding: '2px 8px' }}>
                                {a.signature} ×{a.count}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {(p.source_evidence.feedback?.length ?? 0) > 0 && (
                        <div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginBottom: 2 }}>Operator feedback</div>
                          <div style={{ display: 'grid', gap: spacing.xs }}>
                            {p.source_evidence.feedback!.map((f, i) => (
                              <div key={i} style={{ fontSize: fontSize.xs }}>
                                <span style={{ color: f.action === 'confirm' ? colors.good : f.action === 'dismiss' ? colors.bad : colors.textMuted, fontWeight: 600 }}>{f.action}</span>
                                {f.reason ? <span style={{ color: colors.textMuted }}> — “{f.reason}”</span> : null}
                                {f.memory_id ? <span style={{ color: colors.textDim, fontFamily: 'monospace' }}> · {f.memory_id.slice(0, 8)}</span> : null}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <GraphQueryLink
                        query={memoriesGraphQuery((p.source_evidence.feedback ?? []).map((f) => f.memory_id).filter((x): x is string => Boolean(x)))}
                        label="View these events in the graph"
                        style={{ color: colors.accent, textDecoration: 'none', fontSize: fontSize.xs }}
                      />
                    </div>
                  </details>
                )}

                {p.notes.length > 0 && (
                  <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginTop: spacing.xs }}>{p.notes.join(' · ')}</div>
                )}

                {/* Decision controls */}
                {!decided && (
                  <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.sm, marginTop: spacing.md }}>
                    <button type="button" style={btn('accept')} disabled={busy} onClick={() => decide(p, 'accept')}>Approve</button>
                    <button type="button" style={btn('reject')} disabled={busy} onClick={() => decide(p, 'reject')}>Reject</button>
                    {feedDelta && (
                      <div style={{ alignItems: 'center', display: 'flex', gap: spacing.xs, marginLeft: spacing.md }}>
                        <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>Fine-tune feed −%</span>
                        <input
                          type="number"
                          min={0}
                          max={15}
                          step={1}
                          value={currentFeed}
                          onChange={(e) => setFeedEdits((s) => ({ ...s, [p.proposal_id]: Number(e.target.value) }))}
                          style={{ width: 64, background: colors.surface, border: `1px solid ${colors.border}`, borderRadius: radii.sm, color: colors.text, fontSize: fontSize.sm, padding: '4px 6px' }}
                        />
                        <button type="button" style={btn('neutral')} disabled={busy} onClick={() => decide(p, 'modify')}>Apply adjustment</button>
                      </div>
                    )}
                  </div>
                )}
              </section>
            )
          })}
        </div>
      )}
      </>
      )}
    </div>
  )
}

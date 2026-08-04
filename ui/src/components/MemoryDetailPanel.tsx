import React, { useMemo, useState } from 'react'
import { humanPattern, humanReason, patternOrigin } from '../utils/patternNames'
import { DocLinksSection } from './DocLinksSection'
import { MemoryGraphLink } from './MemoryGraphLink'
import type { DocLink } from '../types'

// ── Cutting-context field labels for human display ──────────────────────────
const CUTTING_LABELS: Record<string, string> = {
  ground_truth_label: 'Ground truth label',
  ground_truth_index: 'Ground truth sample',
  tool_type: 'Tool',
  spindle_speed: 'Spindle speed',
  num_teeth: 'Number of teeth',
  axial_depth: 'Axial depth of cut',
  radial_depth: 'Radial depth of cut',
  workpiece_material: 'Material',
  operating_regime: 'Operating regime',
  machine_type: 'Machine type',
  feed_rate: 'Feed rate',
  coolant: 'Coolant',
}

const CUTTING_UNITS: Record<string, string> = {
  spindle_speed: 'RPM',
  axial_depth: 'mm',
  radial_depth: 'mm',
  feed_rate: 'mm/min',
}

const METRIC_LABELS: Record<string, string> = {
  rms: 'RMS values',
  dominant_freq: 'Dominant freq.',
  total_energy: 'Total energy',
  spectral_centroids: 'Spectral centroids',
}

function fmtValue(key: string, val: unknown): string {
  if (val === null || val === undefined) return '–'
  if (Array.isArray(val)) return val.map((v) => (typeof v === 'number' ? v.toFixed(2) : String(v))).join(', ')
  if (typeof val === 'object') return JSON.stringify(val)
  const unit = CUTTING_UNITS[key]
  return unit ? `${val} ${unit}` : String(val)
}

function asString(v: unknown): string {
  if (v === null || v === undefined) return ''
  return typeof v === 'string' ? v : JSON.stringify(v)
}

function fmtTs(v: unknown): string {
  const s = typeof v === 'string' ? v : ''
  if (!s) return ''
  const d = new Date(s)
  if (Number.isNaN(d.getTime())) return s
  return `${d.toLocaleString()} (${s})`
}

function kvEntries(obj: Record<string, unknown> | null | undefined, keys: string[]): Array<[string, unknown]> {
  if (!obj || typeof obj !== 'object') return []
  const out: Array<[string, unknown]> = []
  for (const k of keys) {
    if (obj[k] !== undefined) out.push([k, obj[k]])
  }
  return out
}

type Props = {
  memory: Record<string, unknown> | null
  feedback: Record<string, unknown> | null
  traces: Record<string, unknown>[]
  alert?: Record<string, unknown> | null
  docLinks?: DocLink[]
}

type TraceSummary = {
  id: string
  created_at: string
  trace_type: string
  score?: number
  action?: string
  reasons?: string[]
  returnedCount?: number
  topReturned?: { memory_id: string; score?: number; reasons?: string[] } | null
  raw: unknown
}

function toNum(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

function asArray<T = unknown>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : []
}

/** Safely access nested record fields */
function rec(o: unknown): Record<string, unknown> {
  return (o && typeof o === 'object' ? o : {}) as Record<string, unknown>
}

/** Filter out null/undefined/empty-object entries and flatten nested 'extra' dicts */
function flatEntries(obj: Record<string, unknown>): Array<[string, unknown]> {
  const out: Array<[string, unknown]> = []
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue
    // Flatten 'extra' sub-dict into parent level
    if (k === 'extra' && v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [ek, ev] of Object.entries(v as Record<string, unknown>)) {
        if (ev !== null && ev !== undefined) out.push([ek, ev])
      }
      continue
    }
    // Skip empty objects
    if (typeof v === 'object' && !Array.isArray(v) && v !== null && Object.keys(v as object).length === 0) continue
    out.push([k, v])
  }
  return out
}

export function MemoryDetailPanel({ memory, feedback, traces, alert, docLinks = [] }: Props) {
  const [showRaw, setShowRaw] = useState(false)
  const [expandedTraceId, setExpandedTraceId] = useState<string>('')
  const [showExplanation, setShowExplanation] = useState(false)

  const patterns = useMemo(() => {
    const p = memory?.pattern_keys
    if (Array.isArray(p)) {
      // pattern_keys might be [{key, pattern_type}, ...] or strings.
      return p.map((x: unknown) => (typeof x === 'string' ? x : (x as Record<string, unknown>)?.key)).filter(Boolean)
    }
    const ps = memory?.patterns
    if (Array.isArray(ps)) return ps
    return []
  }, [memory])

  const meta = rec(memory?.metadata)
  const sigScore = meta.significance_score
  const sigAction = meta.significance_action

  const cuttingContext = meta.cutting_context
  const alertContext = rec(alert).context
  const alertMetrics = rec(alert).metrics
  const hasLiveAlert = Boolean(alert && Object.keys(rec(alert)).length > 0)

  const overview = useMemo(() => {
    const base = kvEntries(memory, ['id', 'session_id', 'created_at', 'created_by', 'label'])
    return base.map(([k, v]) => [k, k === 'created_at' ? fmtTs(v) : v] as const)
  }, [memory])

  const tags = Array.isArray(memory?.tags) ? (memory.tags as string[]) : []

  const feedbackStats = rec(feedback).stats || rec(feedback).feedback_stats
  const feedbackEvents = rec(feedback).events
  const persistedDocLinks = Array.isArray(docLinks) ? docLinks : []

  const traceSummaries = useMemo<TraceSummary[]>(() => {
    const arr = Array.isArray(traces) ? traces : []
    const out: TraceSummary[] = []
    for (const t of arr) {
      const id = String(t?.id || '')
      const trace_type = String(t?.trace_type || '')
      const created_at = String(t?.created_at || '')
      const payload = rec(t?.payload)

      const sum: TraceSummary = {
        id,
        trace_type,
        created_at,
        raw: t,
      }

      if (trace_type === 'score') {
        const sig = rec(payload.significance)
        sum.score = toNum(sig.score)
        sum.action = typeof sig.action === 'string' ? sig.action : undefined
        sum.reasons = asArray<string>(sig.reasons).filter((x) => typeof x === 'string')
      }

      if (trace_type === 'retrieve') {
        const returned = asArray<Record<string, unknown>>(payload.returned)
        sum.returnedCount = returned.length
        const top = returned[0]
        if (top && typeof top === 'object') {
          sum.topReturned = {
            memory_id: String(top?.memory_id || ''),
            score: toNum(top?.score),
            reasons: asArray<string>(top?.reasons).filter((x) => typeof x === 'string'),
          }
        } else {
          sum.topReturned = null
        }
      }

      out.push(sum)
    }

    // Newest first
    out.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
    return out
  }, [traces])

  const latestScore = traceSummaries.find((t) => t.trace_type === 'score')
  const latestRetrieve = traceSummaries.find((t) => t.trace_type === 'retrieve')

  return (
    <div>
      <div className="hrow" style={{ justifyContent: 'space-between' }}>
        <div style={{ fontWeight: 700 }}>Memory detail</div>
        <div className="hrow">
          <button onClick={() => setShowRaw((v) => !v)}>{showRaw ? 'Hide raw' : 'Show raw'}</button>
        </div>
      </div>

      <div className="hr" />

      <div className="kvTable">
        {overview.map(([k, v]) => (
          <React.Fragment key={k}>
            <div className="k">{k}</div>
            <div className="v">{typeof v === 'string' ? v : asString(v)}</div>
          </React.Fragment>
        ))}

        <div className="k">significance</div>
        <div className="v">
          {sigAction ? <span className="badge">{String(sigAction)}</span> : null}
          {typeof sigScore === 'number' ? <span className="badge">score={sigScore.toFixed(3)}</span> : null}
        </div>

        <div className="k">tags</div>
        <div className="v">{tags.length ? tags.map((t) => <span key={t} className="badge">{t}</span>) : <span className="small">(none)</span>}</div>

        <div className="k">patterns</div>
        <div className="v">
          {patterns.length ? patterns.map((p) => <span key={p} className="badge" title={p}>{humanPattern(p)}</span>) : <span className="small">(none)</span>}
        </div>
      </div>

      <div className="hr" />

      <div style={{ fontWeight: 700 }}>Annotation</div>
      <div className="small">Stored as `annotation_text` on the memory.</div>
      <pre>{String(memory?.annotation_text || '')}</pre>

      {!hasLiveAlert && (alertContext || alertMetrics || cuttingContext) ? (
        <>
          <div className="hr" />
          <div style={{ fontWeight: 700 }}>Operator context</div>
          <div className="small">What the system knew when it generated this alert.</div>

          {alert?.summary ? (
            <>
              <div className="small" style={{ marginTop: 8, fontWeight: 650 }}>
                Alert summary {alert?.summary_source ? <span className="badge" style={{ opacity: 0.7 }}>{String(alert.summary_source)}</span> : null}
              </div>
              <div style={{ padding: '8px 0', fontWeight: 600, lineHeight: 1.4 }}>{String(alert.summary)}</div>
            </>
          ) : null}

          {cuttingContext && typeof cuttingContext === 'object' ? (() => {
            const entries = flatEntries(cuttingContext as Record<string, unknown>)
            return entries.length > 0 ? (
              <>
                <div className="small" style={{ marginTop: 8, fontWeight: 650 }}>Cutting conditions</div>
                <div className="kvTable" style={{ marginTop: 4 }}>
                  {entries.map(([k, v]) => (
                    <React.Fragment key={k}>
                      <div className="k">{CUTTING_LABELS[k] || k.replace(/_/g, ' ')}</div>
                      <div className="v">{fmtValue(k, v)}</div>
                    </React.Fragment>
                  ))}
                </div>
              </>
            ) : null
          })() : null}

          {alertContext && typeof alertContext === 'object' ? (() => {
            const entries = flatEntries(alertContext as Record<string, unknown>)
            return entries.length > 0 ? (
              <>
                <div className="small" style={{ marginTop: 8, fontWeight: 650 }}>Alert context</div>
                <div className="kvTable" style={{ marginTop: 4 }}>
                  {entries.map(([k, v]) => (
                    <React.Fragment key={k}>
                      <div className="k">{CUTTING_LABELS[k] || k.replace(/_/g, ' ')}</div>
                      <div className="v">{fmtValue(k, v)}</div>
                    </React.Fragment>
                  ))}
                </div>
              </>
            ) : null
          })() : null}

          {alertMetrics && typeof alertMetrics === 'object' ? (() => {
            const entries = flatEntries(alertMetrics as Record<string, unknown>)
            return entries.length > 0 ? (
              <>
                <div className="small" style={{ marginTop: 8, fontWeight: 650 }}>Signal metrics</div>
                <div className="kvTable" style={{ marginTop: 4 }}>
                  {entries.map(([k, v]) => (
                    <React.Fragment key={k}>
                      <div className="k">{METRIC_LABELS[k] || k.replace(/_/g, ' ')}</div>
                      <div className="v">{fmtValue(k, v)}</div>
                    </React.Fragment>
                  ))}
                </div>
              </>
            ) : null
          })() : null}
        </>
      ) : null}

      {persistedDocLinks.length > 0 ? (
        <>
          <div className="hr" />
          <div style={{ fontWeight: 700 }}>Documentation links</div>
          <div className="small" style={{ color: 'var(--muted)' }}>
            Persisted citations proposed for this memory.
          </div>
          <div style={{ marginTop: 8 }}>
            <DocLinksSection docLinks={persistedDocLinks} limit={5} memoryId={typeof memory?.id === 'string' ? memory.id : null} />
          </div>
        </>
      ) : null}

      {/* ── LLM Explanation (grounded, detailed) ───────────────────── */}
      {!hasLiveAlert && (() => {
        // Explanation may come from the alert (WS push) or the memory metadata (persisted)
        const explanation = String((alert as Record<string, unknown>)?.explanation || meta.explanation || '')
        const explanationSource = String((alert as Record<string, unknown>)?.explanation_source || meta.explanation_source || '')
        const alertLine = typeof meta.alert_line === 'string' ? meta.alert_line : ''
        const alertLineSource = typeof meta.alert_line_source === 'string' ? meta.alert_line_source : ''
        if (!explanation && !alertLine) return null

        return (
          <>
            <div className="hr" />
            <div className="hrow" style={{ justifyContent: 'space-between' }}>
              <div style={{ fontWeight: 700 }}>
                LLM Explanation
                {explanationSource ? <span className="badge" style={{ marginLeft: 6, opacity: 0.7 }}>{String(explanationSource)}</span> : null}
              </div>
              {explanation && (
                <button className="small" onClick={() => setShowExplanation(v => !v)}>
                  {showExplanation ? 'Collapse' : 'Expand details'}
                </button>
              )}
            </div>

            {/* Short alert line (always visible) */}
            {alertLine ? (
              <div style={{ padding: '6px 0', fontWeight: 600, lineHeight: 1.4 }}>
                {alertLine}
                {alertLineSource ? <span className="badge" style={{ marginLeft: 6, opacity: 0.5, fontSize: 10 }}>{alertLineSource}</span> : null}
              </div>
            ) : null}

            {/* Detailed grounded explanation (expandable) */}
            {showExplanation && explanation ? (
              <div style={{
                padding: '10px 12px',
                background: 'rgba(122,162,247,0.06)',
                borderRadius: 6,
                marginTop: 4,
                lineHeight: 1.55,
                fontSize: 13,
                whiteSpace: 'pre-wrap',
              }}>
                {explanation}
              </div>
            ) : null}
          </>
        )
      })()}

      {/* ── Feature Evidence ───────────────────────────────────────── */}
      {(() => {
        const sigObj = meta.significance && typeof meta.significance === 'object' ? meta.significance as Record<string, unknown> : null
        const evidence = meta.feature_evidence || (sigObj ? sigObj.feature_evidence : null)
        if (!evidence || typeof evidence !== 'object') return null
        const entries = Object.entries(evidence as Record<string, unknown[]>).filter(([, v]) => Array.isArray(v) && v.length > 0)
        if (!entries.length) return null

        return (
          <>
            <div className="hr" />
            <div className="hrow" style={{ alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
              <div style={{ fontWeight: 700 }}>Feature evidence</div>
              <MemoryGraphLink
                memoryId={typeof memory?.id === 'string' ? memory.id : null}
                style={{ fontSize: 11, color: '#7aa2f7', textDecoration: 'none', whiteSpace: 'nowrap' }}
              />
            </div>
            <div className="small" style={{ color: 'var(--muted)' }}>Sensor features that triggered each pattern detection.</div>
            {entries.map(([pattern, feats]) => (
              <div key={pattern} style={{ marginTop: 8 }}>
                <div className="small" style={{ fontWeight: 650 }}>{humanPattern(pattern)}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px 80px 70px', gap: '2px 8px', fontSize: 11, marginTop: 4 }}>
                  <div className="small" style={{ fontWeight: 600, color: 'var(--muted)' }}>Feature</div>
                  <div className="small" style={{ fontWeight: 600, color: 'var(--muted)', textAlign: 'right' }}>Value</div>
                  <div className="small" style={{ fontWeight: 600, color: 'var(--muted)', textAlign: 'right' }}>Threshold</div>
                  <div className="small" style={{ fontWeight: 600, color: 'var(--muted)', textAlign: 'right' }}>Exceeded</div>
                  {(feats as Array<Record<string, unknown>>).map((f, i) => {
                    const val = typeof f.value === 'number' ? f.value : 0
                    const thr = typeof f.threshold === 'number' ? f.threshold : 0
                    const pct = thr > 0 ? ((val - thr) / thr * 100) : 0
                    const feature = String(f.feature || '').replace(/_/g, ' ')
                    return (
                      <React.Fragment key={i}>
                        <div className="small">{feature}</div>
                        <div className="small" style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{val.toFixed(2)}</div>
                        <div className="small" style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{thr > 0 ? thr.toFixed(2) : '–'}</div>
                        <div className="small" style={{
                          textAlign: 'right',
                          fontVariantNumeric: 'tabular-nums',
                          color: pct > 0 ? 'var(--danger)' : pct < 0 ? 'var(--ok)' : undefined,
                          fontWeight: 600,
                        }}>
                          {thr > 0 ? `${pct > 0 ? '+' : ''}${pct.toFixed(0)}%` : '–'}
                        </div>
                      </React.Fragment>
                    )
                  })}
                </div>
              </div>
            ))}
          </>
        )
      })()}

      {/* ── Discovered / Suppressed pattern badges ─────────────────── */}
      {(() => {
        const discovered = patterns.filter((p: string) => patternOrigin(String(p)) === 'detected' || String(p).startsWith('discovered:'))
        const suppressed = patterns.filter((p: string) => String(p).startsWith('suppressed:'))
        if (!discovered.length && !suppressed.length) return null

        return (
          <>
            <div className="hr" />
            <div style={{ fontWeight: 700 }}>Pattern discovery</div>
            <div className="small" style={{ color: 'var(--muted)' }}>Patterns learned from operator feedback.</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {discovered.map((p: string) => (
                <span key={p} className="badge" style={{ color: 'var(--ok)', borderColor: 'rgba(158,206,106,0.3)' }} title={p}>
                  ✦ {humanPattern(p)}
                </span>
              ))}
              {suppressed.map((p: string) => (
                <span key={p} className="badge" style={{ color: 'var(--danger)', borderColor: 'rgba(247,118,142,0.3)', textDecoration: 'line-through' }} title={p}>
                  ✗ {humanPattern(p)}
                </span>
              ))}
            </div>
          </>
        )
      })()}

      <div className="hr" />

      <div className="row">
        <div>
          <div style={{ fontWeight: 700 }}>Feedback stats</div>
          {feedbackStats && typeof feedbackStats === 'object' ? (
            <div className="kvTable" style={{ marginTop: 4 }}>
              {Object.entries(feedbackStats as Record<string, unknown>)
                .filter(([, v]) => v !== null && v !== undefined)
                .map(([k, v]) => (
                  <React.Fragment key={k}>
                    <div className="k">{k.replace(/_/g, ' ')}</div>
                    <div className="v" style={{
                      fontVariantNumeric: 'tabular-nums',
                      color: k === 'net_significance' ? (Number(v) > 0 ? 'var(--ok)' : Number(v) < 0 ? 'var(--danger)' : undefined) : undefined,
                    }}>{String(v)}</div>
                  </React.Fragment>
                ))}
            </div>
          ) : <div className="small">(no stats yet)</div>}
        </div>
        <div>
          <div className="hrow" style={{ justifyContent: 'space-between' }}>
            <div style={{ fontWeight: 700 }}>Trace summary</div>
            <div className="small">count={traceSummaries.length}</div>
          </div>

          {traceSummaries.length === 0 ? (
            <div className="small">No traces recorded yet.</div>
          ) : (
            <>
              <div className="small">
                Latest score: {latestScore?.action || '(n/a)'} {typeof latestScore?.score === 'number' ? `score=${latestScore.score.toFixed(3)}` : ''}
              </div>
              <div className="small">
                Latest retrieve: {typeof latestRetrieve?.returnedCount === 'number' ? `returned=${latestRetrieve.returnedCount}` : '(n/a)'}
              </div>

              <div className="hr" />

              <div className="traceList">
                {traceSummaries.slice(0, 20).map((t) => {
                  const isExpanded = expandedTraceId === t.id
                  return (
                    <div key={t.id || `${t.trace_type}-${t.created_at}`}
                      className="traceCard"
                    >
                      <div className="hrow" style={{ justifyContent: 'space-between' }}>
                        <div style={{ fontWeight: 650 }}>
                          <span className="badge">{t.trace_type}</span>
                          {t.action ? <span className="badge">{t.action}</span> : null}
                          {typeof t.score === 'number' ? <span className="badge">score={t.score.toFixed(3)}</span> : null}
                          {typeof t.returnedCount === 'number' ? <span className="badge">returned={t.returnedCount}</span> : null}
                        </div>
                        <div className="small">{fmtTs(t.created_at) || t.created_at}</div>
                      </div>

                      {t.trace_type === 'score' && t.reasons && t.reasons.length ? (
                        <div className="small">reasons: {t.reasons.slice(0, 3).map(humanReason).join(' • ')}{t.reasons.length > 3 ? ' …' : ''}</div>
                      ) : null}

                      {t.trace_type === 'retrieve' && t.topReturned?.memory_id ? (
                        <div className="small">
                          top: {t.topReturned.memory_id.slice(0, 10)}
                          {typeof t.topReturned.score === 'number' ? ` (score=${t.topReturned.score.toFixed(3)})` : ''}
                        </div>
                      ) : null}

                      <div className="hrow" style={{ marginTop: 8 }}>
                        <button onClick={() => setExpandedTraceId(isExpanded ? '' : t.id)}>{isExpanded ? 'Hide raw' : 'Show raw'}</button>
                      </div>

                      {isExpanded ? <pre>{JSON.stringify(t.raw, null, 2)}</pre> : null}
                    </div>
                  )
                })}
              </div>
              {traceSummaries.length > 20 ? <div className="small">Showing newest 20 traces.</div> : null}
            </>
          )}
        </div>
      </div>

      {Array.isArray(feedbackEvents) && feedbackEvents.length > 0 ? (
        <>
          <div className="hr" />
          <div style={{ fontWeight: 700 }}>Feedback events</div>
          <div className="small">count={feedbackEvents.length}</div>
          <div className="traceList" style={{ marginTop: 4 }}>
            {feedbackEvents.map((ev: Record<string, unknown>, idx: number) => {
              const action = String(ev?.action || 'unknown')
              const userId = String(ev?.user_id || '')
              const ts = fmtTs(ev?.created_at || ev?.timestamp)
              const comment = typeof ev?.comment === 'string' ? ev.comment.trim() : ''
              const reason = typeof ev?.reason === 'string' ? ev.reason.trim() : ''
              const evLabel = typeof ev?.label === 'string' ? ev.label.trim() : ''
              const tags = Array.isArray(ev?.tags) ? (ev.tags as string[]) : []
              const actionColor =
                action === 'confirm' ? 'var(--ok)' :
                action === 'dismiss' ? 'var(--danger)' :
                undefined

              return (
                <div key={String(ev?.id || `fb-${idx}`)} className="traceCard">
                  <div className="hrow" style={{ justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                      <span className="badge" style={{ color: actionColor }}>{action}</span>
                      {userId ? <span className="badge">{userId}</span> : null}
                    </div>
                    <div className="small">{ts}</div>
                  </div>
                  {reason ? <div className="small" style={{ marginTop: 4 }}>reason: {reason}</div> : null}
                  {comment ? <div className="small" style={{ marginTop: 2 }}>comment: {comment}</div> : null}
                  {evLabel ? <div className="small" style={{ marginTop: 2 }}>label: {evLabel}</div> : null}
                  {tags.length > 0 ? (
                    <div className="small" style={{ marginTop: 2 }}>
                      tags: {tags.map((t: string) => <span key={t} className="badge" style={{ marginLeft: 2 }}>{t}</span>)}
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        </>
      ) : null}

      {showRaw ? (
        <>
          <div className="hr" />
          <div style={{ fontWeight: 700 }}>Raw</div>
          <div className="small">Full payloads (debug)</div>
          <pre>{JSON.stringify({ memory, feedback, traces }, null, 2)}</pre>
        </>
      ) : null}
    </div>
  )
}

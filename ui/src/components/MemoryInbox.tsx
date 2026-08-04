/**
 * MemoryInbox — memory list with filters + review history.
 *
 * Extracted from the right panel of App.tsx's detailed view.
 */
import React, { useMemo, useState } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import type { SignificantEventAlert } from '../state/alertsStore'
import type { ListMemoriesResponse } from '../contexts/AppContext'
import { humanPatterns, humanReason } from '../utils/patternNames'
import { alertHistoryBadge } from '../utils/alerts'
import { severity } from '../hooks/useFeedback'
import type { ReviewEntry, ReviewHistoryEntry } from '../types'

interface Props {
  memoriesQuery: UseQueryResult<ListMemoriesResponse>
  alerts: SignificantEventAlert[]
  streamSessionId: string
  selectedMemoryId: string
  setSelectedMemoryId: (id: string) => void
  autoOpenAlerts: boolean
  setAutoOpenAlerts: (v: boolean) => void
  reviewedById: Record<string, ReviewEntry>
  reviewHistory: ReviewHistoryEntry[]
}

export function MemoryInbox({
  memoriesQuery,
  alerts,
  streamSessionId,
  selectedMemoryId,
  setSelectedMemoryId,
  autoOpenAlerts,
  setAutoOpenAlerts,
  reviewedById,
  reviewHistory,
}: Props) {
  const [showReviewed, setShowReviewed] = useState(false)
  const [onlyAlerted, setOnlyAlerted] = useState(false)
  const [maxVisibleMemories, setMaxVisibleMemories] = useState(15)

  const alertedById = useMemo(() => {
    const out: Record<string, SignificantEventAlert> = {}
    for (const a of alerts || []) {
      if (a && typeof a.event_id === 'string') out[a.event_id] = a
    }
    return out
  }, [alerts])

  const memoryRows = useMemo(() => {
    const all = (memoriesQuery.data?.memories || [])
      .filter((m) => m && typeof m.id === 'string')
      .map((m) => ({
        ...m,
        _reviewed: Boolean(reviewedById[m.id]),
        _reviewAction: reviewedById[m.id]?.action,
        _alert: alertedById[m.id],
      }))

    const filtered = all
      .filter((m) => (showReviewed ? true : !m._reviewed))
      .filter((m) => (onlyAlerted ? Boolean(m._alert) : true))

    // newest first
    filtered.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
    return filtered.slice(0, Math.max(1, maxVisibleMemories))
  }, [memoriesQuery.data, reviewedById, alertedById, showReviewed, onlyAlerted, maxVisibleMemories])

  return (
    <>
      <div className="hrow" style={{ justifyContent: 'space-between' }}>
        <div style={{ fontWeight: 700 }}>Alerts + Memories</div>
        <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input type="checkbox" checked={autoOpenAlerts} onChange={(e) => setAutoOpenAlerts(e.target.checked)} />
          Auto-open new alerts
        </label>
      </div>

      <div className="hr" />

      {/* Inline AlertsPanel replacement — delegated via import */}
      {/* The AlertsPanel is rendered in the parent to keep its props interface stable */}

      <div className="hr" />

      <div style={{ fontWeight: 700 }}>Memory inbox</div>
      <div className="small">
        session={streamSessionId || '(none)'} • fetched={memoriesQuery.data?.memories?.length ?? 0} • total={memoriesQuery.data?.total_count ?? 0}
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input type="checkbox" checked={showReviewed} onChange={(e) => setShowReviewed(e.target.checked)} />
          Show reviewed
        </label>
        <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input type="checkbox" checked={onlyAlerted} onChange={(e) => setOnlyAlerted(e.target.checked)} />
          Only alerted
        </label>
        <div>
          <div className="small">Max visible</div>
          <select
            value={String(maxVisibleMemories)}
            onChange={(e) => setMaxVisibleMemories(Number(e.target.value) || 15)}
          >
            <option value="10">10</option>
            <option value="15">15</option>
            <option value="25">25</option>
            <option value="50">50</option>
          </select>
        </div>
        <div />
      </div>

      {memoriesQuery.isError && (
        <div className="panel" style={{ marginTop: 10, borderColor: 'rgba(247, 118, 142, 0.35)' }}>
          <div style={{ fontWeight: 700 }}>Memories load error</div>
          <div className="small">{String(memoriesQuery.error)}</div>
          <div className="hrow" style={{ marginTop: 8 }}>
            <button onClick={() => memoriesQuery.refetch()}>Retry memories</button>
          </div>
        </div>
      )}

      <div className="hr" />

      {memoryRows.length === 0 ? (
        <div className="small">No matching memories yet. Wait for alerts or capture a window.</div>
      ) : (
        <div className="memoryList">
          {memoryRows.map((m: Record<string, unknown> & { id: string; created_at: string; _reviewed: boolean; _reviewAction?: string; _alert?: SignificantEventAlert; patterns?: string[]; tags?: string[]; label?: string | null; annotation_preview?: string | null }) => {
            const isSelected = selectedMemoryId && m.id === selectedMemoryId
            const alert = m._alert
            const isAlerted = Boolean(alert)
            const isUnread = Boolean(alert?._unread)
            const sigScoreNum = typeof alert?.significance?.score === 'number' ? alert.significance.score : undefined
            const sigScore = typeof sigScoreNum === 'number' ? sigScoreNum.toFixed(2) : ''
            const sev = alert?.severity || severity(sigScoreNum)
            const sevLabel = typeof sev === 'string' ? sev : (sev as { label: string }).label
            const sevColor = sevLabel === 'CRITICAL' ? 'var(--danger)' : sevLabel === 'WARNING' ? 'var(--accent)' : 'var(--muted)'
            const cat = alert?.category || ''
            const historyBadge = alert ? alertHistoryBadge(alert) : null
            const alertSummary = typeof alert?.summary === 'string' ? alert.summary.trim() : ''
            const alertTopReason = Array.isArray(alert?.significance?.reasons) ? humanReason(String(alert.significance.reasons[0] || '')) : ''
            const headline = alertSummary || alertTopReason || ''
            const humanPats = humanPatterns(((m.patterns as string[]) || []).slice(0, 3))
            const tags = ((m.tags as string[]) || []).slice(0, 4).join(', ')

            return (
              <button
                key={m.id}
                className={`memoryItem${isAlerted ? ' alerted' : ''}${m._reviewed ? ' reviewed' : ''}${isSelected ? ' selected' : ''}`}
                onClick={() => setSelectedMemoryId(m.id)}
                title={headline || (m.annotation_preview as string) || ''}
              >
                {headline ? <div style={{ fontWeight: 650, lineHeight: 1.35 }}>{headline}</div> : null}

                <div className="hrow" style={{ justifyContent: 'space-between', marginTop: headline ? 4 : 0 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                    {isAlerted ? <span className="badge" style={{ color: sevColor, borderColor: 'rgba(255,255,255,0.10)' }}>{sevLabel}</span> : null}
                    {isAlerted && cat ? <span className="badge">{cat}</span> : null}
                    {sigScore ? <span className="badge">{sigScore}</span> : null}
                    {isAlerted && historyBadge ? <span className="badge">{historyBadge}</span> : null}
                    {isAlerted && isUnread ? <span className="badge" style={{ color: 'var(--accent)' }}>new</span> : null}
                    {m.label ? <span className="badge">{m.label}</span> : null}
                    {m._reviewed ? <span className="badge">{m._reviewAction}</span> : null}
                  </div>
                  <div className="small">{String(m.created_at).replace('T', ' ').slice(0, 19)}</div>
                </div>

                {!headline && humanPats.length ? <div className="small" style={{ marginTop: 2 }}>{humanPats.join(' · ')}</div> : null}
                {tags ? <div className="small" style={{ marginTop: 2, opacity: 0.7 }}>tags: {tags}</div> : null}
                {m.annotation_preview && !headline ? <div className="small" style={{ marginTop: 2 }}>{m.annotation_preview}</div> : null}
              </button>
            )
          })}
        </div>
      )}

      {reviewHistory.length > 0 && !showReviewed ? (
        <>
          <div className="hr" />
          <div style={{ fontWeight: 700 }}>History</div>
          <div className="small">Recently reviewed (click to reopen)</div>
          <div className="memoryList" style={{ maxHeight: 180 }}>
            {reviewHistory.slice(0, 30).map((h) => (
              <button key={`${h.id}-${h.at}`} className="memoryItem reviewed" onClick={() => setSelectedMemoryId(h.id)}>
                <div className="hrow" style={{ justifyContent: 'space-between' }}>
                  <div style={{ fontWeight: 650 }}>
                    <span className="badge">{h.action}</span>
                    {h.reason ? <span className="badge">{h.reason}</span> : null}
                  </div>
                  <div className="small">{new Date(h.at).toLocaleTimeString()}</div>
                </div>
                <div className="small">{h.id.slice(0, 10)}</div>
              </button>
            ))}
          </div>
        </>
      ) : null}
    </>
  )
}

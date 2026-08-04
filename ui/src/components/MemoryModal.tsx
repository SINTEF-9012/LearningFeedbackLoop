/**
 * MemoryModal — modal backdrop + feedback form + prior diff display
 *              + MemoryDetailPanel + shared alert detail + ChatPanel.
 *
 * Extracted from the `selectedMemoryId &&` modal block in App.tsx.
 */
import React, { useRef, useState, useEffect } from 'react'
import { AlertDetailContent } from './AlertDetailContent'
import { MemoryDetailPanel } from './MemoryDetailPanel'
import ChatPanel from './ChatPanel'
import { humanPattern } from '../utils/patternNames'
import { extractMemoryPatterns } from '../hooks/useFeedback'
import type { SignificantEventAlert } from '../state/alertsStore'
import type { UseQueryResult } from '@tanstack/react-query'
import type { PriorDiffRow, MemoryDetailResponse, FeedbackHistoryResponse, TraceListResponse } from '../contexts/AppContext'

// One-click example notes (T1.6) — illustrate how an operator's floor
// observation maps to feedback on an alert. Click a chip to fill the comment.
const SUGGESTED_NOTES: string[] = [
  'Confirmed — chatter matches audible tool noise on the floor.',
  'False alarm — vibration is from the fixture, not the cut.',
  'Eased off feed and it settled — good catch.',
]

interface Props {
  selectedMemoryId: string
  onClose: () => void

  /* feedback form state (from useFeedback hook) */
  feedbackReason: string
  setFeedbackReason: (v: string) => void
  feedbackComment: string
  setFeedbackComment: (v: string) => void
  feedbackLabel: string
  setFeedbackLabel: (v: string) => void
  feedbackTags: string
  setFeedbackTags: (v: string) => void
  feedbackPending: boolean
  feedbackMsg: { kind: 'ok' | 'err' | 'info'; text: string } | null
  lastPriorDiff: PriorDiffRow[]
  lastPriorDiffAt: number
  lastFeedbackMeta: { action: string; memoryId: string } | null
  sendFeedback: (action: 'confirm' | 'dismiss', aspect?: 'explanation' | 'recommendation') => Promise<void>
  applyMetadataOnly: () => Promise<void>
  reportMissedEvent: (sessionId: string, patternKeys: string[]) => Promise<void>

  /* queries */
  memoryDetailQuery: UseQueryResult<MemoryDetailResponse>
  feedbackHistoryQuery: UseQueryResult<FeedbackHistoryResponse>
  tracesQuery: UseQueryResult<TraceListResponse>

  /* alert */
  selectedAlert?: SignificantEventAlert

  /* pause-by-alert */
  pausedByAlert: { at: number; memoryId: string } | null
  onHandleAlert: (memoryId: string) => void
  onClearPausedByAlert: () => void
  onResume: () => Promise<void>

  /* highlight */
  detailHighlight?: boolean
}

export function MemoryModal({
  selectedMemoryId,
  onClose,
  feedbackReason, setFeedbackReason,
  feedbackComment, setFeedbackComment,
  feedbackLabel, setFeedbackLabel,
  feedbackTags, setFeedbackTags,
  feedbackPending, feedbackMsg,
  lastPriorDiff, lastPriorDiffAt, lastFeedbackMeta,
  sendFeedback, applyMetadataOnly, reportMissedEvent,
  memoryDetailQuery, feedbackHistoryQuery, tracesQuery,
  selectedAlert,
  pausedByAlert, onHandleAlert, onClearPausedByAlert, onResume,
  detailHighlight,
}: Props) {
  const detailRef = useRef<HTMLDivElement>(null)

  // Close modal with Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      if (pausedByAlert?.memoryId) onHandleAlert(pausedByAlert.memoryId)
      onClearPausedByAlert()
      onClose()
    }
  }

  const handleCloseClick = () => {
    if (pausedByAlert?.memoryId) onHandleAlert(pausedByAlert.memoryId)
    onClearPausedByAlert()
    onClose()
  }

  return (
    <div className="modalBackdrop" onClick={handleBackdropClick}>
      <div
        ref={detailRef}
        className={`modalContent${detailHighlight ? ' detailHighlight' : ''}`}
      >
        <div className="hrow" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <div style={{ fontWeight: 750, fontSize: 16 }}>Memory Detail</div>
          <button
            onClick={handleCloseClick}
            style={{ padding: '4px 12px', fontSize: 18, lineHeight: 1 }}
            title="Close"
          >
            x
          </button>
        </div>

        {/* Status message */}
        {feedbackMsg ? (
          <div
            className="panel"
            style={{
              marginBottom: 10,
              borderColor:
                feedbackMsg.kind === 'ok'
                  ? 'rgba(158, 206, 106, 0.35)'
                  : feedbackMsg.kind === 'err'
                    ? 'rgba(247, 118, 142, 0.35)'
                    : 'rgba(122, 162, 247, 0.25)',
            }}
          >
            <div style={{ fontWeight: 700 }}>
              {feedbackMsg.kind === 'ok' ? 'Success' : feedbackMsg.kind === 'err' ? 'Error' : 'Status'}
            </div>
            <div className="small">{feedbackMsg.text}</div>
          </div>
        ) : null}

        {selectedAlert ? (
          <>
            <div className="hr" />
            <AlertDetailContent
              alert={selectedAlert}
              onAspectFeedback={(action, aspect) => { void sendFeedback(action, aspect) }}
              aspectPending={feedbackPending}
            />
            <div className="hr" />
          </>
        ) : null}

        {/* Suggested notes (T1.6): one-click example feedback so the operator —
            and a demo viewer — can see how a note maps to what's on the floor. */}
        <div className="hrow" style={{ gap: 6, flexWrap: 'wrap', marginBottom: 6, alignItems: 'center' }}>
          <span className="small" style={{ color: 'var(--muted)' }}>Suggested notes:</span>
          {SUGGESTED_NOTES.map((note) => (
            <button
              key={note}
              type="button"
              onClick={() => setFeedbackComment(note)}
              title="Insert this example note"
              className="small"
              style={{
                padding: '2px 10px',
                borderRadius: 999,
                border: '1px solid var(--border)',
                background: 'transparent',
                color: 'var(--muted)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              {note}
            </button>
          ))}
        </div>

        {/* Feedback form — lead with one clear "your feedback" field so it's
            obvious where the operator adds detail; the operational
            reason/label/tags are secondary and tucked into a disclosure so the
            form doesn't overwhelm on open. */}
        <div
          style={{
            marginBottom: 10,
            padding: '10px 12px',
            borderLeft: '3px solid var(--accent)',
            background: 'rgba(122, 162, 247, 0.06)',
            borderRadius: 6,
          }}
        >
          <label
            htmlFor="feedbackComment"
            style={{ display: 'block', fontWeight: 700, fontSize: 13, marginBottom: 4 }}
          >
            Your feedback
            <span className="small" style={{ fontWeight: 400, color: 'var(--muted)' }}>
              {' '}— what did you see or hear on the floor? Optional, but it sharpens future alerts.
            </span>
          </label>
          <textarea
            id="feedbackComment"
            value={feedbackComment}
            onChange={(e) => setFeedbackComment(e.target.value)}
            placeholder="e.g., Chatter matched audible tool noise — eased off the feed and it settled."
            rows={2}
            style={{ width: '100%', resize: 'vertical' }}
          />

          <details style={{ marginTop: 8 }}>
            <summary className="small" style={{ cursor: 'pointer', color: 'var(--muted)' }}>
              Advanced fields (reason · label · tags)
            </summary>
            <div className="row" style={{ marginTop: 8 }}>
              <div>
                <div className="small">Reason (used for confirm/dismiss)</div>
                <input value={feedbackReason} onChange={(e) => setFeedbackReason(e.target.value)} placeholder="e.g., false positive: vibration modulation" />
              </div>
              <div>
                <div className="small">Label</div>
                <input value={feedbackLabel} onChange={(e) => setFeedbackLabel(e.target.value)} placeholder="e.g., spindle shift review" />
              </div>
              <div>
                <div className="small">Tags (comma-separated)</div>
                <input value={feedbackTags} onChange={(e) => setFeedbackTags(e.target.value)} placeholder="e.g., operator, reviewed" />
              </div>
            </div>
          </details>
        </div>

        {/* Action buttons */}
        <div className="hrow">
          <button className="primary" onClick={() => sendFeedback('confirm')} disabled={feedbackPending}>
            {feedbackPending ? 'Working…' : 'Confirm'}
          </button>
          <button className="danger" onClick={() => sendFeedback('dismiss')} disabled={feedbackPending}>
            {feedbackPending ? 'Working…' : 'Dismiss'}
          </button>
          <button
            onClick={() => {
              const mem = memoryDetailQuery.data?.memory as Record<string, unknown> | undefined
              const sessionId = String(mem?.session_id || '')
              const pkeys = extractMemoryPatterns(mem)
              reportMissedEvent(sessionId, pkeys)
            }}
            disabled={feedbackPending}
            title="Report that the system should have flagged this event more aggressively. Lowers detection thresholds and boosts pattern priors."
            style={{ color: 'var(--accent)' }}
          >
            {feedbackPending ? 'Working…' : 'Report missed event'}
          </button>
          <button onClick={applyMetadataOnly} disabled={feedbackPending}>
            {feedbackPending ? 'Working…' : 'Apply note/label/tags'}
          </button>
          {pausedByAlert ? (
            <button
              style={{ marginLeft: 'auto' }}
              onClick={() => {
                if (pausedByAlert.memoryId) onHandleAlert(pausedByAlert.memoryId)
                onClearPausedByAlert()
                void onResume()
              }}
            >
              Resume playback
            </button>
          ) : null}
        </div>

        {/* Prior diff table */}
        {lastPriorDiff.length > 0 ? (
          <div className="panel" style={{ marginTop: 10, borderColor: 'rgba(122, 162, 247, 0.25)' }}>
            <div className="hrow" style={{ justifyContent: 'space-between' }}>
              <div style={{ fontWeight: 700 }}>Priors impact (from last feedback)</div>
              <div className="small">{lastPriorDiffAt ? new Date(lastPriorDiffAt).toLocaleTimeString() : ''}</div>
            </div>
            <div className="small">
              {lastFeedbackMeta ? `${lastFeedbackMeta.action} • ${lastFeedbackMeta.memoryId.slice(0, 10)}` : ''}
            </div>
            <div className="kvTable" style={{ marginTop: 8, gridTemplateColumns: '1fr 110px' }}>
              {lastPriorDiff.map((r) => (
                <React.Fragment key={r.pattern}>
                  <div
                    className="k"
                    title={r.pattern}
                    style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                  >
                    {humanPattern(r.pattern)}
                  </div>
                  <div
                    className="v"
                    style={{
                      textAlign: 'right',
                      fontVariantNumeric: 'tabular-nums',
                      color: r.delta > 0 ? 'var(--ok)' : 'var(--danger)',
                    }}
                  >
                    {r.delta > 0 ? `+${r.delta.toFixed(3)}` : r.delta.toFixed(3)}
                  </div>
                </React.Fragment>
              ))}
            </div>
          </div>
        ) : null}

        <div className="hr" />

        {/* Detail load errors */}
        {(memoryDetailQuery.isError || feedbackHistoryQuery.isError || tracesQuery.isError) && (
          <div className="panel" style={{ marginBottom: 10, borderColor: 'rgba(247, 118, 142, 0.35)' }}>
            <div style={{ fontWeight: 700 }}>Detail load error</div>
            <div className="small">
              {memoryDetailQuery.isError ? `memory: ${String(memoryDetailQuery.error)}` : ''}
            </div>
            <div className="small">
              {feedbackHistoryQuery.isError ? `feedback: ${String(feedbackHistoryQuery.error)}` : ''}
            </div>
            <div className="small">
              {tracesQuery.isError ? `traces: ${String(tracesQuery.error)}` : ''}
            </div>
            <div className="hrow" style={{ marginTop: 8 }}>
              <button onClick={() => memoryDetailQuery.refetch()} disabled={!memoryDetailQuery.isError}>
                Retry memory
              </button>
              <button onClick={() => feedbackHistoryQuery.refetch()} disabled={!feedbackHistoryQuery.isError}>
                Retry feedback
              </button>
              <button onClick={() => tracesQuery.refetch()} disabled={!tracesQuery.isError}>
                Retry traces
              </button>
            </div>
          </div>
        )}

        {/* Memory detail panel */}
        <MemoryDetailPanel
          memory={memoryDetailQuery.data?.memory ?? {}}
          feedback={feedbackHistoryQuery.data ?? {}}
          traces={tracesQuery.data?.traces ?? []}
          alert={selectedAlert}
          docLinks={selectedAlert?.doc_links?.length ? [] : (memoryDetailQuery.data?.doc_links ?? [])}
        />

        {/* LLM Chat */}
        <ChatPanel
          memoryId={selectedMemoryId}
          patterns={extractMemoryPatterns(memoryDetailQuery.data?.memory as Record<string, unknown>) || []}
        />
      </div>
    </div>
  )
}

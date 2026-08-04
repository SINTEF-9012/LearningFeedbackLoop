/**
 * OperatorPage — route wrapper for the Operator view.
 */
import React from 'react'
import { OperatorView } from '../components/OperatorView'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { useAppContext } from '../contexts/AppContext'
import { useAlertsStore } from '../state/alertsStore'
import { api } from '../api/http'

export default function OperatorPage() {
  const ctx = useAppContext()
  const lastAlert = useAlertsStore((s) => s.lastPushed)
  const markAllAlertsRead = useAlertsStore((s) => s.markAllRead)
  const removeAlert = useAlertsStore((s) => s.removeByEventId)
  const sessionInfo = (ctx.sessionInfoQuery.data as Record<string, unknown> | null) ?? null

  const refetchLearningQueries = async (memoryId?: string) => {
    const shouldRefreshSelectedDetail = Boolean(memoryId && ctx.selectedMemoryId === memoryId)

    await Promise.all([
      ctx.memoriesQuery.refetch(),
      ctx.priorsQuery.refetch(),
      shouldRefreshSelectedDetail ? ctx.memoryDetailQuery.refetch() : Promise.resolve(ctx.memoryDetailQuery),
      shouldRefreshSelectedDetail ? ctx.feedbackHistoryQuery.refetch() : Promise.resolve(ctx.feedbackHistoryQuery),
    ])
  }

  return (
    <ErrorBoundary label="Operator View">
      <OperatorView
        sessionId={ctx.streamSessionId}
        sessionInfo={sessionInfo}
        wsConnected={ctx.wsStatus === 'connected'}
        alerts={ctx.alerts}
        latestAlert={lastAlert}
        paused={sessionInfo?.paused === true}
        sessionStatus={typeof sessionInfo?.status === 'string' ? sessionInfo.status : null}
        sessionStatusLabel={typeof sessionInfo?.status_label === 'string' ? sessionInfo.status_label : null}
        sessionProgress={typeof sessionInfo?.progress === 'number' ? sessionInfo.progress : null}
        sessionLastError={typeof sessionInfo?.last_error === 'string' ? sessionInfo.last_error : null}
        feedbackPending={ctx.feedbackPending}
        onConfirm={async (eventId: string, reason?: string) => {
          const id = encodeURIComponent(eventId)
          // Episode-level learning dedup (plan 1.4).
          const episode_id = useAlertsStore.getState().alerts.find((a) => a.event_id === eventId)?.recurrence?.episode_id ?? undefined
          try {
            await api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'confirm', user_id: 'ui', reason: reason || null, episode_id })
            // Visible acknowledgement: drop the reviewed alert from the list.
            removeAlert(eventId)
            await refetchLearningQueries(eventId)
          } catch (e) {
            console.error('confirm feedback failed', e)
          }
        }}
        onDismiss={async (eventId: string, reason?: string, extra?: { severity_target?: 'info' | 'warning' | 'critical' }) => {
          const id = encodeURIComponent(eventId)
          const episode_id = useAlertsStore.getState().alerts.find((a) => a.event_id === eventId)?.recurrence?.episode_id ?? undefined
          try {
            await api(`/agent/memory/${id}/feedback`, 'PATCH', {
              action: 'dismiss',
              user_id: 'ui',
              reason: reason || null,
              severity_target: extra?.severity_target ?? null,
              episode_id,
            })
            // Visible acknowledgement: drop the reviewed alert from the list.
            removeAlert(eventId)
            await refetchLearningQueries(eventId)
          } catch (e) {
            console.error('dismiss feedback failed', e)
          }
        }}
        onMuteSignature={(alert, muted) => {
          return api('/agent/memory/alerts/signature-mute', 'POST', {
            session_id: alert.session_id,
            signature: alert.recurrence?.signature || null,
            muted,
            source: 'operator_ui',
            reason: muted
              ? 'operator muted recurring signature from operator view'
              : 'operator restored recurring signature alerts from operator view',
          }).then(() => undefined)
        }}
        onExplanationFeedback={(alert, helpful) => {
          return api('/agent/memory/feedback/explanation', 'POST', {
            memory_id: alert.event_id,
            helpful,
            operator_id: 'ui',
            session_id: alert.session_id,
            signature: alert.recurrence?.signature || null,
            summary_source: alert.summary_source || null,
            explanation_source: alert.explanation_source || null,
          }).then(() => undefined)
        }}
        onResume={() => void ctx.resume()}
        onCapture={() => {}}
        freezeOnAlert={ctx.freezeOnAlert}
        onToggleFreeze={ctx.setFreezeOnAlert}
        unreadCount={ctx.unreadCount}
        onMarkAllRead={markAllAlertsRead}
        focusMemoryId={ctx.selectedMemoryId}
        detailRequest={ctx.detailRequest}
        onDetailConsumed={ctx.clearAlertDetail}
      />
    </ErrorBoundary>
  )
}

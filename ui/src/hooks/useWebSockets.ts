/**
 * useWebSockets — Extract WS connection logic from App.tsx.
 *
 * Manages the three WebSocket connections: sensor stream, alerts, inference.
 * Uses the real `connectWs(url, onMsg, onStatus?)` signature from `../api/ws`.
 */
import { useEffect, useRef } from 'react'
import { baseUrl, wsUrl } from '../api/http'
import { connectWs } from '../api/ws'
import { useStreamStore } from '../state/streamStore'
import { useAlertsStore, type SignificantEventAlert } from '../state/alertsStore'
import { useInferenceStore, type InferencePoint } from '../state/inferenceStore'
import { useLiveScoreStore } from '../state/liveScoreStore'

function resolveStreamIndex(
  rangeIndex: number | undefined,
  currentX: number[],
  fs: number,
): number | undefined {
  const currentCursor = currentX.length > 0 ? currentX[currentX.length - 1] : undefined
  if (typeof rangeIndex !== 'number') return currentCursor
  if (typeof currentCursor !== 'number') return rangeIndex

  const currentStart = currentX[0]
  const maxExpectedLag = Math.max(10, Math.round(fs * 5))
  const isOutsideBufferedWindow = rangeIndex < currentStart || rangeIndex > currentCursor
  const isFarBehindLiveCursor = currentCursor - rangeIndex > maxExpectedLag

  return isOutsideBufferedWindow || isFarBehindLiveCursor ? currentCursor : rangeIndex
}

export function useWebSockets(sessionId: string) {
  const appendFrame = useStreamStore((s) => s.appendFrame)
  const resetStream = useStreamStore((s) => s.reset)
  const setWsStatus = useStreamStore((s) => s.setWsStatus)
  const setSessionId = useStreamStore((s) => s.setSessionId)
  const streamDownsample = useStreamStore((s) => s.streamDownsample)
  const pushAlert = useAlertsStore((s) => s.pushAlert)
  const pushScoredEvent = useAlertsStore((s) => s.pushScoredEvent)
  const consolidateAlert = useAlertsStore((s) => s.consolidateAlert)
  const updateExplanation = useAlertsStore((s) => s.updateExplanation)
  const clearAlerts = useAlertsStore((s) => s.clear)
  const pushInferencePoint = useInferenceStore((s) => s.push)
  const clearInference = useInferenceStore((s) => s.clear)
  const apiUrl = baseUrl()

  const pushLiveScores = (alert: SignificantEventAlert) => {
    const metrics = (alert.metrics ?? {}) as Record<string, number>
    const sig = alert.significance ?? {}
    const t = alert.timestamp ? new Date(alert.timestamp).getTime() / 1000 : Date.now() / 1000
    useLiveScoreStore.getState().push({
      t,
      significance_score: sig.score ?? 0,
      prior_boost: sig.prior_boost ?? metrics.prior_boost ?? 0,
      n_rules_triggered: metrics.n_rules_triggered ?? 0,
      anomaly_detector_score: metrics.anomaly_detector_score ?? 0,
    })
    if (sig.pattern_priors && typeof sig.pattern_priors === 'object') {
      useLiveScoreStore.getState().pushPriors({ t, priors: sig.pattern_priors })
    }
    // Keep the latest abstracted process snapshot (cutting context + metrics)
    // for at-a-glance overview surfaces like the landing "process pulse".
    useLiveScoreStore.getState().setLatest({
      t,
      context: (alert.context ?? {}) as Record<string, unknown>,
      metrics,
    })
  }

  // Stream WS
  useEffect(() => {
    if (!sessionId) return
    resetStream()
    // Append ?downsample=N when the store requests server-side LTTB
    // thinning. Backend contract: any value <= 2 (or missing) is
    // passthrough. See backend/routers/_stream_downsample.py.
    const q = streamDownsample > 2 ? `?downsample=${streamDownsample}` : ''
    const url = wsUrl(`/streams/${encodeURIComponent(sessionId)}${q}`)
    const c = connectWs<Record<string, unknown>>(
      url,
      (msg) => appendFrame(msg),
      (s) => {
        setWsStatus(s)
        // Server says session is gone — clear stale session immediately
        if (s === 'rejected') { setSessionId(''); resetStream() }
      },
    )
    return () => c.stop()
  }, [sessionId, apiUrl, streamDownsample, appendFrame, resetStream, setWsStatus, setSessionId])

  // Alerts WS
  const lastClearedSession = useRef<string | null>(null)
  useEffect(() => {
    if (!sessionId) return
    // Only wipe accumulated alerts when the session genuinely changes — NOT on
    // every reconnect (a transient WS drop / 403 must not erase the alerts the
    // operator is reviewing, e.g. an injected demo event).
    if (lastClearedSession.current !== sessionId) {
      clearAlerts()
      useLiveScoreStore.getState().clear()
      lastClearedSession.current = sessionId
    }
    const url = wsUrl(`/agent/memory/alerts/${encodeURIComponent(sessionId)}`)
    const c = connectWs<Record<string, unknown>>(
      url,
      (msg) => {
        const alert = msg as unknown as SignificantEventAlert
        if (!alert || typeof alert.event_id !== 'string') return
        const streamState = useStreamStore.getState().series
        const currentX = streamState.x
        const range = alert.time_range && typeof alert.time_range === 'object'
          ? alert.time_range
          : null
        const rangeIndex = typeof range?.i1 === 'number'
          ? Math.max(0, range.i1 - 1)
          : typeof range?.i0 === 'number'
            ? range.i0
            : undefined
        const streamIndex = resolveStreamIndex(rangeIndex, currentX, streamState.fs)
        if (alert.type === 'explanation_update') {
          // Background LLM explanation arrived — patch existing alert/scored event
          updateExplanation(alert.event_id, {
            explanation: alert.explanation,
            explanation_source: alert.explanation_source,
            summary: alert.summary,
            summary_source: alert.summary_source,
            recommendation: alert.recommendation,
          })
        } else if (alert.type === 'alert_consolidated') {
          // Fold into existing alert — no new toast
          consolidateAlert(alert.event_id, {
            patterns: alert.patterns,
            severity: alert.severity,
            significance: alert.significance,
            metrics: alert.metrics,
            _consolidated_count: (alert as any).consolidated_count,
            _consolidated_ids: (alert as any).consolidated_ids,
          })
        } else if (alert.type === 'scored_event') {
          pushScoredEvent(alert)
          pushLiveScores(alert)
        } else {
          pushAlert(alert, streamIndex)
        }
      },
      () => {},
    )
    return () => c.stop()
  }, [sessionId, apiUrl, pushAlert, pushScoredEvent, consolidateAlert, updateExplanation, clearAlerts])

  // Inference WS
  useEffect(() => {
    if (!sessionId) return
    clearInference()
    const url = wsUrl(`/sessions/${encodeURIComponent(sessionId)}/inference`)
    const c = connectWs<Record<string, unknown>>(
      url,
      (msg) => {
        if (msg && typeof (msg as Record<string, unknown>).t === 'number' && (msg as Record<string, unknown>).scores) {
          pushInferencePoint(msg as unknown as InferencePoint)
        }
      },
      () => {},
    )
    return () => c.stop()
  }, [sessionId, apiUrl, pushInferencePoint, clearInference])
}

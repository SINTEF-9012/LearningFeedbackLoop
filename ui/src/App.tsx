import { useCallback, useEffect, useMemo, useRef, useState, Suspense } from 'react'
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom'

import { api, wsUrl } from './api/http'
import { StreamPlot } from './components/StreamPlot'
import { PriorsChart } from './components/PriorsChart'
import { BackendReadyBadge } from './components/BackendReadyBadge'
import { AlertsPanel } from './components/AlertsPanel'
import { FaultIndicatorPanel } from './components/FaultIndicatorPanel'
import { ErrorBoundary } from './components/ErrorBoundary'
import { SessionControls } from './components/SessionControls'
import { PlaybackControls } from './components/PlaybackControls'
import { CapturePanel } from './components/CapturePanel'
import { MemoryInbox } from './components/MemoryInbox'
import { MemoryModal } from './components/MemoryModal'
import { DemoDirector } from './components/DemoDirector'
import { useStreamStore } from './state/streamStore'
import { useAlertsStore, type SignificantEventAlert } from './state/alertsStore'
import { alertCuttingContextEntries, alertEvidenceBadges, alertHeadline, alertOperatorExplanation, severityToneColor } from './utils/alerts'
import {
  LandingPage,
  BatchReviewPage,
  OperatorPage,
  InferencePage,
  HarmonicsPage,
  ExperimentPage,
  DatasetExplorerPage,
  DocumentRetrievalPage,
  DevelopmentPage,
  GraphPage,
  SinditPage,
  LearningsPage,
  SettingsPage,
} from './pages'
import { AppContext, type SessionSummary } from './contexts/AppContext'

import { useSessionQueries } from './hooks/useSessionQueries'
import { useWebSockets } from './hooks/useWebSockets'
import { useFeedback, severity } from './hooks/useFeedback'
import type { ReviewEntry, ReviewHistoryEntry } from './types'

function sessionStatusColor(status?: string): string {
  if (status === 'live') return 'var(--ok)'
  if (status === 'paused') return 'var(--accent)'
  if (status === 'completed') return '#7dcfff'
  if (status === 'error') return 'var(--danger)'
  return 'var(--muted)'
}

function formatSessionOptionLabel(session: SessionSummary): string {
  const sourceLabel = session.source_label || [session.case_dir, session.operation_id].filter(Boolean).join(' / ') || session.source || ''
  const progress =
    typeof session.progress === 'number' && Number.isFinite(session.progress) && session.total_samples > 0
      ? session.progress <= 0
        ? '0%'
        : session.progress < 0.001
          ? '<0.1%'
          : session.progress < 0.01
            ? `${(session.progress * 100).toFixed(1)}%`
            : `${Math.round(session.progress * 100)}%`
      : ''
  return [sourceLabel, session.session_id, session.status_label, progress].filter(Boolean).join(' · ')
}

type ViewMode =
  | 'home'
  | 'detailed'
  | 'operator'
  | 'batch'
  | 'inference'
  | 'harmonics'
  | 'experiment'
  | 'dataset'
  | 'documents'
  | 'development'
  | 'graph'
  | 'sindit'
  | 'learnings'
  | 'settings'

type ViewTab = {
  mode: ViewMode
  label: string
  operatorVisible: boolean
}

// operatorVisible = shown in the clean "Operator view" (audience A). The Operator
// path is deliberately short and value-forward: monitor a session, start one,
// and see what the feedback becomes (Learnings/MaaS). Inference/Harmonics are
// redundant live-score surfaces (see UI eval ISS-29) and live in Full view only.
// Full view (toggle off) exposes everything incl. Knowledge Graph + Digital Twin
// for "how does it work" Q&A.
const VIEW_TABS: ViewTab[] = [
  { mode: 'home', label: 'Home', operatorVisible: true },
  { mode: 'operator', label: 'Monitoring', operatorVisible: true },
  { mode: 'detailed', label: 'Detailed', operatorVisible: true },
  { mode: 'batch', label: 'Batch Review', operatorVisible: true },
  { mode: 'inference', label: 'Inference', operatorVisible: false },
  { mode: 'harmonics', label: 'Harmonics', operatorVisible: false },
  { mode: 'experiment', label: 'Experiment', operatorVisible: false },
  { mode: 'dataset', label: 'Dataset', operatorVisible: false },
  { mode: 'documents', label: 'Documents', operatorVisible: false },
  { mode: 'development', label: 'Development', operatorVisible: false },
  { mode: 'graph', label: 'Knowledge Graph', operatorVisible: false },
  { mode: 'sindit', label: 'Digital Twin', operatorVisible: false },
  { mode: 'learnings', label: 'Learnings', operatorVisible: true },
  { mode: 'settings', label: 'Settings', operatorVisible: true },
]

const OPERATOR_MODE_STORAGE_KEY = 'operatorMode'

function readStoredBoolean(key: string, fallback: boolean): boolean {
  if (typeof window === 'undefined') return fallback
  const raw = window.localStorage.getItem(key)
  if (raw == null) return fallback
  return raw === 'true'
}

export function App() {
  /* ── Stream store selectors ─────────────── */
  const streamSessionId = useStreamStore((s) => s.sessionId)
  const setStreamSessionId = useStreamStore((s) => s.setSessionId)
  const wsStatus = useStreamStore((s) => s.wsStatus)
  const resetStream = useStreamStore((s) => s.reset)
  const followTail = useStreamStore((s) => s.followTail)
  const setFollowTail = useStreamStore((s) => s.setFollowTail)
  const windowSeconds = useStreamStore((s) => s.windowSeconds)
  const setWindowSeconds = useStreamStore((s) => s.setWindowSeconds)

  /* ── Alert store selectors ──────────────── */
  const alerts = useAlertsStore((s) => s.alerts)
  const lastAlertAt = useAlertsStore((s) => s.lastPushedAt)
  const lastAlert = useAlertsStore((s) => s.lastPushed)
  const clearAlerts = useAlertsStore((s) => s.clear)
  const markAlertRead = useAlertsStore((s) => s.markRead)
  const markAllAlertsRead = useAlertsStore((s) => s.markAllRead)

  /* ── Routing ────────────────────────────── */
  const navigate = useNavigate()
  const location = useLocation()
  const viewMode = (location.pathname.replace(/^\//, '') || 'home') as ViewMode
  const setViewMode = (mode: ViewMode) => navigate(`/${mode}`)

  /* ── Local state ────────────────────────── */
  const [selectedMemoryId, setSelectedMemoryId] = useState('')
  const [detailRequest, setDetailRequest] = useState<{ id: string; at: number } | null>(null)
  const requestAlertDetail = useCallback((id: string) => {
    if (!id) return
    setDetailRequest({ id, at: Date.now() })
  }, [])
  const clearAlertDetail = useCallback(() => setDetailRequest(null), [])
  const [captureSel, setCaptureSel] = useState<{ i0: number; i1: number } | null>(null)

  const [operatorMode, setOperatorMode] = useState(() => readStoredBoolean(OPERATOR_MODE_STORAGE_KEY, false))
  const [autoOpenAlerts, setAutoOpenAlerts] = useState(true)
  const [freezeOnAlert, setFreezeOnAlertState] = useState(false)
  const [pausedByAlert, setPausedByAlert] = useState<null | { at: number; memoryId: string }>(null)
  const [handledAlertIds] = useState(() => new Set<string>())

  const [reviewedById, setReviewedById] = useState<Record<string, ReviewEntry>>({})
  const [reviewHistory, setReviewHistory] = useState<ReviewHistoryEntry[]>([])

  // Toast STACK: new alerts fold the previous one (kept, viewable) instead of
  // replacing it. The newest is expanded/focused; older ones minimize to a chip
  // and auto-hide after ~10s. See TOAST_EXPIRY_MS sweep below.
  type ToastItem = {
    id: string
    memoryId?: string
    severityLabel: string
    category?: string
    headline: string
    detail?: string
    preview?: boolean
    at: number
  }
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const [expandedToastId, setExpandedToastId] = useState<string | null>(null)
  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])
  const [alertsFlashUntil, setAlertsFlashUntil] = useState(0)
  const [detailHighlight, setDetailHighlight] = useState(false)
  const detailRef = useRef<HTMLDivElement>(null)

  /* ── Extracted hooks ────────────────────── */
  const {
    sessionsQuery, sessionInfoQuery, memoriesQuery, priorsQuery,
    memoryDetailQuery, tracesQuery, feedbackHistoryQuery, sessionOptions, sessionSummaries,
  } = useSessionQueries(selectedMemoryId)

  useWebSockets(streamSessionId)

  const feedbackState = useFeedback({
    selectedMemoryId,
    priorsQuery,
    memoryDetailQuery,
    feedbackHistoryQuery,
    memoriesQuery,
  })

  /* ── Playback actions ───────────────────── */
  const pause = async () => {
    if (!streamSessionId) return
    await api(`/sessions/${encodeURIComponent(streamSessionId)}/pause`, 'POST')
    await sessionInfoQuery.refetch()
  }

  const resume = async () => {
    if (!streamSessionId) return
    const info = sessionInfoQuery.data
    if (info && (info as Record<string, unknown>).running === false) {
      await api(`/sessions/${encodeURIComponent(streamSessionId)}/start`, 'POST')
    }
    await api(`/sessions/${encodeURIComponent(streamSessionId)}/resume`, 'POST')
    await sessionInfoQuery.refetch()
  }

  const replay = async () => {
    if (!streamSessionId) return
    resetStream()
    await api(`/sessions/${encodeURIComponent(streamSessionId)}/replay`, 'POST', { speed: 1.0 })
    await sessionInfoQuery.refetch()
  }

  /* ── Mark alert read when selecting memory ─ */
  useEffect(() => {
    if (!selectedMemoryId) return
    markAlertRead(selectedMemoryId)
  }, [selectedMemoryId, markAlertRead])

  /* ── Reset per-session UI state ─────────── */
  useEffect(() => {
    setSelectedMemoryId('')
    setReviewedById({})
    setReviewHistory([])
    setToasts([])
    setExpandedToastId(null)
    setPausedByAlert(null)
  }, [streamSessionId])

  useEffect(() => {
    const cfg = (sessionInfoQuery.data as Record<string, unknown> | null)?.config as Record<string, unknown> | undefined
    if (!cfg) return
    setFreezeOnAlertState(cfg.pause_on_alert === true)
  }, [streamSessionId, sessionInfoQuery.data])

  const setFreezeOnAlert = (next: boolean) => {
    const previous = freezeOnAlert
    setFreezeOnAlertState(next)
    if (!streamSessionId) return
    void api(`/sessions/${encodeURIComponent(streamSessionId)}/config`, 'PATCH', { pause_on_alert: next })
      .then(() => sessionInfoQuery.refetch())
      .catch(() => setFreezeOnAlertState(previous))
  }

  /* ── Surface new alerts (toast + flash + auto-open) ── */
  useEffect(() => {
    if (!lastAlertAt || !lastAlert) return
    const memoryId = lastAlert.event_id
    const alertContext = lastAlert.context && typeof lastAlert.context === 'object'
      ? lastAlert.context as Record<string, unknown>
      : null
    const isPreviewAlert = alertContext?.operator_preview === true
    const matchesSelectedSession = Boolean(streamSessionId) && lastAlert.session_id === streamSessionId

    if (!matchesSelectedSession && !isPreviewAlert) return

    if (handledAlertIds.has(memoryId)) return

    const scoreNum = typeof lastAlert.significance?.score === 'number' ? lastAlert.significance.score : undefined
    const sev = lastAlert.severity || severity(scoreNum).label
    const cat = lastAlert.category || ''
    const headline = alertHeadline(lastAlert)
    const detail = alertOperatorExplanation(lastAlert)

    const now = Date.now()
    const item: ToastItem = {
      id: `${memoryId}:${now}`,
      memoryId,
      severityLabel: sev,
      category: cat || undefined,
      headline,
      detail: detail && detail !== headline ? detail : undefined,
      preview: isPreviewAlert,
      at: now,
    }
    // Fold (don't replace): append the new toast, drop any earlier toast for the
    // same memory, cap the stack. The newest becomes the expanded/focused one;
    // the previous one minimizes but stays viewable until it auto-expires.
    setToasts((prev) => [...prev.filter((t) => t.memoryId !== memoryId), item].slice(-5))
    setExpandedToastId(item.id)
    setAlertsFlashUntil(Date.now() + 2500)
    if (autoOpenAlerts && memoryId && !isPreviewAlert) setSelectedMemoryId(memoryId)

    if (freezeOnAlert && matchesSelectedSession && streamSessionId) {
      setPausedByAlert({ at: Date.now(), memoryId })
      void sessionInfoQuery.refetch()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastAlertAt])

  /* ── Auto-expire toasts (~10s each, folded or focused) ────── */
  const TOAST_EXPIRY_MS = 10_000
  useEffect(() => {
    if (toasts.length === 0) return
    const t = window.setInterval(() => {
      setToasts((prev) => prev.filter((x) => Date.now() - x.at < TOAST_EXPIRY_MS))
    }, 1000)
    return () => window.clearInterval(t)
  }, [toasts.length])

  /* ── Track reviewed memories via feedback ── */
  const originalSendFeedback = feedbackState.sendFeedback
  const wrappedSendFeedback = async (action: 'confirm' | 'dismiss', aspect?: 'explanation' | 'recommendation') => {
    const reviewedId = selectedMemoryId
    const ok = await originalSendFeedback(action, aspect)
    if (!ok) return
    // Per-aspect ratings (explanation / recommendation "helpful?") record quality
    // feedback only — they don't adjudicate the alert, so keep the modal open and
    // don't mark it reviewed. Only whole-alert confirm/dismiss does that.
    if (aspect) return
    setReviewedById((m) => ({ ...m, [reviewedId]: { action, at: Date.now() } }))
    setReviewHistory((h) => [
      { id: reviewedId, action, at: Date.now(), reason: feedbackState.feedbackReason || undefined },
      ...h,
    ].slice(0, 200))
    // Auto-close the alert once feedback is recorded — brief delay so the
    // operator sees the "Confirmed/Dismissed ✓" acknowledgement first.
    window.setTimeout(() => {
      setSelectedMemoryId((current) => (current === reviewedId ? '' : current))
    }, 900)
  }

  /* ── Derived values ─────────────────────── */
  const alertsFlash = alertsFlashUntil > Date.now()
  const unreadCount = alerts.filter((a) => a._unread).length

  const alertedById = useMemo(() => {
    const out: Record<string, SignificantEventAlert> = {}
    for (const a of alerts || []) {
      if (a && typeof a.event_id === 'string') out[a.event_id] = a
    }
    return out
  }, [alerts])

  // Open one toast → focus/act on its event, then remove that toast from the
  // stack (the alert itself lives on in the store / Recent list / graph).
  const openToast = (item: ToastItem) => {
    const isPreview = item.memoryId && alertedById[item.memoryId]?.context
      && typeof alertedById[item.memoryId]?.context === 'object'
      ? (alertedById[item.memoryId]!.context as Record<string, unknown>).operator_preview === true
      : Boolean(item.preview)
    const canOpen = Boolean(item.memoryId) && !isPreview
    if (canOpen && item.memoryId) {
      setSelectedMemoryId(item.memoryId)
      if (viewMode !== 'detailed') {
        requestAlertDetail(item.memoryId)
        setViewMode('operator')
        dismissToast(item.id)
        return
      }
      dismissToast(item.id)
      requestAnimationFrame(() => {
        setDetailHighlight(true)
        detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        setTimeout(() => setDetailHighlight(false), 1800)
      })
      return
    }
    if (isPreview) {
      setViewMode('operator')
      dismissToast(item.id)
    }
  }

  const selectedAlert = useMemo(() => {
    if (!selectedMemoryId) return undefined
    return alertedById[selectedMemoryId]
  }, [selectedMemoryId, alertedById])

  const selectedSessionSummary = useMemo(
    () => sessionSummaries.find((session) => session.session_id === streamSessionId) || null,
    [sessionSummaries, streamSessionId],
  )

  const visibleViewTabs = useMemo(
    () => (operatorMode ? VIEW_TABS.filter((tab) => tab.operatorVisible) : VIEW_TABS),
    [operatorMode],
  )

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(OPERATOR_MODE_STORAGE_KEY, String(operatorMode))
  }, [operatorMode])

  useEffect(() => {
    if (!operatorMode) return
    const currentTab = VIEW_TABS.find((tab) => tab.mode === viewMode)
    if (!currentTab || currentTab.operatorVisible) return
    navigate('/operator', { replace: true })
  }, [navigate, operatorMode, viewMode])

  /* ── AppContext value ────────────────────── */
  const appCtx = useMemo(() => ({
    streamSessionId, setStreamSessionId, sessionOptions, sessionInfoQuery,
    memoriesQuery, priorsQuery, memoryDetailQuery, feedbackHistoryQuery, tracesQuery,
    alerts, unreadCount, wsStatus, selectedMemoryId, setSelectedMemoryId,
    detailRequest, requestAlertDetail, clearAlertDetail,
    pause, resume, replay, sendFeedback: wrappedSendFeedback,
    feedbackPending: feedbackState.feedbackPending,
    feedbackMsg: feedbackState.feedbackMsg,
    lastPriorDiff: feedbackState.lastPriorDiff,
    lastPriorDiffAt: feedbackState.lastPriorDiffAt,
    lastFeedbackMeta: feedbackState.lastFeedbackMeta,
    operatorMode, setOperatorMode,
    autoOpenAlerts, setAutoOpenAlerts, freezeOnAlert, setFreezeOnAlert,
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [
    streamSessionId, sessionOptions, wsStatus, selectedMemoryId,
    detailRequest, requestAlertDetail, clearAlertDetail,
    feedbackState.feedbackPending, feedbackState.feedbackMsg,
    alerts, unreadCount,
    feedbackState.lastPriorDiff, feedbackState.lastPriorDiffAt, feedbackState.lastFeedbackMeta,
    operatorMode, autoOpenAlerts, freezeOnAlert,
  ])

  return (
    <AppContext.Provider value={appCtx}>
    <div className={viewMode === 'detailed' ? 'grid' : 'gridSingle'}>

      {/* ── View switcher tabs ─────────────── */}
      <div className="viewTabs">
        {visibleViewTabs.map((tab) => (
          <button
            key={tab.mode}
            className={`viewTab ${viewMode === tab.mode ? 'active' : ''}`}
            onClick={() => navigate(`/${tab.mode}`)}
          >
            {tab.label}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Operator / Full view toggle — always visible so the presenter can
              flip between the clean operator story and the deep-dive (KG, DT,
              experiments) for live Q&A. */}
          <div
            role="group"
            aria-label="View mode"
            style={{ display: 'inline-flex', flexShrink: 0, border: '1px solid var(--border)', borderRadius: 999, overflow: 'hidden' }}
            title={operatorMode
              ? 'Operator view: the clean operator + MaaS-learnings path. Switch to Full view for Knowledge Graph, Digital Twin and experiments.'
              : 'Full view: all surfaces incl. Knowledge Graph and Digital Twin. Switch to Operator view for the clean operator story.'}
          >
            {([['Operator', true], ['Full', false]] as const).map(([label, op]) => (
              <button
                key={label}
                onClick={() => setOperatorMode(op)}
                className="small"
                style={{
                  border: 'none',
                  cursor: 'pointer',
                  flexShrink: 0,
                  whiteSpace: 'nowrap',
                  padding: '3px 12px',
                  fontWeight: operatorMode === op ? 700 : 400,
                  background: operatorMode === op ? 'var(--accent)' : 'transparent',
                  color: operatorMode === op ? '#fff' : 'var(--muted)',
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {viewMode !== 'detailed' && viewMode !== 'settings' && (
            <>
              <select
                value={streamSessionId}
                onChange={(e) => setStreamSessionId(e.target.value)}
                style={{ fontSize: 12, padding: '3px 8px' }}
              >
                <option value="">(select session)</option>
                {sessionSummaries.map((session) => (
                  <option key={session.session_id} value={session.session_id}>
                    {formatSessionOptionLabel(session)}
                  </option>
                ))}
              </select>
              {selectedSessionSummary && (
                <span
                  className="small"
                  style={{
                    color: sessionStatusColor(selectedSessionSummary.status),
                    border: `1px solid ${sessionStatusColor(selectedSessionSummary.status)}`,
                    borderRadius: 999,
                    padding: '2px 8px',
                  }}
                  title={selectedSessionSummary.last_error || `Session is ${selectedSessionSummary.status_label.toLowerCase()}`}
                >
                  {selectedSessionSummary.status_label}
                  {typeof selectedSessionSummary.progress === 'number' && Number.isFinite(selectedSessionSummary.progress) && selectedSessionSummary.total_samples > 0
                    ? selectedSessionSummary.progress <= 0
                      ? ' 0%'
                      : selectedSessionSummary.progress < 0.001
                        ? ' <0.1%'
                        : selectedSessionSummary.progress < 0.01
                          ? ` ${(selectedSessionSummary.progress * 100).toFixed(1)}%`
                          : ` ${Math.round(selectedSessionSummary.progress * 100)}%`
                    : ''}
                </span>
              )}
              <span className="small" style={{ color: 'var(--muted)' }}>WS: {wsStatus}</span>
              <BackendReadyBadge />
            </>
          )}
        </div>
      </div>

      {/* ── Routed views (lazy-loaded) ──────── */}
      <Suspense fallback={<div className="panel"><div className="small">Loading…</div></div>}>
        <Routes>
          <Route path="/home" element={<LandingPage />} />
          <Route path="/batch" element={<BatchReviewPage />} />
          <Route path="/operator" element={<OperatorPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/harmonics" element={<HarmonicsPage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
          <Route path="/dataset" element={<DatasetExplorerPage />} />
          <Route path="/documents" element={<DocumentRetrievalPage />} />
          <Route path="/development" element={<DevelopmentPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/sindit" element={<SinditPage />} />
          <Route path="/learnings" element={<LearningsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/detailed" element={null} />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </Suspense>

      {/* ── Demo Director (presenter-only, Ctrl+Shift+D) ── */}
      <DemoDirector />

      {/* ── Toast stack (fold, don't replace) ─────────── */}
      {toasts.length > 0 && (
        <div className="toastStack">
          {toasts.map((item) => {
            const alert = item.memoryId ? alertedById[item.memoryId] : undefined
            const isExpanded = (expandedToastId ?? toasts[toasts.length - 1]?.id) === item.id
            const severityLabel = alert?.severity || item.severityLabel
            const accent = severityToneColor(severityLabel)
            const headline = alert ? alertHeadline(alert) : item.headline
            const isPreview = alert?.context && typeof alert.context === 'object'
              ? (alert.context as Record<string, unknown>).operator_preview === true
              : Boolean(item.preview)
            const canOpen = Boolean(item.memoryId) && !isPreview

            if (!isExpanded) {
              // Folded / minimized — one line. Click to bring back into focus.
              return (
                <button
                  key={item.id}
                  type="button"
                  className="toastMini"
                  style={{ borderColor: accent }}
                  onClick={() => setExpandedToastId(item.id)}
                  title="Show this alert"
                >
                  <span className="toastMiniDot" style={{ background: accent }} />
                  <span className="toastMiniSeverity" style={{ color: accent }}>{severityLabel}</span>
                  <span className="toastMiniHeadline">{headline}</span>
                </button>
              )
            }

            const detail = alert ? alertOperatorExplanation(alert) : item.detail
            const badges = alert ? alertEvidenceBadges(alert).slice(0, 3) : []
            const ctxEntries = alert ? alertCuttingContextEntries(alert).slice(0, 3) : []
            const ctxLine = ctxEntries.map((e) => `${e.label}: ${e.value}`).join(' · ')
            const category = alert?.category || item.category || ''
            return (
              <div
                key={item.id}
                className="toast toastInStack"
                style={{
                  borderColor: accent,
                  boxShadow: `0 10px 30px rgba(0,0,0,0.35), 0 0 0 1px ${accent}`,
                }}
              >
                <div className="toastHeaderRow">
                  <div style={{ minWidth: 0 }}>
                    <div className="toastEyebrow">
                      <span className="toastSignal" style={{ background: accent, boxShadow: `0 0 10px ${accent}` }} />
                      <span className="toastSeverity" style={{ color: accent, borderColor: accent }}>
                        {severityLabel}
                      </span>
                      {category && <span className="toastCategory">{category}</span>}
                      {isPreview && <span className="toastCategory">Preview</span>}
                    </div>
                    <div className="toastHeadline">{headline}</div>
                  </div>
                  <button className="toastClose" onClick={() => dismissToast(item.id)} title="Dismiss">×</button>
                </div>
                {detail && detail !== headline && <div className="toastDetail">{detail}</div>}
                {badges.length > 0 && (
                  <div className="toastBadges">
                    {badges.map((badge) => (
                      <span key={badge} className="toastBadge">{badge}</span>
                    ))}
                  </div>
                )}
                {ctxLine && <div className="toastContext">Cutting context: {ctxLine}</div>}
                <div className="toastActionsRow">
                  <div className="hrow" style={{ gap: 8 }}>
                    <button
                      className="primary"
                      onClick={() => openToast(item)}
                      disabled={!(canOpen || isPreview)}
                    >
                      {isPreview ? 'Open Monitoring' : 'Open'}
                    </button>
                    <button onClick={() => dismissToast(item.id)}>Dismiss</button>
                  </div>
                  <div className="small">Folds when the next alert arrives · auto-hides in 10s</div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Detailed View ──────────────────── */}
      {viewMode === 'detailed' && (
        <ErrorBoundary label="Detailed View">
          <>
            {/* Left panel: stream + controls */}
            <div className="panel">
              <div className="hrow" style={{ justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>LFL realtime UI (desktop-ready)</div>
                  <div className="small">time stream + alerts + capture / memory / feedback / priors</div>
                </div>
                <div className="small">WS: {wsStatus}</div>
              </div>

              <div className="hr" />

              <SessionControls
                sessionsQuery={sessionsQuery}
                priorsQuery={priorsQuery}
                sessionOptions={sessionOptions}
                sessionSummaries={sessionSummaries}
              />

              <PlaybackControls
                sessionInfoQuery={sessionInfoQuery}
                freezeOnAlert={freezeOnAlert}
                setFreezeOnAlert={setFreezeOnAlert}
                pausedByAlert={pausedByAlert}
                onPause={pause}
                onResume={resume}
                onReplay={replay}
              />

              <div className="hrow" style={{ justifyContent: 'space-between' }}>
                <div style={{ fontWeight: 700 }}>Stream</div>
                <div className="small">
                  WS: {streamSessionId ? wsUrl(`/streams/${encodeURIComponent(streamSessionId)}`) : '(select session)'}
                </div>
              </div>

              <div className="row" style={{ marginTop: 8 }}>
                <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input type="checkbox" checked={followTail} onChange={(e) => setFollowTail(e.target.checked)} />
                  Follow tail
                </label>
                <div>
                  <div className="small">Window (seconds)</div>
                  <input value={String(windowSeconds)} onChange={(e) => setWindowSeconds(Number(e.target.value))} placeholder="5" />
                </div>
              </div>

              <StreamPlot height={360} onSelect={(i0, i1) => setCaptureSel({ i0, i1 })} />

              {captureSel && (
                <CapturePanel
                  captureSel={captureSel}
                  onClose={() => setCaptureSel(null)}
                  memoriesQuery={memoriesQuery}
                  priorsQuery={priorsQuery}
                />
              )}
            </div>

            {/* Right panel: alerts + memories + priors */}
            <div className="panel">
              <FaultIndicatorPanel />
              <div className="hr" />

              <AlertsPanel
                alerts={alerts}
                selectedMemoryId={selectedMemoryId}
                onSelectMemoryId={(memoryId) => setSelectedMemoryId(memoryId)}
                onClear={clearAlerts}
                onMarkAllRead={markAllAlertsRead}
                flash={alertsFlash}
              />

              <div className="hr" />

              <MemoryInbox
                memoriesQuery={memoriesQuery}
                alerts={alerts}
                streamSessionId={streamSessionId}
                selectedMemoryId={selectedMemoryId}
                setSelectedMemoryId={setSelectedMemoryId}
                autoOpenAlerts={autoOpenAlerts}
                setAutoOpenAlerts={setAutoOpenAlerts}
                reviewedById={reviewedById}
                reviewHistory={reviewHistory}
              />

              {/* Memory detail modal */}
              {selectedMemoryId && (
                <MemoryModal
                  selectedMemoryId={selectedMemoryId}
                  onClose={() => {
                    if (pausedByAlert?.memoryId) handledAlertIds.add(pausedByAlert.memoryId)
                    setPausedByAlert(null)
                    setSelectedMemoryId('')
                  }}
                  {...feedbackState}
                  sendFeedback={wrappedSendFeedback}
                  memoryDetailQuery={memoryDetailQuery}
                  feedbackHistoryQuery={feedbackHistoryQuery}
                  tracesQuery={tracesQuery}
                  selectedAlert={selectedAlert}
                  pausedByAlert={pausedByAlert}
                  onHandleAlert={(id) => handledAlertIds.add(id)}
                  onClearPausedByAlert={() => setPausedByAlert(null)}
                  onResume={resume}
                  detailHighlight={detailHighlight}
                />
              )}

              <div className="hr" />

              <div style={{ fontWeight: 700 }}>Priors (top)</div>
              <PriorsChart priors={priorsQuery.data?.priors ?? []} maxRows={30} />
            </div>
          </>
        </ErrorBoundary>
      )}
    </div>
    </AppContext.Provider>
  )
}

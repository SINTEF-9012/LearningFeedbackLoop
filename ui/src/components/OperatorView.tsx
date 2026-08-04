import React, { useEffect, useMemo, useRef, useState } from 'react'
import type { SignificantEventAlert } from '../state/alertsStore'
import { useInferenceStore } from '../state/inferenceStore'
import { useStreamStore } from '../state/streamStore'
import { AlertSummaryCard } from './AlertSummaryCard'
import { InferenceChart } from './InferenceChart'
import { StreamPlot } from './StreamPlot'
import { AlertDetailContent } from './AlertDetailContent'
import ChatPanel from './ChatPanel'
import { alertHeadline } from '../utils/alerts'
import { getDefaultHiddenPlotChannels, getDefaultVisiblePlotChannels, sortPlotChannelsByImportance } from '../utils/plotChannels'

/* ── Severity helpers ────────────────────────────────────── */
function sevColor(sev?: string) {
  if (sev === 'CRITICAL') return 'var(--danger)'
  if (sev === 'WARNING') return 'var(--accent)'
  return 'var(--ok)'
}

type SeverityTarget = 'info' | 'warning' | 'critical'

type DismissExtra = {
  severity_target?: SeverityTarget
}

type MutedSignatureEntry = {
  alert: SignificantEventAlert
  headline: string
  mutedAt: number
}

const dismissPresets = [
  { label: 'False alarm', reason: 'false alarm', requiresSeverityTarget: false },
  { label: 'Already aware', reason: 'already aware', requiresSeverityTarget: false },
  { label: 'Known condition', reason: 'known recurring condition', requiresSeverityTarget: false },
  { label: 'Wrong severity', reason: 'wrong severity', requiresSeverityTarget: true },
] as const

// Confirm-with-reason presets — let the operator record *why* the alert is real
// in one tap, mirroring the dismiss menu. Plain "Confirm" (no reason) still works.
const confirmPresets = [
  { label: 'Matches shop-floor signs', reason: 'confirmed — matches shop-floor observation' },
  { label: 'Tool wear / damage found', reason: 'confirmed — tool wear or damage found' },
  { label: 'Adjusted the process', reason: 'confirmed — operator adjusted the process' },
  { label: 'Flag for maintenance', reason: 'confirmed — flagged for maintenance' },
] as const

const severityTargets = [
  { label: 'INFO', value: 'info' },
  { label: 'WARNING', value: 'warning' },
  { label: 'CRITICAL', value: 'critical' },
] as const

function sessionStatusColor(status?: string | null) {
  if (status === 'live') return 'var(--ok)'
  if (status === 'paused') return 'var(--accent)'
  if (status === 'completed') return '#7dcfff'
  if (status === 'error') return 'var(--danger)'
  if (status === 'stopped') return '#f0a050'
  return 'var(--muted)'
}

function harmonicScoreColor(score?: number | null, dangerThreshold = 0.7) {
  if (typeof score !== 'number' || !Number.isFinite(score)) return 'var(--muted)'
  if (score >= dangerThreshold) return 'var(--danger)'
  if (score > 0.4) return '#f0a050'
  return 'var(--ok)'
}

function harmonicStatusLabel(status?: string | null) {
  if (status === 'zero_input') return 'no signal'
  if (status === 'no_pair_columns') return 'no columns'
  if (status === 'no_harmonic_columns') return 'no columns'
  if (status === 'warming_up') return 'warming up'
  if (status === 'nan_logit' || status === 'invalid_score') return 'invalid'
  return 'unavailable'
}

function harmonicStatusMessage(kind: 'context' | 'pair', status?: string | null) {
  const modelLabel = kind === 'pair' ? 'pair model' : 'context model'
  if (status === 'zero_input') {
    return `Live ${modelLabel} score unavailable because the current window has no usable harmonic signal.`
  }
  if (status === 'no_pair_columns') {
    return 'Live pair score unavailable because the current dataset does not expose matching pair columns.'
  }
  if (status === 'no_harmonic_columns') {
    return 'Live harmonic context score unavailable because the current dataset does not expose matching harmonic columns.'
  }
  if (status === 'warming_up') {
    return `Live ${modelLabel} score is warming up and needs more windows before it can score.`
  }
  if (status === 'nan_logit' || status === 'invalid_score') {
    return `Live ${modelLabel} score was suppressed because the model produced a non-finite value.`
  }
  return `Live ${modelLabel} score from the latest inference window.`
}

function harmonicDatasetLabel(dataset?: string | null): string {
  if (!dataset) return 'Pair'
  if (dataset === 'pair_lfl') return 'Pair LFL'
  if (dataset === 'pair_casedata') return 'Pair casedata'
  if (dataset === 'pair_raw') return 'Pair raw'
  return dataset.replace(/_/g, ' ')
}

function resolveActivePairDataset(sessionInfo?: Record<string, unknown> | null): string | null {
  const cfg = sessionInfo?.config && typeof sessionInfo.config === 'object'
    ? sessionInfo.config as Record<string, unknown>
    : {}
  const metadata = sessionInfo?.metadata && typeof sessionInfo.metadata === 'object'
    ? sessionInfo.metadata as Record<string, unknown>
    : {}
  const explicit = [cfg.harmonic_dataset, metadata.harmonic_dataset, metadata.harmonic_dataset_name]
    .find((value): value is string => typeof value === 'string' && value.trim().length > 0)

  if (explicit && explicit.startsWith('pair_')) return explicit
  if (explicit === 'casedata') return 'pair_lfl'

  const casedata = metadata.casedata && typeof metadata.casedata === 'object'
    ? metadata.casedata as Record<string, unknown>
    : {}
  const sourceHints = [
    typeof metadata.source === 'string' ? metadata.source.toLowerCase() : '',
    typeof casedata.root === 'string' ? casedata.root.toLowerCase() : '',
    typeof casedata.case_dir === 'string' ? casedata.case_dir.toLowerCase() : '',
    typeof metadata.machine_id === 'string' ? metadata.machine_id.toLowerCase() : '',
  ].filter(Boolean).join(' ')

  if (sourceHints.includes('site_c') || sourceHints.includes('site_b') || sourceHints.includes('casedata')) {
    return 'pair_lfl'
  }

  return null
}

function formatSessionProgress(progress?: number | null) {
  if (typeof progress !== 'number' || !Number.isFinite(progress)) return ''
  const percent = progress * 100
  if (percent <= 0) return '0%'
  if (percent < 0.1) return '<0.1%'
  if (percent < 1) return `${percent.toFixed(1)}%`
  return `${Math.round(percent)}%`
}

function sessionStatusMessage(
  status?: string | null,
  progress?: number | null,
  lastError?: string | null,
  wsConnected?: boolean,
) {
  if (lastError) return `Issue detected: ${lastError}`
  if (status === 'live' && wsConnected === false) {
    return 'Playback is live, but the stream connection is disconnected.'
  }
  if (status === 'live') {
    return `Playback is running${formatSessionProgress(progress) ? ` at ${formatSessionProgress(progress)}` : ''}.`
  }
  if (status === 'paused') {
    return `Playback is paused${formatSessionProgress(progress) ? ` at ${formatSessionProgress(progress)}` : ''}.`
  }
  if (status === 'completed') {
    return 'Playback reached the end of the session.'
  }
  if (status === 'stopped') {
    return `Playback stopped before completion${formatSessionProgress(progress) ? ` at ${formatSessionProgress(progress)}` : ''}.`
  }
  if (status === 'error') {
    return 'Playback failed.'
  }
  if (status === 'idle') {
    return 'Session is loaded and ready to start.'
  }
  return 'Session state is unavailable.'
}

/* ── Props ───────────────────────────────────────────────── */
interface OperatorViewProps {
  sessionId: string | null
  sessionInfo?: Record<string, unknown> | null
  wsConnected: boolean
  alerts: SignificantEventAlert[]
  latestAlert: SignificantEventAlert | null | undefined
  paused: boolean
  sessionStatus?: string | null
  sessionStatusLabel?: string | null
  sessionProgress?: number | null
  sessionLastError?: string | null
  feedbackPending: boolean
  onConfirm: (eventId: string, reason?: string) => void | Promise<void>
  onDismiss: (eventId: string, reason?: string, extra?: DismissExtra) => void | Promise<void>
  onMuteSignature: (alert: SignificantEventAlert, muted: boolean) => void | Promise<void>
  onExplanationFeedback: (alert: SignificantEventAlert, helpful: boolean) => void | Promise<void>
  onResume: () => void
  onCapture: (i0: number, i1: number) => void
  freezeOnAlert: boolean
  onToggleFreeze: (v: boolean) => void
  unreadCount: number
  onMarkAllRead: () => void
  /** When set (e.g. after clicking "Open" on an alert toast), open this event's
   *  detail panel here in the Operator view instead of the complex Detailed tab. */
  focusMemoryId?: string | null
  /** Explicit request to POP the detail modal for a given event (toast "Open").
   *  `at` is a nonce so repeat requests for the same id still fire. */
  detailRequest?: { id: string; at: number } | null
  /** Called once a detailRequest has been consumed (modal popped) so the parent
   *  can clear it — otherwise a stale request re-pops the modal when this view
   *  remounts on returning to the Monitoring tab. */
  onDetailConsumed?: () => void
}

/* ── Main Component ──────────────────────────────────────── */
export function OperatorView({
  sessionId, sessionInfo = null, wsConnected, alerts, latestAlert,
  paused, feedbackPending,
  sessionStatus, sessionStatusLabel, sessionProgress, sessionLastError,
  onConfirm, onDismiss, onMuteSignature, onExplanationFeedback, onResume, onCapture,
  freezeOnAlert, onToggleFreeze,
  unreadCount, onMarkAllRead, focusMemoryId = null, detailRequest = null,
  onDetailConsumed,
}: OperatorViewProps) {
  // Pin by event_id (NOT a snapshot object) so the displayed strip / detail modal
  // always reflect the LATEST store content — e.g. the real LLM explanation that
  // arrives a beat after the alert. Pinning a frozen object froze the pre-
  // explanation text and made the LLM reasoning "disappear".
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detailId, setDetailId] = useState<string | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [showChannels, setShowChannels] = useState(false)
  const [showMoreChannels, setShowMoreChannels] = useState(false)
  const [dismissMenuOpen, setDismissMenuOpen] = useState(false)
  const [confirmMenuOpen, setConfirmMenuOpen] = useState(false)
  const [pendingSeverityReason, setPendingSeverityReason] = useState<string | null>(null)
  const [feedbackNote, setFeedbackNote] = useState('')
  // Feedback controls shown inside the detail modal (separate from the strip menu).
  const [modalNote, setModalNote] = useState('')
  const [modalWrongSeverity, setModalWrongSeverity] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [mutedSignatures, setMutedSignatures] = useState<Record<string, MutedSignatureEntry>>({})
  const [explanationVotes, setExplanationVotes] = useState<Record<string, 'helpful' | 'unhelpful'>>({})
  const [auxActionPending, setAuxActionPending] = useState<null | 'mute' | 'explanation'>(null)
  // Resolve an id to the live alert object (from the store-backed `alerts`
  // array, falling back to `latestAlert`). Always returns the freshest copy.
  const findAlert = (id: string | null): SignificantEventAlert | null => {
    if (!id) return null
    return alerts.find((a) => a.event_id === id)
      || (latestAlert?.event_id === id ? latestAlert : null)
  }
  const current = findAlert(selectedId) || latestAlert || null
  const detailAlert = findAlert(detailId)
  const activeAlert = showDetails ? detailAlert : current

  const openDetails = () => {
    if (!current) return
    setDetailId(current.event_id)
    setShowDetails(true)
  }

  // Pin the alert the operator is looking at so a newly-arriving alert (e.g. from
  // live playback, or the next injected demo event) does NOT steal focus. Once an
  // alert is selected it stays put until the operator dismisses it or explicitly
  // clicks another in the list. When nothing is selected we pin the requested
  // focus (toast "Open") if present, else the latest alert.
  const appliedFocusRef = useRef<string | null>(null)
  useEffect(() => {
    // An explicit focus request (e.g. clicking "Open" on a toast) must win even
    // over an already-pinned alert — otherwise Open silently does nothing. We
    // only react when the requested id actually CHANGES, so a passively-arriving
    // alert still can't steal focus.
    if (focusMemoryId && focusMemoryId !== appliedFocusRef.current) {
      const exists = alerts.some((a) => a.event_id === focusMemoryId)
        || latestAlert?.event_id === focusMemoryId
      if (exists) {
        appliedFocusRef.current = focusMemoryId
        setSelectedId(focusMemoryId)
        return
      }
    }
    if (!focusMemoryId) appliedFocusRef.current = null
    // Default when nothing is pinned: follow the latest alert.
    if (!selectedId && latestAlert?.event_id) setSelectedId(latestAlert.event_id)
  }, [selectedId, focusMemoryId, alerts, latestAlert])

  // Explicit "Open" (toast) → pop the detail modal for that event, pinned. Only
  // fires on a NEW request nonce, so ordinary alert arrivals never auto-open the
  // (intentionally heavier) modal.
  const appliedDetailReqRef = useRef<number | null>(null)
  useEffect(() => {
    if (!detailRequest || detailRequest.at === appliedDetailReqRef.current) return
    const exists = alerts.some((a) => a.event_id === detailRequest.id)
      || latestAlert?.event_id === detailRequest.id
    if (!exists) return
    appliedDetailReqRef.current = detailRequest.at
    setSelectedId(detailRequest.id)
    setDetailId(detailRequest.id)
    setShowDetails(true)
    // Clear the request now it is consumed, so returning to this tab later
    // (which remounts the view and resets the applied-nonce ref) does not
    // re-pop the modal from a stale request.
    onDetailConsumed?.()
  }, [detailRequest, alerts, latestAlert, onDetailConsumed])

  const closeDetails = () => {
    setShowDetails(false)
    setDetailId(null)
    setChatOpen(false)
  }


  useEffect(() => {
    setDismissMenuOpen(false)
    setPendingSeverityReason(null)
  }, [activeAlert?.event_id])

  useEffect(() => {
    if (!showDetails && !dismissMenuOpen) return undefined
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (dismissMenuOpen) {
          const shouldSubmitPendingSeverity = Boolean(pendingSeverityReason)
          setDismissMenuOpen(false)
          setPendingSeverityReason(null)
          if (shouldSubmitPendingSeverity && activeAlert) {
            void Promise.resolve(onDismiss(activeAlert.event_id, 'wrong severity'))
          }
          return
        }
        closeDetails()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [showDetails, dismissMenuOpen, pendingSeverityReason, activeAlert, onDismiss])

  useEffect(() => {
    if (!dismissMenuOpen) return undefined
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest('.alertDismissGroup')) {
        const shouldSubmitPendingSeverity = Boolean(pendingSeverityReason)
        setDismissMenuOpen(false)
        setPendingSeverityReason(null)
        if (shouldSubmitPendingSeverity && activeAlert) {
          void Promise.resolve(onDismiss(activeAlert.event_id, 'wrong severity'))
        }
      }
    }
    window.addEventListener('mousedown', handlePointerDown)
    return () => window.removeEventListener('mousedown', handlePointerDown)
  }, [dismissMenuOpen, pendingSeverityReason, activeAlert, onDismiss])

  /* ── Channel selector state ────────────────────── */
  const yByChannel = useStreamStore((s) => s.series.yByChannel)
  const allChannels = useMemo(() => Object.keys(yByChannel), [yByChannel])
  const activeChannels = useStreamStore((s) => s.channels)
  const setChannels = useStreamStore((s) => s.setChannels)
  const windowSeconds = useStreamStore((s) => s.windowSeconds)
  const setWindowSeconds = useStreamStore((s) => s.setWindowSeconds)
  const fs = useStreamStore((s) => s.series.fs)
  const latestInference = useInferenceStore((s) => (s.points.length > 0 ? s.points[s.points.length - 1] : null))
  const defaultVisibleChannels = useMemo(() => getDefaultVisiblePlotChannels(yByChannel), [yByChannel])
  const defaultHiddenChannels = useMemo(() => getDefaultHiddenPlotChannels(yByChannel), [yByChannel])
  const orderedChannels = useMemo(() => sortPlotChannelsByImportance(allChannels), [allChannels])
  const suggestedChannels = useMemo(
    () => orderedChannels.filter((channel) => defaultVisibleChannels.includes(channel)),
    [orderedChannels, defaultVisibleChannels],
  )
  const additionalChannels = useMemo(
    () => orderedChannels.filter((channel) => !defaultVisibleChannels.includes(channel)),
    [orderedChannels, defaultVisibleChannels],
  )

  const visibleSet = useMemo(
    () => new Set(activeChannels.includes('__none__') ? [] : (activeChannels.length ? activeChannels : defaultVisibleChannels)),
    [activeChannels, defaultVisibleChannels],
  )

  useEffect(() => {
    if (!showChannels) {
      setShowMoreChannels(false)
    }
  }, [showChannels])

  const toggleChannel = (ch: string) => {
    const next = new Set(visibleSet)
    if (next.has(ch)) {
      next.delete(ch)
    } else {
      next.add(ch)
    }
    const nextChannels = [...next].filter((channel) => allChannels.includes(channel))
    if (nextChannels.length === 0) {
      setChannels(['__none__'])
      return
    }

    const matchesDefaultSelection = nextChannels.length === defaultVisibleChannels.length
      && nextChannels.every((channel) => defaultVisibleChannels.includes(channel))

    setChannels(matchesDefaultSelection ? [] : nextChannels)
  }

  const selectAll = () => setChannels(allChannels)
  const selectSuggested = () => setChannels([])
  const selectNone = () => setChannels(['__none__'])

  // Friendly short name for channel pills
  const shortName = (ch: string) =>
    ch.replace(/^(Monit_chatter_detection_|Cnc_Override_|Axis_FeedRate_|Accel_Severity_)/, '')
      .replace(/_/g, ' ')

  const sessionBadgeColor = sessionStatusColor(sessionStatus)
  const liveHarmonicScore = latestInference?.scores.harmonic_context_score
  const liveHarmonicScoreColor = harmonicScoreColor(liveHarmonicScore)
  const liveHarmonicStatus = latestInference?.harmonic_status?.context
  const livePairScore = latestInference?.scores.harmonic_pair_score
  const livePairThreshold = latestInference?.harmonic_thresholds?.pair
  const livePairScoreColor = harmonicScoreColor(livePairScore, typeof livePairThreshold === 'number' ? livePairThreshold : 0.7)
  const livePairStatus = latestInference?.harmonic_status?.pair
  const activePairDataset = resolveActivePairDataset(sessionInfo)
  const pairModelLabel = harmonicDatasetLabel(activePairDataset)
  const sessionProgressLabel = formatSessionProgress(sessionProgress)
  const sessionMessage = sessionStatusMessage(
    sessionStatus,
    sessionProgress,
    sessionLastError,
    wsConnected,
  )
  const currentSignature = activeAlert?.recurrence?.signature || null
  const isCurrentMuted = currentSignature ? Boolean(mutedSignatures[currentSignature]) : false
  const mutedSignatureEntries = useMemo(
    () => Object.entries(mutedSignatures)
      .map(([signature, entry]) => ({ signature, ...entry }))
      .sort((a, b) => b.mutedAt - a.mutedAt),
    [mutedSignatures],
  )
  const currentExplanationVote = activeAlert ? explanationVotes[activeAlert.event_id] : undefined
  const auxBusy = auxActionPending !== null

  const handleDismiss = async (reason?: string, extra?: DismissExtra) => {
    if (!activeAlert) return
    const actedId = activeAlert.event_id
    setDismissMenuOpen(false)
    setPendingSeverityReason(null)
    await Promise.resolve(onDismiss(actedId, reason, extra))
    setFeedbackNote('')
    setModalNote('')
    setModalWrongSeverity(false)
    // Visible acknowledgement: close details and drop the event from the card so
    // the reviewed alert clearly disappears (the store also removes it from the list).
    if (detailId === actedId) closeDetails()
    setSelectedId((prev) => (prev === actedId ? null : prev))
  }

  const handleConfirm = async (reason?: string) => {
    if (!activeAlert) return
    const actedId = activeAlert.event_id
    setConfirmMenuOpen(false)
    await Promise.resolve(onConfirm(actedId, reason))
    setFeedbackNote('')
    setModalNote('')
    setModalWrongSeverity(false)
    if (detailId === actedId) closeDetails()
    setSelectedId((prev) => (prev === actedId ? null : prev))
  }

  const handleMuteToggle = async () => {
    if (!activeAlert || !currentSignature) return
    const nextMuted = !isCurrentMuted
    setAuxActionPending('mute')
    try {
      await Promise.resolve(onMuteSignature(activeAlert, nextMuted))
      setMutedSignatures((prev) => {
        const next = { ...prev }
        if (nextMuted) {
          next[currentSignature] = {
            alert: activeAlert,
            headline: alertHeadline(activeAlert),
            mutedAt: Date.now(),
          }
        } else {
          delete next[currentSignature]
        }
        return next
      })
    } finally {
      setAuxActionPending(null)
    }
  }

  const handleUnmuteSignature = async (signature: string) => {
    const entry = mutedSignatures[signature]
    if (!entry) return

    setAuxActionPending('mute')
    try {
      await Promise.resolve(onMuteSignature(entry.alert, false))
      setMutedSignatures((prev) => {
        const next = { ...prev }
        delete next[signature]
        return next
      })
    } finally {
      setAuxActionPending(null)
    }
  }

  const handleExplanationVote = async (helpful: boolean) => {
    if (!activeAlert) return
    setAuxActionPending('explanation')
    try {
      await Promise.resolve(onExplanationFeedback(activeAlert, helpful))
      setExplanationVotes((prev) => ({
        ...prev,
        [activeAlert.event_id]: helpful ? 'helpful' : 'unhelpful',
      }))
    } finally {
      setAuxActionPending(null)
    }
  }

  return (
    <div className="operatorGrid">
      {/* ── Status Bar ───────────────────────────────── */}
      <div className="operatorStatus">
        <div className="operatorStatusLeft" style={{ display: 'grid', gap: 4 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span className={`wsDot ${wsConnected ? 'wsOk' : 'wsOff'}`} />
            <span style={{ fontSize: 12, color: 'var(--muted)' }}>
              {sessionId ? `Session: ${sessionId.slice(0, 8)}…` : 'No session'}
            </span>
            {unreadCount > 0 && (
              <span className="unreadBadge">{unreadCount} new</span>
            )}
          </div>
          {sessionId && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span
                style={{
                  color: sessionBadgeColor,
                  border: `1px solid ${sessionBadgeColor}`,
                  borderRadius: 999,
                  padding: '2px 8px',
                  fontSize: 11,
                  fontWeight: 700,
                }}
                title={sessionLastError || undefined}
              >
                {sessionStatusLabel || 'Unknown'}
                {sessionProgressLabel ? ` ${sessionProgressLabel}` : ''}
              </span>
              <span
                style={{
                  color: liveHarmonicScoreColor,
                  border: `1px solid ${liveHarmonicScoreColor}`,
                  borderRadius: 999,
                  padding: '2px 8px',
                  fontSize: 11,
                  fontWeight: 700,
                }}
                title={harmonicStatusMessage('context', liveHarmonicStatus)}
              >
                Harmonic {typeof liveHarmonicScore === 'number' ? liveHarmonicScore.toFixed(3) : latestInference ? harmonicStatusLabel(liveHarmonicStatus) : 'waiting'}
              </span>
              <span
                style={{
                  color: livePairScoreColor,
                  border: `1px solid ${livePairScoreColor}`,
                  borderRadius: 999,
                  padding: '2px 8px',
                  fontSize: 11,
                  fontWeight: 700,
                }}
                title={typeof livePairThreshold === 'number'
                  ? `${pairModelLabel}. ${harmonicStatusMessage('pair', livePairStatus)} Threshold ${livePairThreshold.toFixed(3)}.`
                  : harmonicStatusMessage('pair', livePairStatus)}
              >
                {pairModelLabel} {typeof livePairScore === 'number' ? livePairScore.toFixed(3) : latestInference ? harmonicStatusLabel(livePairStatus) : 'waiting'}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: sessionLastError ? 'var(--danger)' : 'var(--muted)',
                }}
                title={sessionLastError || sessionMessage}
              >
                {sessionMessage}
              </span>
            </div>
          )}
        </div>
        <div className="operatorStatusRight">
          <label style={{ fontSize: 12, color: 'var(--muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
            <input
              type="checkbox" checked={freezeOnAlert}
              onChange={e => onToggleFreeze(e.target.checked)}
            />
            Pause on alert
          </label>
          {paused && (
            <button className="opBtn opBtnAccent" onClick={onResume}>Resume</button>
          )}
        </div>
      </div>

      {!sessionId && (
        <div
          className="panelCard"
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: '32px 20px', textAlign: 'center' }}
        >
          <div style={{ fontSize: 15, fontWeight: 700 }}>No session running</div>
          <div style={{ fontSize: 13, color: 'var(--muted)', maxWidth: 460 }}>
            Start a demo session to see live inference scores, the sensor stream, and operator alerts here.
          </div>
          <a
            href="#/detailed"
            className="opBtn opBtnAccent"
            style={{ textDecoration: 'none', marginTop: 4, padding: '6px 16px', borderRadius: 6 }}
          >
            Start a demo session →
          </a>
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            Detailed → Configuration → Start Demo
          </div>
        </div>
      )}

      <div className="panelCard">
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 8,
            marginBottom: 8,
            flexWrap: 'wrap',
          }}
        >
          <div>
            <h4 style={{ margin: 0, fontSize: 12, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Live Inference Scores
            </h4>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
              Includes ensemble, baseline models, harmonic context, and harmonic pair scores per inference window.
            </div>
          </div>
        </div>
        <InferenceChart height={180} />
      </div>

      {/* ── Plot toolbar: channel picker + window size ── */}
      <div className="operatorToolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
          <button
            className="opBtn"
            onClick={() => setShowChannels(v => !v)}
            style={{ fontSize: 11, padding: '3px 8px' }}
            title={defaultHiddenChannels.length > 0 ? `${defaultHiddenChannels.length} channel${defaultHiddenChannels.length === 1 ? '' : 's'} hidden by default because of implausibly high spindle-command values` : undefined}
          >
            {showChannels ? '▾ Variables' : '▸ Variables'}
            <span style={{ marginLeft: 4, fontSize: 10, color: 'var(--muted)' }}>
              ({visibleSet.size}/{allChannels.length})
            </span>
          </button>

          {showChannels && (
            <div style={{ display: 'grid', gap: 6, minWidth: 0, width: '100%' }}>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                <button
                  onClick={selectSuggested}
                  style={{
                    fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                    border: '1px solid var(--border)', background: 'transparent',
                    color: 'var(--muted)',
                  }}
                >
                  Suggested
                </button>
                <button
                  onClick={selectAll}
                  style={{
                    fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                    border: '1px solid var(--border)', background: 'transparent',
                    color: 'var(--muted)',
                  }}
                >
                  All
                </button>
                <button
                  onClick={selectNone}
                  style={{
                    fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                    border: '1px solid var(--border)', background: 'transparent',
                    color: 'var(--muted)',
                  }}
                >
                  None
                </button>
                {additionalChannels.length > 0 && (
                  <button
                    onClick={() => setShowMoreChannels((value) => !value)}
                    style={{
                      fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                      border: '1px solid var(--border)', background: 'transparent',
                      color: 'var(--muted)',
                    }}
                  >
                    {showMoreChannels ? `Hide extras (${additionalChannels.length})` : `More variables (${additionalChannels.length})`}
                  </button>
                )}
              </div>

              <div style={{ display: 'grid', gap: 4 }}>
                <div className="small" style={{ fontSize: 10, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                  Suggested by default
                </div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                  {suggestedChannels.map((ch) => {
                    const isSel = visibleSet.has(ch)
                    return (
                      <button
                        key={ch}
                        onClick={() => toggleChannel(ch)}
                        style={{
                          fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                          border: `1px solid ${isSel ? '#7aa2f7' : 'var(--border)'}`,
                          background: isSel ? 'rgba(122,162,247,0.15)' : 'transparent',
                          color: isSel ? '#7aa2f7' : 'var(--muted)',
                        }}
                        title={ch}
                      >
                        {shortName(ch)}
                      </button>
                    )
                  })}
                </div>
              </div>

              {showMoreChannels && additionalChannels.length > 0 && (
                <div style={{ display: 'grid', gap: 4 }}>
                  <div className="small" style={{ fontSize: 10, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                    Additional variables
                  </div>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                    {additionalChannels.map((ch) => {
                      const isSel = visibleSet.has(ch)
                      return (
                        <button
                          key={ch}
                          onClick={() => toggleChannel(ch)}
                          style={{
                            fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                            border: `1px solid ${isSel ? '#7aa2f7' : 'var(--border)'}`,
                            background: isSel ? 'rgba(122,162,247,0.15)' : 'transparent',
                            color: isSel ? '#7aa2f7' : 'var(--muted)',
                          }}
                          title={ch}
                        >
                          {shortName(ch)}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Window size slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          <span style={{ fontSize: 10, color: 'var(--muted)', whiteSpace: 'nowrap' }}>
            Window: {windowSeconds < 60
              ? `${windowSeconds.toFixed(0)}s`
              : `${(windowSeconds / 60).toFixed(1)}m`}
          </span>
          <input
            type="range"
            min={fs <= 10 ? 30 : 1}
            max={fs <= 10 ? 1800 : 30}
            step={fs <= 10 ? 10 : 0.5}
            value={windowSeconds}
            onChange={e => setWindowSeconds(Number(e.target.value))}
            style={{ width: 90 }}
          />
        </div>
      </div>

      {/* ── Stream Plot ──────────────────────────────── */}
      <div className="operatorPlot">
        <StreamPlot height={380} onSelect={onCapture} />
      </div>

      {/* ── Alert Strip ──────────────────────────────── */}
      {current ? (
        <AlertSummaryCard
          variant="strip"
          alert={current}
          actions={(
            <>
              <div className="alertDismissGroup">
                <button
                  className="opBtn opBtnConfirm"
                  disabled={feedbackPending || auxBusy}
                  onClick={() => void handleConfirm()}
                >
                  Confirm
                </button>
                <button
                  className="opBtn opBtnConfirm opBtnSplit"
                  disabled={feedbackPending || auxBusy}
                  aria-label="Confirm options"
                  aria-expanded={confirmMenuOpen}
                  onClick={() => {
                    setDismissMenuOpen(false)
                    setConfirmMenuOpen((v) => !v)
                  }}
                >
                  ▾
                </button>
                {confirmMenuOpen && (
                  <div className="alertDismissMenu">
                    {confirmPresets.map((preset) => (
                      <button
                        key={preset.reason}
                        className="alertDismissMenuItem"
                        disabled={feedbackPending || auxBusy}
                        onClick={() => void handleConfirm(preset.reason)}
                      >
                        {preset.label}
                      </button>
                    ))}
                    <div style={{ borderTop: '1px solid var(--border)', marginTop: 4, paddingTop: 6, display: 'grid', gap: 4 }}>
                      <textarea
                        value={feedbackNote}
                        onChange={(e) => setFeedbackNote(e.target.value)}
                        placeholder="Write feedback…"
                        rows={2}
                        style={{ width: '100%', resize: 'vertical', fontSize: 12, padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--panel)', color: 'var(--text)' }}
                      />
                      <button
                        className="alertDismissMenuItem"
                        disabled={feedbackPending || auxBusy || !feedbackNote.trim()}
                        onClick={() => void handleConfirm(feedbackNote.trim())}
                      >
                        Confirm with note
                      </button>
                    </div>
                  </div>
                )}
              </div>
              <div className="alertDismissGroup">
                <button
                  className="opBtn opBtnDismiss"
                  disabled={feedbackPending || auxBusy}
                  onClick={() => void handleDismiss()}
                >
                  Dismiss
                </button>
                <button
                  className="opBtn opBtnDismiss opBtnSplit"
                  disabled={feedbackPending || auxBusy}
                  aria-label="Dismiss options"
                  aria-expanded={dismissMenuOpen}
                  onClick={() => {
                    if (dismissMenuOpen) {
                      setDismissMenuOpen(false)
                      setPendingSeverityReason(null)
                      return
                    }
                    setConfirmMenuOpen(false)
                    setDismissMenuOpen(true)
                  }}
                >
                  ▾
                </button>
                {dismissMenuOpen && (
                  <div className="alertDismissMenu">
                    {dismissPresets.map((preset) => (
                      <button
                        key={preset.reason}
                        className="alertDismissMenuItem"
                        disabled={feedbackPending || auxBusy}
                        aria-pressed={pendingSeverityReason === preset.reason}
                        onClick={() => {
                          if (preset.requiresSeverityTarget) {
                            setPendingSeverityReason(preset.reason)
                            return
                          }
                          void handleDismiss(preset.reason)
                        }}
                      >
                        {preset.label}
                      </button>
                    ))}
                    {pendingSeverityReason === 'wrong severity' && (
                      <div className="alertSeverityPickerRow">
                        <span className="alertSeverityPickerLabel">Should be:</span>
                        <div className="alertSeverityPickerChips">
                          {severityTargets.map((target) => (
                            <button
                              key={target.value}
                              className="alertSeverityChip"
                              disabled={feedbackPending || auxBusy}
                              onClick={() => void handleDismiss('wrong severity', { severity_target: target.value })}
                            >
                              {target.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    <div style={{ borderTop: '1px solid var(--border)', marginTop: 4, paddingTop: 6, display: 'grid', gap: 4 }}>
                      <textarea
                        value={feedbackNote}
                        onChange={(e) => setFeedbackNote(e.target.value)}
                        placeholder="Write feedback…"
                        rows={2}
                        style={{ width: '100%', resize: 'vertical', fontSize: 12, padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--panel)', color: 'var(--text)' }}
                      />
                      <button
                        className="alertDismissMenuItem"
                        disabled={feedbackPending || auxBusy || !feedbackNote.trim()}
                        onClick={() => void handleDismiss(feedbackNote.trim())}
                      >
                        Dismiss with note
                      </button>
                    </div>
                  </div>
                )}
              </div>
              <button
                className="opBtn"
                onClick={openDetails}
                style={{ fontSize: 12 }}
              >
                Details
              </button>
            </>
          )}
        />
      ) : (
        <div className="alertStripEmpty">
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>
            Monitoring — no alerts yet
          </span>
        </div>
      )}

      {/* ── Recent alerts list (compact) ─────────────── */}
      {alerts.length > 0 && (
        <div className="operatorAlertList">
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', marginBottom: 6,
          }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--muted)' }}>
              Recent ({alerts.length})
            </span>
            {unreadCount > 0 && (
              <button className="opBtnSmall" onClick={onMarkAllRead}>Mark read</button>
            )}
          </div>
          <div style={{ maxHeight: 200, overflow: 'auto' }}>
            {[...alerts].reverse().slice(0, 20).map(a => (
              <AlertSummaryCard
                key={a.event_id}
                variant="mini"
                alert={a}
                selected={a.event_id === current?.event_id}
                unread={Boolean(a._unread)}
                onClick={() => setSelectedId(a.event_id === selectedId ? null : a.event_id)}
              />
            ))}
          </div>
        </div>
      )}

      {mutedSignatureEntries.length > 0 && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Muted signatures</span>
            <span className="alertDisclosureHint">{`${mutedSignatureEntries.length} active`}</span>
          </summary>
          <div className="alertDisclosureBody">
            <div style={{ display: 'grid', gap: 8 }}>
              {mutedSignatureEntries.map((entry) => (
                <div
                  key={entry.signature}
                  style={{
                    alignItems: 'center',
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 8,
                    display: 'flex',
                    gap: 10,
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.4 }}>{entry.headline}</div>
                    <div className="small" style={{ color: 'var(--muted)', fontFamily: 'monospace', marginTop: 2 }}>
                      {entry.signature}
                    </div>
                  </div>
                  <button
                    className="opBtnSmall"
                    disabled={auxActionPending === 'mute'}
                    onClick={() => void handleUnmuteSignature(entry.signature)}
                    style={{ whiteSpace: 'nowrap' }}
                  >
                    Unmute
                  </button>
                </div>
              ))}
            </div>
          </div>
        </details>
      )}

      {activeAlert && showDetails && (
        <div
          className="modalBackdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              closeDetails()
            }
          }}
        >
          <div className="modalContent operatorAlertModal">
            <div className="operatorAlertModalBody">
            <AlertDetailContent
              alert={activeAlert}
              isMuted={isCurrentMuted}
              compact
              headerAction={(
                <button
                  onClick={closeDetails}
                  style={{ padding: '4px 12px', fontSize: 18, lineHeight: 1 }}
                  title="Close"
                >
                  x
                </button>
              )}
              controls={(
                <>
                  <div className="alertDetailToolbar">
                    <button
                      className="opBtn"
                      disabled={!currentSignature || auxActionPending === 'mute'}
                      onClick={() => void handleMuteToggle()}
                    >
                      {isCurrentMuted ? 'Resume Signature Alerts' : 'Mute Signature This Session'}
                    </button>
                  </div>
                  <div className="feedbackThumbRow">
                    <span style={{ fontSize: 12, color: 'var(--muted)' }}>Explanation helpful?</span>
                    <button
                      className={`feedbackThumb${currentExplanationVote === 'helpful' ? ' feedbackThumbActive' : ''}`}
                      disabled={auxActionPending === 'explanation'}
                      onClick={() => void handleExplanationVote(true)}
                    >
                      Helpful
                    </button>
                    <button
                      className={`feedbackThumb${currentExplanationVote === 'unhelpful' ? ' feedbackThumbActive' : ''}`}
                      disabled={auxActionPending === 'explanation'}
                      onClick={() => void handleExplanationVote(false)}
                    >
                      Not helpful
                    </button>
                  </div>
                </>
              )}
            />

            {/* Ask-the-assistant — hidden until requested (kept out of the way). */}
            <div className="operatorModalChat">
              <button
                type="button"
                className="alertTechToggle"
                aria-expanded={chatOpen}
                onClick={() => setChatOpen((v) => !v)}
              >
                {chatOpen ? '− Hide assistant' : '💬 Ask about this alert'}
              </button>
              {chatOpen && activeAlert && (
                <ChatPanel
                  memoryId={activeAlert.event_id}
                  patterns={activeAlert.patterns}
                />
              )}
            </div>
            </div>

            {/* Operator feedback — right here in the modal, no trip to Detailed.
                The detail note is always available; quick reasons are optional. */}
            <div className="operatorModalFeedback">
              <div className="operatorModalFeedbackHead">
                <span className="operatorModalFeedbackTitle">Your feedback</span>
                <span className="operatorModalFeedbackHint">Confirm or dismiss — add detail if useful</span>
              </div>
              <textarea
                value={modalNote}
                onChange={(e) => setModalNote(e.target.value)}
                placeholder="Add detail — what's actually happening on the floor, what you did, why it's (not) a real alert… (optional)"
                rows={3}
                className="operatorModalNote"
              />
              <details className="operatorModalReasons">
                <summary>Quick reasons</summary>
                <div className="operatorModalReasonsBody">
                  <div className="operatorModalChips">
                    <span className="operatorModalChipsLabel">Confirm because…</span>
                    {confirmPresets.map((preset) => (
                      <button
                        key={preset.reason}
                        className="reasonChip reasonChipConfirm"
                        disabled={feedbackPending || auxBusy}
                        onClick={() => void handleConfirm(preset.reason)}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                  <div className="operatorModalChips">
                    <span className="operatorModalChipsLabel">Dismiss because…</span>
                    {dismissPresets.map((preset) => (
                      <button
                        key={preset.reason}
                        className="reasonChip reasonChipDismiss"
                        disabled={feedbackPending || auxBusy}
                        aria-pressed={preset.requiresSeverityTarget ? modalWrongSeverity : undefined}
                        onClick={() => {
                          if (preset.requiresSeverityTarget) {
                            setModalWrongSeverity((v) => !v)
                            return
                          }
                          void handleDismiss(preset.reason)
                        }}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                  {modalWrongSeverity && (
                    <div className="operatorModalChips">
                      <span className="operatorModalChipsLabel">Should be…</span>
                      {severityTargets.map((target) => (
                        <button
                          key={target.value}
                          className="reasonChip"
                          disabled={feedbackPending || auxBusy}
                          onClick={() => void handleDismiss('wrong severity', { severity_target: target.value })}
                        >
                          {target.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </details>
              <div className="operatorModalFeedbackPrimary">
                <button
                  className="opBtn opBtnConfirm"
                  disabled={feedbackPending || auxBusy}
                  onClick={() => void handleConfirm(modalNote.trim() || undefined)}
                >
                  {modalNote.trim() ? 'Confirm with note' : 'Confirm'}
                </button>
                <button
                  className="opBtn opBtnDismiss"
                  disabled={feedbackPending || auxBusy}
                  onClick={() => void handleDismiss(modalNote.trim() || undefined)}
                >
                  {modalNote.trim() ? 'Dismiss with note' : 'Dismiss'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

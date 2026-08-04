import { create } from 'zustand'
import type { DocLink } from '../types'

export type SignificantEventAlert = {
  type?: string
  event_id: string
  session_id: string
  timestamp?: string
  time_range?: {
    i0?: number
    i1?: number
    t0?: number
    t1?: number
    fs?: number
  } | null
  severity?: string   // CRITICAL | WARNING | INFO (server-provided)
  category?: string   // Vibration Modulation | Anomaly | Frequency | … (server-provided)
  persistence_label?: 'candidate' | 'recurring' | null
  recurrence?: {
    signature?: string
    episode_id?: string
    first_seen?: string
    last_seen?: string
    occurrences?: number
    suppressed_since_last_emit?: number
  } | null
  primary_observation_key?: string | null
  primary_observation_label?: string | null
  indicators_present?: number | null
  indicators_required?: number | null
  indicator_details?: Array<{
    key: string
    label?: string | null
    confidence?: number | null
    reason?: string | null
    source_metric?: string | null
  }>
  significance?: {
    score?: number
    action?: string
    reasons?: string[]
    prior_boost?: number
    historical_prior?: number
    prior_evidence_count?: number
    prior_damping_factor?: number
    prior_factor?: number
    prior_mode?: 'additive' | 'multiplicative' | string
    pattern_priors?: Record<string, number>
    score_trace?: Array<{
      component: string
      value: number
      source: string
    }>
  }
  patterns?: string[]
  summary?: string | null
  summary_source?: string | null
  explanation?: string | null
  explanation_source?: string | null
  // Two-tier recommendation model (2026-07-12): the immediate breakage-avoidance
  // action, distinct from the explanation. Arrives via explanation_update.
  recommendation?: string | null
  similar_memories?: string[]
  similar_history?: Array<{
    id?: string
    created_at?: string
    annotation_text?: string | null
    label?: string | null
    shared_pattern_keys?: string[]
    shared_pattern_details?: Array<{
      key: string
      query_strength?: number
      query_source_metric?: string | null
      candidate_strength?: number
      candidate_source_metric?: string | null
    }>
    feedback?: {
      confirm_count?: number
      dismiss_count?: number
      last_action?: string | null
      last_comment?: string | null
      last_action_ts?: string | null
    }
  }>
  context?: Record<string, unknown> | null
  metrics?: Record<string, unknown> | null
  doc_links?: DocLink[]

  // UI-only
  _received_at?: number
  _unread?: boolean
  _streamIndex?: number  // Stream sample index at time of receipt
  _consolidated_count?: number  // how many events folded into this alert
  _consolidated_ids?: string[]  // memory IDs that were consolidated
}

type AlertsState = {
  maxAlerts: number
  alerts: SignificantEventAlert[]
  scoredEvents: SignificantEventAlert[]   // ALL events (alerts + sub-threshold) for inference panel
  _recentKeys: string[]

  // UI-only convenience signals
  lastPushed?: SignificantEventAlert
  lastPushedAt?: number

  pushAlert: (a: SignificantEventAlert, streamIndex?: number) => void
  pushScoredEvent: (a: SignificantEventAlert) => void
  consolidateAlert: (eventId: string, patch: Partial<SignificantEventAlert>) => void
  updateExplanation: (eventId: string, patch: Pick<SignificantEventAlert, 'explanation' | 'explanation_source' | 'summary' | 'summary_source' | 'recommendation'>) => void
  clear: () => void
  markRead: (eventId: string) => void
  markAllRead: () => void

  removeByEventId: (eventId: string) => void
}

export const useAlertsStore = create<AlertsState>((set, get) => ({
  maxAlerts: 50,
  alerts: [],
  scoredEvents: [],
  _recentKeys: [],
  lastPushed: undefined,
  lastPushedAt: undefined,

  pushAlert: (a, streamIndex) => {
    const maxAlerts = get().maxAlerts
    const recent = get()._recentKeys

    // Best-effort de-dupe: same event_id+timestamp+action within recent window.
    const key = `${a?.event_id || ''}|${a?.timestamp || ''}|${a?.significance?.action || ''}`
    if (key.trim() && recent.includes(key)) return

    const next: SignificantEventAlert = {
      ...a,
      _received_at: Date.now(),
      _unread: true,
      _streamIndex: typeof streamIndex === 'number' ? streamIndex : undefined,
    }

    set((s) => {
      const alerts = [...s.alerts, next]
      while (alerts.length > maxAlerts) alerts.shift()
      // Also add to scoredEvents for inference panel
      const scoredEvents = [...s.scoredEvents, next]
      while (scoredEvents.length > 200) scoredEvents.shift()
      const nextRecent = [...s._recentKeys, key].filter((k) => k.trim())
      while (nextRecent.length > 80) nextRecent.shift()
      return { alerts, scoredEvents, _recentKeys: nextRecent, lastPushed: next, lastPushedAt: next._received_at }
    })
  },

  pushScoredEvent: (a) => {
    // De-dupe by event_id
    const existing = get().scoredEvents
    if (existing.some(e => e.event_id === a.event_id)) return

    const next: SignificantEventAlert = {
      ...a,
      _received_at: Date.now(),
      _unread: false,
    }
    set((s) => {
      const scoredEvents = [...s.scoredEvents, next]
      while (scoredEvents.length > 200) scoredEvents.shift()
      return { scoredEvents }
    })
  },

  consolidateAlert: (eventId, patch) =>
    set((s) => {
      const update = (a: SignificantEventAlert): SignificantEventAlert =>
        a.event_id === eventId
          ? {
              ...a,
              patterns: patch.patterns ?? a.patterns,
              severity: patch.severity ?? a.severity,
              significance: patch.significance ?? a.significance,
              metrics: patch.metrics ?? a.metrics,
              _consolidated_count: patch._consolidated_count ?? a._consolidated_count,
              _consolidated_ids: patch._consolidated_ids ?? a._consolidated_ids,
            }
          : a
      return {
        alerts: s.alerts.map(update),
        scoredEvents: s.scoredEvents.map(update),
        // Do NOT update lastPushed/lastPushedAt — no new toast for consolidation
      }
    }),

  clear: () => set({ alerts: [], scoredEvents: [], _recentKeys: [], lastPushed: undefined, lastPushedAt: undefined }),

  updateExplanation: (eventId, patch) =>
    set((s) => {
      const patchAlert = (a: SignificantEventAlert): SignificantEventAlert => {
        if (a.event_id !== eventId) return a
        // A background explanation_update delivers the EXPLANATION; it must not
        // re-write a summary we already trust. Two cases we guard against:
        //  1. Once we hold an LLM summary, keep it — the update often carries a
        //     different, terser LLM summary that churns the headline a beat later
        //     ("Modulated vibration…" → "High force ratio occurring again").
        //  2. A fallback-sourced explanation must not clobber a real LLM one.
        const incomingExplIsFallback = patch.explanation_source != null && patch.explanation_source !== 'llm'
        const haveLlmSummary = a.summary_source === 'llm' && Boolean(a.summary)
        const haveLlmExplanation = a.explanation_source === 'llm' && Boolean(a.explanation)

        const keepSummary = haveLlmSummary
        const keepExplanation = incomingExplIsFallback && haveLlmExplanation

        return {
          ...a,
          summary: keepSummary ? a.summary : (patch.summary ?? a.summary),
          summary_source: keepSummary ? a.summary_source : (patch.summary_source ?? a.summary_source),
          explanation: keepExplanation ? a.explanation : (patch.explanation ?? a.explanation),
          explanation_source: keepExplanation ? a.explanation_source : (patch.explanation_source ?? a.explanation_source),
          recommendation: patch.recommendation ?? a.recommendation,
        }
      }
      return {
        alerts: s.alerts.map(patchAlert),
        scoredEvents: s.scoredEvents.map(patchAlert),
        lastPushed: s.lastPushed?.event_id === eventId
          ? patchAlert(s.lastPushed)
          : s.lastPushed,
      }
    }),

  markRead: (eventId) =>
    set((s) => ({
      alerts: s.alerts.map((a) => (a.event_id === eventId ? { ...a, _unread: false } : a)),
    })),

  markAllRead: () => set((s) => ({ alerts: s.alerts.map((a) => ({ ...a, _unread: false })) })),

  removeByEventId: (eventId) =>
    set((s) => {
      const wasLast = s.lastPushed?.event_id === eventId
      return {
        alerts: s.alerts.filter((a) => a.event_id !== eventId),
        scoredEvents: s.scoredEvents.filter((a) => a.event_id !== eventId),
        lastPushed: wasLast ? undefined : s.lastPushed,
        lastPushedAt: wasLast ? undefined : s.lastPushedAt,
      }
    }),
}))

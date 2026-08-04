/**
 * AppContext — shared state for all views/pages.
 *
 * Extracted from the monolithic App.tsx so that page components can
 * access session info, queries, alerts, and callbacks without prop-drilling.
 */
import React, { createContext, useContext } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'
import type { SignificantEventAlert } from '../state/alertsStore'
import type { DocLink } from '../types'

/* ── Shared helper types (moved from App.tsx) ─────────────── */

export type MemorySummary = {
  id: string
  session_id: string
  created_at: string
  patterns: string[]
  label?: string | null
  tags: string[]
  significance_score?: number | null
  annotation_preview?: string | null
}

export type PriorSeverityCalibration = {
  average_delta?: number
  weight_total?: number
  targets?: Partial<Record<'info' | 'warning' | 'critical', number>>
}

export type PriorRow = {
  pattern: string
  prior: number
  effective_weight_total?: number
  passive_outcome_count?: number
  severity_correction_count?: number
  severity_calibration?: PriorSeverityCalibration
}
export type PriorDiffRow = { pattern: string; before: number; after: number; delta: number }
export type ListMemoriesResponse = { memories: MemorySummary[]; total_count: number }
export type PriorsResponse = { priors: PriorRow[] }
export type MemoryDetailResponse = {
  memory: Record<string, unknown>
  feedback_stats: Record<string, unknown>
  doc_links: DocLink[]
}
export type FeedbackHistoryResponse = { events: Record<string, unknown>[]; stats: Record<string, unknown> }
export type TraceListResponse = { traces: Record<string, unknown>[] }
export type SessionSummary = {
  session_id: string
  status: string
  status_label: string
  running: boolean
  paused: boolean
  position: number
  total_samples: number
  progress: number | null
  last_error?: string | null
  loading?: boolean
  source?: string
  source_label?: string | null
  case_dir?: string | null
  operation_id?: string | null
  tool_id?: string | null
  resolved_start_position?: number | null
  requested_start_position?: number | null
  start_at_first_cutting_row?: boolean
}

/* ── Context shape ────────────────────────────────────────── */

export interface AppContextValue {
  /* session */
  streamSessionId: string
  setStreamSessionId: (id: string) => void
  sessionOptions: string[]
  sessionInfoQuery: UseQueryResult<Record<string, unknown> | null>

  /* queries */
  memoriesQuery: UseQueryResult<ListMemoriesResponse>
  priorsQuery: UseQueryResult<PriorsResponse>
  memoryDetailQuery: UseQueryResult<MemoryDetailResponse>
  feedbackHistoryQuery: UseQueryResult<FeedbackHistoryResponse>
  tracesQuery: UseQueryResult<TraceListResponse>

  /* alerts */
  alerts: SignificantEventAlert[]
  unreadCount: number

  /* WebSocket */
  wsStatus: string

  /* selected memory / feedback */
  selectedMemoryId: string
  setSelectedMemoryId: (id: string) => void

  /* explicit "open the detail modal for this event" request (e.g. toast "Open").
   * Distinct from selectedMemoryId, which merely focuses/pins an alert without
   * popping the modal. `at` is a nonce so repeat requests for the same id fire. */
  detailRequest: { id: string; at: number } | null
  requestAlertDetail: (id: string) => void
  clearAlertDetail: () => void

  /* playback */
  pause: () => Promise<void>
  resume: () => Promise<void>
  replay: () => Promise<void>

  /* feedback */
  sendFeedback: (action: 'confirm' | 'dismiss', aspect?: 'explanation' | 'recommendation') => Promise<void>
  feedbackPending: boolean
  feedbackMsg: { kind: 'ok' | 'err' | 'info'; text: string } | null

  /* prior diffs */
  lastPriorDiff: PriorDiffRow[]
  lastPriorDiffAt: number
  lastFeedbackMeta: { action: string; memoryId: string } | null

  /* config */
  operatorMode: boolean
  setOperatorMode: (v: boolean) => void
  autoOpenAlerts: boolean
  setAutoOpenAlerts: (v: boolean) => void
  freezeOnAlert: boolean
  setFreezeOnAlert: (v: boolean) => void
}

export const AppContext = createContext<AppContextValue | null>(null)

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useAppContext must be used inside <AppContext.Provider>')
  return ctx
}

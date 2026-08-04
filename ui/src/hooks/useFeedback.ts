/**
 * useFeedback — Extract feedback logic from App.tsx.
 *
 * Manages all state & actions for the confirm/dismiss/comment/label/tag workflow,
 * plus the prior-diff visualisation after each action.
 */
import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/http'
import { useAlertsStore } from '../state/alertsStore'
import type { UseQueryResult } from '@tanstack/react-query'
import type { PriorRow, PriorDiffRow, MemoryDetailResponse, PriorsResponse, FeedbackHistoryResponse, ListMemoriesResponse } from '../contexts/AppContext'

/* ── Utility helpers ───────────────────────── */

export function priorsToMap(priors: PriorRow[] | undefined): Record<string, number> {
  const out: Record<string, number> = {}
  for (const p of priors || []) {
    if (!p || typeof p.pattern !== 'string') continue
    out[p.pattern] = Number(p.prior) || 0
  }
  return out
}

export function extractMemoryPatterns(memory: Record<string, unknown> | undefined): string[] {
  if (!memory) return []
  const p = memory.pattern_keys
  if (Array.isArray(p))
    return p.map((x: unknown) => (typeof x === 'string' ? x : (x as Record<string, unknown>)?.key)).filter(Boolean) as string[]
  const ps = memory.patterns
  if (Array.isArray(ps)) return ps.filter((x: unknown) => typeof x === 'string') as string[]
  return []
}

export function diffPriors(
  before: PriorRow[] | undefined,
  after: PriorRow[] | undefined,
  focusPatterns: string[] = [],
): PriorDiffRow[] {
  const b = priorsToMap(before)
  const a = priorsToMap(after)
  const keys = new Set([...Object.keys(b), ...Object.keys(a)])
  const rows: PriorDiffRow[] = []
  for (const k of keys) {
    const bv = typeof b[k] === 'number' ? b[k] : 0
    const av = typeof a[k] === 'number' ? a[k] : 0
    const d = av - bv
    if (Math.abs(d) < 1e-8) continue
    rows.push({ pattern: k, before: bv, after: av, delta: d })
  }
  const focus = new Set(focusPatterns)
  rows.sort((r1, r2) => {
    const f1 = focus.has(r1.pattern) ? 1 : 0
    const f2 = focus.has(r2.pattern) ? 1 : 0
    if (f1 !== f2) return f2 - f1
    return Math.abs(r2.delta) - Math.abs(r1.delta)
  })
  return rows
}

export function severity(score?: number): { label: string; color: string } {
  const s = typeof score === 'number' && Number.isFinite(score) ? score : 0
  if (s >= 0.9) return { label: 'CRITICAL', color: 'var(--danger)' }
  if (s >= 0.75) return { label: 'WARNING', color: 'var(--accent)' }
  return { label: 'INFO', color: 'var(--muted)' }
}

function parseTags(s: string): string[] {
  return Array.from(new Set((s || '').split(',').map((x) => x.trim()).filter(Boolean)))
}

/* ── Hook ──────────────────────────────────── */

interface UseFeedbackDeps {
  selectedMemoryId: string
  priorsQuery: UseQueryResult<PriorsResponse>
  memoryDetailQuery: UseQueryResult<MemoryDetailResponse>
  feedbackHistoryQuery: UseQueryResult<FeedbackHistoryResponse>
  memoriesQuery: UseQueryResult<ListMemoriesResponse>
}

export interface FeedbackState {
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
  sendFeedback: (action: 'confirm' | 'dismiss', aspect?: 'explanation' | 'recommendation') => Promise<boolean>
  applyMetadataOnly: () => Promise<void>
  reportMissedEvent: (sessionId: string, patternKeys: string[]) => Promise<void>
}

export function useFeedback(deps: UseFeedbackDeps): FeedbackState {
  const { selectedMemoryId, priorsQuery, memoryDetailQuery, feedbackHistoryQuery, memoriesQuery } = deps

  const [feedbackReason, setFeedbackReason] = useState('')
  const [feedbackComment, setFeedbackComment] = useState('')
  const [feedbackLabel, setFeedbackLabel] = useState('')
  const [feedbackTags, setFeedbackTags] = useState('')
  const [feedbackPending, setFeedbackPending] = useState(false)
  const [feedbackMsg, setFeedbackMsg] = useState<{ kind: 'ok' | 'err' | 'info'; text: string } | null>(null)
  const [lastPriorDiff, setLastPriorDiff] = useState<PriorDiffRow[]>([])
  const [lastPriorDiffAt, setLastPriorDiffAt] = useState(0)
  const [lastFeedbackMeta, setLastFeedbackMeta] = useState<{ action: string; memoryId: string } | null>(null)

  // Reset draft feedback when switching memories.
  useEffect(() => {
    setFeedbackReason('')
    setFeedbackComment('')
    setFeedbackLabel('')
    setFeedbackTags('')
    setFeedbackMsg(null)
    setLastPriorDiff([])
    setLastFeedbackMeta(null)
  }, [selectedMemoryId])

  const applyMetadataOnly = useCallback(async () => {
    if (!selectedMemoryId) return

    setFeedbackPending(true)
    setFeedbackMsg({ kind: 'info', text: 'Applying metadata…' })

    const id = encodeURIComponent(selectedMemoryId)
    const comment = feedbackComment.trim() || null
    const nextLabel = feedbackLabel.trim() || null
    const tags = parseTags(feedbackTags)

    if (!comment && !nextLabel && !tags.length) {
      setFeedbackPending(false)
      setFeedbackMsg({ kind: 'info', text: 'Nothing to apply.' })
      return
    }

    try {
      if (comment) {
        await api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'comment', user_id: 'ui', comment })
      }
      if (nextLabel) {
        await api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'label', user_id: 'ui', label: nextLabel })
      }
      if (tags.length) {
        await api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'tag', user_id: 'ui', tags })
      }

      await memoryDetailQuery.refetch()
      await feedbackHistoryQuery.refetch()
      await priorsQuery.refetch()
      await memoriesQuery.refetch()

      setFeedbackMsg({ kind: 'ok', text: 'Applied.' })
    } catch (e) {
      setFeedbackMsg({ kind: 'err', text: `Apply failed: ${String(e)}` })
    } finally {
      setFeedbackPending(false)
    }

    setFeedbackComment('')
    setFeedbackReason('')
  }, [selectedMemoryId, feedbackComment, feedbackLabel, feedbackTags, memoryDetailQuery, feedbackHistoryQuery, priorsQuery, memoriesQuery])

  const sendFeedback = useCallback(async (action: 'confirm' | 'dismiss', aspect?: 'explanation' | 'recommendation') => {
    if (!selectedMemoryId) return false

    setFeedbackPending(true)
    setFeedbackMsg({ kind: 'info', text: action === 'confirm' ? 'Confirming…' : 'Dismissing…' })
    setLastPriorDiff([])
    setLastPriorDiffAt(0)
    setLastFeedbackMeta({ action, memoryId: selectedMemoryId })

    const id = encodeURIComponent(selectedMemoryId)
    const reason = feedbackReason.trim() || null
    const comment = feedbackComment.trim() || null
    const nextLabel = feedbackLabel.trim() || null
    const tags = parseTags(feedbackTags)

    // Episode-level learning dedup (plan 1.4): pass the alert's episode_id so a
    // multi-window episode only nudges the priors once. Undefined for alerts
    // without recurrence tracking → backend falls back to per-memory behaviour.
    const alert = useAlertsStore.getState().alerts.find((a) => a.event_id === selectedMemoryId)
    const episode_id = alert?.recurrence?.episode_id ?? undefined

    const priorsBefore = priorsQuery.data?.priors
    const focusPatterns = extractMemoryPatterns(memoryDetailQuery.data?.memory as Record<string, unknown> | undefined)

    let ok = false
    try {
      // Core confirm/dismiss lands first — it's the label that matters.
      // `aspect` (explanation | recommendation) is recorded when the operator
      // rates one facet of the alert independently; omitted → whole-alert feedback.
      await api(`/agent/memory/${id}/feedback`, 'PATCH', { action, user_id: 'ui', reason, episode_id, ...(aspect ? { aspect } : {}) })

      // Optional metadata actions are independent — send them concurrently.
      const metaPatches: Promise<unknown>[] = []
      if (comment) metaPatches.push(api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'comment', user_id: 'ui', comment }))
      if (nextLabel) metaPatches.push(api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'label', user_id: 'ui', label: nextLabel }))
      if (tags.length) metaPatches.push(api(`/agent/memory/${id}/feedback`, 'PATCH', { action: 'tag', user_id: 'ui', tags }))
      if (metaPatches.length) await Promise.all(metaPatches)

      // Success is known now — surface it immediately instead of waiting on refetches.
      ok = true
      setFeedbackMsg({ kind: 'ok', text: `${action === 'confirm' ? 'Confirmed' : 'Dismissed'} ✓` })

      // Refresh the dependent views concurrently; the prior-diff needs the post-priors.
      const [, , priorsAfterRes] = await Promise.all([
        memoryDetailQuery.refetch(),
        feedbackHistoryQuery.refetch(),
        priorsQuery.refetch(),
        memoriesQuery.refetch(),
      ])

      const priorsAfter = priorsAfterRes.data?.priors
      const diffs = diffPriors(priorsBefore, priorsAfter, focusPatterns)
      setLastPriorDiff(diffs.slice(0, 12))
      setLastPriorDiffAt(Date.now())
    } catch (e) {
      setFeedbackMsg({ kind: 'err', text: `${action} failed: ${String(e)}` })
    } finally {
      setFeedbackPending(false)
    }

    setFeedbackComment('')
    return ok
  }, [selectedMemoryId, feedbackReason, feedbackComment, feedbackLabel, feedbackTags, priorsQuery, memoryDetailQuery, feedbackHistoryQuery, memoriesQuery])

  const reportMissedEvent = useCallback(async (sessionId: string, patternKeys: string[]) => {
    if (!sessionId) return

    setFeedbackPending(true)
    setFeedbackMsg({ kind: 'info', text: 'Reporting missed event…' })

    const reason = feedbackReason.trim() || 'Operator-reported false negative'

    try {
      await api('/agent/memory/feedback/missed-event', 'POST', {
        session_id: sessionId,
        pattern_keys: patternKeys,
        user_id: 'ui',
        reason,
      })

      await priorsQuery.refetch()
      await memoriesQuery.refetch()

      setFeedbackMsg({ kind: 'ok', text: 'Missed event reported — thresholds adjusted ✓' })
    } catch (e) {
      setFeedbackMsg({ kind: 'err', text: `Report failed: ${String(e)}` })
    } finally {
      setFeedbackPending(false)
    }
  }, [feedbackReason, priorsQuery, memoriesQuery])

  return {
    feedbackReason, setFeedbackReason,
    feedbackComment, setFeedbackComment,
    feedbackLabel, setFeedbackLabel,
    feedbackTags, setFeedbackTags,
    feedbackPending,
    feedbackMsg,
    lastPriorDiff, lastPriorDiffAt, lastFeedbackMeta,
    sendFeedback, applyMetadataOnly, reportMissedEvent,
  }
}

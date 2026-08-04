/**
 * useSessionQueries — Extract query hooks from App.tsx.
 *
 * Centralises all React Query hooks for sessions, memories, priors,
 * memory detail, traces, and feedback history.
 */
import { useMemo, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, baseUrl } from '../api/http'
import { useStreamStore } from '../state/streamStore'
import type {
  ListMemoriesResponse,
  PriorsResponse,
  MemoryDetailResponse,
  FeedbackHistoryResponse,
  TraceListResponse,
  SessionSummary,
} from '../contexts/AppContext'

type SessionsResponse = {
  sessions: string[]
  session_summaries?: SessionSummary[]
}

export function useSessionQueries(selectedMemoryId: string) {
  const streamSessionId = useStreamStore((s) => s.sessionId)
  const setStreamSessionId = useStreamStore((s) => s.setSessionId)
  const resetStream = useStreamStore((s) => s.reset)
  const apiUrlApplied = baseUrl()

  const sessionsQuery = useQuery({
    queryKey: ['sessions', apiUrlApplied],
    queryFn: () => api<SessionsResponse>('/sessions'),
    refetchInterval: 5000,
  })

  const sessionInfoQuery = useQuery({
    queryKey: ['session-info', apiUrlApplied, streamSessionId],
    queryFn: async () => {
      if (!streamSessionId) return null
      return api<Record<string, unknown>>(`/sessions/${encodeURIComponent(streamSessionId)}`)
    },
    enabled: Boolean(streamSessionId),
    refetchInterval: 2000,
  })

  const memoriesQuery = useQuery({
    queryKey: ['memories', apiUrlApplied, streamSessionId],
    queryFn: async () => {
      if (!streamSessionId) return { memories: [], total_count: 0 } as ListMemoriesResponse
      return api<ListMemoriesResponse>(
        `/agent/memory/session/${encodeURIComponent(streamSessionId)}?limit=50&offset=0`,
      )
    },
    enabled: Boolean(streamSessionId),
    refetchInterval: 3000,
  })

  const priorsQuery = useQuery({
    queryKey: ['priors', apiUrlApplied],
    queryFn: () => api<PriorsResponse>('/agent/memory/scorer/priors?limit=50'),
    refetchInterval: 2000,
  })

  const memoryDetailQuery = useQuery({
    queryKey: ['memory', apiUrlApplied, selectedMemoryId],
    queryFn: () =>
      api<MemoryDetailResponse>(`/agent/memory/${encodeURIComponent(selectedMemoryId)}`),
    enabled: Boolean(selectedMemoryId),
  })

  const tracesQuery = useQuery({
    queryKey: ['traces', apiUrlApplied, selectedMemoryId],
    queryFn: () =>
      api<TraceListResponse>(
        `/agent/memory/${encodeURIComponent(selectedMemoryId)}/traces?limit=50`,
      ),
    enabled: Boolean(selectedMemoryId),
    refetchInterval: 8000,
  })

  const feedbackHistoryQuery = useQuery({
    queryKey: ['feedback', apiUrlApplied, selectedMemoryId],
    queryFn: () =>
      api<FeedbackHistoryResponse>(
        `/agent/memory/${encodeURIComponent(selectedMemoryId)}/feedback?limit=200`,
      ),
    enabled: Boolean(selectedMemoryId),
    refetchInterval: 8000,
  })

  const sessionOptions = useMemo(
    () => (sessionsQuery.data?.session_summaries || []).map((session) => session.session_id),
    [sessionsQuery.data],
  )

  const sessionSummaries = useMemo(() => {
    if (Array.isArray(sessionsQuery.data?.session_summaries)) {
      return sessionsQuery.data.session_summaries
    }
    return (sessionsQuery.data?.sessions || []).map((session_id) => ({
      session_id,
      status: 'idle',
      status_label: 'Ready',
      running: false,
      paused: false,
      position: 0,
      total_samples: 0,
      progress: null,
      last_error: null,
      loading: false,
      source: 'simulated_file',
      source_label: null,
      case_dir: null,
      operation_id: null,
      tool_id: null,
      resolved_start_position: null,
      requested_start_position: null,
      start_at_first_cutting_row: false,
    }))
  }, [sessionsQuery.data])

  // Clear stale session id when the backend no longer recognizes it, but do not
  // drop a freshly started session during the window before /sessions catches up.
  useEffect(() => {
    if (!streamSessionId) return
    if (!sessionsQuery.data) return
    if (sessionOptions.includes(streamSessionId)) return
    if (sessionInfoQuery.isLoading || sessionInfoQuery.isFetching) return
    // Gate on the *error* state, not on stale data: React Query keeps the last
    // successful `data` after a query starts erroring, so a dead session (e.g.
    // after a backend restart) still has `data` and would otherwise never clear
    // — leaving the UI hammering WS endpoints with 403s.
    if (!sessionInfoQuery.isError) return

    const errorMessage = sessionInfoQuery.error instanceof Error
      ? sessionInfoQuery.error.message
      : String(sessionInfoQuery.error || '')
    if (!errorMessage.includes('404')) return

    setStreamSessionId('')
    resetStream()
  }, [
    streamSessionId,
    sessionOptions,
    sessionsQuery.data,
    sessionInfoQuery.isLoading,
    sessionInfoQuery.isFetching,
    sessionInfoQuery.isError,
    sessionInfoQuery.error,
    setStreamSessionId,
    resetStream,
  ])

  return {
    sessionsQuery,
    sessionInfoQuery,
    memoriesQuery,
    priorsQuery,
    memoryDetailQuery,
    tracesQuery,
    feedbackHistoryQuery,
    sessionOptions,
    sessionSummaries,
  }
}

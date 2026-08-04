import { useDeferredValue, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api } from '../api/http'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { useAppContext } from '../contexts/AppContext'
import {
  buildChatPrompt,
  buildSearchStatusMessage,
  buildTwinEvidenceSummary,
  buildWeakMatchMessage,
  filterGroundedMatches,
  type DocsMatch,
} from './documentRetrievalHelpers'
import { colors, fontSize, radii, shadows, spacing } from '../styles/tokens'

interface DispatchEnvelope<T> {
  ok: boolean
  agent: string
  dispatch_id: string
  result: T
}

interface DocsStatusResult {
  backend: string
  ready: boolean
  document_count?: number
  entity_count?: number
  mention_count?: number
  relation_count?: number
  docs_with_mentions?: number
  docs_without_mentions?: number
  semantic_coverage_ratio?: number
  semantic_gap_usecases?: string[]
  sources?: string[]
  machines?: string[]
  twin_health?: {
    status?: 'ok' | 'warning' | 'error'
    headline?: string
    summary?: string
    semantic_ready_usecases?: number
    total_usecases?: number
    canonical_entity_count?: number
    semantic_coverage_ratio?: number
    semantic_gap_usecases?: string[]
  }
  usecase_coverage?: Array<{
    usecase: string
    document_count: number
    file_count: number
    entity_count: number
    canonical_entity_count: number
    mention_count: number
    relation_count: number
    docs_with_mentions: number
    docs_without_mentions: number
    semantic_coverage_ratio: number
    semantic_ready: boolean
  }>
  message?: string
}

interface SubgraphIntegrity {
  healthy: boolean
  mixed_label_nodes: number
  disallowed_cross_graph_edges: number
  disallowed_relationship_types?: string[]
  memory_labels?: string[]
  knowledge_labels?: string[]
  allowed_cross_relationships?: string[]
  error?: string
}

interface MemoryGraphStatus {
  configured_backend?: string
  resolved_backend?: string | null
  store_class?: string | null
  database?: string | null
  db_path?: string | null
  count?: number | string | null
  subgraph_integrity?: SubgraphIntegrity
}

interface RetrieverStatusResult {
  configured_storage_backend?: string
  storage_backend?: string
  sindit_enabled?: boolean
  sindit_reachable?: boolean
  memory_graph?: MemoryGraphStatus
}

interface DocsSearchResult {
  backend: string
  query: string
  top_k: number
  usecase?: string | null
  source_filter?: string | null
  machine?: string | null
  document_type?: string | null
  matches: DocsMatch[]
  message?: string
  timestamp?: string
}

interface DocsStructuredResult {
  backend: string
  mode: string
  records: DocsMatch[]
  ready: boolean
  message?: string
  timestamp?: string
}

interface LlmResult {
  answer?: string
  llm_available?: boolean
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  source: 'user' | 'llm' | 'fallback'
  matches?: DocsMatch[]
}

const STARTER_QUESTIONS = [
  'What maintenance steps does the manual recommend before startup?',
  'How does the documentation describe chatter or vibration issues?',
  'What alarms or warnings should the operator look for?',
]

const pageStyle: React.CSSProperties = {
  background: colors.bg,
  color: colors.text,
  minHeight: '100%',
  padding: spacing.xl,
}

const titleStyle: React.CSSProperties = {
  color: colors.text,
  fontSize: fontSize.xxl,
  fontWeight: 600,
  marginBottom: spacing.md,
}

const panelStyle: React.CSSProperties = {
  background: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.lg,
  boxShadow: shadows.panel,
  padding: spacing.lg,
}

const summaryGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.md,
  gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
}

const summaryCardStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  padding: spacing.md,
}

const filterGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.md,
  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
  alignItems: 'end',
}

const labelStyle: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.xs,
  letterSpacing: '0.06em',
  marginBottom: spacing.xs,
  textTransform: 'uppercase',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.text,
  fontSize: fontSize.sm,
  padding: '8px 10px',
}

const buttonStyle = (enabled: boolean): React.CSSProperties => ({
  background: enabled ? colors.accent : colors.surfaceAlt,
  border: `1px solid ${enabled ? colors.accent : colors.border}`,
  borderRadius: radii.sm,
  color: enabled ? '#fff' : colors.textDim,
  cursor: enabled ? 'pointer' : 'not-allowed',
  fontSize: fontSize.sm,
  fontWeight: 600,
  padding: '8px 12px',
})

const subtleButtonStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.text,
  cursor: 'pointer',
  fontSize: fontSize.sm,
  padding: '8px 12px',
}

const contentGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.lg,
  gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
  alignItems: 'start',
}

const chatListStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  display: 'grid',
  gap: spacing.sm,
  minHeight: 360,
  maxHeight: 640,
  overflowY: 'auto',
  padding: spacing.md,
}

const helperTextStyle: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.sm,
}

async function dispatchAgent<T>(
  sessionId: string,
  agent: string,
  action: string,
  args: Record<string, unknown>,
): Promise<T> {
  const response = await api<DispatchEnvelope<T>>(
    `/agent/dispatch/${encodeURIComponent(sessionId)}`,
    'POST',
    { agent, action, args, stream: false },
  )
  return response.result
}

function trimOrUndefined(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

function shorten(text: string | null | undefined, limit = 220): string {
  const value = String(text || '').trim()
  if (value.length <= limit) return value
  return `${value.slice(0, limit - 3).trimEnd()}...`
}

function buildFallbackReply(question: string, matches: DocsMatch[], backendMessage?: string): string {
  if (!matches.length) {
    return [
      `I could not find relevant document chunks for "${question}" in the current scope.`,
      backendMessage || 'Try broadening the usecase, machine, or document-type filters, or verify that the document graph has been ingested.',
    ].join(' ')
  }

  const lines = matches.slice(0, 3).map((match, index) => {
    const citation = match.citation || match.file_name || `Document ${index + 1}`
    const preview = shorten(match.text, 180)
    return `${index + 1}. ${citation}${match.score != null ? ` (score ${match.score.toFixed(2)})` : ''} — ${preview}`
  })

  return [
    'I found relevant documentation, but the assistant model is unavailable so I cannot synthesize a grounded answer right now.',
    'Closest excerpts:',
    ...lines,
    'Start the configured LLM provider to get a synthesized answer over these citations.',
  ].join('\n\n')
}

function uniqueValues(values: Array<string | null | undefined>, limit = 8): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const text = String(value || '').trim()
    if (!text || seen.has(text)) continue
    seen.add(text)
    out.push(text)
    if (out.length >= limit) break
  }
  return out
}

export default function DocumentRetrievalPage() {
  const { streamSessionId } = useAppContext()
  const effectiveSessionId = streamSessionId || 'documents-ui'

  const [draftQuestion, setDraftQuestion] = useState('')
  const [usecase, setUsecase] = useState('')
  const [machine, setMachine] = useState('')
  const [documentType, setDocumentType] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [submitError, setSubmitError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastSearchMessage, setLastSearchMessage] = useState<string>('')

  const deferredUsecase = useDeferredValue(usecase)
  const deferredMachine = useDeferredValue(machine)
  const deferredDocumentType = useDeferredValue(documentType)

  const statusQuery = useQuery({
    queryKey: ['docs-status', effectiveSessionId],
    queryFn: () => dispatchAgent<DocsStatusResult>(effectiveSessionId, 'retriever', 'docs.status', {}),
    staleTime: 15000,
  })

  const retrieverStatusQuery = useQuery({
    queryKey: ['retriever-status', effectiveSessionId],
    queryFn: () => dispatchAgent<RetrieverStatusResult>(effectiveSessionId, 'retriever', 'status', {}),
    staleTime: 15000,
  })

  const structuredQuery = useQuery({
    queryKey: ['docs-structured', effectiveSessionId, deferredUsecase, deferredMachine, deferredDocumentType],
    queryFn: () => dispatchAgent<DocsStructuredResult>(effectiveSessionId, 'retriever', 'docs.structured', {
      mode: 'structured',
      limit: 10,
      usecase: trimOrUndefined(deferredUsecase),
      machine: trimOrUndefined(deferredMachine),
      document_type: trimOrUndefined(deferredDocumentType),
    }),
    staleTime: 10000,
  })

  const latestMatches = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index]
      if (message.role === 'assistant' && message.matches && message.matches.length > 0) {
        return message.matches
      }
    }
    return [] as DocsMatch[]
  }, [messages])

  const latestCitations = uniqueValues(latestMatches.map((match) => match.citation || match.file_name))
  const availableSources = uniqueValues(statusQuery.data?.sources || [])
  const availableMachines = uniqueValues(statusQuery.data?.machines || [])
  const memoryGraph = retrieverStatusQuery.data?.memory_graph
  const integrity = memoryGraph?.subgraph_integrity
  const semanticGapUsecases = uniqueValues(statusQuery.data?.semantic_gap_usecases || [])
  const usecaseCoverage = statusQuery.data?.usecase_coverage || []
  const integrityIssueCount = (integrity?.mixed_label_nodes ?? 0) + (integrity?.disallowed_cross_graph_edges ?? 0)
  const integrityTone = integrity?.error
    ? colors.warn
    : integrity?.healthy
      ? colors.good
      : colors.bad
  const twinHealth = statusQuery.data?.twin_health
  const semanticCoverageLabel = twinHealth?.headline || '—'
  const twinHealthSummary = twinHealth?.summary || statusQuery.data?.message || 'no twin health summary reported'
  const twinHealthTone = twinHealth?.status === 'warning'
    ? colors.warn
    : twinHealth?.status === 'error'
      ? colors.bad
      : statusQuery.data?.ready
        ? colors.good
        : colors.textMuted
  const filtersSummary = [
    trimOrUndefined(usecase) ? `usecase=${trimOrUndefined(usecase)}` : null,
    trimOrUndefined(machine) ? `machine=${trimOrUndefined(machine)}` : null,
    trimOrUndefined(documentType) ? `type=${trimOrUndefined(documentType)}` : null,
  ].filter(Boolean).join(' · ') || 'all documents'

  const askQuestion = async () => {
    const question = draftQuestion.trim()
    if (!question || isSubmitting) return

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      source: 'user',
    }

    setMessages((current) => [...current, userMessage])
    setSubmitError('')
    setIsSubmitting(true)

    try {
      const searchResult = await dispatchAgent<DocsSearchResult>(effectiveSessionId, 'retriever', 'docs.search', {
        query: question,
        top_k: 5,
        usecase: trimOrUndefined(usecase),
        machine: trimOrUndefined(machine),
        document_type: trimOrUndefined(documentType),
      })

      const rawMatches = searchResult.matches || []
      const matches = filterGroundedMatches(rawMatches)
      setLastSearchMessage(buildSearchStatusMessage(rawMatches, matches, searchResult.message))

      let answer = ''
      let source: ChatMessage['source'] = 'fallback'

      if (matches.length > 0) {
        try {
          const llmResult = await dispatchAgent<LlmResult>(effectiveSessionId, 'llm.rag', 'query', {
            question: buildChatPrompt(question, matches),
          })
          const candidate = String(llmResult.answer || '').trim()
          if (llmResult.llm_available && candidate) {
            answer = candidate
            source = 'llm'
          }
        } catch {
          // Fallback rendering below handles LLM failures.
        }
      }

      if (!answer) {
        answer = matches.length > 0
          ? buildFallbackReply(question, matches, searchResult.message)
          : rawMatches.length > 0
            ? buildWeakMatchMessage(question, rawMatches)
            : buildFallbackReply(question, matches, searchResult.message)
      }

      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: answer,
          source,
          matches,
        },
      ])
      setDraftQuestion('')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Document search failed.'
      setSubmitError(message)
      setMessages((current) => [
        ...current,
        {
          id: `assistant-error-${Date.now()}`,
          role: 'assistant',
          content: `The document chat request failed before retrieval completed. ${message}`,
          source: 'fallback',
          matches: [],
        },
      ])
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <ErrorBoundary label="Document Chat">
      <div style={pageStyle}>
        <h1 style={titleStyle}>Document Chat</h1>
        <p style={{ color: colors.textMuted, marginTop: -spacing.sm, marginBottom: spacing.lg, maxWidth: 860 }}>
          Ask questions against the document graph, inspect the retrieved source chunks, and verify the citations the assistant used.
        </p>

        <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
          <div style={summaryGridStyle}>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Active Session</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>{effectiveSessionId}</div>
              <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
                {streamSessionId ? 'using current stream session' : 'standalone document chat scope'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Docs Backend</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {statusQuery.data?.backend || 'loading'}
              </div>
              <div style={{ ...helperTextStyle, color: statusQuery.data?.ready ? colors.good : colors.warn, marginTop: spacing.xs }}>
                {statusQuery.isLoading ? 'checking index...' : statusQuery.data?.message || 'status unavailable'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Indexed Chunks</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {statusQuery.data?.document_count ?? 0}
              </div>
              <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
                {availableSources.length > 0 ? availableSources.slice(0, 3).join(', ') : 'no sources reported'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Semantic Layer</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {typeof statusQuery.data?.entity_count === 'number' ? statusQuery.data.entity_count.toLocaleString() : '—'} entities
              </div>
              <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
                {typeof statusQuery.data?.mention_count === 'number' && typeof statusQuery.data?.relation_count === 'number'
                  ? `${statusQuery.data.mention_count.toLocaleString()} mentions · ${statusQuery.data.relation_count.toLocaleString()} relations`
                  : 'semantic counts unavailable'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Coverage Gaps</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {typeof statusQuery.data?.docs_without_mentions === 'number'
                  ? statusQuery.data.docs_without_mentions.toLocaleString()
                  : '—'} ungrounded chunks
              </div>
              <div
                style={{
                  ...helperTextStyle,
                  color: semanticGapUsecases.length > 0 ? colors.warn : colors.textMuted,
                  marginTop: spacing.xs,
                }}
              >
                {semanticGapUsecases.length > 0
                  ? `No entities for ${semanticGapUsecases.join(', ')}`
                  : 'all reported usecases have some semantic coverage'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Current Scope</div>
              <div style={{ fontSize: fontSize.md, fontWeight: 600 }}>{filtersSummary}</div>
              <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
                {latestCitations.length > 0 ? `${latestCitations.length} recent citations loaded` : 'ask a question to retrieve context'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Memory Graph</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {memoryGraph?.resolved_backend || 'loading'}
              </div>
              <div
                style={{
                  ...helperTextStyle,
                  color: memoryGraph?.resolved_backend === 'neo4j' ? colors.good : colors.warn,
                  marginTop: spacing.xs,
                }}
              >
                {retrieverStatusQuery.isLoading
                  ? 'checking graph store...'
                  : `${typeof memoryGraph?.count === 'number' ? memoryGraph.count.toLocaleString() : memoryGraph?.count ?? '—'} memories · ${memoryGraph?.store_class || 'store unavailable'}`}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Subgraph Integrity</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {retrieverStatusQuery.isLoading
                  ? 'loading'
                  : integrity?.error
                    ? 'warning'
                    : integrity?.healthy
                      ? 'healthy'
                      : `${integrityIssueCount} issues`}
              </div>
              <div style={{ ...helperTextStyle, color: integrityTone, marginTop: spacing.xs }}>
                {integrity?.error
                  ? integrity.error
                  : integrity
                    ? `${integrity.mixed_label_nodes} mixed labels · ${integrity.disallowed_cross_graph_edges} disallowed edges`
                    : 'no integrity summary reported'}
              </div>
            </div>
            <div style={summaryCardStyle}>
              <div style={labelStyle}>Twin Health</div>
              <div style={{ fontSize: fontSize.lg, fontWeight: 600 }}>
                {statusQuery.isLoading ? 'loading' : semanticCoverageLabel}
              </div>
              <div style={{ ...helperTextStyle, color: twinHealthTone, marginTop: spacing.xs }}>
                {twinHealthSummary}
              </div>
            </div>
          </div>
          {(integrity || statusQuery.data || retrieverStatusQuery.data) ? (
            <div style={{ ...helperTextStyle, marginTop: spacing.md }}>
              Cross-graph edges allowed: {integrity?.allowed_cross_relationships?.length ? integrity.allowed_cross_relationships.join(', ') : 'none'}.
              {integrity && !integrity.error ? ` Violating relationships: ${integrity.disallowed_relationship_types?.join(', ') || 'none'}.` : ''}
              {semanticGapUsecases.length > 0 ? ` Semantic gaps: ${semanticGapUsecases.join(', ')}.` : ' Semantic layer ready for reported usecases.'}
              {retrieverStatusQuery.data?.sindit_enabled ? ` SINDIT reachable: ${retrieverStatusQuery.data?.sindit_reachable ? 'yes' : 'no'}.` : ' SINDIT integration disabled.'}
            </div>
          ) : null}
        </section>

        <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
          <div style={filterGridStyle}>
            <div>
              <div style={labelStyle}>Usecase</div>
              <input
                value={usecase}
                onChange={(event) => setUsecase(event.target.value)}
                placeholder="site_b / site_a / site_c"
                style={inputStyle}
              />
            </div>
            <div>
              <div style={labelStyle}>Machine</div>
              <input
                value={machine}
                onChange={(event) => setMachine(event.target.value)}
                placeholder="MACHINE_B1, MACHINE_A1, c1001..."
                style={inputStyle}
              />
            </div>
            <div>
              <div style={labelStyle}>Document Type</div>
              <input
                value={documentType}
                onChange={(event) => setDocumentType(event.target.value)}
                placeholder="manual / procedure / spreadsheet"
                style={inputStyle}
              />
            </div>
            <div>
              <div style={labelStyle}>Refresh</div>
              <button
                type="button"
                onClick={() => {
                  void statusQuery.refetch()
                  void retrieverStatusQuery.refetch()
                  void structuredQuery.refetch()
                }}
                style={subtleButtonStyle}
              >
                Reload status
              </button>
            </div>
          </div>

          <div style={{ ...helperTextStyle, marginTop: spacing.md }}>
            Available sources: {availableSources.join(', ') || 'none reported'}
          </div>
          <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
            Known machines: {availableMachines.join(', ') || 'none reported'}
          </div>
          <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
            Usecase coverage: {usecaseCoverage.length > 0
              ? usecaseCoverage
                .map((entry) => `${entry.usecase} ${entry.docs_with_mentions}/${entry.document_count} grounded · ${entry.entity_count} entities`)
                .join(' | ')
              : 'none reported'}
          </div>
        </section>

        <div style={contentGridStyle}>
          <section style={panelStyle}>
            <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, justifyContent: 'space-between', marginBottom: spacing.md }}>
              <div>
                <h2 style={{ color: colors.text, fontSize: fontSize.lg, margin: 0 }}>Chat</h2>
                <div style={{ ...helperTextStyle, marginTop: spacing.xs }}>
                  The page first retrieves matching chunks from Neo4j, filters out weak hits, then asks the LLM to answer over the remaining snippets.
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  setMessages([])
                  setSubmitError('')
                  setLastSearchMessage('')
                }}
                style={subtleButtonStyle}
              >
                New chat
              </button>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: spacing.xs, marginBottom: spacing.md }}>
              {STARTER_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => setDraftQuestion(question)}
                  style={{
                    background: colors.surfaceAlt,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 999,
                    color: colors.textMuted,
                    cursor: 'pointer',
                    fontSize: fontSize.xs,
                    padding: '6px 10px',
                  }}
                >
                  {question}
                </button>
              ))}
            </div>

            <div style={chatListStyle}>
              {messages.length === 0 ? (
                <div style={{ ...helperTextStyle, alignSelf: 'center', justifySelf: 'center', maxWidth: 520, textAlign: 'center' }}>
                  Ask a manual question to retrieve relevant chunks, see the citations, and verify whether the answer was synthesized by the LLM or returned in fallback mode.
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    style={{
                      justifySelf: message.role === 'user' ? 'end' : 'start',
                      maxWidth: '92%',
                    }}
                  >
                    <div
                      style={{
                        background: message.role === 'user' ? colors.accent : colors.surface,
                        border: `1px solid ${message.role === 'user' ? colors.accent : colors.border}`,
                        borderRadius: radii.md,
                        color: '#fff',
                        padding: spacing.md,
                        whiteSpace: 'pre-wrap',
                      }}
                    >
                      <div style={{ fontSize: fontSize.xs, fontWeight: 700, marginBottom: spacing.xs, opacity: 0.8, textTransform: 'uppercase' }}>
                        {message.role === 'user' ? 'Operator' : message.source === 'llm' ? 'Assistant · llm' : 'Assistant · fallback'}
                      </div>
                      <div style={{ color: message.role === 'user' ? '#fff' : colors.text, fontSize: fontSize.sm, lineHeight: 1.55 }}>
                        {message.content}
                      </div>
                    </div>

                    {message.matches && message.matches.length > 0 ? (
                      <div style={{ display: 'grid', gap: spacing.xs, marginTop: spacing.sm }}>
                        {message.matches.slice(0, 5).map((match, index) => (
                          (() => {
                            const twinEvidence = buildTwinEvidenceSummary(match)
                            return (
                          <div
                            key={`${message.id}-${match.id || index}`}
                            style={{
                              background: colors.surfaceAlt,
                              border: `1px solid ${colors.border}`,
                              borderRadius: radii.sm,
                              padding: spacing.sm,
                            }}
                          >
                            <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, justifyContent: 'space-between' }}>
                              <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600 }}>
                                <span style={{ color: colors.accent, marginRight: spacing.xs }}>[{index + 1}]</span>
                                {match.citation || match.file_name || `Source ${index + 1}`}
                              </div>
                              <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                                {match.score != null ? `score ${match.score.toFixed(2)}` : match.document_type || 'document'}
                              </div>
                            </div>
                            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                              {shorten(match.text, 220)}
                            </div>
                            {twinEvidence ? (
                              <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                                Twin evidence: {twinEvidence}
                              </div>
                            ) : null}
                          </div>
                            )
                          })()
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))
              )}

              {isSubmitting ? (
                <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>Retrieving matching chunks and asking the assistant...</div>
              ) : null}
            </div>

            {submitError ? (
              <div style={{ background: 'rgba(224, 100, 94, 0.14)', border: `1px solid ${colors.bad}`, borderRadius: radii.sm, color: colors.bad, fontSize: fontSize.sm, marginTop: spacing.md, padding: spacing.sm }}>
                {submitError}
              </div>
            ) : null}

            <form
              onSubmit={(event) => {
                event.preventDefault()
                void askQuestion()
              }}
              style={{ display: 'grid', gap: spacing.sm, marginTop: spacing.md }}
            >
              <textarea
                value={draftQuestion}
                onChange={(event) => setDraftQuestion(event.target.value)}
                placeholder="Ask a question about the manuals, alarms, procedures, or troubleshooting steps..."
                rows={4}
                style={{
                  ...inputStyle,
                  minHeight: 112,
                  resize: 'vertical',
                }}
              />
              <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, justifyContent: 'space-between', flexWrap: 'wrap' }}>
                <div style={helperTextStyle}>
                  Last backend message: {lastSearchMessage || 'none yet'}
                </div>
                <button type="submit" disabled={isSubmitting || draftQuestion.trim().length === 0} style={buttonStyle(!isSubmitting && draftQuestion.trim().length > 0)}>
                  Ask documents
                </button>
              </div>
            </form>
          </section>

          <div style={{ display: 'grid', gap: spacing.lg }}>
            <section style={panelStyle}>
              <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                <h2 style={{ color: colors.text, fontSize: fontSize.lg, margin: 0 }}>Latest retrieval</h2>
                <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{latestMatches.length} hits</span>
              </div>

              {latestMatches.length === 0 ? (
                <div style={helperTextStyle}>No retrieval snapshot yet. Ask a question to inspect the supporting chunks.</div>
              ) : (
                <div style={{ display: 'grid', gap: spacing.sm }}>
                  {latestMatches.map((match, index) => (
                    (() => {
                      const twinEvidence = buildTwinEvidenceSummary(match)
                      return (
                    <div
                      key={`latest-${match.id || index}`}
                      style={{
                        background: colors.surfaceAlt,
                        border: `1px solid ${colors.border}`,
                        borderRadius: radii.md,
                        padding: spacing.md,
                      }}
                    >
                      <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600 }}>
                        {match.citation || match.file_name || `Result ${index + 1}`}
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                        {match.machine || 'machine unknown'} · {match.document_type || 'document'} · {match.language || 'language unknown'}
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.sm, lineHeight: 1.5, marginTop: spacing.sm }}>
                        {shorten(match.text, 320)}
                      </div>
                      {twinEvidence ? (
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          Twin evidence: {twinEvidence}
                        </div>
                      ) : null}
                    </div>
                      )
                    })()
                  ))}
                </div>
              )}
            </section>

            <section style={panelStyle}>
              <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                <h2 style={{ color: colors.text, fontSize: fontSize.lg, margin: 0 }}>Structured library view</h2>
                <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                  {structuredQuery.data?.records?.length ?? 0} records
                </span>
              </div>

              {structuredQuery.isLoading ? (
                <div style={helperTextStyle}>Loading document records...</div>
              ) : structuredQuery.isError ? (
                <div style={{ color: colors.bad, fontSize: fontSize.sm }}>
                  {(structuredQuery.error as Error)?.message || 'Failed to load structured records.'}
                </div>
              ) : structuredQuery.data?.records?.length ? (
                <div style={{ display: 'grid', gap: spacing.sm }}>
                  {structuredQuery.data.records.map((record, index) => (
                    <div
                      key={`record-${record.id || index}`}
                      style={{
                        background: colors.surfaceAlt,
                        border: `1px solid ${colors.border}`,
                        borderRadius: radii.md,
                        padding: spacing.md,
                      }}
                    >
                      <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600 }}>
                        {record.citation || record.file_name || `Record ${index + 1}`}
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                        {[
                          record.file_name && record.citation && record.file_name !== record.citation ? record.file_name : null,
                          record.machine || 'machine unknown',
                          record.document_type || 'document',
                          record.language || 'language unknown',
                        ].filter(Boolean).join(' | ')}
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.sm, lineHeight: 1.5, marginTop: spacing.sm }}>
                        {shorten(record.text, 240)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={helperTextStyle}>
                  {structuredQuery.data?.message || 'No structured document records available for the current scope.'}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}

export interface DocsMatch {
  id?: string | null
  text?: string | null
  source?: string | null
  usecase?: string | null
  file_name?: string | null
  page?: number | string | null
  machine?: string | null
  document_type?: string | null
  language?: string | null
  score?: number | null
  citation?: string | null
  helpful_count?: number | null
  not_helpful_count?: number | null
  feedback_score?: number | null
  ranking_score?: number | null
  graph_support?: number | null
  evidence_entities?: Array<Record<string, unknown>>
}

export const MIN_GROUNDED_SCORE = 0.18

function shorten(text: string | null | undefined, limit = 220): string {
  const value = String(text || '').trim()
  if (value.length <= limit) return value
  return `${value.slice(0, limit - 3).trimEnd()}...`
}

function evidenceEntityLabels(match: DocsMatch, limit = 4): string[] {
  return (match.evidence_entities || [])
    .map((entity) => {
      const name = String(entity.name || entity.id || '').trim()
      const entityType = String(entity.type || '').trim()
      if (name && entityType) return `${name} (${entityType})`
      return name
    })
    .filter(Boolean)
    .slice(0, limit)
}

export function buildTwinEvidenceSummary(match: DocsMatch): string {
  const details: string[] = []
  const graphSupport = typeof match.graph_support === 'number' && Number.isFinite(match.graph_support)
    ? match.graph_support
    : null
  const helpfulCount = typeof match.helpful_count === 'number' && Number.isFinite(match.helpful_count)
    ? match.helpful_count
    : 0
  const notHelpfulCount = typeof match.not_helpful_count === 'number' && Number.isFinite(match.not_helpful_count)
    ? match.not_helpful_count
    : 0
  const entityLabels = evidenceEntityLabels(match)

  if (graphSupport != null && graphSupport > 0) {
    details.push(`graph support ${graphSupport}`)
  }
  if (entityLabels.length > 0) {
    details.push(`entities ${entityLabels.join(', ')}`)
  }
  if (helpfulCount > 0 || notHelpfulCount > 0) {
    details.push(`feedback helpful ${helpfulCount}, not helpful ${notHelpfulCount}`)
  }

  return details.join(' | ')
}

export function buildChatPrompt(question: string, matches: DocsMatch[]): string {
  const contexts = matches
    .slice(0, 5)
    .map((match, index) => {
      const source = match.citation || match.file_name || `Document ${index + 1}`
      const snippet = shorten(match.text, 700)
      const twinEvidence = buildTwinEvidenceSummary(match)
      return [
        `[${index + 1}] ${source}`,
        twinEvidence ? `Twin evidence: ${twinEvidence}` : '',
        snippet,
      ].filter(Boolean).join('\n')
    })
    .join('\n\n')

  return [
    'You are answering a question about CNC manuals and machine documents.',
    'Use only the excerpts and twin evidence below. If the answer is not supported by them, say that directly.',
    'Treat graph support, entity labels, and feedback as grounding hints, not as substitutes for missing source text.',
    'Cite sources inline using only their bracketed number (e.g. [1], [2]). Do NOT write out document names, file names, or page numbers inline — the full numbered source list is shown to the operator separately.',
    '',
    contexts,
    '',
    `Question: ${question}`,
    'Answer concisely and cite the most relevant sources by their bracketed number.',
  ].join('\n')
}

export function getMatchScore(match: DocsMatch): number | null {
  return typeof match.score === 'number' && Number.isFinite(match.score) ? match.score : null
}

export function filterGroundedMatches(matches: DocsMatch[]): DocsMatch[] {
  return matches.filter((match) => {
    const score = getMatchScore(match)
    return score == null || score >= MIN_GROUNDED_SCORE
  })
}

export function buildWeakMatchMessage(question: string, matches: DocsMatch[]): string {
  const bestScore = matches.reduce<number | null>((best, match) => {
    const score = getMatchScore(match)
    if (score == null) return best
    if (best == null || score > best) return score
    return best
  }, null)

  const scoreText = bestScore == null ? '' : ` Best score: ${bestScore.toFixed(2)}.`
  return [
    `I found document chunks for "${question}", but they are too weakly related to treat as grounded evidence.`,
    `The current relevance gate is ${MIN_GROUNDED_SCORE.toFixed(2)} and the retrieved matches stayed below it.${scoreText}`,
    'Narrow the scope with usecase, machine, or document type filters, or ask with terms that should appear directly in the source material.',
  ].join(' ')
}

export function buildSearchStatusMessage(rawMatches: DocsMatch[], groundedMatches: DocsMatch[], backendMessage?: string): string {
  if (rawMatches.length > 0 && groundedMatches.length === 0) {
    const bestScore = rawMatches.reduce<number | null>((best, match) => {
      const score = getMatchScore(match)
      if (score == null) return best
      if (best == null || score > best) return score
      return best
    }, null)
    return [
      `Suppressed ${rawMatches.length} weak matches below ${MIN_GROUNDED_SCORE.toFixed(2)}.`,
      bestScore == null ? '' : `Best score ${bestScore.toFixed(2)}.`,
      backendMessage || '',
    ].filter(Boolean).join(' ')
  }
  return backendMessage || ''
}
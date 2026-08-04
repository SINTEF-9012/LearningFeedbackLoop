import React, { useEffect, useState } from 'react'

import { api } from '../api/http'
import type { DocLink } from '../types'

type DocLinkFeedback = 'helpful' | 'not_helpful'

function shortenText(value: string | null | undefined, limit = 220): string {
  const text = String(value || '').trim()
  if (!text) return ''
  if (text.length <= limit) return text
  return `${text.slice(0, limit - 3).trimEnd()}...`
}

function linkTitle(link: DocLink, index: number): string {
  return String(link.citation || link.file_name || `Document ${index + 1}`)
}

function entityNames(link: DocLink): string[] {
  return (link.evidence_entities || [])
    .map((entity) => {
      const record = entity as Record<string, unknown>
      return String(record.name || record.id || '').trim()
    })
    .filter(Boolean)
    .slice(0, 4)
}

function feedbackSummary(link: DocLink): string {
  const parts: string[] = []
  if (link.doc_feedback) parts.push(`rated ${link.doc_feedback}`)
  if (typeof link.helpful_count === 'number' && link.helpful_count > 0) parts.push(`helpful ${link.helpful_count}`)
  if (typeof link.not_helpful_count === 'number' && link.not_helpful_count > 0) parts.push(`not helpful ${link.not_helpful_count}`)
  if (typeof link.feedback_score === 'number' && parts.length > 0) parts.push(`feedback score ${link.feedback_score.toFixed(1)}`)
  return parts.join(' | ')
}

function linkKey(link: DocLink, index: number): string {
  return `${link.id || link.citation || link.file_name || link.query_used}-${index}`
}

function applyFeedback(link: DocLink, feedback: DocLinkFeedback): DocLink {
  const previous = String(link.doc_feedback || '').trim()
  let helpfulCount = typeof link.helpful_count === 'number' ? link.helpful_count : 0
  let notHelpfulCount = typeof link.not_helpful_count === 'number' ? link.not_helpful_count : 0

  if (previous === 'helpful') helpfulCount = Math.max(0, helpfulCount - 1)
  if (previous === 'not_helpful') notHelpfulCount = Math.max(0, notHelpfulCount - 1)
  if (feedback === 'helpful') helpfulCount += 1
  if (feedback === 'not_helpful') notHelpfulCount += 1

  return {
    ...link,
    doc_feedback: feedback,
    helpful_count: helpfulCount,
    not_helpful_count: notHelpfulCount,
    feedback_score: helpfulCount - notHelpfulCount,
  }
}

function replaceLink(links: DocLink[], rowKey: string, nextLink: DocLink): DocLink[] {
  return links.map((candidate, index) => (linkKey(candidate, index) === rowKey ? nextLink : candidate))
}

type DocLinksSectionProps = {
  docLinks: DocLink[]
  limit?: number
  memoryId?: string | null
  userId?: string
}

export function DocLinksSection({ docLinks, limit = 5, memoryId = null, userId = 'ui' }: DocLinksSectionProps) {
  const [itemsState, setItemsState] = useState<DocLink[]>(Array.isArray(docLinks) ? docLinks : [])
  const [pendingState, setPendingState] = useState<{ rowKey: string; feedback: DocLinkFeedback } | null>(null)
  const [errorMessage, setErrorMessage] = useState<string>('')

  useEffect(() => {
    setItemsState(Array.isArray(docLinks) ? docLinks : [])
  }, [docLinks])

  const items = Array.isArray(itemsState) ? itemsState.slice(0, limit) : []
  if (!items.length) return null

  const memoryKey = String(memoryId || '').trim()

  const handleFeedback = async (link: DocLink, index: number, feedback: DocLinkFeedback) => {
    const docId = String(link.id || '').trim()
    const rowKey = linkKey(link, index)
    if (!memoryKey || !docId || pendingState) return

    const previousItems = itemsState.slice()
    setPendingState({ rowKey, feedback })
    setErrorMessage('')
    setItemsState((current) => replaceLink(current, rowKey, applyFeedback(link, feedback)))

    try {
      const encodedMemoryId = encodeURIComponent(memoryKey)
      const encodedDocId = encodeURIComponent(docId)
      const response = await api<{ doc_link?: DocLink }>(
        `/agent/memory/${encodedMemoryId}/doc_links/${encodedDocId}/feedback`,
        'PATCH',
        { feedback, user_id: userId, reason: null },
      )

      if (response.doc_link) {
        setItemsState((current) => replaceLink(current, rowKey, response.doc_link as DocLink))
      }

      const refreshed = await api<{ doc_links?: DocLink[] }>(`/agent/memory/${encodedMemoryId}`)
      if (Array.isArray(refreshed.doc_links)) {
        setItemsState(refreshed.doc_links)
      }
    } catch (error) {
      setItemsState(previousItems)
      setErrorMessage(error instanceof Error ? error.message : 'Unable to save document feedback.')
    } finally {
      setPendingState(null)
    }
  }

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {errorMessage ? (
        <div className="small" style={{ color: 'var(--danger)' }}>
          {errorMessage}
        </div>
      ) : null}
      {items.map((link, index) => {
        const title = linkTitle(link, index)
        const excerpt = shortenText(link.text, 260)
        const entities = entityNames(link)
        const feedback = feedbackSummary(link)
        const rowKey = linkKey(link, index)
        const docId = String(link.id || '').trim()
        const feedbackEnabled = Boolean(memoryKey && docId)
        const pending = pendingState?.rowKey === rowKey

        return (
          <div
            key={rowKey}
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 8,
              padding: '10px 12px',
            }}
          >
            <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'space-between' }}>
              <div style={{ fontWeight: 600, lineHeight: 1.4 }}>{title}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {link.document_type ? <span className="badge">{link.document_type}</span> : null}
                {typeof link.score === 'number' ? <span className="badge">score={link.score.toFixed(2)}</span> : null}
              </div>
            </div>

            <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.5 }}>
              {`pattern: ${link.pattern_key} | query: ${link.query_used}`}
            </div>

            {entities.length > 0 ? (
              <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                {`graph entities: ${entities.join(', ')}`}
              </div>
            ) : null}

            {feedback ? (
              <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                {feedback}
              </div>
            ) : null}

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              <button
                type="button"
                aria-pressed={link.doc_feedback === 'helpful'}
                disabled={!feedbackEnabled || pending}
                onClick={() => void handleFeedback(link, index, 'helpful')}
                style={{
                  background: link.doc_feedback === 'helpful' ? 'rgba(123, 212, 141, 0.18)' : 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 999,
                  color: 'inherit',
                  cursor: !feedbackEnabled || pending ? 'not-allowed' : 'pointer',
                  fontSize: 12,
                  padding: '5px 10px',
                }}
                title={feedbackEnabled ? 'Mark this document link as helpful' : 'Document feedback unavailable for this row'}
              >
                {pending && pendingState?.feedback === 'helpful' ? 'Saving…' : 'Helpful'}
              </button>
              <button
                type="button"
                aria-pressed={link.doc_feedback === 'not_helpful'}
                disabled={!feedbackEnabled || pending}
                onClick={() => void handleFeedback(link, index, 'not_helpful')}
                style={{
                  background: link.doc_feedback === 'not_helpful' ? 'rgba(247, 118, 142, 0.18)' : 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 999,
                  color: 'inherit',
                  cursor: !feedbackEnabled || pending ? 'not-allowed' : 'pointer',
                  fontSize: 12,
                  padding: '5px 10px',
                }}
                title={feedbackEnabled ? 'Mark this document link as not helpful' : 'Document feedback unavailable for this row'}
              >
                {pending && pendingState?.feedback === 'not_helpful' ? 'Saving…' : 'Not helpful'}
              </button>
            </div>

            {excerpt ? (
              <div className="small" style={{ lineHeight: 1.55, marginTop: 8 }}>
                {excerpt}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}
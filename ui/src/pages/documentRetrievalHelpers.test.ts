import { describe, expect, it } from 'vitest'

import {
  buildChatPrompt,
  buildSearchStatusMessage,
  buildTwinEvidenceSummary,
  buildWeakMatchMessage,
  filterGroundedMatches,
  type DocsMatch,
} from './documentRetrievalHelpers'

describe('documentRetrievalHelpers', () => {
  it('builds twin evidence summaries from graph support, entities, and feedback', () => {
    const summary = buildTwinEvidenceSummary({
      graph_support: 3,
      helpful_count: 4,
      not_helpful_count: 1,
      evidence_entities: [
        { name: 'Spindle', type: 'component' },
        { id: 'ALM-21', type: 'alarm' },
      ],
    })

    expect(summary).toBe(
      'graph support 3 | entities Spindle (component), ALM-21 (alarm) | feedback helpful 4, not helpful 1',
    )
  })

  it('builds a chat prompt with twin evidence and only the first five sources', () => {
    const matches: DocsMatch[] = Array.from({ length: 6 }, (_, index) => ({
      citation: `Manual ${index + 1}`,
      text: `Excerpt ${index + 1}`,
      graph_support: index + 1,
      evidence_entities: [{ name: `Entity ${index + 1}`, type: 'concept' }],
    }))

    const prompt = buildChatPrompt('What is the warm-up sequence?', matches)

    expect(prompt).toContain('Question: What is the warm-up sequence?')
    expect(prompt).toContain('[1] Manual 1')
    expect(prompt).toContain('Twin evidence: graph support 1 | entities Entity 1 (concept)')
    expect(prompt).toContain('[5] Manual 5')
    expect(prompt).not.toContain('[6] Manual 6')
  })

  it('reports weak match suppression and weak grounding guidance', () => {
    const rawMatches: DocsMatch[] = [
      { citation: 'Manual A', score: 0.11, text: 'A' },
      { citation: 'Manual B', score: 0.17, text: 'B' },
    ]

    const groundedMatches = filterGroundedMatches(rawMatches)
    const status = buildSearchStatusMessage(rawMatches, groundedMatches, 'retriever backend note')
    const message = buildWeakMatchMessage('spindle warm-up', rawMatches)

    expect(groundedMatches).toHaveLength(0)
    expect(status).toContain('Suppressed 2 weak matches below 0.18.')
    expect(status).toContain('Best score 0.17.')
    expect(status).toContain('retriever backend note')
    expect(message).toContain('spindle warm-up')
    expect(message).toContain('Best score: 0.17.')
  })
})
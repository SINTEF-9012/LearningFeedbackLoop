/**
 * MemoryGraphLink — a small "View in memory graph" link that opens the Neo4j
 * Browser with a Cypher query for the given memory pre-loaded. Renders nothing
 * until the Neo4j URL is known (or if there is no memory id), so it degrades
 * silently when the graph backend URL is unavailable.
 */
import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'

import { getRuntimeUrls } from '../api/config'
import { memoryGraphUrl, neo4jBrowserBase } from '../utils/graphLink'

interface Props {
  memoryId?: string | null
  className?: string
  style?: CSSProperties
  label?: string
}

export function MemoryGraphLink({ memoryId, className, style, label = 'View in memory graph' }: Props) {
  const [base, setBase] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getRuntimeUrls()
      .then((urls) => { if (alive) setBase(neo4jBrowserBase(urls.neo4j)) })
      .catch(() => { if (alive) setBase(null) })
    return () => { alive = false }
  }, [])

  if (!memoryId || !base) return null

  return (
    <a
      href={memoryGraphUrl(base, memoryId)}
      target="_blank"
      rel="noreferrer"
      className={className}
      style={style}
      title="Open this event's neighbourhood (patterns, feedback, context) in the Neo4j graph browser"
    >
      🔎 {label} ↗
    </a>
  )
}

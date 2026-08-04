/**
 * GraphQueryLink — a small "view in the graph" link that opens the Neo4j Browser
 * with an arbitrary Cypher query pre-loaded (e.g. the patterns behind a
 * capability, or the memories behind a batch proposal). Renders nothing until
 * the Neo4j URL is known, so it degrades silently when the graph backend is
 * unavailable. Generic sibling of MemoryGraphLink.
 */
import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'

import { getRuntimeUrls } from '../api/config'
import { graphBrowserUrl, neo4jBrowserBase } from '../utils/graphLink'

interface Props {
  query: string
  label?: string
  className?: string
  style?: CSSProperties
}

export function GraphQueryLink({ query, label = 'View in memory graph', className, style }: Props) {
  const [base, setBase] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getRuntimeUrls()
      .then((urls) => { if (alive) setBase(neo4jBrowserBase(urls.neo4j)) })
      .catch(() => { if (alive) setBase(null) })
    return () => { alive = false }
  }, [])

  if (!base || !query) return null

  return (
    <a
      href={graphBrowserUrl(base, query)}
      target="_blank"
      rel="noreferrer"
      className={className}
      style={style}
      title="Open these elements in the Neo4j graph browser"
    >
      🔎 {label} ↗
    </a>
  )
}

export default GraphQueryLink

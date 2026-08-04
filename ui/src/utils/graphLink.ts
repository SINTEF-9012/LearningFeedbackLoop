/**
 * Deep-links into the Neo4j Browser for the memory graph.
 *
 * The runtime config exposes the Neo4j *bolt* URL (e.g. bolt://host:7687); the
 * Browser is served over HTTP on :7474. We derive that, then build a Browser
 * deep-link that pre-loads a Cypher query into the editor (cmd=edit) so a click
 * on an alert lands on that event's neighbourhood in the graph.
 */

/** bolt://host:7687 (or neo4j://…) → http://host:7474, or null if unparseable. */
export function neo4jBrowserBase(neo4j?: string | null): string | null {
  if (!neo4j) return null
  const match = String(neo4j).match(/^[a-zA-Z+]+:\/\/([^:/]+)(?::\d+)?/)
  const host = match?.[1]
  if (!host) return null
  return `http://${host}:7474`
}

/** Strip characters that would break a single-quoted Cypher string literal. */
function sanitizeId(id: string): string {
  return String(id).replace(/['"\\]/g, '')
}

/**
 * Cypher that returns the event's *relevant* neighbourhood: its patterns, the
 * feedback about it, and similar events (memories sharing a pattern). We do NOT
 * use a blind 1–2 hop expansion — that traverses the Session hub and floods the
 * result with every sibling memory in the session (and truncates out the
 * patterns). Each branch is targeted and the similar-events branch is bounded.
 */
export function memoryGraphQuery(memoryId: string): string {
  const id = sanitizeId(memoryId)
  return [
    `MATCH path=(m:Memory {id:'${id}'})-[:HAS_PATTERN]->(:Pattern) RETURN path`,
    `UNION MATCH path=(:Feedback)-[:ABOUT]->(:Memory {id:'${id}'}) RETURN path`,
    `UNION MATCH path=(:Memory {id:'${id}'})-[:HAS_PATTERN]->(:Pattern)<-[:HAS_PATTERN]-(sim:Memory) `
      + `WHERE sim.id <> '${id}' RETURN path LIMIT 60`,
  ].join(' ')
}

/** Generic Neo4j Browser URL with an arbitrary Cypher query pre-loaded. */
export function graphBrowserUrl(base: string, cypher: string): string {
  return `${base}/browser/?cmd=edit&arg=${encodeURIComponent(cypher)}`
}

/** Full Neo4j Browser URL with the memory query pre-loaded in the editor. */
export function memoryGraphUrl(base: string, memoryId: string): string {
  return graphBrowserUrl(base, memoryGraphQuery(memoryId))
}

/** Cypher for a set of Pattern keys and their 1–2 hop neighbourhood (the
 *  memories and feedback that reference them) — used to trace an aggregate
 *  (capability / fleet prior) back to the graph elements behind it. */
export function patternsGraphQuery(keys: string[]): string {
  const list = keys.filter(Boolean).map(sanitizeId).slice(0, 25)
  if (list.length === 0) return 'MATCH (p:Pattern) RETURN p LIMIT 50'
  const inList = list.map((k) => `'${k}'`).join(', ')
  // Patterns + co-occurring patterns + the feedback on them (the learned
  // knowledge), plus a bounded sample of the memories that carry them — without
  // a blind expansion that would drown the view in memories.
  return [
    `MATCH path=(p:Pattern)-[:CO_OCCURS_WITH]-(:Pattern) WHERE p.key IN [${inList}] RETURN path`,
    `UNION MATCH path=(:Feedback)-[:ON_PATTERN]->(p:Pattern) WHERE p.key IN [${inList}] RETURN path LIMIT 80`,
    `UNION MATCH path=(m:Memory)-[:HAS_PATTERN]->(p:Pattern) WHERE p.key IN [${inList}] RETURN path LIMIT 60`,
  ].join(' ')
}

/** Cypher for a set of Memory ids — the exact events behind a batch proposal /
 *  evidence record, with their patterns and feedback (no Session-hub flood). */
export function memoriesGraphQuery(ids: string[]): string {
  const list = ids.filter(Boolean).map(sanitizeId).slice(0, 25)
  if (list.length === 0) return 'MATCH (m:Memory) RETURN m LIMIT 50'
  const inList = list.map((i) => `'${i}'`).join(', ')
  return [
    `MATCH path=(m:Memory)-[:HAS_PATTERN]->(:Pattern) WHERE m.id IN [${inList}] RETURN path`,
    `UNION MATCH path=(:Feedback)-[:ABOUT]->(m:Memory) WHERE m.id IN [${inList}] RETURN path LIMIT 100`,
  ].join(' ')
}

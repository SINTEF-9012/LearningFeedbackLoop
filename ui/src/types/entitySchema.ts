/**
 * EntitySchema — TypeScript mirror of
 * ``backend/agents/schema/entity_schema.py``.
 *
 * Kept intentionally minimal so adding a new field server-side never
 * needs a coordinated UI change: unknown keys in `fields` or
 * `metrics` are accepted verbatim.
 */

export type EntityKind =
  | 'memory'
  | 'feedback'
  | 'pattern'
  | 'model'
  | 'session'
  | 'experiment'
  | 'sindit_asset'
  | 'knowledge_pack'

export interface EntityRelationship {
  kind: string
  id: string
  role: string
}

export interface EntitySchema {
  kind: EntityKind
  id: string
  label?: string | null
  fields: Record<string, unknown>
  tags: string[]
  metrics: Record<string, number>
  relationships: EntityRelationship[]
}

/** Accepts an unknown object and coerces it into an EntitySchema. */
export function toEntitySchema(raw: unknown): EntitySchema | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const kind = typeof r.kind === 'string' ? (r.kind as EntityKind) : null
  const id = typeof r.id === 'string' ? r.id : null
  if (!kind || !id) return null
  const fields =
    r.fields && typeof r.fields === 'object'
      ? (r.fields as Record<string, unknown>)
      : {}
  const tags = Array.isArray(r.tags) ? (r.tags as unknown[]).filter((t): t is string => typeof t === 'string') : []
  const metricsRaw =
    r.metrics && typeof r.metrics === 'object'
      ? (r.metrics as Record<string, unknown>)
      : {}
  const metrics: Record<string, number> = {}
  for (const [k, v] of Object.entries(metricsRaw)) {
    if (typeof v === 'number' && Number.isFinite(v)) metrics[k] = v
  }
  const relsRaw = Array.isArray(r.relationships) ? r.relationships : []
  const relationships: EntityRelationship[] = []
  for (const rr of relsRaw) {
    if (rr && typeof rr === 'object') {
      const o = rr as Record<string, unknown>
      if (typeof o.kind === 'string' && typeof o.id === 'string' && typeof o.role === 'string') {
        relationships.push({ kind: o.kind, id: o.id, role: o.role })
      }
    }
  }
  return {
    kind,
    id,
    label: typeof r.label === 'string' ? r.label : null,
    fields,
    tags,
    metrics,
    relationships,
  }
}

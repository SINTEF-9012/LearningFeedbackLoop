/**
 * EntityView — Agent K Phase 2 (2026-04-24).
 *
 * Generic schema-driven detail view. Consumes the Agent K Phase 1
 * `EntitySchema` contract. Adding a new `fields` or `metrics` key
 * server-side shows up here automatically — no code changes required.
 */

import type { CSSProperties } from 'react'
import type { EntitySchema } from '../types/entitySchema'
import { colors, fontSize, radii, spacing } from '../styles/tokens'

export interface EntityViewProps {
  schema: EntitySchema
  /** Called when the user clicks a relationship; receives target. */
  onRelationshipClick?: (rel: { kind: string; id: string; role: string }) => void
  /** Compact mode: smaller spacing, no headers on empty sections. */
  compact?: boolean
}

const panel: CSSProperties = {
  background: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  color: colors.text,
  fontSize: fontSize.sm,
  padding: spacing.md,
}

const headerStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'baseline',
  justifyContent: 'space-between',
  gap: spacing.sm,
  marginBottom: spacing.sm,
}

const kindBadge: CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.textMuted,
  fontSize: fontSize.xs,
  padding: '2px 6px',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const sectionHeader: CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.xs,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  margin: `${spacing.sm}px 0 ${spacing.xs}px`,
}

const rowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'minmax(100px, 30%) 1fr',
  gap: spacing.sm,
  padding: `2px 0`,
}

const keyStyle: CSSProperties = { color: colors.textDim }
const valueStyle: CSSProperties = {
  color: colors.text,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
}

const tagStyle: CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.text,
  display: 'inline-block',
  fontSize: fontSize.xs,
  margin: `2px 4px 2px 0`,
  padding: '2px 6px',
}

const metricStyle: CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.text,
  display: 'inline-flex',
  flexDirection: 'column',
  fontSize: fontSize.xs,
  margin: '2px 6px 2px 0',
  minWidth: 80,
  padding: '4px 8px',
}

const relStyle: CSSProperties = {
  background: 'transparent',
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.accent,
  cursor: 'pointer',
  fontSize: fontSize.xs,
  margin: '2px 4px 2px 0',
  padding: '2px 6px',
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value)
    // Keep short integers as-is; round floats to 4 sig figs.
    if (Number.isInteger(value)) return String(value)
    return value.toPrecision(4)
  }
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function formatMetric(v: number): string {
  if (!Number.isFinite(v)) return String(v)
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v)
  if (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-2)) {
    return v.toExponential(3)
  }
  return v.toPrecision(4)
}

export function EntityView({ schema, onRelationshipClick, compact }: EntityViewProps) {
  const showEmpty = !compact
  const fieldEntries = Object.entries(schema.fields)
  const metricEntries = Object.entries(schema.metrics)
  const panelStyle: CSSProperties = compact
    ? { ...panel, padding: spacing.sm }
    : panel

  return (
    <div style={panelStyle} data-testid={`entity-view-${schema.kind}-${schema.id}`}>
      <div style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: spacing.sm, minWidth: 0 }}>
          <strong style={{ color: colors.text, fontSize: fontSize.md, ...valueStyle }}>
            {schema.label || schema.id}
          </strong>
          <span style={kindBadge}>{schema.kind.replace('_', ' ')}</span>
        </div>
        <span style={{ color: colors.textDim, fontSize: fontSize.xs }}>#{schema.id}</span>
      </div>

      {fieldEntries.length > 0 || showEmpty ? (
        <>
          <div style={sectionHeader}>Fields</div>
          {fieldEntries.length === 0 ? (
            <div style={{ color: colors.textDim, fontSize: fontSize.xs }}>None</div>
          ) : (
            fieldEntries.map(([k, v]) => (
              <div key={k} style={rowStyle}>
                <span style={keyStyle}>{k}</span>
                <span style={valueStyle} title={formatValue(v)}>
                  {formatValue(v)}
                </span>
              </div>
            ))
          )}
        </>
      ) : null}

      {schema.tags.length > 0 || showEmpty ? (
        <>
          <div style={sectionHeader}>Tags</div>
          {schema.tags.length === 0 ? (
            <div style={{ color: colors.textDim, fontSize: fontSize.xs }}>None</div>
          ) : (
            <div>
              {schema.tags.map((t) => (
                <span key={t} style={tagStyle}>
                  {t}
                </span>
              ))}
            </div>
          )}
        </>
      ) : null}

      {metricEntries.length > 0 || showEmpty ? (
        <>
          <div style={sectionHeader}>Metrics</div>
          {metricEntries.length === 0 ? (
            <div style={{ color: colors.textDim, fontSize: fontSize.xs }}>None</div>
          ) : (
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {metricEntries.map(([k, v]) => (
                <div key={k} style={metricStyle}>
                  <span style={{ color: colors.textDim }}>{k}</span>
                  <strong>{formatMetric(v)}</strong>
                </div>
              ))}
            </div>
          )}
        </>
      ) : null}

      {schema.relationships.length > 0 || showEmpty ? (
        <>
          <div style={sectionHeader}>Relationships</div>
          {schema.relationships.length === 0 ? (
            <div style={{ color: colors.textDim, fontSize: fontSize.xs }}>None</div>
          ) : (
            <div>
              {schema.relationships.map((r) => (
                <button
                  key={`${r.kind}:${r.id}:${r.role}`}
                  type="button"
                  style={relStyle}
                  onClick={() => onRelationshipClick?.(r)}
                  title={`${r.role} → ${r.kind}#${r.id}`}
                >
                  {r.role} · {r.kind}#{r.id}
                </button>
              ))}
            </div>
          )}
        </>
      ) : null}
    </div>
  )
}

export default EntityView

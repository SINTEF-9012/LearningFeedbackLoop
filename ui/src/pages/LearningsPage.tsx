/**
 * LearningsPage — operator feedback, tool learnings, and knowledge export.
 */

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, wsUrl } from '../api/http'
import { getRuntimeUrls } from '../api/config'
import { connectWs } from '../api/ws'
import { toEntitySchema, type EntitySchema } from '../types/entitySchema'
import { EntityView } from '../components/EntityView'
import { GraphQueryLink } from '../components/GraphQueryLink'
import { patternsGraphQuery } from '../utils/graphLink'
import { PriorsChart } from '../components/PriorsChart'
import { useAppContext } from '../contexts/AppContext'
import { useLiveScoreStore } from '../state/liveScoreStore'
import { colors, fontSize, radii, shadows, spacing } from '../styles/tokens'
import { humanPattern } from '../utils/patternNames'

type SinkName = 'file' | 'mqtt'

interface KnowledgePushResponse {
  site: string
  summary: Record<string, number>
  sinks: Record<string, boolean>
}

interface FeedbackOperatorSummary {
  operator_id: string
  total: number
  actions: Record<string, number>
}

interface FeedbackOperatorSummaryResponse {
  operators: FeedbackOperatorSummary[]
}

interface FeedbackOutboxEvent {
  memory_id: string
  action: string
  operator_id: string
  created_at: string
  sequence: number
  data: Record<string, unknown>
}

interface FeedbackOutboxResponse {
  pending: number
  head: FeedbackOutboxEvent[]
}

interface ModelTrustScope {
  context: string
  model_confidence: number
  confirmed: number
  dismissed: number
  evidence_count: number
}

interface ModelTrust {
  model_confidence?: number
  smoothed_precision?: number
  evidence_count?: number
  true_positives?: number
  false_positives?: number
  scopes?: ModelTrustScope[]
  scope_count?: number
}

interface LoopMetricsResponse {
  model_trust?: ModelTrust
}

interface CounterfactualArm {
  alerts: number
  false_alerts: number
  true_alerts: number
  broken_episodes_alerted: number
  broken_episodes_total: number
  healthy_episodes_alerting: number
}

interface CounterfactualResponse {
  available: boolean
  off?: CounterfactualArm
  on?: CounterfactualArm
  burden_reduction?: number
  false_alarm_reduction?: number
  coverage_preserved?: boolean
  measured?: boolean
}

interface MaasEvidenceRecord {
  plant_id?: string
  supplier_id?: string
  capability?: string
  declared?: boolean
  confirmed?: number
  dismissed?: number
  confirm_rate?: number
  confidence?: number
  co2_avoided_kg_per_confirmed_catch?: number
  co2_avoided_kg_total?: number
  realised_energy_kwh_per_good_part?: number
  realised_co2_kg_per_good_part?: number
  dpp_source?: string
  lead_time_s_median?: number | null
  window?: string
  context?: { machine_family?: string; tool_type?: string; material?: string }
}

interface MaasEvidenceResponse {
  available: boolean
  records?: MaasEvidenceRecord[]
  count?: number
  illustrative?: boolean
}

interface FaultEntry { fault: string; confirmed: number; dismissed: number; lead_time_s_median: number | null }
interface FaultRecord {
  plant_id?: string
  capability?: string
  faults?: FaultEntry[]
  window?: string
  confidence?: number
  context?: { machine_family?: string; tool_type?: string; material?: string }
}

interface AvailabilityRecord {
  plant_id?: string
  declared_availability_pct?: number | null
  confirmed_stoppages?: number
  operating_hours?: number
  mean_hours_between_stoppages?: number | null
  availability_adjustment_pct?: number | null
  window?: string
  confidence?: number
  context?: { machine_family?: string; tool_type?: string; material?: string }
}

interface SustainabilityRecord {
  plant_id?: string
  declared_energy_kwh_per_good_part?: number | null
  realised_energy_kwh_per_good_part?: number | null
  co2_factor_kg_per_kwh?: number | null
  co2_kg_per_good_part?: number | null
  co2_avoided_kg_per_confirmed_catch?: number | null
  co2_avoided_kg_total?: number | null
  confirmed_catches?: number
  scrap_rate?: number | null
  good_parts?: number | null
  window?: string
  confidence?: number
}

interface FacetSummary<T> { records?: T[]; count?: number; source?: string }
interface MaasFacetsResponse {
  facets?: {
    capability?: FacetSummary<MaasEvidenceRecord> | null
    fault?: FacetSummary<FaultRecord> | null
    availability?: FacetSummary<AvailabilityRecord> | null
    sustainability?: FacetSummary<SustainabilityRecord> | null
  }
}

interface ToolAuditAnomalyStats {
  scored_count?: number
  significant_count?: number
  alerted_count?: number
  confirmed_count?: number
  dismissed_count?: number
  last_score?: number | null
  last_memory_id?: string | null
  last_patterns?: string[]
  last_feedback_patterns?: string[]
  last_feedback_action?: string | null
  last_operator_id?: string | null
  last_event_at?: string | null
  last_feedback_at?: string | null
  pattern_counts?: Record<string, number>
}

interface ToolAuditRuntime {
  session_ids?: string[]
  machine_ids?: string[]
  tool_id?: string | null
  tool_uri?: string | null
  seen_count?: number
  first_seen_at?: string | null
  last_seen_at?: string | null
  effective_ctx?: Record<string, unknown>
  anomaly_stats?: ToolAuditAnomalyStats
}

interface ToolAuditRow {
  machine_family: string
  tool_number: number
  flags: string[]
  harmonic_ready: boolean
  runtime?: ToolAuditRuntime | null
  master?: {
    description?: string | null
    tool_type?: string | null
  } | null
}

interface ToolAuditListResponse {
  sindit_available: boolean
  total: number
  items: ToolAuditRow[]
  detail?: string
}

interface LearningsConfigResponse {
  mqtt_enabled: boolean
  mqtt_configured: boolean
  mqtt_transport_available: boolean
  mqtt_forwarding_active: boolean
  mqtt_state: string
  mqtt_topic?: string | null
  mqtt_broker_host: string
  mqtt_broker_port: number
  mqtt_qos: number
  published_count: number
  last_published_at?: number | null
  last_error?: string | null
}

interface LearningEnvelope {
  kind: string
  session_id?: string
  source?: string
  ts_unix?: number
  payload?: Record<string, unknown>
}

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

const controlBar: React.CSSProperties = {
  alignItems: 'center',
  background: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  display: 'flex',
  flexWrap: 'wrap',
  gap: spacing.md,
  marginBottom: spacing.lg,
  padding: spacing.md,
}

const labelStyle: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.xs,
  marginRight: spacing.xs,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const inputStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.sm,
  color: colors.text,
  fontSize: fontSize.sm,
  padding: '4px 8px',
  minWidth: 120,
}

const btnStyle = (enabled: boolean): React.CSSProperties => ({
  background: enabled ? colors.accent : colors.surfaceAlt,
  border: `1px solid ${enabled ? colors.accent : colors.border}`,
  borderRadius: radii.sm,
  color: enabled ? '#fff' : colors.textDim,
  cursor: enabled ? 'pointer' : 'not-allowed',
  fontSize: fontSize.sm,
  fontWeight: 500,
  padding: '6px 12px',
})

const statusStyle: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.sm,
  padding: spacing.sm,
}

const sectionTitleStyle: React.CSSProperties = {
  color: colors.text,
  fontSize: fontSize.lg,
  fontWeight: 600,
  margin: 0,
}

const panelStyle: React.CSSProperties = {
  background: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.lg,
  boxShadow: shadows.panel,
  padding: spacing.lg,
}

const gridStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.lg,
  gridTemplateColumns: 'minmax(0, 1.15fr) minmax(320px, 0.85fr)',
  alignItems: 'start',
}

const summaryGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.md,
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
}

const summaryCardStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  padding: spacing.md,
}

// Map a capability to the fault-pattern keys it monitors, so its evidence can
// deep-link to the actual Pattern (and the memories/feedback around them) in
// the graph. Empty → the link falls back to a broad Pattern query.
function capabilityPatternKeys(capability?: string): string[] {
  const c = (capability || '').toLowerCase()
  if (c.includes('tool') && c.includes('wear')) return ['TOOL_WEAR_RISK', 'BREAKAGE_RISK_HIGH']
  if (c.includes('vibration') || c.includes('chatter')) return ['ANOMALY_HIGH_VIBRATION', 'VIB_SEVERITY_HIGH']
  if (c.includes('breakage')) return ['BREAKAGE_RISK_HIGH']
  if (c.includes('spindle') || c.includes('power') || c.includes('overload')) return ['POWER_SPIKE_SUSTAINED']
  return []
}

// Small pill that tags a claim as measured / modeled / illustrative.
const claimBadge = (color: string): React.CSSProperties => ({
  border: `1px solid ${color}`,
  borderRadius: 999,
  color,
  fontSize: fontSize.xs,
  padding: '2px 10px',
})

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: fontSize.sm,
}

const thStyle: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: fontSize.xs,
  fontWeight: 600,
  letterSpacing: '0.06em',
  padding: '0 0 10px 0',
  textAlign: 'left',
  textTransform: 'uppercase',
}

const tdStyle: React.CSSProperties = {
  borderTop: `1px solid ${colors.border}`,
  color: colors.text,
  padding: '10px 0',
  verticalAlign: 'top',
}

const feedListStyle: React.CSSProperties = {
  display: 'grid',
  gap: spacing.sm,
  maxHeight: 420,
  overflowY: 'auto',
}

const feedItemStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  padding: spacing.md,
}

type TabKey = 'gains' | 'fleet' | 'diagnostics'

const TAB_DEFS: { key: TabKey; label: string; hint: string }[] = [
  { key: 'fleet', label: 'MaaS evidence', hint: 'What propagates to the matchmaking platform' },
  { key: 'gains', label: 'Maintenance gains', hint: 'What operator feedback bought you' },
  { key: 'diagnostics', label: 'Diagnostics', hint: 'Model trust, priors & live bus' },
]

const tabBarStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: spacing.xs,
  borderBottom: `1px solid ${colors.border}`,
  marginBottom: spacing.lg,
}

const tabButtonStyle = (active: boolean): React.CSSProperties => ({
  background: 'transparent',
  border: 'none',
  borderBottom: `2px solid ${active ? colors.accent : 'transparent'}`,
  color: active ? colors.text : colors.textMuted,
  cursor: 'pointer',
  fontSize: fontSize.md,
  fontWeight: active ? 600 : 500,
  padding: '10px 16px',
  marginBottom: -1,
})

const heroCardStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  padding: spacing.lg,
  textAlign: 'center',
}

const loopStepStyle: React.CSSProperties = {
  background: colors.surfaceAlt,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.md,
  color: colors.text,
  flex: '1 1 130px',
  fontSize: fontSize.sm,
  fontWeight: 500,
  padding: spacing.md,
  textAlign: 'center',
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function formatAgo(isoOrSeconds?: string | number | null): string {
  if (typeof isoOrSeconds === 'number' && Number.isFinite(isoOrSeconds)) {
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - isoOrSeconds))
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    return `${Math.floor(seconds / 3600)}h ago`
  }
  if (typeof isoOrSeconds !== 'string' || !isoOrSeconds) return 'never'
  const deltaSeconds = Math.max(0, Math.round((Date.now() - new Date(isoOrSeconds).getTime()) / 1000))
  if (deltaSeconds < 60) return `${deltaSeconds}s ago`
  if (deltaSeconds < 3600) return `${Math.floor(deltaSeconds / 60)}m ago`
  return `${Math.floor(deltaSeconds / 3600)}h ago`
}

function compactNumber(value: unknown): string {
  const numeric = numberOrZero(value)
  return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(2)
}

function sortTopPatterns(patternCounts?: Record<string, number>): string[] {
  return Object.entries(patternCounts || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([pattern]) => humanPattern(pattern))
}

function toolRowKey(row: Pick<ToolAuditRow, 'machine_family' | 'tool_number'>): string {
  return `${row.machine_family}:${row.tool_number}`
}

function numericToolNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function LearningKindPill({ kind }: { kind: string }) {
  const tone = kind === 'tool_event'
    ? colors.accent
    : kind === 'feedback_event'
      ? colors.good
      : kind === 'scored_event'
        ? colors.warn
        : colors.textMuted
  return (
    <span
      style={{
        border: `1px solid ${tone}`,
        borderRadius: 999,
        color: tone,
        display: 'inline-flex',
        fontSize: fontSize.xs,
        fontWeight: 600,
        letterSpacing: '0.04em',
        padding: '2px 8px',
        textTransform: 'uppercase',
      }}
    >
      {kind.replace(/_/g, ' ')}
    </span>
  )
}

function renderLearningSummary(item: LearningEnvelope): string {
  const payload = item.payload || {}
  if (item.kind === 'tool_event') {
    const toolNumber = payload.tool_number
    const action = typeof payload.action === 'string' ? payload.action : 'updated'
    const patterns = Array.isArray(payload.patterns) ? payload.patterns.slice(0, 2).map((entry) => humanPattern(String(entry))).join(', ') : ''
    return `Tool T${toolNumber ?? '?'} ${action}${patterns ? ` · ${patterns}` : ''}`
  }
  if (item.kind === 'feedback_event') {
    return `Feedback ${String(payload.action || 'event')} · ${String(payload.operator_id || 'operator')}`
  }
  if (item.kind === 'scored_event') {
    const significance = payload.significance && typeof payload.significance === 'object'
      ? payload.significance as Record<string, unknown>
      : {}
    return `Scored event · score ${compactNumber(significance.score)}`
  }
  if (item.kind === 'insight_event') {
    return `Insight · ${String(payload.alert_line || payload.explanation || 'explanation')}`
  }
  return item.kind
}

function packToEntity(resp: KnowledgePushResponse): EntitySchema | null {
  // Hand-roll a knowledge_pack schema that matches the Python adapter.
  const metrics: Record<string, number> = {}
  for (const [k, v] of Object.entries(resp.summary)) {
    if (typeof v === 'number' && Number.isFinite(v)) metrics[k] = v
  }
  const fields: Record<string, unknown> = {
    site: resp.site,
    sinks_succeeded: Object.entries(resp.sinks).filter(([, ok]) => ok).map(([n]) => n).join(', ') || '—',
    sinks_failed: Object.entries(resp.sinks).filter(([, ok]) => !ok).map(([n]) => n).join(', ') || '—',
  }
  return toEntitySchema({
    kind: 'knowledge_pack',
    id: resp.site,
    label: `Knowledge pack · ${resp.site}`,
    fields,
    tags: [],
    metrics,
    relationships: [],
  })
}

// ── MaaS evidence objects ──────────────────────────────────────────────────
// The loop exposes local learning to the platform as structured evidence
// objects: aggregated, context-conditioned, confidence-scored summaries a
// matchmaking / quotation service can consume. Each object carries a purpose,
// an example platform use, what it is built from, and the fields it reports.
// Capability evidence additionally renders live records from operator feedback.
interface EvidenceObject {
  key: string
  title: string
  purpose: string
  use: string
  builtFrom: string
  fields: [string, string][]
}

const MAAS_EVIDENCE_OBJECTS: EvidenceObject[] = [
  {
    key: 'capability',
    title: 'Capability evidence',
    purpose: 'Whether a plant has demonstrated a capability in a specific machine, material and operation context.',
    use: 'Matchmaking and partner selection — a supplier is described not only by what it declares, but by what has been observed and validated in prior production.',
    builtFrom: 'Operator confirm/dismiss feedback, aggregated per context and shrunk toward a prior so a few uncertain reviews carry less weight.',
    fields: [
      ['capability', 'the demonstrated capability, e.g. vibration control'],
      ['context', 'machine family · tool type · material'],
      ['confirmations / dismissals', 'operator review counts in the window'],
      ['confidence', 'volume-shrunk toward the prior'],
    ],
  },
  {
    key: 'fault',
    title: 'Fault & lead-time evidence',
    purpose: 'Confirmed fault patterns and their observed effect on response or lead time.',
    use: 'Quotation risk assessment and route ranking — confirmed tool-breakage or chatter history makes fault-related risk visible for a machine, material or operation.',
    builtFrom: 'Confirmed fault feedback and the observed timing between detection and operator response.',
    fields: [
      ['faults', 'confirmed fault types with confirm / dismiss counts'],
      ['lead_time', 'observed time from detection to response'],
      ['context', 'machine · material · operation'],
      ['confidence', 'scaled by evidence volume'],
    ],
  },
  {
    key: 'availability',
    title: 'Availability-adjustment evidence',
    purpose: 'Operational evidence that modifies declared availability from observed behaviour.',
    use: 'Availability calculation and production planning — observed stoppages can reduce the effective availability used during quotation.',
    builtFrom: 'Stoppage detection accumulated over operating hours.',
    fields: [
      ['declared_availability', 'the supplier-declared figure'],
      ['confirmed_stoppages', 'count over operating hours'],
      ['availability_adjustment', 'signed correction from observation'],
      ['confidence', 'scaled by evidence volume'],
    ],
  },
  {
    key: 'sustainability',
    title: 'Realised-sustainability evidence',
    purpose: 'Compares declared and realised energy, CO₂, scrap or quality per good part.',
    use: 'Footprint estimation and KPI monitoring — realised energy and scrap refine sustainability and cost estimates.',
    builtFrom: 'Per-part energy and good/scrap counts, which come from MES, ERP or inspection systems outside the loop; the loop defines the target interface.',
    fields: [
      ['energy_per_good_part', 'realised against declared'],
      ['co2_per_good_part', 'derived footprint'],
      ['scrap_rate', 'from inspection counts'],
      ['confidence', 'scaled by evidence volume'],
    ],
  },
]

// Fields every evidence object carries, regardless of facet — the common
// interface a platform service can rely on. Shown on request.
const EVIDENCE_INTERFACE_FIELDS: [string, string][] = [
  ['identifier', 'plant, supplier, machine or asset'],
  ['context', 'material, operation, tool, machine family or production order'],
  ['indicator', 'the capability, fault, KPI or availability being reported'],
  ['evidence counts', 'confirmations, dismissals, stoppages, good or scrapped parts'],
  ['window', 'the period over which the evidence was collected'],
  ['confidence', 'so uncertain, thin evidence carries less weight'],
  ['provenance', 'links back to the memories, feedback and twin entities behind it'],
  ['validation status', 'candidate, active, validated, deprecated or archived'],
]

function FieldGrid({ fields }: { fields: [string, string][] }) {
  return (
    <div style={{ display: 'grid', gap: spacing.sm, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
      {fields.map(([k, v], i) => (
        <div key={k || i} style={summaryCardStyle}>
          <div style={{ color: colors.text, fontFamily: 'monospace', fontSize: fontSize.xs, fontWeight: 600 }}>{k}</div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>{v}</div>
        </div>
      ))}
    </div>
  )
}

function EvidenceStructure({ object }: { object: EvidenceObject }) {
  return (
    <details style={{ marginTop: spacing.md }}>
      <summary style={{ color: colors.textMuted, cursor: 'pointer', fontSize: fontSize.sm }}>Show structure</summary>
      <div style={{ marginTop: spacing.sm }}>
        <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginBottom: spacing.sm }}>
          Built from: {object.builtFrom}
        </div>
        <FieldGrid fields={object.fields} />
      </div>
    </details>
  )
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div>
      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ color: tone ?? colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>{value}</div>
      {sub ? <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>{sub}</div> : null}
    </div>
  )
}

const statGrid: React.CSSProperties = {
  display: 'grid',
  gap: spacing.sm,
  gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
  marginTop: spacing.md,
}

function contextLine(ctx?: { machine_family?: string; tool_type?: string; material?: string }, plant?: string): string {
  return [plant, ctx?.machine_family, ctx?.tool_type, ctx?.material].filter(Boolean).join(' · ')
}

function pending(): React.ReactElement {
  return <span style={{ color: colors.textDim }}>pending MES / stoppage log</span>
}

function FaultCard({ record }: { record: FaultRecord }) {
  return (
    <div style={{ ...summaryCardStyle, marginTop: spacing.md }}>
      <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{contextLine(record.context, record.plant_id)}</div>
      <div style={{ display: 'grid', gap: spacing.xs, marginTop: spacing.sm }}>
        {(record.faults ?? []).map((f) => {
          const n = (f.confirmed ?? 0) + (f.dismissed ?? 0)
          const rate = n > 0 ? Math.round((f.confirmed / n) * 100) : 0
          return (
            <div key={f.fault} style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', flexWrap: 'wrap' }}>
              <span style={{ color: colors.text, fontWeight: 600 }}>{humanPattern(f.fault)}</span>
              <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                <span style={{ color: colors.good }}>{f.confirmed}✓</span>{' / '}
                <span style={{ color: colors.bad }}>{f.dismissed}✗</span>{' · '}{rate}% confirmed{' · '}
                lead {typeof f.lead_time_s_median === 'number' ? `${f.lead_time_s_median}s` : 'not recorded'}
              </span>
            </div>
          )
        })}
      </div>
      <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginTop: spacing.sm }}>
        confidence {typeof record.confidence === 'number' ? record.confidence.toFixed(3) : '—'} (volume-shrunk) · window {record.window ?? '—'}
      </div>
    </div>
  )
}

function AvailabilityCard({ record }: { record: AvailabilityRecord }) {
  const adj = record.availability_adjustment_pct
  return (
    <div style={{ ...summaryCardStyle, marginTop: spacing.md }}>
      <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{contextLine(record.context, record.plant_id)}</div>
      <div style={statGrid}>
        <Stat label="Declared availability" value={typeof record.declared_availability_pct === 'number' ? `${record.declared_availability_pct}%` : '—'} sub="from catalogue" />
        <Stat
          label="Confirmed stoppages"
          value={String(record.confirmed_stoppages ?? 0)}
          sub={typeof record.operating_hours === 'number' && record.operating_hours > 0 ? `over ${Math.round(record.operating_hours)} op. hrs` : `window ${record.window ?? '—'}`}
        />
        {typeof record.mean_hours_between_stoppages === 'number' && (
          <Stat label="Mean hrs between stoppages" value={`${record.mean_hours_between_stoppages}`} sub="observed" />
        )}
        <div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Availability adjustment</div>
          <div style={{ color: typeof adj === 'number' && adj < 0 ? colors.warn : colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>
            {typeof adj === 'number' ? `${adj > 0 ? '+' : ''}${adj}%` : pending()}
          </div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>
            {typeof adj === 'number' ? 'applied to declared availability' : 'needs a confirmed-stoppage log'}
          </div>
        </div>
      </div>
    </div>
  )
}

function SustainabilityCard({ record }: { record: SustainabilityRecord }) {
  const num = (v?: number | null, unit = '') => (typeof v === 'number' ? `${Math.round(v)}${unit}` : null)
  return (
    <div style={{ ...summaryCardStyle, marginTop: spacing.md }}>
      <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{record.plant_id ?? ''}</div>
      <div style={statGrid}>
        <Stat
          label="CO₂ at stake (modeled)"
          value={num(record.co2_avoided_kg_total, ' kg') ? `~${num(record.co2_avoided_kg_total, ' kg')}` : '—'}
          sub={`~${num(record.co2_avoided_kg_per_confirmed_catch, ' kg') ?? '—'} / catch · ${record.confirmed_catches ?? 0} catches · not a realised saving`}
          tone={colors.warn}
        />
        <Stat
          label="Energy / good part"
          value={num(record.realised_energy_kwh_per_good_part ?? record.declared_energy_kwh_per_good_part, ' kWh') ?? '—'}
          sub={typeof record.declared_energy_kwh_per_good_part === 'number' ? `realised · declared ${Math.round(record.declared_energy_kwh_per_good_part)}` : 'realised'}
        />
        <Stat
          label="CO₂ / good part"
          value={num(record.co2_kg_per_good_part, ' kg') ?? '—'}
          sub={`factor ${record.co2_factor_kg_per_kwh ?? '—'} kg/kWh`}
        />
        {typeof record.good_parts === 'number' && (
          <Stat label="Good parts" value={String(record.good_parts)} sub={`window ${record.window ?? '—'}`} />
        )}
        <div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Scrap rate</div>
          <div style={{ color: colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>
            {typeof record.scrap_rate === 'number' ? `${(record.scrap_rate * 100).toFixed(1)}%` : pending()}
          </div>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>
            {typeof record.scrap_rate === 'number' ? 'from inspection' : 'needs inspection counts'}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LearningsPage() {
  const ctx = useAppContext()
  const [activeTab, setActiveTab] = useState<TabKey>('fleet')
  const [site, setSite] = useState('default')
  const [dataDir, setDataDir] = useState('data')
  const [fileSink, setFileSink] = useState(true)
  const [mqttSink, setMqttSink] = useState(false)
  const [busy, setBusy] = useState(false)
  const [lastResult, setLastResult] = useState<KnowledgePushResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [upstreamUrl, setUpstreamUrl] = useState<string | null>(null)
  const [toolScope, setToolScope] = useState<'current' | 'all'>('current')
  const [liveEvents, setLiveEvents] = useState<LearningEnvelope[]>([])
  const [selectedToolKey, setSelectedToolKey] = useState<string | null>(null)
  const liveScores = useLiveScoreStore((state) => state.points)
  const livePriorSnapshots = useLiveScoreStore((state) => state.priorSnapshots)

  useEffect(() => {
    // Fetch once; never blocks page render.
    getRuntimeUrls()
      .then((urls) => setUpstreamUrl(urls.upstream_knowledge || null))
      .catch(() => setUpstreamUrl(null))
  }, [])

  const toolQuery = useQuery({
    queryKey: ['learnings-tools', toolScope, ctx.streamSessionId],
    queryFn: () => {
      const query = toolScope === 'current' && ctx.streamSessionId
        ? `?session_id=${encodeURIComponent(ctx.streamSessionId)}`
        : ''
      return api<ToolAuditListResponse>(`/sindit/tools${query}`)
    },
    refetchInterval: 5000,
  })

  const feedbackOperatorsQuery = useQuery({
    queryKey: ['feedback-operators'],
    queryFn: () => api<FeedbackOperatorSummaryResponse>('/agent/memory/feedback/operators'),
    refetchInterval: 5000,
  })

  const feedbackOutboxQuery = useQuery({
    queryKey: ['feedback-outbox'],
    queryFn: () => api<FeedbackOutboxResponse>('/agent/memory/feedback/outbox?head=5'),
    refetchInterval: 5000,
  })

  const modelTrustQuery = useQuery({
    queryKey: ['loop-metrics-model-trust'],
    queryFn: () => api<LoopMetricsResponse>('/agent/memory/loop_metrics'),
    refetchInterval: 15000,
  })

  const counterfactualQuery = useQuery({
    queryKey: ['counterfactual'],
    queryFn: () => api<CounterfactualResponse>('/agent/memory/counterfactual'),
  })

  const maasEvidenceQuery = useQuery({
    queryKey: ['maas-evidence'],
    queryFn: () => api<MaasEvidenceResponse>('/agent/memory/maas-evidence'),
  })

  const maasFacetsQuery = useQuery({
    queryKey: ['maas-evidence-facets'],
    queryFn: () => api<MaasFacetsResponse>('/agent/memory/maas-evidence/facets'),
  })

  // Fleet transfer: the family-level aggregate of per-site learning for the demo
  // context. k-anonymity gates it (no aggregate below the site threshold).
  type FleetPackResponse = {
    context: Record<string, string | null>
    pack_count: number
    site_count: number
    k_anonymity_threshold: number
    k_anonymity_met: boolean
    source_sites: string[]
    pattern_priors: Record<string, { prior: number; site_count: number; evidence_count: number }>
    discovery_families: Array<Record<string, unknown>>
    notes: string[]
  }
  const FLEET_DEMO_CTX = { machine_type: 'gantry_mill', tool_type: 'face_mill', material: 'casting_steel', regime: 'roughing' }
  const fleetQuery = useQuery({
    queryKey: ['fleet-pack', FLEET_DEMO_CTX],
    queryFn: () => {
      const q = new URLSearchParams({ ...FLEET_DEMO_CTX, min_sites: '3' }).toString()
      return api<FleetPackResponse>(`/fleet/pack?${q}`)
    },
  })
  const [fleetBusy, setFleetBusy] = useState(false)
  const populateFleet = async () => {
    setFleetBusy(true)
    try {
      await api('/demo-director/seed-fleet', 'POST', { ...FLEET_DEMO_CTX, sites: 3 })
      await fleetQuery.refetch()
    } finally {
      setFleetBusy(false)
    }
  }

  // Scope: this operation (batch, default) vs the all-time total/aggregate below.
  const [learnScope, setLearnScope] = useState<'batch' | 'total'>('batch')
  const batchSummaryQuery = useQuery({
    queryKey: ['learnings-batch-summary', ctx.streamSessionId],
    queryFn: () => api<{ n_memories: number; total_confirmed: number; total_dismissed: number; n_contexts: number }>(
      `/reconfig/batch/${encodeURIComponent(ctx.streamSessionId)}/summary`,
    ),
    enabled: Boolean(ctx.streamSessionId),
    refetchInterval: 5000,
  })

  const learningsConfigQuery = useQuery({
    queryKey: ['config-learnings'],
    queryFn: () => api<LearningsConfigResponse>('/config/learnings'),
    refetchInterval: 3000,
  })

  useEffect(() => {
    setLiveEvents([])
    const path = toolScope === 'current' && ctx.streamSessionId
      ? `/learnings/ws/${encodeURIComponent(ctx.streamSessionId)}`
      : '/learnings/ws'
    const connection = connectWs<LearningEnvelope>(
      wsUrl(path),
      (msg) => {
        if (!msg || typeof msg.kind !== 'string') return
        setLiveEvents((current) => [msg, ...current].slice(0, 40))
      },
    )
    return () => connection.stop()
  }, [toolScope, ctx.streamSessionId])

  const sinks: SinkName[] = []
  if (fileSink) sinks.push('file')
  if (mqttSink) sinks.push('mqtt')

  const runExport = async () => {
    if (!site || sinks.length === 0 || busy) return
    setBusy(true)
    setError(null)
    try {
      const resp = await api<KnowledgePushResponse>('/knowledge/push', 'POST', {
        site,
        data_dir: dataDir,
        sinks,
      })
      setLastResult(resp)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const entity = lastResult ? packToEntity(lastResult) : null

  const topTools = useMemo(() => {
    return [...(toolQuery.data?.items || [])]
      .map((row) => {
        const stats = row.runtime?.anomaly_stats || {}
        return {
          ...row,
          stats,
          score: numberOrZero(stats.last_score),
          confirmCount: numberOrZero(stats.confirmed_count),
          dismissCount: numberOrZero(stats.dismissed_count),
          alertCount: numberOrZero(stats.alerted_count),
          scoredCount: numberOrZero(stats.scored_count),
        }
      })
      .filter((row) => row.scoredCount > 0 || row.confirmCount > 0 || row.dismissCount > 0)
      .sort((a, b) => {
        const byFeedback = (b.confirmCount + b.dismissCount) - (a.confirmCount + a.dismissCount)
        if (byFeedback !== 0) return byFeedback
        const byAlerts = b.alertCount - a.alertCount
        if (byAlerts !== 0) return byAlerts
        return b.scoredCount - a.scoredCount
      })
      .slice(0, 12)
  }, [toolQuery.data])

  useEffect(() => {
    if (topTools.length === 0) {
      setSelectedToolKey(null)
      return
    }
    if (!selectedToolKey || !topTools.some((row) => toolRowKey(row) === selectedToolKey)) {
      setSelectedToolKey(toolRowKey(topTools[0]))
    }
  }, [selectedToolKey, topTools])

  const selectedTool = useMemo(() => {
    if (topTools.length === 0) return null
    return topTools.find((row) => toolRowKey(row) === selectedToolKey) || topTools[0]
  }, [selectedToolKey, topTools])

  const selectedToolPatterns = useMemo(() => {
    if (!selectedTool) return []
    const stats = selectedTool.runtime?.anomaly_stats
    const explicit = [
      ...(stats?.last_patterns || []),
      ...(stats?.last_feedback_patterns || []),
    ]
      .filter((value, index, all) => Boolean(value) && all.indexOf(value) === index)
      .slice(0, 6)
      .map((pattern) => humanPattern(pattern))
    if (explicit.length > 0) return explicit
    return sortTopPatterns(stats?.pattern_counts).slice(0, 6)
  }, [selectedTool])

  const selectedToolEvents = useMemo(() => {
    if (!selectedTool) return []
    return liveEvents.filter((item) => {
      if (item.kind !== 'tool_event') return false
      const payload = item.payload || {}
      const toolNumber = numericToolNumber(payload.tool_number)
      if (toolNumber !== selectedTool.tool_number) return false
      const machineFamily = typeof payload.machine_family === 'string' ? payload.machine_family : null
      return !machineFamily || machineFamily === selectedTool.machine_family
    }).slice(0, 5)
  }, [liveEvents, selectedTool])

  const operatorRows = useMemo(() => {
    return [...(feedbackOperatorsQuery.data?.operators || [])]
      .sort((a, b) => b.total - a.total)
      .slice(0, 6)
  }, [feedbackOperatorsQuery.data])

  const learningCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const item of liveEvents) {
      counts[item.kind] = (counts[item.kind] || 0) + 1
    }
    return counts
  }, [liveEvents])

  const recentPriorDiff = ctx.lastPriorDiff.slice(0, 5)

  const liveFeed = useMemo(() => liveEvents.slice(0, 20), [liveEvents])
  const mqttStatus = learningsConfigQuery.data
  const mqttTone = !mqttStatus
    ? colors.textMuted
    : mqttStatus.mqtt_state === 'connected'
      ? colors.good
      : mqttStatus.mqtt_state === 'error' || (mqttStatus.mqtt_enabled && !mqttStatus.mqtt_configured)
        ? colors.bad
        : mqttStatus.mqtt_forwarding_active || mqttStatus.mqtt_enabled
          ? colors.warn
          : colors.textMuted
  const mqttHeadline = !mqttStatus
    ? 'Checking transport'
    : !mqttStatus.mqtt_enabled
      ? 'Local bus only'
      : mqttStatus.mqtt_state === 'connected'
        ? 'Forwarding live'
        : mqttStatus.mqtt_state === 'error'
          ? 'Forwarding error'
          : mqttStatus.mqtt_configured
            ? 'Configured, waiting'
            : 'MQTT misconfigured'

  const activeHint = TAB_DEFS.find((t) => t.key === activeTab)?.hint ?? ''

  const scopeStrip = (
    <div style={{ ...controlBar, marginBottom: spacing.lg }}>
      <span style={labelStyle}>Session</span>
      <span style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600 }}>
        {ctx.streamSessionId || 'None selected'}
      </span>
      <div style={{ flex: 1 }} />
      <span style={labelStyle}>Tool scope</span>
      <button
        type="button"
        style={btnStyle(toolScope !== 'current' || Boolean(ctx.streamSessionId))}
        disabled={toolScope === 'current' && !ctx.streamSessionId}
        onClick={() => setToolScope('current')}
      >
        Current session
      </button>
      <button
        type="button"
        style={btnStyle(toolScope !== 'all')}
        onClick={() => setToolScope('all')}
      >
        All sessions
      </button>
    </div>
  )

  return (
    <div style={pageStyle}>
      <h1 style={titleStyle}>Learnings</h1>
      <p style={{ color: colors.textMuted, marginTop: -spacing.sm, marginBottom: spacing.md }}>
        {learnScope === 'batch'
          ? 'This operation — what this run of the demo produced.'
          : activeHint}
        {learnScope === 'total' && activeTab === 'fleet' && upstreamUrl ? ` · Upstream target: ${upstreamUrl}` : ''}
      </p>

      {/* Scope: this operation (batch, default) vs total (all operations). */}
      <div style={{ display: 'flex', gap: spacing.sm, marginBottom: spacing.lg }}>
        {(['batch', 'total'] as const).map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setLearnScope(s)}
            style={tabButtonStyle(learnScope === s)}
          >
            {s === 'batch' ? 'This batch' : 'Total (all operations)'}
          </button>
        ))}
      </div>

      {learnScope === 'batch' && (() => {
        const d = batchSummaryQuery.data
        const perCatch = maasEvidenceQuery.data?.records?.[0]?.co2_avoided_kg_per_confirmed_catch ?? 921
        return (
          <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
            <div style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', flexWrap: 'wrap' }}>
              <h2 style={sectionTitleStyle}>This operation</h2>
              <a href="#/batch" style={{ color: colors.accent, textDecoration: 'none', fontSize: fontSize.sm }}>Review batch recommendation ›</a>
            </div>
            {!ctx.streamSessionId ? (
              <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.sm }}>
                No active operation. Switch to <em>Total</em> for the all-time picture.
              </div>
            ) : d ? (
              <>
                <div style={{ color: colors.text, fontSize: fontSize.md, marginTop: spacing.sm }}>
                  <span style={{ color: colors.good }}>{d.total_confirmed} confirmed</span>
                  {' / '}<span style={{ color: colors.bad }}>{d.total_dismissed} dismissed</span>
                  <span style={{ color: colors.textMuted, fontWeight: 400 }}> across {d.n_memories} events · {d.n_contexts} contexts</span>
                </div>
                {d.total_confirmed > 0 && (
                  <div style={{ color: colors.warn, fontSize: fontSize.sm, marginTop: spacing.xs }}>
                    CO₂ at stake this operation ~{Math.round(d.total_confirmed * perCatch)} kg
                    <span style={{ color: colors.textMuted }}> · modeled · {d.total_confirmed} catches × ~{Math.round(perCatch)} kg</span>
                  </div>
                )}
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.sm }}>
                  Rolls up into the all-time <em>Total</em>.
                </div>
              </>
            ) : (
              <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.sm }}>Loading this operation…</div>
            )}
          </section>
        )
      })()}

      {learnScope === 'total' && (
      <>
      {/* ── Tab bar ── */}
      <div style={tabBarStyle} role="tablist">
        {TAB_DEFS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.key}
            style={tabButtonStyle(activeTab === tab.key)}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════
          TAB 1 — MAINTENANCE GAINS (operator-facing)
          ══════════════════════════════════════════════════════════ */}
      {activeTab === 'gains' && (
        <>
          {scopeStrip}

          {/* Hero: what operator feedback bought you */}
          {counterfactualQuery.data?.available && counterfactualQuery.data.off && counterfactualQuery.data.on ? (() => {
            const cf = counterfactualQuery.data
            const off = cf.off!
            const on = cf.on!
            const burdenPct = Math.round((cf.burden_reduction ?? 0) * 100)
            const fpPct = Math.round((cf.false_alarm_reduction ?? 0) * 100)
            return (
              <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>What your feedback bought you</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>measured · case study</span>
                </div>
                <div style={{ display: 'grid', gap: spacing.md, gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
                  <div style={heroCardStyle}>
                    <div style={{ color: colors.good, fontSize: 34, fontWeight: 700, lineHeight: 1 }}>−{burdenPct}%</div>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.sm }}>Nuisance alerts avoided</div>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                      {off.alerts} → {on.alerts} alerts to check
                    </div>
                  </div>
                  <div style={heroCardStyle}>
                    <div style={{ color: cf.coverage_preserved ? colors.good : colors.bad, fontSize: 34, fontWeight: 700, lineHeight: 1 }}>
                      {on.broken_episodes_alerted}/{on.broken_episodes_total}
                    </div>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.sm }}>Tool breakages still caught</div>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                      {cf.coverage_preserved ? 'No real catch was lost' : 'Coverage dropped'}
                    </div>
                  </div>
                  <div style={heroCardStyle}>
                    <div style={{ color: colors.good, fontSize: 34, fontWeight: 700, lineHeight: 1 }}>−{fpPct}%</div>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.sm }}>False alarms</div>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                      {off.false_alerts} → {on.false_alerts} false alerts
                    </div>
                  </div>
                </div>
                <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginTop: spacing.md }}>
                  Confirm/dismiss feedback cut the alert load without dropping a real catch. Measured on the
                  Site_a_line2 breakage case study (calibrated model + co-occurrence gating + context-scoped
                  feedback), not a live re-run of the current session.
                </div>
              </section>
            )
          })() : (
            <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
              <h2 style={sectionTitleStyle}>What your feedback bought you</h2>
              <div style={{ ...statusStyle, marginTop: spacing.sm }}>
                Gains appear once operators confirm or dismiss enough alerts for the system to adapt.
              </div>
            </section>
          )}

          {/* How the loop improves */}
          <section style={{ ...panelStyle, marginBottom: spacing.lg }}>
            <h2 style={{ ...sectionTitleStyle, marginBottom: spacing.md }}>How the loop improves</h2>
            <div style={{ alignItems: 'stretch', display: 'flex', flexWrap: 'wrap', gap: spacing.sm }}>
              {[
                { n: '1', t: 'An alert fires', s: 'model + patterns flag an event' },
                { n: '2', t: 'You confirm or dismiss', s: 'was it real on the floor?' },
                { n: '3', t: 'The system adapts', s: 'priors & thresholds shift' },
                { n: '4', t: 'Fewer false alarms', s: 'next shift is quieter' },
              ].map((step, i, all) => (
                <div key={step.n} style={{ alignItems: 'center', display: 'flex', flex: '1 1 130px', gap: spacing.sm }}>
                  <div style={loopStepStyle}>
                    <div style={{ color: colors.accent, fontSize: fontSize.xs, fontWeight: 700 }}>{step.n}</div>
                    <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: 2 }}>{step.t}</div>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: 2 }}>{step.s}</div>
                  </div>
                  {i < all.length - 1 && (
                    <span style={{ color: colors.textDim, fontSize: fontSize.lg }} aria-hidden>→</span>
                  )}
                </div>
              ))}
            </div>
          </section>

          <div style={gridStyle}>
            {/* Tools the system is watching */}
            <section style={panelStyle}>
              <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                <h2 style={sectionTitleStyle}>Tools the system is watching</h2>
                <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                  {topTools.length} learning · {toolQuery.data?.total ?? 0} loaded
                </span>
              </div>
              {toolQuery.isLoading ? (
                <div style={statusStyle}>Loading tool learnings…</div>
              ) : toolQuery.isError ? (
                <div style={{ ...statusStyle, color: colors.bad }}>{(toolQuery.error as Error)?.message || 'Failed to load tool learnings.'}</div>
              ) : topTools.length > 0 ? (
                <>
                  <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginBottom: spacing.sm }}>
                    Select a tool to see its recent activity and your review history.
                  </div>
                  <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Tool</th>
                      <th style={thStyle}>Alerts</th>
                      <th style={thStyle}>Your reviews</th>
                      <th style={thStyle}>Last activity</th>
                      <th style={thStyle}>Recent patterns</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topTools.map((row) => (
                      <tr
                        key={toolRowKey(row)}
                        onClick={() => setSelectedToolKey(toolRowKey(row))}
                        style={{
                          background: selectedTool && toolRowKey(row) === toolRowKey(selectedTool)
                            ? colors.surfaceAlt
                            : 'transparent',
                          cursor: 'pointer',
                        }}
                      >
                        <td style={tdStyle}>
                          <div style={{ fontWeight: 600 }}>{`T${row.tool_number}`}</div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{row.machine_family}</div>
                        </td>
                        <td style={tdStyle}>
                          <div>{row.alertCount} alerted</div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{row.scoredCount} scored</div>
                        </td>
                        <td style={tdStyle}>
                          <div style={{ color: colors.good }}>confirm {row.confirmCount}</div>
                          <div style={{ color: colors.bad }}>dismiss {row.dismissCount}</div>
                        </td>
                        <td style={tdStyle}>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                            evt {formatAgo(row.stats.last_event_at)} · fb {formatAgo(row.stats.last_feedback_at)}
                          </div>
                        </td>
                        <td style={tdStyle}>
                          <div>{sortTopPatterns(row.stats.pattern_counts).join(' · ') || '—'}</div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{(row.master?.description || row.master?.tool_type || 'No master description')}</div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  </table>
                </>
              ) : (
                <div style={statusStyle}>No tools have accumulated anomaly or feedback learnings yet.</div>
              )}
            </section>

            <div style={{ display: 'grid', gap: spacing.lg }}>
              {/* Selected tool detail */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Selected tool detail</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                    {selectedTool ? `T${selectedTool.tool_number} · ${selectedTool.machine_family}` : 'No tool selected'}
                  </span>
                </div>
                {selectedTool ? (
                  <div style={{ display: 'grid', gap: spacing.md }}>
                    <div style={{ ...summaryGridStyle, gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))' }}>
                      <div style={summaryCardStyle}>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Last feedback</div>
                        <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.xs }}>
                          {selectedTool.runtime?.anomaly_stats?.last_feedback_action || 'No review yet'}
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          {String(selectedTool.runtime?.anomaly_stats?.last_operator_id || '—')} · {formatAgo(selectedTool.runtime?.anomaly_stats?.last_feedback_at)}
                        </div>
                      </div>
                      <div style={summaryCardStyle}>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Review history</div>
                        <div style={{ color: colors.good, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.xs }}>
                          confirm {numberOrZero(selectedTool.runtime?.anomaly_stats?.confirmed_count)}
                        </div>
                        <div style={{ color: colors.bad, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          dismiss {numberOrZero(selectedTool.runtime?.anomaly_stats?.dismissed_count)}
                        </div>
                      </div>
                      <div style={summaryCardStyle}>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Recent score</div>
                        <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 600, marginTop: spacing.xs }}>
                          {compactNumber(selectedTool.runtime?.anomaly_stats?.last_score)}
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          event {formatAgo(selectedTool.runtime?.anomaly_stats?.last_event_at)}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', marginBottom: spacing.xs, textTransform: 'uppercase' }}>
                        Recent patterns
                      </div>
                      {selectedToolPatterns.length > 0 ? (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: spacing.xs }}>
                          {selectedToolPatterns.map((pattern) => (
                            <span
                              key={pattern}
                              style={{
                                background: colors.surfaceAlt,
                                border: `1px solid ${colors.border}`,
                                borderRadius: 999,
                                color: colors.text,
                                fontSize: fontSize.xs,
                                padding: '4px 10px',
                              }}
                            >
                              {pattern}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>No recent pattern evidence for this tool yet.</div>
                      )}
                    </div>

                    <div style={{ display: 'grid', gap: spacing.xs }}>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        Context and description
                      </div>
                      <div style={{ color: colors.text, fontSize: fontSize.sm }}>
                        {selectedTool.master?.description || selectedTool.master?.tool_type || 'No master description'}
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                        Seen {numberOrZero(selectedTool.runtime?.seen_count)} times · last seen {formatAgo(selectedTool.runtime?.last_seen_at)}
                      </div>
                    </div>

                    <div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', marginBottom: spacing.xs, textTransform: 'uppercase' }}>
                        Recent tool events
                      </div>
                      {selectedToolEvents.length > 0 ? (
                        <div style={{ display: 'grid', gap: spacing.sm }}>
                          {selectedToolEvents.map((item, index) => (
                            <div key={`${item.kind}-${item.ts_unix || index}-${index}`} style={feedItemStyle}>
                              <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, justifyContent: 'space-between', marginBottom: spacing.xs }}>
                                <LearningKindPill kind={item.kind} />
                                <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{formatAgo(item.ts_unix || null)}</span>
                              </div>
                              <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 500 }}>{renderLearningSummary(item)}</div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>No live tool_event envelopes seen for this selection yet.</div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div style={statusStyle}>Select a tool from the table to inspect its detail.</div>
                )}
              </section>

              {/* Operator feedback summary */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Operator feedback</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{operatorRows.length} operators</span>
                </div>
                {feedbackOperatorsQuery.isLoading ? (
                  <div style={statusStyle}>Loading operator summaries…</div>
                ) : operatorRows.length > 0 ? (
                  <div style={{ display: 'grid', gap: spacing.sm }}>
                    {operatorRows.map((row) => (
                      <div key={row.operator_id} style={summaryCardStyle}>
                        <div style={{ alignItems: 'center', display: 'flex', justifyContent: 'space-between', gap: spacing.md }}>
                          <div style={{ color: colors.text, fontWeight: 600 }}>{row.operator_id}</div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{row.total} actions</div>
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.xs }}>
                          confirm {row.actions.confirm || 0} · dismiss {row.actions.dismiss || 0} · comment {row.actions.comment || 0}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={statusStyle}>No feedback recorded yet.</div>
                )}
              </section>
            </div>
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════════════════════
          TAB 2 — FLEET & MAAS (business / EC-facing)
          ══════════════════════════════════════════════════════════ */}
      {activeTab === 'fleet' && (
        <div style={{ display: 'grid', gap: spacing.lg }}>
          {/* Framing */}
          <section style={panelStyle}>
            <h2 style={sectionTitleStyle}>Learnings that propagate to the platform</h2>
            <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.sm }}>
              Only summarized, context-conditioned evidence propagates — never raw signals, recipes or operator know-how.
            </div>
            <details style={{ marginTop: spacing.md }}>
              <summary style={{ color: colors.textMuted, cursor: 'pointer', fontSize: fontSize.sm }}>
                Common interface — the fields every object carries
              </summary>
              <div style={{ marginTop: spacing.sm }}>
                <FieldGrid fields={EVIDENCE_INTERFACE_FIELDS} />
              </div>
            </details>
          </section>

          {/* Capability evidence — with live records from operator feedback */}
          <section style={panelStyle}>
            <h2 style={{ ...sectionTitleStyle, marginBottom: spacing.sm }}>{MAAS_EVIDENCE_OBJECTS[0].title}</h2>
            <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{MAAS_EVIDENCE_OBJECTS[0].purpose}</div>
            <div style={{ color: colors.textDim, fontSize: fontSize.sm, marginTop: spacing.xs }}>{MAAS_EVIDENCE_OBJECTS[0].use}</div>
            {maasEvidenceQuery.isLoading ? (
              <div style={{ ...statusStyle, marginTop: spacing.md }}>Loading evidence…</div>
            ) : maasEvidenceQuery.data?.available && (maasEvidenceQuery.data.records?.length ?? 0) > 0 ? (
              <>
                <div style={{ display: 'grid', gap: spacing.md, marginTop: spacing.md }}>
                  {[...(maasEvidenceQuery.data.records ?? [])]
                    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
                    .map((r, i) => (
                    <div key={`${r.capability ?? ''}-${r.plant_id ?? i}`} style={summaryCardStyle}>
                      <div style={{ alignItems: 'center', display: 'flex', justifyContent: 'space-between', gap: spacing.md, flexWrap: 'wrap' }}>
                        <div style={{ color: colors.text, fontWeight: 600 }}>{r.capability ?? 'Capability'}</div>
                        <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm }}>
                          {r.declared && (
                            <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>declared</span>
                          )}
                          <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>→</span>
                          <span style={{ color: colors.good, fontSize: fontSize.sm, fontWeight: 700 }}>observed</span>
                        </div>
                      </div>
                      <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                        {r.plant_id ? `${r.plant_id} · ` : ''}
                        {[r.context?.machine_family, r.context?.tool_type, r.context?.material].filter(Boolean).join(' · ')}
                      </div>
                      <div style={{ display: 'grid', gap: spacing.sm, gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', marginTop: spacing.md }}>
                        <div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Confirm-rate</div>
                          <div style={{ color: colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>
                            {typeof r.confirm_rate === 'number' ? `${Math.round(r.confirm_rate * 100)}%` : '—'}
                            <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}>
                              {' '}({r.confirmed ?? 0}✓/{r.dismissed ?? 0}✗)
                            </span>
                          </div>
                        </div>
                        <div>
                          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Confidence</div>
                          <div style={{ color: colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>
                            {typeof r.confidence === 'number' ? r.confidence.toFixed(2) : '—'}
                            <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}> (volume-shrunk)</span>
                          </div>
                        </div>
                        {typeof r.lead_time_s_median === 'number' && (
                          <div>
                            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Lead time</div>
                            <div style={{ color: colors.text, fontSize: fontSize.lg, fontWeight: 700 }}>
                              {Math.round(r.lead_time_s_median)}s
                              <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}> warning</span>
                            </div>
                          </div>
                        )}
                        {typeof r.co2_avoided_kg_per_confirmed_catch === 'number' && (
                          <div>
                            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>CO₂ at stake / catch</div>
                            <div style={{ color: colors.warn, fontSize: fontSize.lg, fontWeight: 700 }}>
                              ~{Math.round(r.co2_avoided_kg_per_confirmed_catch)} kg
                              <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}> · modeled</span>
                            </div>
                          </div>
                        )}
                        {typeof r.co2_avoided_kg_total === 'number' && (
                          <div>
                            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, textTransform: 'uppercase', letterSpacing: '0.06em' }}>CO₂ at stake (window)</div>
                            <div style={{ color: colors.warn, fontSize: fontSize.lg, fontWeight: 700 }}>
                              ~{Math.round(r.co2_avoided_kg_total)} kg
                              <span style={{ color: colors.textMuted, fontSize: fontSize.xs, fontWeight: 400 }}>
                                {r.window ? ` · ${r.window}` : ''} · modeled
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                      <details style={{ marginTop: spacing.sm }}>
                        <summary style={{ color: colors.accent, cursor: 'pointer', fontSize: fontSize.xs }}>Trace</summary>
                        <div style={{ display: 'grid', gap: 4, fontSize: fontSize.xs, color: colors.textMuted, marginTop: spacing.xs }}>
                          <div>
                            Confirm-rate <strong style={{ color: colors.text }}>{Math.round((r.confirm_rate ?? 0) * 100)}%</strong>
                            {' '}← {r.confirmed ?? 0} confirmed / {r.dismissed ?? 0} dismissed
                          </div>
                          {typeof r.co2_avoided_kg_total === 'number' && (
                            <div>
                              CO₂ at stake <strong style={{ color: colors.warn }}>~{Math.round(r.co2_avoided_kg_total)} kg</strong>
                              {' '}← {r.confirmed ?? 0} catches × ~{Math.round(r.co2_avoided_kg_per_confirmed_catch ?? 0)} kg
                              {' '}(≈ one part's CO₂{r.dpp_source ? `, ${r.dpp_source}` : ''})
                            </div>
                          )}
                          <GraphQueryLink
                            query={patternsGraphQuery(capabilityPatternKeys(r.capability))}
                            label="View contributing events in the graph"
                            style={{ color: colors.accent, textDecoration: 'none' }}
                          />
                        </div>
                      </details>
                    </div>
                  ))}
                </div>
              </>
            ) : null}
            <EvidenceStructure object={MAAS_EVIDENCE_OBJECTS[0]} />
          </section>

          {/* Transfers across the fleet — the family-level aggregate */}
          <section style={panelStyle}>
            <div style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', flexWrap: 'wrap' }}>
              <h2 style={sectionTitleStyle}>Transfers across the fleet (same machine family)</h2>
              <button type="button" style={btnStyle(!fleetBusy)} disabled={fleetBusy} onClick={populateFleet}>
                {fleetBusy ? 'Populating…' : 'Populate demo fleet'}
              </button>
            </div>
            <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.sm }}>
              Per-site priors aggregate into a family-level prior, gated by k-anonymity so no single site is identifiable.
            </div>
            {fleetQuery.isLoading ? (
              <div style={{ ...statusStyle, marginTop: spacing.md }}>Loading fleet aggregate…</div>
            ) : fleetQuery.data ? (
              <div style={{ display: 'grid', gap: spacing.md, marginTop: spacing.md }}>
                <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.sm }}>
                  <span style={claimBadge(fleetQuery.data.k_anonymity_met ? colors.good : colors.warn)}>
                    {fleetQuery.data.k_anonymity_met ? '✓ k-anonymity met' : '○ below k-anonymity'} · {fleetQuery.data.site_count}/{fleetQuery.data.k_anonymity_threshold} sites
                  </span>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                    {[FLEET_DEMO_CTX.machine_type, FLEET_DEMO_CTX.tool_type, FLEET_DEMO_CTX.material, FLEET_DEMO_CTX.regime].join(' · ')}
                  </span>
                </div>
                {fleetQuery.data.k_anonymity_met ? (
                  <>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                      Contributing sites: {fleetQuery.data.source_sites.join(', ')}
                    </div>
                    <div style={{ display: 'grid', gap: spacing.xs }}>
                      {Object.entries(fleetQuery.data.pattern_priors).slice(0, 6).map(([k, v]) => (
                        <div key={k} style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', fontSize: fontSize.sm }}>
                          <span style={{ color: colors.text, fontFamily: 'monospace' }}>{k}</span>
                          <span style={{ color: colors.textMuted }}>
                            family prior <span style={{ color: colors.good, fontWeight: 700 }}>{v.prior.toFixed(3)}</span> · {v.site_count} sites
                          </span>
                        </div>
                      ))}
                    </div>
                    <GraphQueryLink
                      query={patternsGraphQuery(Object.keys(fleetQuery.data.pattern_priors))}
                      label="View these patterns in the graph"
                      style={{ color: colors.accent, textDecoration: 'none', fontSize: fontSize.xs }}
                    />
                  </>
                ) : (
                  <div style={statusStyle}>
                    No shared aggregate yet — fewer than {fleetQuery.data.k_anonymity_threshold} sites have contributed for this
                    context. Click <em>Populate demo fleet</em> to add synthetic sites and see the aggregate form.
                  </div>
                )}
              </div>
            ) : null}
          </section>

          {/* Secondary evidence objects + export — collapsed so the tab leads
              with capability + fleet transfer. */}
          <details>
            <summary style={{ color: colors.text, cursor: 'pointer', fontSize: fontSize.md, fontWeight: 600, padding: `${spacing.sm}px 0` }}>
              More evidence objects &amp; knowledge-pack export
            </summary>
            <div style={{ display: 'grid', gap: spacing.lg, marginTop: spacing.md }}>
          {/* Fault/lead-time, Availability, Sustainability objects */}
          {MAAS_EVIDENCE_OBJECTS.slice(1).map((obj) => {
            const facets = maasFacetsQuery.data?.facets
            const faultRec = obj.key === 'fault' ? facets?.fault?.records?.[0] : undefined
            const availRec = obj.key === 'availability' ? facets?.availability?.records?.[0] : undefined
            const sustRec = obj.key === 'sustainability' ? facets?.sustainability?.records?.[0] : undefined
            return (
              <section key={obj.key} style={panelStyle}>
                <h2 style={{ ...sectionTitleStyle, marginBottom: spacing.sm }}>{obj.title}</h2>
                <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{obj.purpose}</div>
                <div style={{ color: colors.textDim, fontSize: fontSize.sm, marginTop: spacing.xs }}>{obj.use}</div>
                {faultRec ? <FaultCard record={faultRec} /> : null}
                {availRec ? <AvailabilityCard record={availRec} /> : null}
                {sustRec ? <SustainabilityCard record={sustRec} /> : null}
                <EvidenceStructure object={obj} />
              </section>
            )
          })}

          {/* Knowledge-pack export */}
          <section style={panelStyle}>
            <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
              <h2 style={sectionTitleStyle}>Export knowledge pack</h2>
              <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>bundle learned state for upstream sharing</span>
            </div>
            <div style={controlBar}>
              <label>
                <span style={labelStyle}>Site</span>
                <input
                  style={inputStyle}
                  value={site}
                  onChange={(e) => setSite(e.target.value)}
                  placeholder="e.g. lab-1"
                />
              </label>
              <label>
                <span style={labelStyle}>Data dir</span>
                <input
                  style={inputStyle}
                  value={dataDir}
                  onChange={(e) => setDataDir(e.target.value)}
                />
              </label>
              <label style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                <input
                  type="checkbox"
                  checked={fileSink}
                  onChange={(e) => setFileSink(e.target.checked)}
                  style={{ marginRight: spacing.xs }}
                />
                File sink
              </label>
              <label style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                <input
                  type="checkbox"
                  checked={mqttSink}
                  onChange={(e) => setMqttSink(e.target.checked)}
                  style={{ marginRight: spacing.xs }}
                />
                MQTT sink
              </label>
              <button
                type="button"
                style={btnStyle(!busy && sinks.length > 0 && site.length > 0)}
                disabled={busy || sinks.length === 0 || site.length === 0}
                onClick={runExport}
              >
                {busy ? 'Exporting…' : 'Export'}
              </button>
              {error ? (
                <span style={{ color: colors.bad, fontSize: fontSize.sm }}>{error}</span>
              ) : null}
            </div>
            {entity ? (
              <div style={{ marginTop: spacing.md }}>
                <EntityView schema={entity} />
              </div>
            ) : (
              <div style={statusStyle}>
                No export yet. Fill the form above and click “Export” to build a pack.
              </div>
            )}
          </section>
            </div>
          </details>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════
          TAB 3 — DIAGNOSTICS (engineer-facing)
          ══════════════════════════════════════════════════════════ */}
      {activeTab === 'diagnostics' && (
        <>
          {scopeStrip}
          <div style={gridStyle}>
            <div style={{ display: 'grid', gap: spacing.lg }}>
              {/* MQTT forwarding */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>MQTT forwarding</h2>
                  <span style={{ color: mqttTone, fontSize: fontSize.sm, fontWeight: 600 }}>{mqttHeadline}</span>
                </div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                  {mqttStatus?.mqtt_topic
                    ? `${mqttStatus.mqtt_broker_host}:${mqttStatus.mqtt_broker_port} · ${mqttStatus.mqtt_topic}`
                    : 'No learnings topic configured'}
                </div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                  {mqttStatus?.last_error
                    ? mqttStatus.last_error
                    : `published ${mqttStatus?.published_count ?? 0} · last ${formatAgo(mqttStatus?.last_published_at ?? null)}`}
                </div>
              </section>

              {/* Pattern priors */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Pattern priors</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                    {ctx.priorsQuery.data?.priors?.length ?? 0} tracked patterns
                  </span>
                </div>
                <PriorsChart priors={ctx.priorsQuery.data?.priors || []} maxRows={10} />
                <div style={{ marginTop: spacing.md }}>
                  <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', marginBottom: spacing.xs, textTransform: 'uppercase' }}>
                    Last prior delta
                  </div>
                  {recentPriorDiff.length > 0 ? (
                    <div style={{ display: 'grid', gap: spacing.xs }}>
                      {recentPriorDiff.map((row) => (
                        <div key={row.pattern} style={{ alignItems: 'center', display: 'grid', gap: spacing.sm, gridTemplateColumns: '1fr auto', fontSize: fontSize.sm }}>
                          <span style={{ color: colors.text }}>{humanPattern(row.pattern)}</span>
                          <span style={{ color: row.delta >= 0 ? colors.good : colors.bad, fontVariantNumeric: 'tabular-nums' }}>
                            {row.delta >= 0 ? '+' : ''}{row.delta.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>No feedback delta captured yet in this client.</div>
                  )}
                </div>
              </section>

              {/* Feedback outbox */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Feedback outbox</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{feedbackOutboxQuery.data?.pending ?? 0} pending</span>
                </div>
                {feedbackOutboxQuery.data?.head?.length ? (
                  <div style={{ display: 'grid', gap: spacing.xs }}>
                    {feedbackOutboxQuery.data.head.map((item) => (
                      <div key={item.sequence} style={{ color: colors.text, display: 'grid', fontSize: fontSize.sm, gap: 2 }}>
                        <div>{item.action} · {item.operator_id}</div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{item.memory_id} · seq {item.sequence}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>Outbox head is empty.</div>
                )}
              </section>
            </div>

            <div style={{ display: 'grid', gap: spacing.lg }}>
              {/* Model trust */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Model trust</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                    feedback-driven confidence
                  </span>
                </div>
                {modelTrustQuery.isLoading ? (
                  <div style={statusStyle}>Loading model-trust state…</div>
                ) : (() => {
                  const mt = modelTrustQuery.data?.model_trust
                  if (!mt) return <div style={statusStyle}>No model-trust data.</div>
                  const globalConf = mt.model_confidence ?? 0.5
                  const confColor = (c: number) => (c >= 0.6 ? colors.good : c <= 0.4 ? colors.bad : colors.textMuted)
                  const scopes = mt.scopes ?? []
                  return (
                    <div style={{ display: 'grid', gap: spacing.md }}>
                      <div style={summaryCardStyle}>
                        <div style={{ alignItems: 'center', display: 'flex', justifyContent: 'space-between', gap: spacing.md }}>
                          <div style={{ color: colors.text, fontWeight: 600 }}>Global (site-wide)</div>
                          <div style={{ color: confColor(globalConf), fontWeight: 700, fontSize: fontSize.lg }}>
                            {globalConf.toFixed(2)}
                          </div>
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.sm, marginTop: spacing.xs }}>
                          confirmed {mt.true_positives ?? 0} · dismissed {mt.false_positives ?? 0} · evidence {mt.evidence_count ?? 0}
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          Scales the classical model's contribution to alert scoring. Repeated dismissals lower it; confirmed catches raise it.
                        </div>
                      </div>

                      <div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', marginBottom: spacing.xs, textTransform: 'uppercase' }}>
                          Per-context trust {scopes.length ? `(${scopes.length})` : ''}
                        </div>
                        {scopes.length > 0 ? (
                          <div style={{ display: 'grid', gap: spacing.xs }}>
                            {scopes.map((s) => (
                              <div key={s.context} style={{ alignItems: 'center', display: 'flex', justifyContent: 'space-between', gap: spacing.md }}>
                                <div style={{ color: colors.text, fontSize: fontSize.sm }}>{s.context}</div>
                                <div style={{ alignItems: 'center', display: 'flex', gap: spacing.md }}>
                                  <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>
                                    {s.confirmed}✓ / {s.dismissed}✗
                                  </span>
                                  <span style={{ color: confColor(s.model_confidence), fontWeight: 700, minWidth: 36, textAlign: 'right' }}>
                                    {s.model_confidence.toFixed(2)}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                            No per-context evidence yet — trust becomes context-specific (regime / tool / material) as operators adjudicate alerts.
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })()}
              </section>

              {/* Live learnings feed */}
              <section style={panelStyle}>
                <div style={{ alignItems: 'baseline', display: 'flex', justifyContent: 'space-between', gap: spacing.md, marginBottom: spacing.md }}>
                  <h2 style={sectionTitleStyle}>Live learnings feed</h2>
                  <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
                    {liveEvents.length} events · scores {liveScores.length} · priors {livePriorSnapshots.length}
                  </span>
                </div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginBottom: spacing.sm }}>
                  tool {learningCounts.tool_event || 0} · feedback {learningCounts.feedback_event || 0} · scored {learningCounts.scored_event || 0}
                </div>
                {liveFeed.length > 0 ? (
                  <div style={feedListStyle}>
                    {liveFeed.map((item, index) => (
                      <div key={`${item.kind}-${item.ts_unix || index}-${index}`} style={feedItemStyle}>
                        <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm, justifyContent: 'space-between', marginBottom: spacing.xs }}>
                          <LearningKindPill kind={item.kind} />
                          <span style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{formatAgo(item.ts_unix || null)}</span>
                        </div>
                        <div style={{ color: colors.text, fontSize: fontSize.sm, fontWeight: 500 }}>{renderLearningSummary(item)}</div>
                        <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.xs }}>
                          {item.session_id ? `session ${item.session_id}` : 'global'} · {item.source || 'unknown source'}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={statusStyle}>Waiting for learnings on the websocket bus…</div>
                )}
              </section>
            </div>
          </div>
        </>
      )}
      </>
      )}
    </div>
  )
}

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/http'

const SINDIT_URL = 'http://localhost:9017'
const GRAPHDB_URL = 'http://localhost:7200'

/* ── Colour/shape map for node kinds ── */
const KIND_STYLES: Record<string, { fill: string; stroke: string; icon: string; radius: number }> = {
  machine:      { fill: '#3d59a1', stroke: '#7aa2f7', icon: '🏭', radius: 28 },
  experiment:   { fill: '#1a1b26', stroke: '#7aa2f7', icon: '🧪', radius: 24 },
  'test-phase': { fill: '#1a1b26', stroke: '#e0af68', icon: '📋', radius: 20 },
  'eval-phase': { fill: '#1a1b26', stroke: '#9ece6a', icon: '✅', radius: 20 },
  operation:    { fill: '#1a1b26', stroke: '#bb9af7', icon: '⚙️', radius: 18 },
  pattern:      { fill: '#1a1b26', stroke: '#f7768e', icon: '🔍', radius: 18 },
  sensor:       { fill: '#1a1b26', stroke: '#73daca', icon: '📡', radius: 16 },
  other:        { fill: '#1a1b26', stroke: '#565f89', icon: '●',  radius: 14 },
}

interface GNode { uri: string; label: string; kind: string; description: string; properties: Record<string, any> }
interface GEdge { source: string; target: string; type: string; label: string }

interface LayoutNode extends GNode { x: number; y: number; vx: number; vy: number }

/**
 * Force-directed knowledge-graph visualisation.
 * Pure React+SVG — no external dependencies.
 */
function KGVisualisation({ nodes, edges }: { nodes: GNode[]; edges: GEdge[] }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const W = 900, H = 500
  const [hoveredUri, setHoveredUri] = useState<string | null>(null)
  const [selectedUri, setSelectedUri] = useState<string | null>(null)
  const [layoutNodes, setLayoutNodes] = useState<LayoutNode[]>([])

  // Initialise positions (radial layout by kind, then force-settle)
  useEffect(() => {
    if (nodes.length === 0) return
    const cx = W / 2, cy = H / 2

    // Group by kind and arrange in concentric rings
    const kindOrder = ['machine', 'experiment', 'test-phase', 'eval-phase', 'operation', 'pattern', 'sensor', 'other']
    const groups: Record<string, GNode[]> = {}
    for (const n of nodes) {
      const k = n.kind || 'other'
      ;(groups[k] = groups[k] || []).push(n)
    }

    const positioned: LayoutNode[] = []
    let ring = 0
    for (const kind of kindOrder) {
      const grp = groups[kind]
      if (!grp) continue
      const r = kind === 'machine' ? 0 : 80 + ring * 65
      grp.forEach((n, i) => {
        const angle = (2 * Math.PI * i) / Math.max(grp.length, 1) - Math.PI / 2
        positioned.push({
          ...n,
          x: cx + r * Math.cos(angle) + (Math.random() - 0.5) * 10,
          y: cy + r * Math.sin(angle) + (Math.random() - 0.5) * 10,
          vx: 0, vy: 0,
        })
      })
      if (grp.length > 0) ring++
    }
    setLayoutNodes(positioned)
  }, [nodes])

  // Simple force simulation (runs 80 iterations on mount / data change)
  useEffect(() => {
    if (layoutNodes.length === 0) return
    const ns = layoutNodes.map(n => ({ ...n }))
    const uriIdx: Record<string, number> = {}
    ns.forEach((n, i) => { uriIdx[n.uri] = i })

    const iterations = 80
    for (let iter = 0; iter < iterations; iter++) {
      const alpha = 1 - iter / iterations
      const repulse = 6000 * alpha
      const attract = 0.03 * alpha

      // Repulsion (all pairs)
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          let dx = ns[j].x - ns[i].x
          let dy = ns[j].y - ns[i].y
          const d2 = dx * dx + dy * dy + 1
          const f = repulse / d2
          const fx = dx * f / Math.sqrt(d2)
          const fy = dy * f / Math.sqrt(d2)
          ns[i].x -= fx; ns[i].y -= fy
          ns[j].x += fx; ns[j].y += fy
        }
      }

      // Attraction (edges)
      for (const e of edges) {
        const si = uriIdx[e.source]
        const ti = uriIdx[e.target]
        if (si === undefined || ti === undefined) continue
        const dx = ns[ti].x - ns[si].x
        const dy = ns[ti].y - ns[si].y
        const d = Math.sqrt(dx * dx + dy * dy) + 0.1
        const f = (d - 120) * attract
        const fx = (dx / d) * f
        const fy = (dy / d) * f
        ns[si].x += fx; ns[si].y += fy
        ns[ti].x -= fx; ns[ti].y -= fy
      }

      // Center gravity
      for (const n of ns) {
        n.x += (W / 2 - n.x) * 0.01 * alpha
        n.y += (H / 2 - n.y) * 0.01 * alpha
        // bounds
        n.x = Math.max(40, Math.min(W - 40, n.x))
        n.y = Math.max(40, Math.min(H - 40, n.y))
      }
    }
    setLayoutNodes(ns)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes.length, edges.length])

  const uriMap: Record<string, LayoutNode> = {}
  for (const n of layoutNodes) uriMap[n.uri] = n

  const selected = selectedUri ? uriMap[selectedUri] : null

  const kindForUri = (uri: string) => uriMap[uri]?.kind || 'other'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 280px' : '1fr', gap: 12 }}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', height: H, background: 'rgba(0,0,0,0.2)', borderRadius: 8, border: '1px solid var(--border)', cursor: 'default' }}
        onClick={() => setSelectedUri(null)}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
            <polygon points="0 0, 10 3.5, 0 7" fill="rgba(128,128,128,0.5)" />
          </marker>
        </defs>

        {/* Edges */}
        {edges.map((e, i) => {
          const s = uriMap[e.source]
          const t = uriMap[e.target]
          if (!s || !t) return null
          const dx = t.x - s.x, dy = t.y - s.y
          const d = Math.sqrt(dx * dx + dy * dy) || 1
          const sr = (KIND_STYLES[s.kind] || KIND_STYLES.other).radius + 4
          const tr = (KIND_STYLES[t.kind] || KIND_STYLES.other).radius + 4
          const x1 = s.x + (dx / d) * sr
          const y1 = s.y + (dy / d) * sr
          const x2 = t.x - (dx / d) * tr
          const y2 = t.y - (dy / d) * tr
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
          const isHl = hoveredUri === e.source || hoveredUri === e.target || selectedUri === e.source || selectedUri === e.target
          return (
            <g key={i} opacity={hoveredUri && !isHl ? 0.15 : 1}>
              <line x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={isHl ? 'var(--accent)' : 'rgba(128,128,128,0.35)'}
                strokeWidth={isHl ? 2 : 1}
                markerEnd="url(#arrow)" />
              <text x={mx} y={my - 5} textAnchor="middle" fontSize={8}
                fill={isHl ? 'var(--fg)' : 'rgba(128,128,128,0.5)'}
                style={{ pointerEvents: 'none' }}>
                {e.label}
              </text>
            </g>
          )
        })}

        {/* Nodes */}
        {layoutNodes.map(n => {
          const s = KIND_STYLES[n.kind] || KIND_STYLES.other
          const isHl = hoveredUri === n.uri || selectedUri === n.uri
          const dimmed = hoveredUri && hoveredUri !== n.uri && !edges.some(e => (e.source === hoveredUri && e.target === n.uri) || (e.target === hoveredUri && e.source === n.uri))
          return (
            <g key={n.uri}
              style={{ cursor: 'pointer' }}
              opacity={dimmed ? 0.2 : 1}
              onMouseEnter={() => setHoveredUri(n.uri)}
              onMouseLeave={() => setHoveredUri(null)}
              onClick={(ev) => { ev.stopPropagation(); setSelectedUri(n.uri === selectedUri ? null : n.uri) }}>
              <circle cx={n.x} cy={n.y} r={s.radius}
                fill={isHl ? s.stroke : s.fill}
                stroke={s.stroke}
                strokeWidth={isHl ? 3 : 1.5}
                opacity={isHl ? 1 : 0.85} />
              <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="central"
                fontSize={s.radius > 20 ? 16 : 12}
                style={{ pointerEvents: 'none' }}>
                {s.icon}
              </text>
              <text x={n.x} y={n.y + s.radius + 12} textAnchor="middle"
                fontSize={10} fontWeight={isHl ? 700 : 400}
                fill={isHl ? '#fff' : 'var(--muted)'}
                style={{ pointerEvents: 'none' }}>
                {n.label.length > 28 ? n.label.slice(0, 26) + '…' : n.label}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Detail panel */}
      {selected && (
        <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 8, border: '1px solid var(--border)', padding: 14, fontSize: 12, overflowY: 'auto', maxHeight: H }}>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>
            {(KIND_STYLES[selected.kind] || KIND_STYLES.other).icon} {selected.label}
          </div>
          <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace', wordBreak: 'break-all', marginBottom: 8 }}>
            {selected.uri}
          </div>
          {selected.description && (
            <div className="small" style={{ color: 'var(--muted)', marginBottom: 8, lineHeight: 1.5 }}>
              {selected.description}
            </div>
          )}
          <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4, color: 'var(--accent)' }}>Kind: {selected.kind}</div>

          {Object.keys(selected.properties).length > 0 && (
            <>
              <div style={{ fontWeight: 600, fontSize: 11, marginTop: 8, marginBottom: 4 }}>Properties</div>
              {Object.entries(selected.properties).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid rgba(128,128,128,0.1)' }}>
                  <span style={{ color: 'var(--muted)' }}>{k}</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                    {typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : String(v)}
                  </span>
                </div>
              ))}
            </>
          )}

          {/* Connected edges */}
          <div style={{ fontWeight: 600, fontSize: 11, marginTop: 10, marginBottom: 4 }}>Connections</div>
          {edges.filter(e => e.source === selected.uri || e.target === selected.uri).map((e, i) => {
            const other = e.source === selected.uri ? e.target : e.source
            const otherNode = uriMap[other]
            const direction = e.source === selected.uri ? '→' : '←'
            return (
              <div key={i} style={{ padding: '3px 0', borderBottom: '1px solid rgba(128,128,128,0.1)', fontSize: 10, display: 'flex', gap: 4, alignItems: 'center', cursor: 'pointer' }}
                onClick={() => setSelectedUri(other)}>
                <span style={{ color: 'var(--accent)' }}>{direction}</span>
                <span style={{ color: 'var(--muted)' }}>{e.label}</span>
                <span style={{ fontWeight: 600, marginLeft: 'auto' }}>{otherNode?.label || other.split(':').pop()}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', gridColumn: '1 / -1' }}>
        {Object.entries(KIND_STYLES).filter(([k]) => nodes.some(n => n.kind === k)).map(([kind, s]) => (
          <div key={kind} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--muted)' }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: s.stroke, border: `1px solid ${s.stroke}` }} />
            {kind}
          </div>
        ))}
        <div style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--muted)' }}>
          {nodes.length} nodes · {edges.length} edges
        </div>
      </div>
    </div>
  )
}

function StatusBadge({ ok, label, detail }: { ok: boolean | null; label: string; detail?: string }) {
  const color = ok === null ? 'var(--muted)' : ok ? 'var(--ok)' : 'var(--danger)'
  const bg = ok === null ? 'rgba(128,128,128,0.1)' : ok ? 'rgba(158,206,106,0.1)' : 'rgba(247,118,142,0.1)'
  const icon = ok === null ? '⏳' : ok ? '✓' : '✗'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: bg, borderRadius: 8, border: `1px solid ${ok === null ? 'var(--border)' : ok ? 'rgba(158,206,106,0.3)' : 'rgba(247,118,142,0.3)'}` }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <div>
        <div style={{ fontWeight: 600, color, fontSize: 13 }}>{label}</div>
        {detail && <div className="small" style={{ color: 'var(--muted)' }}>{detail}</div>}
      </div>
    </div>
  )
}

function FlagPill({ flag }: { flag: string }) {
  const warn = /mismatch|missing|stale|not_ready|duplicate|miss/.test(flag)
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '2px 8px',
      borderRadius: 999,
      border: `1px solid ${warn ? 'rgba(247,118,142,0.35)' : 'var(--border)'}`,
      background: warn ? 'rgba(247,118,142,0.08)' : 'rgba(122,162,247,0.08)',
      color: warn ? 'var(--danger)' : 'var(--muted)',
      fontSize: 10,
      fontFamily: 'monospace',
      whiteSpace: 'nowrap',
    }}>
      {flag}
    </span>
  )
}

function TonePill({ label, tone }: { label: string; tone: 'ok' | 'warn' | 'danger' | 'accent' | 'neutral' }) {
  const styles = {
    ok: { border: 'rgba(158,206,106,0.35)', bg: 'rgba(158,206,106,0.08)', color: 'var(--ok)' },
    warn: { border: 'rgba(224,175,104,0.35)', bg: 'rgba(224,175,104,0.08)', color: '#e0af68' },
    danger: { border: 'rgba(247,118,142,0.35)', bg: 'rgba(247,118,142,0.08)', color: 'var(--danger)' },
    accent: { border: 'rgba(122,162,247,0.35)', bg: 'rgba(122,162,247,0.08)', color: 'var(--accent)' },
    neutral: { border: 'var(--border)', bg: 'rgba(128,128,128,0.08)', color: 'var(--muted)' },
  }[tone]

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      padding: '2px 8px',
      borderRadius: 999,
      border: `1px solid ${styles.border}`,
      background: styles.bg,
      color: styles.color,
      fontSize: 10,
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}

function ToolAuditField({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 10, padding: '4px 0', borderBottom: '1px solid rgba(128,128,128,0.08)' }}>
      <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</div>
      <div style={{ fontSize: 12, color: 'var(--fg)', wordBreak: 'break-word' }}>{value}</div>
    </div>
  )
}

/* ---------- tiny helpers ---------- */

function ago(iso: string | null): string {
  if (!iso) return 'never'
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return `${s}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}

/* ---------- types for /sindit/state ---------- */
interface SinditNode {
  uri?: string; nodeUri?: string; label?: string
  type?: string; nodeType?: string; node_type?: string
  [k: string]: any
}
interface SinditState {
  assets: SinditNode[]; properties: SinditNode[]
  relationships: any[]; node_types: any[]; total_nodes: number
}
interface BridgeStatus {
  available: boolean; running: boolean; detail?: string
  events_received?: number; values_pushed?: number; errors?: number
  last_push_at?: string | null; asset_uri?: string; sensor_fields_tracked?: number
}
interface ToolAuditMaster {
  tool_id?: string | null
  description?: string | null
  tool_type?: string | null
  diameter_mm?: number | null
  teeth?: number | null
  tool_length_mm?: number | null
  tool_material?: string | null
  source_workbook?: string | null
  machine_ids?: string[]
}
interface ToolAuditReferenceDimension {
  tool_label?: string | null
  linked_tool_numbers?: number[]
  arbour_diameter_mm?: number | null
  arbour_length_mm?: number | null
  shaft_length_text?: string | null
  arbour_id_number?: string | null
  head_diameter_mm?: number | null
  head_length_mm?: number | null
  head_length_text?: string | null
  head_id_number?: string | null
  overall_length_mm?: number | null
  tool_weight_kg?: number | null
  tool_weight_text?: string | null
}
interface ToolAuditReference {
  tool_number?: number
  tool_label?: string | null
  linked_tool_numbers?: number[]
  description?: string | null
  drawing_required?: boolean | null
  operations?: Array<{ operation_id?: string | null; title?: string | null }>
  reference_lines?: string[]
  notes?: string[]
  dimensions?: ToolAuditReferenceDimension | null
  sources?: string[]
}
interface ToolAuditProcessPlanEntry {
  use_case_id?: number
  use_case_title?: string | null
  setup?: string | null
  operation_id?: string | null
  head?: string | null
  op_type?: string | null
  description?: string | null
  tool_raw?: string | null
}
interface ToolAuditProcessPlan {
  use_case_ids?: number[]
  use_case_titles?: string[]
  operation_ids?: string[]
  setups?: string[]
  entries?: ToolAuditProcessPlanEntry[]
}
interface ToolAuditSindit {
  asset_uri?: string | null
  label?: string | null
  tool_diameter?: number | null
  num_teeth?: number | null
  tool_type?: string | null
  tool_length?: number | null
  tool_material?: string | null
  last_imported_at?: string | null
  source_workbook?: string | null
  machine_uris?: string[]
  asset_count?: number
  properties?: Record<string, any>
}
interface ToolAuditRuntime {
  session_ids?: string[]
  machine_ids?: string[]
  tool_id?: string | null
  tool_uri?: string | null
  seen_count?: number
  first_seen_at?: string | null
  last_seen_at?: string | null
  effective_ctx?: Record<string, any>
}
interface ToolAuditRow {
  machine_family: string
  tool_number: number
  tool_uri?: string | null
  master?: ToolAuditMaster | null
  sindit?: ToolAuditSindit | null
  runtime?: ToolAuditRuntime | null
  reference?: ToolAuditReference | null
  process_plan?: ToolAuditProcessPlan | null
  flags: string[]
  harmonic_ready: boolean
  sindit_available?: boolean
}
interface ToolAuditSummary {
  sindit_available: boolean
  tools_seen: number
  discrepancies: number
  harmonic_ready: number
  missing_diameter: number
  missing_teeth: number
  missing_sindit_asset: number
  family_resolution_miss: number
  total?: number
}
interface ToolAuditListResponse {
  sindit_available: boolean
  total: number
  items: ToolAuditRow[]
  detail?: string
}
type ToolDatasetDecisionStatus = 'pending' | 'confirmed' | 'rejected'
type ToolDatasetProfileMode = 'default' | 'master' | 'reference' | 'runtime' | 'sindit' | 'manual'
type ToolDatasetScopeFilter = 'all' | 'discrepant' | 'clean' | 'harmonic_ready' | 'harmonics_blocked'
type ToolDatasetCertaintyFilter = 'all' | ToolDatasetOverviewTool['certainty']
type ToolDatasetDecisionFilter = 'all' | ToolDatasetDecisionStatus

interface ToolDatasetProfileField {
  value?: string | number | null
  source?: string | null
}

interface ToolDatasetProfile {
  label: string
  available: boolean
  diameter_mm?: ToolDatasetProfileField
  teeth?: ToolDatasetProfileField
  tool_type?: ToolDatasetProfileField
  tool_length_mm?: ToolDatasetProfileField
  description?: ToolDatasetProfileField
  notes?: string | null
  reference_tool_number?: number | null
}

interface ToolDatasetDecision {
  dataset_id: string
  machine_family: string
  tool_number: number
  selection_mode: ToolDatasetProfileMode
  status: ToolDatasetDecisionStatus
  reference_tool_number?: number | null
  updated_at?: string | null
  updated_by?: string | null
  notes?: string | null
}

interface ToolDatasetPartSummary {
  machine_id?: string | null
  operation_id?: string | null
  label: string
  valid_rows: number
  resolved_dz_rows: number
  harmonic_ready_rows: number
  observed_tools: number[]
  resolved_dz_tools: number[]
  harmonic_ready_tools: number[]
  resolved_dz_row_pct: number
  harmonic_ready_row_pct: number
  resolved_dz_tool_pct: number
  harmonic_ready_tool_pct: number
}

interface ToolDatasetHarmonicSummary {
  observed_tools: number
  resolved_dz_tools: number
  harmonic_ready_tools: number
  resolved_dz_tool_pct: number
  harmonic_ready_tool_pct: number
  valid_rows: number
  resolved_dz_rows: number
  harmonic_ready_rows: number
  resolved_dz_row_pct: number
  harmonic_ready_row_pct: number
  ready_parts: number
  total_parts: number
}

interface ToolDatasetOverviewTool {
  dataset_id: string
  dataset_label: string
  machine_family: string
  tool_number: number
  machine_ids: string[]
  operation_ids: string[]
  operation_count: number
  coverage: {
    observed: boolean
    master: boolean
    diameter: boolean
    teeth: boolean
    diameter_and_teeth: boolean
    harmonic_ready: boolean
  }
  profiles: Record<string, ToolDatasetProfile>
  available_profiles: ToolDatasetProfileMode[]
  recommended_profile: ToolDatasetProfileMode
  selected_profile: ToolDatasetProfileMode
  decision?: ToolDatasetDecision | null
  decision_status: ToolDatasetDecisionStatus
  certainty: 'certain' | 'defaulted' | 'needs_review'
  certainty_reasons: string[]
  review_flags: string[]
  evidence_sources: string[]
  audit: ToolAuditRow
}

interface ToolDatasetOverviewDataset {
  dataset_id: string
  label: string
  machine_ids: string[]
  machine_families: string[]
  shared_workpiece?: boolean
  workpiece_note?: string | null
  operation_count: number
  harmonic_summary?: ToolDatasetHarmonicSummary
  part_summaries?: ToolDatasetPartSummary[]
  summary: {
    tool_count: number
    certain_count: number
    defaulted_count: number
    needs_review_count: number
    confirmed_count: number
    rejected_count: number
    pending_count: number
    master_backed_count: number
  }
  tools: ToolDatasetOverviewTool[]
}

interface ToolDatasetOverviewResponse {
  datasets: ToolDatasetOverviewDataset[]
  total_datasets: number
  total_tools: number
  sindit_available: boolean
  detail?: string
}

interface ToolDatasetDecisionResponse {
  ok: boolean
  decision: ToolDatasetDecision
}

const TOOL_DATASET_PROFILE_LABELS: Record<ToolDatasetProfileMode, string> = {
  default: 'Default',
  master: 'Master',
  reference: 'Reference',
  runtime: 'Runtime',
  sindit: 'SINDIT',
  manual: 'Manual',
}

const TOOL_DATASET_DISCREPANCY_LABELS: Record<string, string> = {
  missing_master_spec: 'The workbook master does not contain a canonical spec for this tool.',
  missing_tool_diameter: 'Diameter is still missing across the available evidence.',
  missing_num_teeth: 'Tooth count is still missing across the available evidence.',
  missing_tool_type: 'Tool type is still missing across the available evidence.',
  diameter_mismatch_mm: 'Sources disagree on the tool diameter.',
  teeth_mismatch: 'Sources disagree on the tooth count.',
  tool_type_mismatch: 'Sources disagree on the tool type.',
  tool_length_mismatch_mm: 'Sources disagree on the tool length.',
  duplicate_tool_assets: 'Multiple SINDIT assets match the same tool number.',
}

const TOOL_SELECTOR_STYLE: React.CSSProperties = {
  background: 'rgba(0,0,0,0.18)',
  color: 'var(--fg)',
  border: '1px solid var(--border)',
  borderRadius: 6,
  padding: '8px 10px',
}

function buildToolAuditQuery(params: {
  sessionId?: string
  machineId?: string
  family?: string
  toolNumber?: string
  onlyDiscrepancies?: boolean
}): string {
  const qs = new URLSearchParams()
  if (params.sessionId?.trim()) qs.set('session_id', params.sessionId.trim())
  if (params.machineId?.trim()) qs.set('machine_id', params.machineId.trim())
  if (params.family?.trim()) qs.set('family', params.family.trim())
  if (params.toolNumber?.trim()) {
    const match = params.toolNumber.trim().match(/\d+/)
    if (match) qs.set('tool_number', match[0])
  }
  if (params.onlyDiscrepancies) qs.set('only_discrepancies', 'true')
  const built = qs.toString()
  return built ? `?${built}` : ''
}

function toolAuditRowKey(row: ToolAuditRow): string {
  return `${row.machine_family}:${row.tool_number}`
}

function toolAuditNumber(...values: Array<number | null | undefined>): string {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(3).replace(/\.000$/, '')
  }
  return '—'
}

function toolAuditText(...values: Array<string | null | undefined>): string {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return '—'
}

function toolAuditBool(value: boolean | null | undefined): string {
  if (value === true) return 'Yes'
  if (value === false) return 'No'
  return '—'
}

function toolAuditDelta(row: ToolAuditRow, field: 'diameter' | 'teeth' | 'length'): string | null {
  if (field === 'diameter') {
    const left = row.master?.diameter_mm
    const right = row.sindit?.tool_diameter
    if (typeof left === 'number' && typeof right === 'number') {
      const delta = right - left
      if (Math.abs(delta) > 1e-3) return `Δd ${delta > 0 ? '+' : ''}${delta.toFixed(3)} mm`
    }
    return null
  }
  if (field === 'teeth') {
    const left = row.master?.teeth
    const right = row.sindit?.num_teeth
    if (typeof left === 'number' && typeof right === 'number' && left !== right) {
      return `Δz ${right > left ? '+' : ''}${right - left}`
    }
    return null
  }
  const left = row.master?.tool_length_mm
  const right = row.sindit?.tool_length
  if (typeof left === 'number' && typeof right === 'number') {
    const delta = right - left
    if (Math.abs(delta) > 1e-3) return `ΔL ${delta > 0 ? '+' : ''}${delta.toFixed(3)} mm`
  }
  return null
}

function toolDatasetRowKey(row: ToolDatasetOverviewTool): string {
  return `${row.dataset_id}:${row.machine_family}:${row.tool_number}`
}

function toolDatasetProfileFor(row: ToolDatasetOverviewTool, mode: ToolDatasetProfileMode): ToolDatasetProfile {
  return row.profiles[mode] || row.profiles.default
}

function toolDatasetDiscrepancyFlags(row?: ToolDatasetOverviewTool | null): string[] {
  if (!row) return []
  return Array.from(new Set(row.review_flags.filter(Boolean)))
}

function toolDatasetDiscrepancyMeaning(flag: string): string {
  return TOOL_DATASET_DISCREPANCY_LABELS[flag] || flag.replace(/_/g, ' ')
}

function toolDatasetDiscrepancyCount(dataset?: ToolDatasetOverviewDataset | null): number {
  if (!dataset) return 0
  return dataset.tools.filter(row => toolDatasetDiscrepancyFlags(row).length > 0).length
}

function toolDatasetDiscrepancyBreakdown(dataset?: ToolDatasetOverviewDataset | null): Array<{ flag: string; count: number }> {
  if (!dataset) return []
  const counts = new Map<string, number>()
  for (const row of dataset.tools) {
    for (const flag of toolDatasetDiscrepancyFlags(row)) {
      counts.set(flag, (counts.get(flag) || 0) + 1)
    }
  }
  return Array.from(counts.entries())
    .map(([flag, count]) => ({ flag, count }))
    .sort((left, right) => right.count - left.count || left.flag.localeCompare(right.flag))
}

function toolDatasetPrimaryNote(row: ToolDatasetOverviewTool): string | null {
  if (row.decision_status === 'confirmed' && row.decision?.notes?.trim()) return row.decision.notes.trim()
  const discrepancy = toolDatasetDiscrepancyFlags(row)[0]
  if (discrepancy) return toolDatasetDiscrepancyMeaning(discrepancy)
  return row.certainty_reasons[0] || null
}

function toolDatasetMatchesScope(row: ToolDatasetOverviewTool, scope: ToolDatasetScopeFilter): boolean {
  if (scope === 'discrepant') return toolDatasetDiscrepancyFlags(row).length > 0
  if (scope === 'clean') return toolDatasetDiscrepancyFlags(row).length === 0
  if (scope === 'harmonic_ready') return row.coverage.harmonic_ready
  if (scope === 'harmonics_blocked') return !row.coverage.harmonic_ready
  return true
}

function toolDatasetFieldNumber(field?: ToolDatasetProfileField | null): string {
  const value = field?.value
  if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(3).replace(/\.000$/, '')
  return '—'
}

function toolDatasetFieldText(field?: ToolDatasetProfileField | null): string {
  const value = field?.value
  if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(3).replace(/\.000$/, '')
  if (typeof value === 'string' && value.trim()) return value.trim()
  return '—'
}

function toolDatasetCertaintyTone(certainty: ToolDatasetOverviewTool['certainty']): 'ok' | 'warn' | 'danger' {
  if (certainty === 'certain') return 'ok'
  if (certainty === 'defaulted') return 'warn'
  return 'danger'
}

function toolDatasetDecisionTone(status: ToolDatasetDecisionStatus): 'neutral' | 'ok' | 'danger' {
  if (status === 'confirmed') return 'ok'
  if (status === 'rejected') return 'danger'
  return 'neutral'
}

function toolDatasetHarmonicReadyToolNumbers(dataset?: ToolDatasetOverviewDataset | null): number[] {
  if (!dataset) return []
  return dataset.tools
    .filter(tool => tool.coverage.harmonic_ready)
    .map(tool => tool.tool_number)
    .sort((left, right) => left - right)
}

function toolDatasetMachineSummary(dataset?: ToolDatasetOverviewDataset | null): string {
  if (!dataset) return 'no machines'
  const machineCount = dataset.machine_ids.length
  if (dataset.shared_workpiece && machineCount > 0) return `${machineCount} machines · shared workpiece`
  if (machineCount > 0) return `${machineCount} machine${machineCount === 1 ? '' : 's'}`
  return 'no machines'
}

function toolDatasetPercent(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0.00%'
  return `${value.toFixed(2)}%`
}

function toolDatasetNumber(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0'
  return value.toLocaleString()
}

function toolDatasetPartSummaries(dataset?: ToolDatasetOverviewDataset | null): ToolDatasetPartSummary[] {
  return dataset?.part_summaries || []
}

type ToolDataSourceGuide = {
  key: string
  label: string
  documents: string[]
  originDocuments?: string[]
  why: string
  how: string
}

function uniqueToolDatasetStrings(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map(value => value?.trim()).filter((value): value is string => Boolean(value))))
}

function toolDatasetMasterDocuments(dataset?: ToolDatasetOverviewDataset | null, row?: ToolDatasetOverviewTool | null): string[] {
  if (row?.audit.master?.source_workbook) return [row.audit.master.source_workbook]
  return uniqueToolDatasetStrings((dataset?.tools || []).map(tool => tool.audit.master?.source_workbook))
}

function toolDatasetSinditDocuments(dataset?: ToolDatasetOverviewDataset | null, row?: ToolDatasetOverviewTool | null): string[] {
  const values = row
    ? [row.audit.sindit?.source_workbook, row.audit.sindit?.asset_uri]
    : (dataset?.tools || []).flatMap(tool => [tool.audit.sindit?.source_workbook, tool.audit.sindit?.asset_uri])
  return uniqueToolDatasetStrings(values)
}

function toolDatasetReferenceOrigins(dataset?: ToolDatasetOverviewDataset | null, row?: ToolDatasetOverviewTool | null): string[] {
  if (row?.audit.reference?.sources?.length) return uniqueToolDatasetStrings(row.audit.reference.sources)
  return uniqueToolDatasetStrings((dataset?.tools || []).flatMap(tool => tool.audit.reference?.sources || []))
}

function toolDatasetSourceGuide(source: string, dataset?: ToolDatasetOverviewDataset | null, row?: ToolDatasetOverviewTool | null, profile?: ToolDatasetProfile | null): ToolDataSourceGuide | null {
  if (source === 'master') {
    const documents = toolDatasetMasterDocuments(dataset, row)
    return {
      key: 'master',
      label: 'Workbook master',
      documents: documents.length > 0 ? documents : ['Workbook path not recorded'],
      why: 'Primary source for tool geometry. This is the first place diameter, teeth, type, and length are taken from when the workbook covers the tool.',
      how: 'Parsed directly from the source workbook into the in-process tool master by backend/agents/processing/tool_lookup.py.',
    }
  }
  if (source === 'reference') {
    return {
      key: 'reference',
      label: 'Critical reference',
      documents: ['data/tools/site_b/critical_tool_reference.json'],
      originDocuments: toolDatasetReferenceOrigins(dataset, row).length > 0
        ? toolDatasetReferenceOrigins(dataset, row)
        : ['data/tools/site_b/Critical tool list.docx', 'data/tools/site_b/Critical Tool List Dimensions.xlsx'],
      why: 'Static secondary evidence used to confirm critical Site_b tools and fill dimension details when the workbook master is incomplete. This tools tab is showing provenance, not live document-graph retrieval results.',
      how: 'scripts/extract_site_b_critical_tool_reference.py normalizes DOCX text blocks and the dimensions workbook into critical_tool_reference.json, which is then loaded by tool_reference_catalog.py.',
    }
  }
  if (source === 'process_plan') {
    return {
      key: 'process_plan',
      label: 'Process-plan mapping',
      documents: ['data/tools/use_case_operation_sequences.json'],
      originDocuments: ['data/tools/UseCasesOperationSequence v2.pptx'],
      why: 'Connects tool numbers to use cases, setups, and operation IDs so the operator can see where each tool should appear in the process.',
      how: 'scripts/extract_use_case_operation_sequence.py parses the PPTX slide XML, extracts table rows, and groups them by use case and tool number.',
    }
  }
  if (source === 'runtime') {
    const sessionIds = row?.audit.runtime?.session_ids || []
    return {
      key: 'runtime',
      label: 'Observed runtime context',
      documents: sessionIds.length > 0 ? sessionIds.map(sessionId => `session:${sessionId}`) : ['Observed cutting-context windows'],
      why: 'Shows what the machine actually reported while running. This is useful as a confirmation layer when workbook or reference data is ambiguous.',
      how: 'record_tool_observation() aggregates cutting-context observations from runtime/session processing into the audit snapshot.',
    }
  }
  if (source === 'sindit') {
    const documents = toolDatasetSinditDocuments(dataset, row)
    return {
      key: 'sindit',
      label: 'SINDIT graph asset',
      documents: documents.length > 0 ? documents : ['SINDIT graph tool asset'],
      why: 'Graph-backed copy of the tool data, useful for cross-system review and for runtime enrichment when SINDIT is enabled.',
      how: 'backend/agents/sindit/import_tool_master.py imports tool-master assets into SINDIT, and the audit reads them back from the graph API.',
    }
  }
  if (source === 'guess') {
    return {
      key: 'guess',
      label: 'Best-guess override',
      documents: ['data/tools/dataset_tool_decisions.json'],
      originDocuments: [
        ...toolDatasetMasterDocuments(dataset, row),
        ...toolDatasetReferenceOrigins(dataset, row),
      ].filter(Boolean),
      why: profile?.notes?.trim() || row?.decision?.notes?.trim() || 'A dataset-scoped best guess is filling the unresolved field until an operator confirms the exact value.',
      how: 'A confirmed dataset snapshot is stored in data/tools/dataset_tool_decisions.json and then applied by resolve_confirmed_tool_context() during runtime enrichment.',
    }
  }
  if (source === 'manual') {
    const referenceTool = profile?.reference_tool_number ?? row?.decision?.reference_tool_number
    return {
      key: 'manual',
      label: 'Manual operator input',
      documents: referenceTool ? [`Reference tool T${referenceTool}`] : ['Manual tool override'],
      originDocuments: ['data/tools/dataset_tool_decisions.json'],
      why: profile?.notes?.trim() || row?.decision?.notes?.trim() || 'An operator supplied a corrected tooth count or copied the default profile from another tool number.',
      how: 'The Dataset Tools view posts a confirmed manual snapshot to /sindit/tools/datasets/decision, and that snapshot is then used by resolve_confirmed_tool_context() the same way as other resolved tool profiles.',
    }
  }
  return null
}

function toolDatasetSourceGuides(dataset?: ToolDatasetOverviewDataset | null): ToolDataSourceGuide[] {
  if (!dataset) return []
  const availableSources = uniqueToolDatasetStrings(dataset.tools.flatMap(tool => tool.evidence_sources))
  return availableSources
    .map(source => toolDatasetSourceGuide(source, dataset, null, null))
    .filter((guide): guide is ToolDataSourceGuide => guide !== null)
}

function toolDatasetSelectedSourceGuides(row?: ToolDatasetOverviewTool | null, mode?: ToolDatasetProfileMode): ToolDataSourceGuide[] {
  if (!row || !mode) return []
  const profile = toolDatasetProfileFor(row, mode)
  const sources = uniqueToolDatasetStrings([
    profile.description?.source,
    profile.diameter_mm?.source,
    profile.teeth?.source,
    profile.tool_type?.source,
    profile.tool_length_mm?.source,
  ])
  return sources
    .map(source => toolDatasetSourceGuide(source, null, row, profile))
    .filter((guide): guide is ToolDataSourceGuide => guide !== null)
}

function toolDatasetRemedyActions(dataset?: ToolDatasetOverviewDataset | null): string[] {
  if (!dataset) return []
  const missingGeometryTools = dataset.tools
    .filter(tool => !tool.coverage.diameter_and_teeth)
    .map(tool => `T${tool.tool_number}`)
  const blockedParts = toolDatasetPartSummaries(dataset)
    .filter(part => part.harmonic_ready_rows <= 0)
    .map(part => part.label)

  const actions: string[] = []
  if (missingGeometryTools.length > 0) {
    actions.push(`Complete diameter and tooth count for unresolved tools: ${missingGeometryTools.join(', ')}.`)
  }
  if (blockedParts.length > 0) {
    actions.push(`Review parts with no harmonic-ready rows: ${blockedParts.join(', ')}.`)
  }

  if (dataset.dataset_id === 'site_b_casedata' || dataset.dataset_id === 'site_b_olddata') {
    actions.push('Start with the Builder_b1 2 workbook master and the reviewed Site_b tool list. The critical-tool DOCX/XLSX entries shown here are static secondary references in the tools audit, not live document retrieval results.')
    actions.push('If a Site_b tool is still unresolved after the documents, ask the operator or tool crib to confirm the pocket-to-tool specification and then record the confirmed choice in this tab.')
  } else if (dataset.dataset_id === 'site_c_casedata') {
    actions.push('SITE_C is blocked mainly by missing tooth counts. The Press_c spreadsheets give diameter and length, but not z, so vendor catalogs or operator annotations are needed for the observed tool numbers.')
    actions.push('Once tooth counts are collected, update the tool master and re-import SINDIT so the harmonic-ready list can move from 0%.')
  } else if (dataset.dataset_id === 'site_a_line2') {
    actions.push('Use Machine_a1.xlsx for tool identity and geometry, and use the DLG6CF CNC_parameters_teeth_num stream to confirm tooth counts where available.')
    actions.push('For remaining Site_a_line2 gaps, collect the OEM tool list or operator confirmation for tool numbers that still lack stable teeth information, then confirm the chosen profile in this tab.')
  }

  actions.push('When a value is trustworthy but not yet canonical, confirm the selected profile here so runtime enrichment uses the resolved snapshot consistently.')
  return actions
}

export function SinditView() {
  // Default to the live asset/twin view rather than the explanatory overview,
  // so the Digital Twin page shows the actual twin (assets + live values) first.
  const [subTab, setSubTab] = useState<'overview' | 'experiments' | 'state' | 'tools' | 'sindit-api' | 'graphdb'>('state')
  const [toolView, setToolView] = useState<'audit' | 'datasets'>('audit')
  const [expandedAssets, setExpandedAssets] = useState<Set<string>>(new Set())
  const [toolSessionFilter, setToolSessionFilter] = useState('')
  const [toolMachineFilter, setToolMachineFilter] = useState('')
  const [toolFamilyFilter, setToolFamilyFilter] = useState('')
  const [toolNumberFilter, setToolNumberFilter] = useState('')
  const [onlyToolDiscrepancies, setOnlyToolDiscrepancies] = useState(false)
  const [selectedToolKey, setSelectedToolKey] = useState<string | null>(null)
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [selectedDatasetToolKey, setSelectedDatasetToolKey] = useState<string | null>(null)
  const [datasetToolScopeFilter, setDatasetToolScopeFilter] = useState<ToolDatasetScopeFilter>('all')
  const [datasetToolCertaintyFilter, setDatasetToolCertaintyFilter] = useState<ToolDatasetCertaintyFilter>('all')
  const [datasetToolDecisionFilter, setDatasetToolDecisionFilter] = useState<ToolDatasetDecisionFilter>('all')
  const [toolDatasetSelections, setToolDatasetSelections] = useState<Record<string, ToolDatasetProfileMode>>({})
  const [toolDatasetManualDrafts, setToolDatasetManualDrafts] = useState<Record<string, { referenceToolNumber: string; teeth: string; notes: string }>>({})
  const queryClient = useQueryClient()

  // Health checks via backend proxy (avoids CORS issues with direct browser→GraphDB/SINDIT)
  const sinditHealthProxyQ = useQuery<{ sindit: boolean; graphdb: boolean }>({
    queryKey: ['sindit-health-proxy'],
    queryFn: async (): Promise<{ sindit: boolean; graphdb: boolean }> => {
      try {
        return await api('/health/sindit') as { sindit: boolean; graphdb: boolean }
      } catch {
        return { sindit: false, graphdb: false }
      }
    },
    refetchInterval: 10000,
    retry: 0,
  })

  // Check backend SINDIT config
  const configQ = useQuery<any>({
    queryKey: ['sindit-config-check'],
    queryFn: () => api('/agent/memory/config').catch(() => null),
    retry: 0,
    staleTime: 30000,
  })

  const sinditOk = sinditHealthProxyQ.data?.sindit ?? null
  const graphdbOk = sinditHealthProxyQ.data?.graphdb ?? null

  // SINDIT KG current state (only fetch when tab is active)
  const stateQ = useQuery<SinditState>({
    queryKey: ['sindit-state'],
    queryFn: () => api('/sindit/state') as Promise<SinditState>,
    enabled: subTab === 'state' && sinditOk === true,
    refetchInterval: subTab === 'state' ? 5000 : false,
    retry: 0,
  })

  // Bridge status
  const bridgeQ = useQuery<BridgeStatus>({
    queryKey: ['sindit-bridge-status'],
    queryFn: () => api('/sindit/bridge/status') as Promise<BridgeStatus>,
    refetchInterval: 3000,
    retry: 0,
  })

  const startBridge = useMutation({
    mutationFn: () => api('/sindit/bridge/start', 'POST'),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['sindit-bridge-status'] }) },
  })
  const stopBridge = useMutation({
    mutationFn: () => api('/sindit/bridge/stop', 'POST'),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['sindit-bridge-status'] }) },
  })

  const toggleAsset = (uri: string) => {
    setExpandedAssets(prev => {
      const next = new Set(prev)
      if (next.has(uri)) next.delete(uri); else next.add(uri)
      return next
    })
  }

  // Experiment graph from SINDIT (uses top-level GNode / GEdge types)
  type ExpGraphData = { nodes: GNode[]; edges: GEdge[]; experiments: GNode[]; count: number }
  const experimentsQ = useQuery<ExpGraphData>({
    queryKey: ['sindit-experiments'],
    queryFn: () => api('/sindit/experiments') as Promise<ExpGraphData>,
    enabled: subTab === 'experiments' && sinditOk === true,
    refetchInterval: subTab === 'experiments' ? 10000 : false,
    retry: 0,
  })

  const toolAuditQuery = buildToolAuditQuery({
    sessionId: toolSessionFilter,
    machineId: toolMachineFilter,
    family: toolFamilyFilter,
    toolNumber: toolNumberFilter,
    onlyDiscrepancies: onlyToolDiscrepancies,
  })

  const toolAuditSummaryQ = useQuery<ToolAuditSummary>({
    queryKey: ['sindit-tools-summary', toolAuditQuery],
    queryFn: () => api(`/sindit/tools/summary${toolAuditQuery}`) as Promise<ToolAuditSummary>,
    enabled: subTab === 'tools' && toolView === 'audit',
    refetchInterval: subTab === 'tools' ? 10000 : false,
    retry: 0,
  })

  const toolAuditRowsQ = useQuery<ToolAuditListResponse>({
    queryKey: ['sindit-tools', toolAuditQuery],
    queryFn: () => api(`/sindit/tools${toolAuditQuery}`) as Promise<ToolAuditListResponse>,
    enabled: subTab === 'tools' && toolView === 'audit',
    refetchInterval: subTab === 'tools' ? 10000 : false,
    retry: 0,
  })

  const toolDatasetsQ = useQuery<ToolDatasetOverviewResponse>({
    queryKey: ['sindit-tool-datasets'],
    queryFn: () => api('/sindit/tools/datasets') as Promise<ToolDatasetOverviewResponse>,
    enabled: subTab === 'tools' && toolView === 'datasets',
    refetchInterval: subTab === 'tools' && toolView === 'datasets' ? 30000 : false,
    retry: 0,
  })

  const toolDatasetDecisionM = useMutation({
    mutationFn: (body: {
      dataset_id: string
      machine_family: string
      tool_number: number
      status: ToolDatasetDecisionStatus
      selection_mode: ToolDatasetProfileMode
      reference_tool_number?: number
      manual_num_teeth?: number
      updated_by?: string | null
      notes?: string | null
    }) => api('/sindit/tools/datasets/decision', 'POST', body) as Promise<ToolDatasetDecisionResponse>,
    onSuccess: (_data, variables) => {
      const key = `${variables.dataset_id}:${variables.machine_family}:${variables.tool_number}`
      setToolDatasetSelections(prev => {
        const next = { ...prev }
        delete next[key]
        return next
      })
      setToolDatasetManualDrafts(prev => {
        const next = { ...prev }
        delete next[key]
        return next
      })
      queryClient.invalidateQueries({ queryKey: ['sindit-tool-datasets'] })
    },
  })

  const toolRows = toolAuditRowsQ.data?.items ?? []
  const selectedTool = selectedToolKey ? toolRows.find(row => toolAuditRowKey(row) === selectedToolKey) ?? null : null
  const toolDatasets = toolDatasetsQ.data?.datasets ?? []
  const selectedDataset = selectedDatasetId ? toolDatasets.find(dataset => dataset.dataset_id === selectedDatasetId) ?? null : toolDatasets[0] ?? null
  const selectedDatasetTools = selectedDataset?.tools ?? []
  const filteredDatasetTools = selectedDatasetTools.filter(row => {
    if (!toolDatasetMatchesScope(row, datasetToolScopeFilter)) return false
    if (datasetToolCertaintyFilter !== 'all' && row.certainty !== datasetToolCertaintyFilter) return false
    if (datasetToolDecisionFilter !== 'all' && row.decision_status !== datasetToolDecisionFilter) return false
    return true
  })
  const selectedDatasetTool = selectedDatasetToolKey ? filteredDatasetTools.find(row => toolDatasetRowKey(row) === selectedDatasetToolKey) ?? null : null
  const selectedDatasetHarmonicReadyTools = toolDatasetHarmonicReadyToolNumbers(selectedDataset)
  const selectedDatasetDiscrepancyCount = toolDatasetDiscrepancyCount(selectedDataset)
  const selectedDatasetDiscrepancyBreakdown = toolDatasetDiscrepancyBreakdown(selectedDataset)
  const selectedDatasetToolDiscrepancyFlags = toolDatasetDiscrepancyFlags(selectedDatasetTool)
  const selectedDatasetPartSummaries = toolDatasetPartSummaries(selectedDataset)
  const selectedDatasetSourceGuides = toolDatasetSourceGuides(selectedDataset)
  const selectedDatasetRemedies = toolDatasetRemedyActions(selectedDataset)

  useEffect(() => {
    if (selectedToolKey && !toolRows.some(row => toolAuditRowKey(row) === selectedToolKey)) {
      setSelectedToolKey(null)
    }
  }, [selectedToolKey, toolRows])

  useEffect(() => {
    if (!toolDatasets.length) {
      if (selectedDatasetId !== null) setSelectedDatasetId(null)
      return
    }
    if (!selectedDatasetId || !toolDatasets.some(dataset => dataset.dataset_id === selectedDatasetId)) {
      setSelectedDatasetId(toolDatasets[0].dataset_id)
    }
  }, [selectedDatasetId, toolDatasets])

  useEffect(() => {
    if (!filteredDatasetTools.length) {
      if (selectedDatasetToolKey !== null) setSelectedDatasetToolKey(null)
      return
    }
    if (!selectedDatasetToolKey || !filteredDatasetTools.some(row => toolDatasetRowKey(row) === selectedDatasetToolKey)) {
      setSelectedDatasetToolKey(toolDatasetRowKey(filteredDatasetTools[0]))
    }
  }, [filteredDatasetTools, selectedDatasetToolKey])

  const currentDatasetProfileMode = useCallback((row: ToolDatasetOverviewTool): ToolDatasetProfileMode => {
    return toolDatasetSelections[toolDatasetRowKey(row)] ?? row.selected_profile
  }, [toolDatasetSelections])

  const manualDraftFor = useCallback((row: ToolDatasetOverviewTool) => {
    const key = toolDatasetRowKey(row)
    const savedReference = row.decision_status === 'confirmed' && row.decision?.selection_mode === 'manual' && typeof row.decision?.reference_tool_number === 'number'
      ? String(row.decision.reference_tool_number)
      : ''
    const savedTeeth = row.decision_status === 'confirmed' && row.decision?.selection_mode === 'manual' && row.profiles.manual?.teeth?.source === 'manual' && typeof row.profiles.manual?.teeth?.value === 'number'
      ? String(row.profiles.manual.teeth.value)
      : ''
    const savedNotes = row.decision_status === 'confirmed' && row.decision?.selection_mode === 'manual'
      ? (row.profiles.manual?.notes || row.decision?.notes || '')
      : ''
    return toolDatasetManualDrafts[key] || {
      referenceToolNumber: savedReference,
      teeth: savedTeeth,
      notes: savedNotes,
    }
  }, [toolDatasetManualDrafts])

  const resolvedDatasetProfile = useCallback((row: ToolDatasetOverviewTool, mode: ToolDatasetProfileMode): ToolDatasetProfile => {
    if (mode !== 'manual') return toolDatasetProfileFor(row, mode)

    const draft = manualDraftFor(row)
    const referenceToolNumber = draft.referenceToolNumber.trim().match(/^\d+$/) ? Number(draft.referenceToolNumber.trim()) : null
    const referenceRow = referenceToolNumber !== null
      ? selectedDatasetTools.find(candidate => candidate.machine_family === row.machine_family && candidate.tool_number === referenceToolNumber) || null
      : null
    const baseProfile = referenceRow ? toolDatasetProfileFor(referenceRow, 'default') : (row.profiles.manual || row.profiles.default)
    const profile: ToolDatasetProfile = {
      ...baseProfile,
      diameter_mm: baseProfile.diameter_mm ? { ...baseProfile.diameter_mm } : undefined,
      teeth: baseProfile.teeth ? { ...baseProfile.teeth } : undefined,
      tool_type: baseProfile.tool_type ? { ...baseProfile.tool_type } : undefined,
      tool_length_mm: baseProfile.tool_length_mm ? { ...baseProfile.tool_length_mm } : undefined,
      description: baseProfile.description ? { ...baseProfile.description } : undefined,
      label: TOOL_DATASET_PROFILE_LABELS.manual,
      available: true,
      notes: draft.notes.trim() || row.profiles.manual?.notes || null,
      reference_tool_number: referenceToolNumber,
    }
    const manualTeeth = draft.teeth.trim().match(/^\d+$/) ? Number(draft.teeth.trim()) : null
    if (manualTeeth !== null) {
      profile.teeth = { value: manualTeeth, source: 'manual' }
    }
    return profile
  }, [manualDraftFor, selectedDatasetTools])

  const buildToolDatasetDecisionBody = useCallback((row: ToolDatasetOverviewTool, status: ToolDatasetDecisionStatus) => {
    const selectionMode = currentDatasetProfileMode(row)
    const profile = resolvedDatasetProfile(row, selectionMode)
    const body: {
      dataset_id: string
      machine_family: string
      tool_number: number
      status: ToolDatasetDecisionStatus
      selection_mode: ToolDatasetProfileMode
      reference_tool_number?: number
      manual_num_teeth?: number
      notes?: string | null
    } = {
      dataset_id: row.dataset_id,
      machine_family: row.machine_family,
      tool_number: row.tool_number,
      status,
      selection_mode: selectionMode,
    }
    if (selectionMode === 'manual') {
      const draft = manualDraftFor(row)
      const referenceToolNumber = draft.referenceToolNumber.trim().match(/^\d+$/) ? Number(draft.referenceToolNumber.trim()) : null
      const manualTeeth = draft.teeth.trim().match(/^\d+$/) ? Number(draft.teeth.trim()) : null
      if (referenceToolNumber !== null) body.reference_tool_number = referenceToolNumber
      if (manualTeeth !== null) body.manual_num_teeth = manualTeeth
      if (draft.notes.trim()) body.notes = draft.notes.trim()
    }
    if (!body.notes && profile.notes?.trim()) body.notes = profile.notes.trim()
    return body
  }, [currentDatasetProfileMode, manualDraftFor, resolvedDatasetProfile])

  const selectedDatasetToolSourceGuides = selectedDatasetTool
    ? (() => {
        const mode = currentDatasetProfileMode(selectedDatasetTool)
        const profile = resolvedDatasetProfile(selectedDatasetTool, mode)
        const sources = uniqueToolDatasetStrings([
          profile.description?.source,
          profile.diameter_mm?.source,
          profile.teeth?.source,
          profile.tool_type?.source,
          profile.tool_length_mm?.source,
        ])
        return sources
          .map(source => toolDatasetSourceGuide(source, null, selectedDatasetTool, profile))
          .filter((guide): guide is ToolDataSourceGuide => guide !== null)
      })()
    : []

  return (
    <div style={{ padding: '12px 20px', display: 'grid', gap: 16, maxWidth: 1200 }}>
      <div>
        <h2 style={{ margin: 0 }}>🌐 SINDIT Digital Twin</h2>
        <p className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
          SINDIT is the digital twin knowledge graph platform — it models physical assets, their properties, connections, and live sensor values.
        </p>
      </div>

      {/* Sub-tabs */}
      <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {[
          {
            key: 'overview',
            label: '📊 Overview',
            active: subTab === 'overview',
            onClick: () => setSubTab('overview'),
          },
          {
            key: 'experiments',
            label: '🧪 Experiments',
            active: subTab === 'experiments',
            onClick: () => setSubTab('experiments'),
          },
          {
            key: 'state',
            label: '🔍 Current State',
            active: subTab === 'state',
            onClick: () => setSubTab('state'),
          },
          {
            key: 'tools-audit',
            label: '🧰 Tool Audit',
            active: subTab === 'tools' && toolView === 'audit',
            onClick: () => {
              setSubTab('tools')
              setToolView('audit')
            },
          },
          {
            key: 'tools-datasets',
            label: '🗂️ Dataset Tools',
            active: subTab === 'tools' && toolView === 'datasets',
            onClick: () => {
              setSubTab('tools')
              setToolView('datasets')
            },
          },
          {
            key: 'sindit-api',
            label: '🔌 SINDIT API',
            active: subTab === 'sindit-api',
            onClick: () => setSubTab('sindit-api'),
          },
          {
            key: 'graphdb',
            label: '🗃️ GraphDB',
            active: subTab === 'graphdb',
            onClick: () => setSubTab('graphdb'),
          },
        ].map(tab => (
          <button
            key={tab.key}
            className="small"
            style={{
              padding: '6px 16px',
              borderRadius: 4,
              border: '1px solid var(--border)',
              cursor: 'pointer',
              background: tab.active ? 'var(--accent)' : 'transparent',
              color: tab.active ? '#fff' : 'var(--fg)',
              fontWeight: tab.active ? 700 : 400,
            }}
            onClick={tab.onClick}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {subTab === 'overview' && (
        <>
          {/* Service status */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>Service Status</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
              <StatusBadge
                ok={sinditOk}
                label={sinditOk ? 'SINDIT API — Connected' : sinditOk === false ? 'SINDIT API — Unreachable' : 'SINDIT API — Checking…'}
                detail={`${SINDIT_URL}`}
              />
              <StatusBadge
                ok={graphdbOk}
                label={graphdbOk ? 'GraphDB — Connected' : graphdbOk === false ? 'GraphDB — Unreachable' : 'GraphDB — Checking…'}
                detail={`${GRAPHDB_URL}`}
              />
              <StatusBadge
                ok={sinditOk !== null ? sinditOk : null}
                label={sinditOk ? 'Context Enrichment — Active' : 'Context Enrichment — Inactive'}
                detail="Events enriched with machine state, spindle speed, feed rate, etc."
              />
            </div>
          </div>

          {/* How it works */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>How SINDIT Context Enrichment Works</div>
            <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
              <p style={{ margin: '0 0 8px' }}>
                When <code>SINDIT_ENABLED=true</code> is set in <code>.env</code>, the backend creates a <strong>SinditContextProvider</strong> at startup.
                For every CNC event processed through the memory pipeline, SINDIT enriches it with:
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8, margin: '8px 0' }}>
                {[
                  { icon: '⚙️', label: 'Spindle Speed', desc: 'RPM from live sensor feed' },
                  { icon: '📏', label: 'Feed Rate', desc: 'mm/min from axis controller' },
                  { icon: '🔧', label: 'Tool ID & Type', desc: 'Current tool from magazine' },
                  { icon: '📊', label: 'Feed Override', desc: 'Operator override percentage' },
                  { icon: '⚡', label: 'Power Level', desc: 'Spindle power consumption' },
                  { icon: '🏭', label: 'Machine State', desc: 'normal / degraded classification' },
                ].map(({ icon, label, desc }) => (
                  <div key={label} style={{ padding: '8px 12px', background: 'rgba(122,162,247,0.06)', borderRadius: 6, border: '1px solid rgba(122,162,247,0.12)' }}>
                    <div style={{ fontWeight: 600, fontSize: 12 }}>{icon} {label}</div>
                    <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>{desc}</div>
                  </div>
                ))}
              </div>
              <p style={{ margin: '8px 0 0' }}>
                When SINDIT is not live, the experiment evaluator <strong>simulates</strong> these fields from the CNC sensor CSV columns,
                so the analysis pipeline works identically in both modes.
              </p>
            </div>
          </div>

          {/* Start instructions */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>
              {sinditOk ? '✅ SINDIT Is Running' : '🚀 Getting Started'}
            </div>
            {sinditOk ? (
              <div className="small" style={{ color: 'var(--ok)' }}>
                SINDIT is connected and enriching events. Use the tabs above to explore the API and GraphDB.
              </div>
            ) : (
              <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                <p style={{ margin: '0 0 8px' }}>SINDIT requires its own Docker services. To start:</p>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '10px 14px', borderRadius: 6, fontFamily: 'monospace', fontSize: 11, lineHeight: 1.8 }}>
                  <div style={{ color: 'var(--muted)' }}># 1. Start SINDIT containers</div>
                  <div>docker compose --profile sindit up -d</div>
                  <div style={{ height: 4 }} />
                  <div style={{ color: 'var(--muted)' }}># 2. Enable SINDIT in .env</div>
                  <div>SINDIT_ENABLED=true</div>
                  <div>SINDIT_API_URL=http://localhost:9017</div>
                  <div style={{ height: 4 }} />
                  <div style={{ color: 'var(--muted)' }}># 3. Restart the backend</div>
                  <div>uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload</div>
                </div>
                <p style={{ margin: '8px 0 0' }}>
                  The SINDIT profile also starts <strong>GraphDB</strong> (RDF triple store at :7200) and <strong>Keycloak</strong> (identity server, optional).
                </p>
              </div>
            )}
          </div>
        </>
      )}

      {subTab === 'tools' && (
        <>
          <div style={{ display: 'flex', gap: 2 }}>
            {([
              { key: 'audit' as const, label: 'Audit Rows' },
              { key: 'datasets' as const, label: 'Dataset Overview' },
            ]).map(({ key, label }) => (
              <button
                key={key}
                type="button"
                className="small"
                style={{
                  padding: '6px 14px',
                  borderRadius: 4,
                  border: '1px solid var(--border)',
                  cursor: 'pointer',
                  background: toolView === key ? 'var(--accent)' : 'transparent',
                  color: toolView === key ? '#fff' : 'var(--fg)',
                  fontWeight: toolView === key ? 700 : 400,
                }}
                onClick={() => setToolView(key)}
              >
                {label}
              </button>
            ))}
          </div>

          {toolView === 'audit' && (
            <>
          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontWeight: 700 }}>Tool Audit</div>
                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                  Compare runtime cutting context, workbook master data, and imported SINDIT tool assets for each machine-family and tool-number pair.
                </div>
              </div>
              <StatusBadge
                ok={toolAuditSummaryQ.data ? sinditOk === true && toolAuditSummaryQ.data.sindit_available : null}
                label={sinditOk === true && toolAuditSummaryQ.data?.sindit_available ? 'SINDIT Graph Snapshot Live' : 'SINDIT Graph Snapshot Unavailable'}
                detail={toolAuditRowsQ.data?.detail || 'Master and runtime data remain available even when the graph is offline.'}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
              {[
                { label: 'Rows', value: toolAuditSummaryQ.data?.total ?? toolAuditRowsQ.data?.total ?? 0, tone: 'var(--accent)' },
                { label: 'Discrepancies', value: toolAuditSummaryQ.data?.discrepancies ?? 0, tone: 'var(--danger)' },
                { label: 'Harmonic Ready', value: toolAuditSummaryQ.data?.harmonic_ready ?? 0, tone: 'var(--ok)' },
                { label: 'Missing Diameter', value: toolAuditSummaryQ.data?.missing_diameter ?? 0, tone: 'var(--danger)' },
                { label: 'Missing Teeth', value: toolAuditSummaryQ.data?.missing_teeth ?? 0, tone: 'var(--danger)' },
                { label: 'Missing Graph Asset', value: toolAuditSummaryQ.data?.missing_sindit_asset ?? 0, tone: 'var(--danger)' },
                { label: 'Family Misses', value: toolAuditSummaryQ.data?.family_resolution_miss ?? 0, tone: '#e0af68' },
              ].map(card => (
                <div key={card.label} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(122,162,247,0.04)' }}>
                  <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--muted)' }}>{card.label}</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: card.tone, marginTop: 4 }}>{card.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 10 }}>Filters</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10, alignItems: 'end' }}>
              <label style={{ display: 'grid', gap: 4 }}>
                <span className="small" style={{ color: 'var(--muted)' }}>Session ID</span>
                <input value={toolSessionFilter} onChange={e => setToolSessionFilter(e.target.value)} placeholder="optional session filter" style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }} />
              </label>
              <label style={{ display: 'grid', gap: 4 }}>
                <span className="small" style={{ color: 'var(--muted)' }}>Machine ID</span>
                <input value={toolMachineFilter} onChange={e => setToolMachineFilter(e.target.value)} placeholder="Site_b - MACHINE_B1 - CASE_B1" style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }} />
              </label>
              <label style={{ display: 'grid', gap: 4 }}>
                <span className="small" style={{ color: 'var(--muted)' }}>Machine Family</span>
                <input value={toolFamilyFilter} onChange={e => setToolFamilyFilter(e.target.value)} placeholder="builder_b12" style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }} />
              </label>
              <label style={{ display: 'grid', gap: 4 }}>
                <span className="small" style={{ color: 'var(--muted)' }}>Tool Number</span>
                <input value={toolNumberFilter} onChange={e => setToolNumberFilter(e.target.value)} placeholder="6" style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }} />
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, paddingBottom: 6 }}>
                <input type="checkbox" checked={onlyToolDiscrepancies} onChange={e => setOnlyToolDiscrepancies(e.target.checked)} />
                <span className="small" style={{ color: 'var(--fg)' }}>Only discrepancies</span>
              </label>
              <button
                type="button"
                onClick={() => {
                  setToolSessionFilter('')
                  setToolMachineFilter('')
                  setToolFamilyFilter('')
                  setToolNumberFilter('')
                  setOnlyToolDiscrepancies(false)
                }}
                style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border)', background: 'transparent', color: 'var(--fg)', cursor: 'pointer' }}
              >
                Clear filters
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: selectedTool ? 'minmax(0, 1.6fr) minmax(320px, 0.9fr)' : '1fr', gap: 12, alignItems: 'start' }}>
            <div className="card" style={{ padding: 16, overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12 }}>
                <div style={{ fontWeight: 700 }}>Audit Rows</div>
                {toolAuditRowsQ.isFetching && <div className="small" style={{ color: 'var(--muted)' }}>refreshing…</div>}
              </div>

              {toolAuditRowsQ.isError && (
                <div style={{ padding: 20, color: 'var(--danger)', textAlign: 'center' }}>
                  Failed to fetch tool audit rows: {(toolAuditRowsQ.error as Error)?.message || 'unknown error'}
                </div>
              )}

              {!toolAuditRowsQ.isError && toolRows.length === 0 && (
                <div style={{ padding: 28, textAlign: 'center', color: 'var(--muted)' }}>
                  No tool rows matched the current filters.
                </div>
              )}

              {!toolAuditRowsQ.isError && toolRows.length > 0 && (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ textAlign: 'left', color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                        <th style={{ padding: '8px 10px' }}>Pair</th>
                        <th style={{ padding: '8px 10px' }}>Effective</th>
                        <th style={{ padding: '8px 10px' }}>Master</th>
                        <th style={{ padding: '8px 10px' }}>SINDIT</th>
                        <th style={{ padding: '8px 10px' }}>Seen</th>
                        <th style={{ padding: '8px 10px' }}>Flags</th>
                      </tr>
                    </thead>
                    <tbody>
                      {toolRows.map(row => {
                        const key = toolAuditRowKey(row)
                        const selected = selectedToolKey === key
                        const effective = row.runtime?.effective_ctx || {}
                        const deltaBits = [toolAuditDelta(row, 'diameter'), toolAuditDelta(row, 'teeth'), toolAuditDelta(row, 'length')].filter(Boolean)
                        const plannedOps = row.process_plan?.entries?.length ?? 0
                        return (
                          <tr
                            key={key}
                            onClick={() => setSelectedToolKey(key)}
                            style={{
                              cursor: 'pointer',
                              background: selected ? 'rgba(122,162,247,0.12)' : 'transparent',
                              borderBottom: '1px solid rgba(128,128,128,0.08)',
                            }}
                          >
                            <td style={{ padding: '10px' }}>
                              <div style={{ fontWeight: 700 }}>{row.machine_family}</div>
                              <div style={{ fontFamily: 'monospace', color: 'var(--muted)' }}>T{row.tool_number}</div>
                              {(plannedOps > 0 || row.reference) && (
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                  {plannedOps > 0 ? `${plannedOps} planned ops` : 'critical ref'}
                                  {plannedOps > 0 && row.reference ? ' · critical ref' : ''}
                                </div>
                              )}
                            </td>
                            <td style={{ padding: '10px', verticalAlign: 'top' }}>
                              <div>d {toolAuditNumber(effective.tool_diameter as number | undefined, row.master?.diameter_mm, row.sindit?.tool_diameter)} mm</div>
                              <div>z {toolAuditNumber(effective.num_teeth as number | undefined, row.master?.teeth, row.sindit?.num_teeth)}</div>
                              <div>{toolAuditText(effective.tool_type as string | undefined, row.master?.tool_type, row.sindit?.tool_type)}</div>
                            </td>
                            <td style={{ padding: '10px', verticalAlign: 'top' }}>
                              <div>{toolAuditText(row.master?.tool_id)}</div>
                              <div>d {toolAuditNumber(row.master?.diameter_mm)} mm</div>
                              <div>z {toolAuditNumber(row.master?.teeth)}</div>
                            </td>
                            <td style={{ padding: '10px', verticalAlign: 'top' }}>
                              <div>{toolAuditText(row.sindit?.label, row.sindit?.asset_uri)}</div>
                              <div>d {toolAuditNumber(row.sindit?.tool_diameter)} mm</div>
                              <div>z {toolAuditNumber(row.sindit?.num_teeth)}</div>
                            </td>
                            <td style={{ padding: '10px', verticalAlign: 'top' }}>
                              <div>runtime {ago(row.runtime?.last_seen_at ?? null)}</div>
                              <div>import {ago(row.sindit?.last_imported_at ?? null)}</div>
                              {deltaBits.length > 0 && <div style={{ color: 'var(--danger)' }}>{deltaBits.join(' · ')}</div>}
                            </td>
                            <td style={{ padding: '10px', verticalAlign: 'top' }}>
                              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                {row.flags.length > 0 ? row.flags.map(flag => <FlagPill key={flag} flag={flag} />) : <FlagPill flag={row.harmonic_ready ? 'harmonic_ready' : 'clean'} />}
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {selectedTool && (
              <div className="card" style={{ padding: 16 }}>
                <div style={{ fontWeight: 700, marginBottom: 10 }}>Tool Detail</div>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>{selectedTool.machine_family} · T{selectedTool.tool_number}</div>
                  <div className="small" style={{ color: 'var(--muted)', fontFamily: 'monospace' }}>{toolAuditText(selectedTool.tool_uri)}</div>
                </div>

                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                  {selectedTool.flags.length > 0 ? selectedTool.flags.map(flag => <FlagPill key={flag} flag={flag} />) : <FlagPill flag="no_discrepancy_flags" />}
                </div>

                <div style={{ display: 'grid', gap: 12 }}>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>Incoming / Runtime</div>
                    <ToolAuditField label="Sessions" value={(selectedTool.runtime?.session_ids || []).join(', ') || '—'} />
                    <ToolAuditField label="Machines" value={(selectedTool.runtime?.machine_ids || []).join(', ') || '—'} />
                    <ToolAuditField label="Seen Count" value={selectedTool.runtime?.seen_count ?? '—'} />
                    <ToolAuditField label="Last Seen" value={selectedTool.runtime?.last_seen_at ? `${selectedTool.runtime.last_seen_at} (${ago(selectedTool.runtime.last_seen_at)})` : '—'} />
                    <ToolAuditField label="Effective d" value={`${toolAuditNumber(selectedTool.runtime?.effective_ctx?.tool_diameter as number | undefined)} mm`} />
                    <ToolAuditField label="Effective z" value={toolAuditNumber(selectedTool.runtime?.effective_ctx?.num_teeth as number | undefined)} />
                    <ToolAuditField label="Effective Type" value={toolAuditText(selectedTool.runtime?.effective_ctx?.tool_type as string | undefined)} />
                    <ToolAuditField label="Spindle / Feed" value={`${toolAuditNumber(selectedTool.runtime?.effective_ctx?.spindle_speed as number | undefined)} rpm · ${toolAuditNumber(selectedTool.runtime?.effective_ctx?.feed_rate as number | undefined)} mm/min`} />
                  </div>

                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>Master</div>
                    <ToolAuditField label="Tool ID" value={toolAuditText(selectedTool.master?.tool_id)} />
                    <ToolAuditField label="Description" value={toolAuditText(selectedTool.master?.description)} />
                    <ToolAuditField label="Geometry" value={`d ${toolAuditNumber(selectedTool.master?.diameter_mm)} mm · z ${toolAuditNumber(selectedTool.master?.teeth)} · ${toolAuditText(selectedTool.master?.tool_type)}`} />
                    <ToolAuditField label="Length" value={`${toolAuditNumber(selectedTool.master?.tool_length_mm)} mm`} />
                    <ToolAuditField label="Material" value={toolAuditText(selectedTool.master?.tool_material)} />
                    <ToolAuditField label="Workbook" value={toolAuditText(selectedTool.master?.source_workbook)} />
                  </div>

                  {(selectedTool.reference || selectedTool.process_plan) && (
                    <>
                      <div>
                        <div style={{ fontWeight: 600, marginBottom: 6 }}>Case-Study Plan</div>
                        <ToolAuditField label="Use Cases" value={(selectedTool.process_plan?.use_case_titles || []).join(', ') || '—'} />
                        <ToolAuditField label="Operations" value={(selectedTool.process_plan?.operation_ids || []).join(', ') || '—'} />
                        <ToolAuditField label="Setups" value={(selectedTool.process_plan?.setups || []).join(', ') || '—'} />
                        {selectedTool.process_plan?.entries && selectedTool.process_plan.entries.length > 0 && (
                          <div style={{ display: 'grid', gap: 6, marginTop: 8 }}>
                            {selectedTool.process_plan.entries.map((entry, index) => (
                              <div key={`${entry.operation_id || 'op'}-${index}`} style={{ border: '1px solid rgba(128,128,128,0.16)', borderRadius: 6, padding: '8px 10px', background: 'rgba(122,162,247,0.04)' }}>
                                <div style={{ fontWeight: 600, fontSize: 12 }}>{toolAuditText(entry.operation_id, entry.use_case_title)}</div>
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 2 }}>
                                  {[entry.use_case_title, entry.setup, entry.head, entry.op_type].filter(Boolean).join(' · ') || '—'}
                                </div>
                                <div className="small" style={{ marginTop: 4 }}>{toolAuditText(entry.description, entry.tool_raw)}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div>
                        <div style={{ fontWeight: 600, marginBottom: 6 }}>Secondary References</div>
                        <ToolAuditField label="Sources" value={(selectedTool.reference?.sources || []).join(', ') || '—'} />
                        <ToolAuditField label="Linked Tools" value={(selectedTool.reference?.linked_tool_numbers || []).map(value => `T${value}`).join(', ') || '—'} />
                        <ToolAuditField label="Critical Description" value={toolAuditText(selectedTool.reference?.description)} />
                        <ToolAuditField label="Drawing Required" value={toolAuditBool(selectedTool.reference?.drawing_required)} />
                        <ToolAuditField label="Reference Lines" value={(selectedTool.reference?.reference_lines || []).join(' | ') || '—'} />
                        <ToolAuditField label="Notes" value={(selectedTool.reference?.notes || []).join(' | ') || '—'} />
                        <ToolAuditField label="Ref Operations" value={(selectedTool.reference?.operations || []).map(op => [op.operation_id, op.title].filter(Boolean).join(': ')).join(', ') || '—'} />
                        {selectedTool.reference?.dimensions && (
                          <>
                            <ToolAuditField label="Arbour" value={`d ${toolAuditNumber(selectedTool.reference.dimensions.arbour_diameter_mm)} mm · L ${toolAuditNumber(selectedTool.reference.dimensions.arbour_length_mm)} mm`} />
                            <ToolAuditField label="Head" value={`d ${toolAuditNumber(selectedTool.reference.dimensions.head_diameter_mm)} mm · L ${toolAuditNumber(selectedTool.reference.dimensions.head_length_mm)} mm · ${toolAuditText(selectedTool.reference.dimensions.head_id_number)}`} />
                            <ToolAuditField label="Overall / Weight" value={`L ${toolAuditNumber(selectedTool.reference.dimensions.overall_length_mm)} mm · ${toolAuditNumber(selectedTool.reference.dimensions.tool_weight_kg)} kg`} />
                          </>
                        )}
                      </div>
                    </>
                  )}

                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>SINDIT</div>
                    <ToolAuditField label="Asset" value={toolAuditText(selectedTool.sindit?.label, selectedTool.sindit?.asset_uri)} />
                    <ToolAuditField label="Graph Geometry" value={`d ${toolAuditNumber(selectedTool.sindit?.tool_diameter)} mm · z ${toolAuditNumber(selectedTool.sindit?.num_teeth)} · ${toolAuditText(selectedTool.sindit?.tool_type)}`} />
                    <ToolAuditField label="Length" value={`${toolAuditNumber(selectedTool.sindit?.tool_length)} mm`} />
                    <ToolAuditField label="Material" value={toolAuditText(selectedTool.sindit?.tool_material)} />
                    <ToolAuditField label="Last Import" value={selectedTool.sindit?.last_imported_at ? `${selectedTool.sindit.last_imported_at} (${ago(selectedTool.sindit.last_imported_at)})` : '—'} />
                    <ToolAuditField label="Workbook" value={toolAuditText(selectedTool.sindit?.source_workbook)} />
                    <ToolAuditField label="Machine Links" value={(selectedTool.sindit?.machine_uris || []).join(', ') || '—'} />
                  </div>
                </div>
              </div>
            )}
          </div>
            </>
          )}

          {toolView === 'datasets' && (
            <>
              <div className="card" style={{ padding: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ fontWeight: 700 }}>Dataset Tool Overview</div>
                    <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                      Review the observed tool set one dataset at a time, see which values are master-backed versus defaulted from secondary evidence, and confirm or reject the proposed tool profile.
                    </div>
                  </div>
                  <StatusBadge
                    ok={toolDatasetsQ.data ? sinditOk === true && toolDatasetsQ.data.sindit_available : null}
                    label={sinditOk === true && toolDatasetsQ.data?.sindit_available ? 'Graph evidence connected' : 'Graph evidence offline'}
                    detail={toolDatasetsQ.data?.detail || 'This tab still works from workbook, runtime, process-plan, and reference evidence. Start SINDIT only if you want graph-backed tool assets as well.'}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                  {[
                    { label: 'Datasets', value: toolDatasetsQ.data?.total_datasets ?? 0, tone: 'var(--accent)' },
                    { label: 'Observed Tools', value: toolDatasetsQ.data?.total_tools ?? 0, tone: 'var(--accent)' },
                    { label: 'Pending', value: toolDatasets.reduce((sum, dataset) => sum + dataset.summary.pending_count, 0), tone: '#e0af68' },
                    { label: 'Confirmed', value: toolDatasets.reduce((sum, dataset) => sum + dataset.summary.confirmed_count, 0), tone: 'var(--ok)' },
                    { label: 'Rejected', value: toolDatasets.reduce((sum, dataset) => sum + dataset.summary.rejected_count, 0), tone: 'var(--danger)' },
                    { label: 'Needs Review', value: toolDatasets.reduce((sum, dataset) => sum + dataset.summary.needs_review_count, 0), tone: 'var(--danger)' },
                  ].map(card => (
                    <div key={card.label} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(122,162,247,0.04)' }}>
                      <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--muted)' }}>{card.label}</div>
                      <div style={{ fontSize: 24, fontWeight: 700, color: card.tone, marginTop: 4 }}>{card.value}</div>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(247,118,142,0.24)', background: 'rgba(247,118,142,0.05)' }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>What discrepancies mean</div>
                  <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                    Discrepancies are tracked per dataset. A tool is discrepant when core geometry is missing in the available evidence, or when workbook, reference, runtime, and SINDIT sources disagree on diameter, teeth, type, or length. Harmonics blocked is separate: it only means the harmonic model is still missing one of d, z, n, or vf.
                  </div>
                </div>
              </div>

              <div className="card" style={{ padding: 16 }}>
                <div style={{ fontWeight: 700, marginBottom: 10 }}>Dataset selectors</div>
                {toolDatasetsQ.isError && (
                  <div style={{ padding: 20, color: 'var(--danger)', textAlign: 'center' }}>
                    Failed to fetch dataset overview: {(toolDatasetsQ.error as Error)?.message || 'unknown error'}
                  </div>
                )}
                {!toolDatasetsQ.isError && toolDatasets.length === 0 && (
                  <div style={{ padding: 28, textAlign: 'center', color: 'var(--muted)' }}>
                    No dataset tool overview is available.
                  </div>
                )}
                {!toolDatasetsQ.isError && toolDatasets.length > 0 && (
                  <div style={{ display: 'grid', gap: 12 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                      <label style={{ display: 'grid', gap: 4 }}>
                        <span className="small" style={{ color: 'var(--muted)' }}>Dataset</span>
                        <select value={selectedDataset?.dataset_id || ''} onChange={e => setSelectedDatasetId(e.target.value || null)} style={TOOL_SELECTOR_STYLE}>
                          {toolDatasets.map(dataset => (
                            <option key={dataset.dataset_id} value={dataset.dataset_id}>
                              {`${dataset.label} (${toolDatasetMachineSummary(dataset)}, ${dataset.summary.tool_count} tools, ${toolDatasetDiscrepancyCount(dataset)} discrepancies)`}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label style={{ display: 'grid', gap: 4 }}>
                        <span className="small" style={{ color: 'var(--muted)' }}>Table scope</span>
                        <select value={datasetToolScopeFilter} onChange={e => setDatasetToolScopeFilter(e.target.value as ToolDatasetScopeFilter)} style={TOOL_SELECTOR_STYLE}>
                          <option value="all">All tools</option>
                          <option value="discrepant">Discrepancies only</option>
                          <option value="clean">No discrepancies</option>
                          <option value="harmonic_ready">Harmonic-ready only</option>
                          <option value="harmonics_blocked">Harmonics blocked only</option>
                        </select>
                      </label>

                      <label style={{ display: 'grid', gap: 4 }}>
                        <span className="small" style={{ color: 'var(--muted)' }}>Certainty</span>
                        <select value={datasetToolCertaintyFilter} onChange={e => setDatasetToolCertaintyFilter(e.target.value as ToolDatasetCertaintyFilter)} style={TOOL_SELECTOR_STYLE}>
                          <option value="all">All certainty levels</option>
                          <option value="certain">Certain</option>
                          <option value="defaulted">Defaulted</option>
                          <option value="needs_review">Needs review</option>
                        </select>
                      </label>

                      <label style={{ display: 'grid', gap: 4 }}>
                        <span className="small" style={{ color: 'var(--muted)' }}>Decision</span>
                        <select value={datasetToolDecisionFilter} onChange={e => setDatasetToolDecisionFilter(e.target.value as ToolDatasetDecisionFilter)} style={TOOL_SELECTOR_STYLE}>
                          <option value="all">All decisions</option>
                          <option value="pending">Pending</option>
                          <option value="confirmed">Confirmed</option>
                          <option value="rejected">Rejected</option>
                        </select>
                      </label>
                    </div>

                    {selectedDataset && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
                        <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.12)' }}>
                          <div style={{ fontWeight: 700 }}>{selectedDataset.label}</div>
                          <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                            {selectedDataset.summary.tool_count} tools · {selectedDataset.operation_count} operations
                          </div>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                            <TonePill label={toolDatasetMachineSummary(selectedDataset)} tone={selectedDataset.shared_workpiece ? 'accent' : 'neutral'} />
                          </div>
                          {selectedDataset.workpiece_note && (
                            <div className="small" style={{ color: 'var(--muted)', marginTop: 8, lineHeight: 1.5 }}>
                              {selectedDataset.workpiece_note}
                            </div>
                          )}
                        </div>
                        <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.12)' }}>
                          <div className="small" style={{ color: 'var(--muted)' }}>Coverage snapshot</div>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                            <TonePill label={`${selectedDataset.summary.certain_count} certain`} tone="ok" />
                            <TonePill label={`${selectedDataset.summary.defaulted_count} defaulted`} tone="warn" />
                            <TonePill label={`${selectedDataset.summary.needs_review_count} review`} tone="danger" />
                            <TonePill label={`${selectedDatasetDiscrepancyCount} discrepancies`} tone={selectedDatasetDiscrepancyCount > 0 ? 'danger' : 'neutral'} />
                            <TonePill label={`${selectedDatasetHarmonicReadyTools.length} harmonic ready`} tone="accent" />
                            <TonePill label={`${selectedDataset.summary.pending_count} pending`} tone="neutral" />
                          </div>
                        </div>
                        <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.12)' }}>
                          <div className="small" style={{ color: 'var(--muted)' }}>Harmonic readiness</div>
                          <div style={{ display: 'grid', gap: 6, marginTop: 8 }}>
                            <div>{toolDatasetNumber(selectedDataset.harmonic_summary?.harmonic_ready_tools)} / {toolDatasetNumber(selectedDataset.harmonic_summary?.observed_tools)} tools ready ({toolDatasetPercent(selectedDataset.harmonic_summary?.harmonic_ready_tool_pct)})</div>
                            <div>{toolDatasetNumber(selectedDataset.harmonic_summary?.harmonic_ready_rows)} / {toolDatasetNumber(selectedDataset.harmonic_summary?.valid_rows)} rows ready ({toolDatasetPercent(selectedDataset.harmonic_summary?.harmonic_ready_row_pct)})</div>
                            <div className="small" style={{ color: 'var(--muted)' }}>{toolDatasetNumber(selectedDataset.harmonic_summary?.ready_parts)} / {toolDatasetNumber(selectedDataset.harmonic_summary?.total_parts)} parts have some harmonic-ready coverage.</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {selectedDataset && (
                <div style={{ display: 'grid', gridTemplateColumns: selectedDatasetTool ? 'minmax(0, 1.55fr) minmax(340px, 0.95fr)' : '1fr', gap: 12, alignItems: 'start' }}>
                  <div className="card" style={{ padding: 16, overflow: 'hidden' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontWeight: 700 }}>{selectedDataset.label}</div>
                        <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                          {selectedDataset.machine_families.join(', ')} · {selectedDataset.machine_ids.join(', ') || 'no machine ids'}
                        </div>
                        {selectedDataset.workpiece_note && (
                          <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.5 }}>
                            {selectedDataset.workpiece_note}
                          </div>
                        )}
                      </div>
                      {toolDatasetsQ.isFetching && <div className="small" style={{ color: 'var(--muted)' }}>refreshing…</div>}
                    </div>

                    <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(247,118,142,0.18)', background: 'rgba(247,118,142,0.06)' }}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                        <TonePill label={`${selectedDatasetDiscrepancyCount} tools with discrepancies`} tone={selectedDatasetDiscrepancyCount > 0 ? 'danger' : 'neutral'} />
                      </div>
                      {selectedDatasetDiscrepancyBreakdown.length > 0 ? (
                        <div style={{ display: 'grid', gap: 8 }}>
                          {selectedDatasetDiscrepancyBreakdown.map(item => (
                            <div key={item.flag}>
                              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                                <FlagPill flag={item.flag} />
                                <TonePill label={`${item.count} tools`} tone="danger" />
                              </div>
                              <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                {toolDatasetDiscrepancyMeaning(item.flag)}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                          No discrepancy flags are active for this dataset.
                        </div>
                      )}
                    </div>

                    <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(158,206,106,0.18)', background: 'rgba(158,206,106,0.06)' }}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
                        <TonePill label={`${selectedDatasetHarmonicReadyTools.length} harmonic-ready tools`} tone="ok" />
                        <TonePill label={`${toolDatasetPercent(selectedDataset.harmonic_summary?.harmonic_ready_tool_pct)} tool coverage`} tone="accent" />
                        <TonePill label={`${toolDatasetPercent(selectedDataset.harmonic_summary?.harmonic_ready_row_pct)} row coverage`} tone="accent" />
                      </div>
                      <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                        {selectedDatasetHarmonicReadyTools.length > 0
                          ? selectedDatasetHarmonicReadyTools.map(toolNumber => `T${toolNumber}`).join(', ')
                          : 'No harmonic-ready tools in this dataset.'}
                      </div>
                      <div className="small" style={{ color: 'var(--muted)', marginTop: 8, lineHeight: 1.6 }}>
                        Tool coverage here means tools where diameter and teeth are both resolved. Row coverage means rows where those tool fields are available for the actual observations in the dataset.
                      </div>
                    </div>

                    <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(122,162,247,0.18)', background: 'rgba(122,162,247,0.05)' }}>
                      <div style={{ fontWeight: 600, marginBottom: 8 }}>Part readiness</div>
                      <div style={{ display: 'grid', gap: 8 }}>
                        {selectedDatasetPartSummaries.length > 0 ? selectedDatasetPartSummaries.map(part => (
                          <div key={part.label} style={{ border: '1px solid rgba(128,128,128,0.12)', borderRadius: 6, padding: '8px 10px', background: 'rgba(0,0,0,0.1)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                              <div style={{ fontWeight: 600 }}>{part.label}</div>
                              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                <TonePill label={`${toolDatasetPercent(part.harmonic_ready_row_pct)} rows`} tone={part.harmonic_ready_row_pct > 0 ? 'ok' : 'danger'} />
                                <TonePill label={`${toolDatasetPercent(part.harmonic_ready_tool_pct)} tools`} tone={part.harmonic_ready_tool_pct > 0 ? 'accent' : 'danger'} />
                              </div>
                            </div>
                            <div className="small" style={{ color: 'var(--muted)', marginTop: 4, lineHeight: 1.5 }}>
                              {toolDatasetNumber(part.harmonic_ready_rows)} / {toolDatasetNumber(part.valid_rows)} rows are harmonic-ready. Ready tools: {part.harmonic_ready_tools.length > 0 ? part.harmonic_ready_tools.map(toolNumber => `T${toolNumber}`).join(', ') : 'none'}.
                            </div>
                          </div>
                        )) : <div className="small" style={{ color: 'var(--muted)' }}>No part-level harmonic summary is available.</div>}
                      </div>
                    </div>

                    <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(122,162,247,0.18)', background: 'rgba(122,162,247,0.05)' }}>
                      <div style={{ fontWeight: 600, marginBottom: 8 }}>Tool data sources</div>
                      <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6, marginBottom: 8 }}>
                        These are the source layers currently contributing tool information for this dataset. The operator can use this to understand what is canonical, what is extracted support material, and where missing values still need to be filled in.
                      </div>
                      <div style={{ display: 'grid', gap: 8 }}>
                        {selectedDatasetSourceGuides.length > 0 ? selectedDatasetSourceGuides.map(guide => (
                          <div key={guide.key} style={{ border: '1px solid rgba(128,128,128,0.12)', borderRadius: 6, padding: '8px 10px', background: 'rgba(0,0,0,0.1)' }}>
                            <div style={{ fontWeight: 600 }}>{guide.label}</div>
                            <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                              Documents: {guide.documents.join(' | ')}
                            </div>
                            {guide.originDocuments && guide.originDocuments.length > 0 && (
                              <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                Static source files: {guide.originDocuments.join(' | ')}
                              </div>
                            )}
                            <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                              Why: {guide.why}
                            </div>
                            <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                              How extracted: {guide.how}
                            </div>
                          </div>
                        )) : <div className="small" style={{ color: 'var(--muted)' }}>No source guidance is available for this dataset yet.</div>}
                      </div>
                    </div>

                    <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(224,175,104,0.24)', background: 'rgba(224,175,104,0.06)' }}>
                      <div style={{ fontWeight: 600, marginBottom: 8 }}>What to do next</div>
                      <div style={{ display: 'grid', gap: 8 }}>
                        {selectedDatasetRemedies.map(remedy => (
                          <div key={remedy} className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                            {remedy}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10, marginBottom: 12 }}>
                      <label style={{ display: 'grid', gap: 4 }}>
                        <span className="small" style={{ color: 'var(--muted)' }}>Selected tool</span>
                        <select value={selectedDatasetToolKey || ''} onChange={e => setSelectedDatasetToolKey(e.target.value || null)} style={TOOL_SELECTOR_STYLE}>
                          {filteredDatasetTools.map(row => {
                            const key = toolDatasetRowKey(row)
                            const discrepancyCount = toolDatasetDiscrepancyFlags(row).length
                            return (
                              <option key={key} value={key}>
                                {`${row.machine_family} · T${row.tool_number} (${row.certainty.replace('_', ' ')}, ${discrepancyCount} discrepancies)`}
                              </option>
                            )
                          })}
                        </select>
                      </label>
                      <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', background: 'rgba(0,0,0,0.12)' }}>
                        <div className="small" style={{ color: 'var(--muted)' }}>Filtered table</div>
                        <div style={{ fontSize: 20, fontWeight: 700, marginTop: 4 }}>{filteredDatasetTools.length}</div>
                        <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                          {filteredDatasetTools.length === selectedDatasetTools.length ? 'Showing all tools in this dataset.' : `Showing ${filteredDatasetTools.length} of ${selectedDatasetTools.length} tools in this dataset.`}
                        </div>
                      </div>
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                        <thead>
                          <tr style={{ textAlign: 'left', color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                            <th style={{ padding: '8px 10px' }}>Tool</th>
                            <th style={{ padding: '8px 10px' }}>Chosen Data</th>
                            <th style={{ padding: '8px 10px' }}>Certainty</th>
                            <th style={{ padding: '8px 10px' }}>Evidence</th>
                            <th style={{ padding: '8px 10px' }}>Decision</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredDatasetTools.map(row => {
                            const key = toolDatasetRowKey(row)
                            const selected = selectedDatasetToolKey === key
                            const profileMode = currentDatasetProfileMode(row)
                            const profile = resolvedDatasetProfile(row, profileMode)
                            const hasUnsavedDraft = profileMode !== row.selected_profile
                            const discrepancyFlags = toolDatasetDiscrepancyFlags(row)
                            return (
                              <tr
                                key={key}
                                onClick={() => setSelectedDatasetToolKey(key)}
                                style={{
                                  cursor: 'pointer',
                                  background: selected ? 'rgba(122,162,247,0.12)' : 'transparent',
                                  borderBottom: '1px solid rgba(128,128,128,0.08)',
                                }}
                              >
                                <td style={{ padding: '10px', verticalAlign: 'top' }}>
                                  <div style={{ fontWeight: 700 }}>{row.machine_family}</div>
                                  <div style={{ fontFamily: 'monospace', color: 'var(--muted)' }}>T{row.tool_number}</div>
                                  <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                    {row.operation_count} ops
                                  </div>
                                </td>
                                <td style={{ padding: '10px', verticalAlign: 'top' }}>
                                  <div>d {toolDatasetFieldNumber(profile.diameter_mm)} mm</div>
                                  <div>z {toolDatasetFieldNumber(profile.teeth)}</div>
                                  <div>{toolDatasetFieldText(profile.tool_type)}</div>
                                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 6 }}>
                                    {(['diameter_mm', 'teeth', 'tool_type'] as const).map(fieldName => {
                                      const field = profile[fieldName]
                                      if (!field?.source) return null
                                      return <TonePill key={fieldName} label={`${fieldName.replace('_mm', '').replace('_', ' ')}: ${field.source}`} tone={field.source === 'master' ? 'ok' : 'warn'} />
                                    })}
                                  </div>
                                </td>
                                <td style={{ padding: '10px', verticalAlign: 'top' }}>
                                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                    <TonePill label={row.certainty.replace('_', ' ')} tone={toolDatasetCertaintyTone(row.certainty)} />
                                    <TonePill label={row.coverage.harmonic_ready ? 'harmonic ready' : 'harmonics blocked'} tone={row.coverage.harmonic_ready ? 'ok' : 'danger'} />
                                    {discrepancyFlags.length > 0 && <TonePill label={`${discrepancyFlags.length} discrepancies`} tone="danger" />}
                                    {hasUnsavedDraft && <TonePill label="unsaved selection" tone="accent" />}
                                  </div>
                                  {toolDatasetPrimaryNote(row) && <div className="small" style={{ color: 'var(--muted)', marginTop: 6 }}>{toolDatasetPrimaryNote(row)}</div>}
                                </td>
                                <td style={{ padding: '10px', verticalAlign: 'top' }}>
                                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                    {row.evidence_sources.map(source => (
                                      <TonePill key={source} label={source} tone={source === 'master' ? 'ok' : source === 'runtime' ? 'accent' : 'neutral'} />
                                    ))}
                                  </div>
                                </td>
                                <td style={{ padding: '10px', verticalAlign: 'top' }}>
                                  <div style={{ marginBottom: 6 }}>
                                    <TonePill label={row.decision_status} tone={toolDatasetDecisionTone(row.decision_status)} />
                                  </div>
                                  <div className="small" style={{ color: 'var(--muted)' }}>{TOOL_DATASET_PROFILE_LABELS[profileMode]}</div>
                                </td>
                              </tr>
                            )
                          })}
                          {filteredDatasetTools.length === 0 && (
                            <tr>
                              <td colSpan={5} style={{ padding: '16px 10px', textAlign: 'center', color: 'var(--muted)' }}>
                                No tools in this dataset match the selected filters.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {selectedDatasetTool && (
                    <div className="card" style={{ padding: 16 }}>
                      <div style={{ fontWeight: 700, marginBottom: 10 }}>Dataset Tool Detail</div>
                      <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 18, fontWeight: 700 }}>{selectedDatasetTool.dataset_label} · {selectedDatasetTool.machine_family} · T{selectedDatasetTool.tool_number}</div>
                        <div className="small" style={{ color: 'var(--muted)' }}>{selectedDatasetTool.operation_ids.join(', ') || 'No operation ids'}</div>
                      </div>

                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                        <TonePill label={selectedDatasetTool.certainty.replace('_', ' ')} tone={toolDatasetCertaintyTone(selectedDatasetTool.certainty)} />
                        <TonePill label={selectedDatasetTool.coverage.harmonic_ready ? 'harmonic ready' : 'harmonics blocked'} tone={selectedDatasetTool.coverage.harmonic_ready ? 'ok' : 'danger'} />
                        {selectedDatasetToolDiscrepancyFlags.length > 0 && <TonePill label={`${selectedDatasetToolDiscrepancyFlags.length} discrepancies`} tone="danger" />}
                        <TonePill label={selectedDatasetTool.decision_status} tone={toolDatasetDecisionTone(selectedDatasetTool.decision_status)} />
                        <TonePill label={`recommended: ${TOOL_DATASET_PROFILE_LABELS[selectedDatasetTool.recommended_profile]}`} tone="accent" />
                      </div>

                      <div style={{ display: 'grid', gap: 12 }}>
                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Decision</div>
                          <div style={{ display: 'grid', gap: 8 }}>
                            <label style={{ display: 'grid', gap: 4 }}>
                              <span className="small" style={{ color: 'var(--muted)' }}>Selected profile</span>
                              <select
                                value={currentDatasetProfileMode(selectedDatasetTool)}
                                onChange={e => {
                                  setToolDatasetSelections(prev => ({
                                    ...prev,
                                    [toolDatasetRowKey(selectedDatasetTool)]: e.target.value as ToolDatasetProfileMode,
                                  }))
                                }}
                                style={TOOL_SELECTOR_STYLE}
                              >
                                {selectedDatasetTool.available_profiles.map(mode => (
                                  <option key={mode} value={mode}>{TOOL_DATASET_PROFILE_LABELS[mode]}</option>
                                ))}
                              </select>
                            </label>
                            {currentDatasetProfileMode(selectedDatasetTool) === 'manual' && (
                              <div style={{ display: 'grid', gap: 8, padding: '10px 12px', borderRadius: 8, border: '1px solid rgba(122,162,247,0.22)', background: 'rgba(122,162,247,0.06)' }}>
                                <div style={{ fontWeight: 600 }}>Manual input</div>
                                <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
                                  Enter a reference tool number to copy that tool's default profile for this dataset, then optionally override the tooth count. The saved snapshot is applied at runtime the same way as the default recommendation.
                                </div>
                                <label style={{ display: 'grid', gap: 4 }}>
                                  <span className="small" style={{ color: 'var(--muted)' }}>Reference tool number</span>
                                  <input
                                    value={manualDraftFor(selectedDatasetTool).referenceToolNumber}
                                    onChange={e => setToolDatasetManualDrafts(prev => ({
                                      ...prev,
                                      [toolDatasetRowKey(selectedDatasetTool)]: {
                                        ...manualDraftFor(selectedDatasetTool),
                                        referenceToolNumber: e.target.value,
                                      },
                                    }))}
                                    placeholder="e.g. 44"
                                    style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}
                                  />
                                </label>
                                <label style={{ display: 'grid', gap: 4 }}>
                                  <span className="small" style={{ color: 'var(--muted)' }}>Correct tooth count</span>
                                  <input
                                    value={manualDraftFor(selectedDatasetTool).teeth}
                                    onChange={e => setToolDatasetManualDrafts(prev => ({
                                      ...prev,
                                      [toolDatasetRowKey(selectedDatasetTool)]: {
                                        ...manualDraftFor(selectedDatasetTool),
                                        teeth: e.target.value,
                                      },
                                    }))}
                                    placeholder="e.g. 7"
                                    style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}
                                  />
                                </label>
                                <label style={{ display: 'grid', gap: 4 }}>
                                  <span className="small" style={{ color: 'var(--muted)' }}>Reason</span>
                                  <textarea
                                    value={manualDraftFor(selectedDatasetTool).notes}
                                    onChange={e => setToolDatasetManualDrafts(prev => ({
                                      ...prev,
                                      [toolDatasetRowKey(selectedDatasetTool)]: {
                                        ...manualDraftFor(selectedDatasetTool),
                                        notes: e.target.value,
                                      },
                                    }))}
                                    placeholder="Why this tool number / tooth count is correct"
                                    rows={3}
                                    style={{ background: 'rgba(0,0,0,0.18)', color: 'var(--fg)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px', resize: 'vertical' }}
                                  />
                                </label>
                              </div>
                            )}
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              <button
                                type="button"
                                disabled={toolDatasetDecisionM.isPending}
                                onClick={() => toolDatasetDecisionM.mutate(buildToolDatasetDecisionBody(selectedDatasetTool, 'confirmed'))}
                                style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid rgba(158,206,106,0.35)', background: 'rgba(158,206,106,0.12)', color: 'var(--ok)', cursor: 'pointer' }}
                              >
                                Confirm selection
                              </button>
                              <button
                                type="button"
                                disabled={toolDatasetDecisionM.isPending}
                                onClick={() => toolDatasetDecisionM.mutate(buildToolDatasetDecisionBody(selectedDatasetTool, 'rejected'))}
                                style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid rgba(247,118,142,0.35)', background: 'rgba(247,118,142,0.12)', color: 'var(--danger)', cursor: 'pointer' }}
                              >
                                Reject selection
                              </button>
                              <button
                                type="button"
                                disabled={toolDatasetDecisionM.isPending}
                                onClick={() => {
                                  setToolDatasetSelections(prev => {
                                    const next = { ...prev }
                                    delete next[toolDatasetRowKey(selectedDatasetTool)]
                                    return next
                                  })
                                  setToolDatasetManualDrafts(prev => {
                                    const next = { ...prev }
                                    delete next[toolDatasetRowKey(selectedDatasetTool)]
                                    return next
                                  })
                                  toolDatasetDecisionM.mutate({
                                    dataset_id: selectedDatasetTool.dataset_id,
                                    machine_family: selectedDatasetTool.machine_family,
                                    tool_number: selectedDatasetTool.tool_number,
                                    status: 'pending',
                                    selection_mode: 'default',
                                  })
                                }}
                                style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border)', background: 'transparent', color: 'var(--fg)', cursor: 'pointer' }}
                              >
                                Reset to pending
                              </button>
                            </div>
                          </div>
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Chosen profile</div>
                          {(() => {
                            const profile = resolvedDatasetProfile(selectedDatasetTool, currentDatasetProfileMode(selectedDatasetTool))
                            return (
                              <>
                                <ToolAuditField label="Description" value={toolDatasetFieldText(profile.description)} />
                                <ToolAuditField label="Diameter" value={`${toolDatasetFieldNumber(profile.diameter_mm)} mm`} />
                                <ToolAuditField label="Diameter src" value={toolDatasetFieldText({ value: profile.diameter_mm?.source })} />
                                <ToolAuditField label="Teeth" value={toolDatasetFieldNumber(profile.teeth)} />
                                <ToolAuditField label="Teeth src" value={toolDatasetFieldText({ value: profile.teeth?.source })} />
                                <ToolAuditField label="Type" value={toolDatasetFieldText(profile.tool_type)} />
                                <ToolAuditField label="Type src" value={toolDatasetFieldText({ value: profile.tool_type?.source })} />
                                <ToolAuditField label="Length" value={`${toolDatasetFieldNumber(profile.tool_length_mm)} mm`} />
                                <ToolAuditField label="Reference tool" value={typeof profile.reference_tool_number === 'number' ? `T${profile.reference_tool_number}` : '—'} />
                                <ToolAuditField label="Rationale" value={toolAuditText(profile.notes)} />
                              </>
                            )
                          })()}
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Chosen source details</div>
                          <div style={{ display: 'grid', gap: 8 }}>
                            {selectedDatasetToolSourceGuides.length > 0 ? selectedDatasetToolSourceGuides.map(guide => (
                              <div key={guide.key} style={{ border: '1px solid rgba(128,128,128,0.16)', borderRadius: 6, padding: '8px 10px', background: 'rgba(122,162,247,0.04)' }}>
                                <div style={{ fontWeight: 600 }}>{guide.label}</div>
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                  Documents: {guide.documents.join(' | ')}
                                </div>
                                {guide.originDocuments && guide.originDocuments.length > 0 && (
                                  <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                    Static source files: {guide.originDocuments.join(' | ')}
                                  </div>
                                )}
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                  Why: {guide.why}
                                </div>
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                                  How extracted: {guide.how}
                                </div>
                              </div>
                            )) : <div className="small" style={{ color: 'var(--muted)' }}>No source document is attached to the currently selected profile.</div>}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Discrepancies</div>
                          <div style={{ display: 'grid', gap: 8 }}>
                            {selectedDatasetToolDiscrepancyFlags.length > 0 ? selectedDatasetToolDiscrepancyFlags.map(flag => (
                              <div key={flag} style={{ border: '1px solid rgba(128,128,128,0.16)', borderRadius: 6, padding: '8px 10px', background: 'rgba(247,118,142,0.05)' }}>
                                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                                  <FlagPill flag={flag} />
                                </div>
                                <div className="small" style={{ color: 'var(--muted)', marginTop: 6 }}>
                                  {toolDatasetDiscrepancyMeaning(flag)}
                                </div>
                              </div>
                            )) : <div className="small" style={{ color: 'var(--muted)' }}>No discrepancy flags for this dataset/tool.</div>}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Why this needs attention</div>
                          <div style={{ display: 'grid', gap: 6 }}>
                            {selectedDatasetTool.certainty_reasons.length > 0 ? selectedDatasetTool.certainty_reasons.map(reason => (
                              <div key={reason} style={{ border: '1px solid rgba(128,128,128,0.16)', borderRadius: 6, padding: '8px 10px', background: 'rgba(122,162,247,0.04)' }}>
                                {reason}
                              </div>
                            )) : <div className="small" style={{ color: 'var(--muted)' }}>No additional review notes.</div>}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Dataset coverage</div>
                          <ToolAuditField label="Machines" value={selectedDatasetTool.machine_ids.join(', ') || '—'} />
                          <ToolAuditField label="Operations" value={selectedDatasetTool.operation_ids.join(', ') || '—'} />
                          <ToolAuditField label="Harmonic ready" value={selectedDatasetTool.coverage.harmonic_ready ? 'Yes' : 'No'} />
                          <ToolAuditField label="Master backed" value={selectedDatasetTool.coverage.master ? 'Yes' : 'No'} />
                          <ToolAuditField label="Diameter / teeth" value={`${selectedDatasetTool.coverage.diameter ? 'diameter' : 'no diameter'} · ${selectedDatasetTool.coverage.teeth ? 'teeth' : 'no teeth'}`} />
                          <ToolAuditField label="Evidence" value={selectedDatasetTool.evidence_sources.join(', ') || '—'} />
                        </div>

                        <div>
                          <div style={{ fontWeight: 600, marginBottom: 6 }}>Audit evidence</div>
                          <ToolAuditField label="Master" value={toolAuditText(selectedDatasetTool.audit.master?.description, selectedDatasetTool.audit.master?.tool_id)} />
                          <ToolAuditField label="Reference" value={toolAuditText(selectedDatasetTool.audit.reference?.description)} />
                          <ToolAuditField label="Plan ops" value={(selectedDatasetTool.audit.process_plan?.operation_ids || []).join(', ') || '—'} />
                          <ToolAuditField label="Runtime" value={`${toolAuditNumber(selectedDatasetTool.audit.runtime?.effective_ctx?.tool_diameter as number | undefined)} mm · z ${toolAuditNumber(selectedDatasetTool.audit.runtime?.effective_ctx?.num_teeth as number | undefined)}`} />
                          <ToolAuditField label="SINDIT" value={toolAuditText(selectedDatasetTool.audit.sindit?.label, selectedDatasetTool.audit.sindit?.asset_uri)} />
                          <ToolAuditField label="Flags" value={selectedDatasetTool.audit.flags.length ? selectedDatasetTool.audit.flags.join(', ') : '—'} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Experiments tab ── */}
      {subTab === 'experiments' && (
        <>
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>
              🧪 Experiment Knowledge Graph
              {experimentsQ.isFetching && <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--muted)' }}>refreshing…</span>}
            </div>

            {!sinditOk && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🔌</div>
                <div style={{ fontWeight: 600 }}>SINDIT is not reachable</div>
                <div className="small" style={{ marginTop: 4 }}>Start the SINDIT containers to view experiment data in the knowledge graph.</div>
              </div>
            )}

            {sinditOk && experimentsQ.isError && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--danger)' }}>
                Failed to fetch experiments: {(experimentsQ.error as Error)?.message || 'unknown error'}
              </div>
            )}

            {sinditOk && experimentsQ.data && experimentsQ.data.nodes.length === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>📭</div>
                <div style={{ fontWeight: 600 }}>No experiment results in SINDIT yet</div>
                <div className="small" style={{ marginTop: 8, maxWidth: 450, marginInline: 'auto', lineHeight: 1.6 }}>
                  Run a <strong>live experiment</strong> (stoppage or breakage) while SINDIT is enabled.
                  Experiment results will be pushed as a connected sub-graph.
                </div>
              </div>
            )}

            {sinditOk && experimentsQ.data && experimentsQ.data.nodes.length > 0 && (
              <KGVisualisation nodes={experimentsQ.data.nodes} edges={experimentsQ.data.edges} />
            )}
          </div>

          {/* Experiment cards */}
          {sinditOk && experimentsQ.data && experimentsQ.data.experiments.length > 0 && (
            <div className="card" style={{ padding: 16 }}>
              <div style={{ fontWeight: 700, marginBottom: 12 }}>Experiment Runs ({experimentsQ.data.experiments.length})</div>
              <div style={{ display: 'grid', gap: 10 }}>
                {experimentsQ.data.experiments.map(exp => {
                  const p = exp.properties
                  const expType = typeof p.experiment_type === 'string' ? p.experiment_type : 'unknown'
                  const evalF1 = typeof p.eval_f1 === 'number' ? p.eval_f1 : (typeof p.f1 === 'number' ? p.f1 : null)
                  const delta = typeof p.delta_f1 === 'number' ? p.delta_f1 : null
                  const pctImprove = typeof p.pct_f1_improvement === 'number' ? p.pct_f1_improvement : null

                  return (
                    <div key={exp.uri} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
                      <span style={{ fontSize: 18 }}>{expType === 'breakage' ? '🔧' : '📊'}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, fontSize: 13 }}>{exp.label}</div>
                        <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)' }}>{exp.uri.replace('urn:lfl:experiment:', '')}</div>
                      </div>
                      {evalF1 !== null && (
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontWeight: 700, fontSize: 18, fontFamily: 'monospace', color: 'var(--accent)' }}>
                            {Number.isFinite(evalF1) ? evalF1.toFixed(3) : '—'}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--muted)' }}>Eval F1</div>
                        </div>
                      )}
                      {delta !== null && (
                        <div style={{ textAlign: 'right' }}>
                          <div style={{
                            fontWeight: 700, fontSize: 14, fontFamily: 'monospace',
                            color: delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--danger)' : 'var(--muted)',
                          }}>
                            {delta > 0 ? '+' : ''}{Number.isFinite(delta) ? delta.toFixed(4) : '0'}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--muted)' }}>ΔF1</div>
                        </div>
                      )}
                      {pctImprove !== null && (
                        <div style={{ textAlign: 'right' }}>
                          <div style={{
                            fontWeight: 600, fontSize: 12, fontFamily: 'monospace',
                            color: pctImprove > 0 ? 'var(--ok)' : 'var(--muted)',
                          }}>
                            {pctImprove > 0 ? '+' : ''}{Number.isFinite(pctImprove) ? pctImprove.toFixed(1) : '0'}%
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--muted)' }}>improvement</div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>How Experiment → SINDIT Works</div>
            <div className="small" style={{ color: 'var(--muted)', lineHeight: 1.6 }}>
              <p style={{ margin: '0 0 8px' }}>
                When a live experiment runs with <code>SINDIT_ENABLED=true</code>:
              </p>
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                <li><strong>Feature bridging</strong> — each event posted via the API publishes to the PubSub <code>features</code> channel, so the SinditBridge updates live sensor properties in real time.</li>
                <li><strong>Graph push</strong> — when the experiment completes, a connected sub-graph is created: machine → experiment → test/eval phases → operations → detected patterns.</li>
                <li><strong>GraphDB</strong> — all nodes and relationships are visible in the GraphDB visual graph explorer. Start from <code>urn:lfl:asset:cnc-machine-1</code> and expand outward.</li>
              </ol>
            </div>
          </div>
        </>
      )}

      {/* Current State */}
      {subTab === 'state' && (
        <>
          {/* Bridge controls */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ fontWeight: 700 }}>🔗 Live Data Bridge</div>
              {bridgeQ.data?.available ? (
                bridgeQ.data.running ? (
                  <button className="small" style={{ padding: '4px 14px', borderRadius: 4, background: 'var(--danger)', color: '#fff', border: 'none', cursor: 'pointer' }} onClick={() => stopBridge.mutate()}>⏹ Stop Bridge</button>
                ) : (
                  <button className="small" style={{ padding: '4px 14px', borderRadius: 4, background: 'var(--ok)', color: '#000', border: 'none', cursor: 'pointer', fontWeight: 600 }} onClick={() => startBridge.mutate()}>▶ Start Bridge</button>
                )
              ) : null}
            </div>
            {bridgeQ.data?.available === false && (
              <div className="small" style={{ color: 'var(--muted)', padding: '8px 12px', background: 'rgba(128,128,128,0.08)', borderRadius: 6 }}>
                ⚠️ {bridgeQ.data.detail || 'SINDIT bridge not configured. Set SINDIT_ENABLED=true in .env and restart.'}
              </div>
            )}
            {bridgeQ.data?.available && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
                <StatusBadge ok={bridgeQ.data.running} label={bridgeQ.data.running ? 'Bridge running' : 'Bridge stopped'} />
                <div style={{ padding: '8px 14px', background: 'rgba(122,162,247,0.06)', borderRadius: 8, border: '1px solid rgba(122,162,247,0.12)' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent)' }}>{bridgeQ.data.events_received ?? 0}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>Events received</div>
                </div>
                <div style={{ padding: '8px 14px', background: 'rgba(158,206,106,0.06)', borderRadius: 8, border: '1px solid rgba(158,206,106,0.12)' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--ok)' }}>{bridgeQ.data.values_pushed ?? 0}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>Values pushed</div>
                </div>
                <div style={{ padding: '8px 14px', background: bridgeQ.data.errors ? 'rgba(247,118,142,0.06)' : 'rgba(128,128,128,0.06)', borderRadius: 8, border: `1px solid ${bridgeQ.data.errors ? 'rgba(247,118,142,0.12)' : 'rgba(128,128,128,0.08)'}` }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: bridgeQ.data.errors ? 'var(--danger)' : 'var(--muted)' }}>{bridgeQ.data.errors ?? 0}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>Errors</div>
                </div>
                <div style={{ padding: '8px 14px', background: 'rgba(128,128,128,0.06)', borderRadius: 8, border: '1px solid rgba(128,128,128,0.08)' }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--fg)' }}>{ago(bridgeQ.data.last_push_at ?? null)}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>Last push</div>
                </div>
              </div>
            )}
            <div className="small" style={{ color: 'var(--muted)', marginTop: 8 }}>
              The bridge subscribes to the sensor PubSub channel and pushes live CNC feature data into SINDIT, creating assets and updating property values in the knowledge graph.
            </div>
          </div>

          {/* Knowledge Graph State */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>
              📊 Knowledge Graph — {stateQ.data?.total_nodes ?? '…'} nodes
              {stateQ.isFetching && <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--muted)' }}>refreshing…</span>}
            </div>

            {!sinditOk && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>🔌</div>
                <div style={{ fontWeight: 600 }}>SINDIT is not reachable</div>
                <div className="small" style={{ marginTop: 4 }}>Start the SINDIT containers to view the knowledge graph.</div>
              </div>
            )}

            {sinditOk && stateQ.isError && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--danger)' }}>
                Failed to fetch SINDIT state: {(stateQ.error as Error)?.message || 'unknown error'}
              </div>
            )}

            {sinditOk && stateQ.data && stateQ.data.total_nodes === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>📭</div>
                <div style={{ fontWeight: 600 }}>Knowledge graph is empty</div>
                <div className="small" style={{ marginTop: 4 }}>
                  No assets in SINDIT yet. Start the live data bridge above to push sensor data, or use the SINDIT API to create assets manually.
                </div>
              </div>
            )}

            {sinditOk && stateQ.data && stateQ.data.total_nodes > 0 && (
              <div style={{ display: 'grid', gap: 10 }}>
                {/* Assets */}
                {stateQ.data.assets.length > 0 && (
                  <div>
                    <div className="small" style={{ fontWeight: 700, color: 'var(--accent)', marginBottom: 6 }}>⚙️ Assets ({stateQ.data.assets.length})</div>
                    {stateQ.data.assets.map(asset => {
                      const uri = asset.uri || asset.nodeUri || ''
                      const label = asset.label || uri.split(/[#/]/).pop() || 'Unknown Asset'
                      const expanded = expandedAssets.has(uri)
                      // Properties belong to an asset via its assetProperties list
                      // (URI refs), not via relationship edges — and the property URI
                      // is {assetUri}:{name}. Match on both (declared list + prefix).
                      const asUri = (v: any): string => typeof v === 'string' ? v : (v?.uri || v?.iri || v?.value || '')
                      const declaredPropUris = new Set(
                        (((asset as any).assetProperties) || []).map((ap: any) => asUri(ap)).filter(Boolean)
                      )
                      const relProps = stateQ.data!.properties.filter(p => {
                        const pu = p.uri || p.nodeUri || ''
                        return declaredPropUris.has(pu) || (!!uri && pu.startsWith(uri + ':'))
                      })

                      return (
                        <div key={uri} style={{ border: '1px solid var(--border)', borderRadius: 8, marginBottom: 6, overflow: 'hidden' }}>
                          <div
                            style={{ padding: '10px 14px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8, background: expanded ? 'rgba(122,162,247,0.06)' : 'transparent' }}
                            onClick={() => toggleAsset(uri)}
                          >
                            <span style={{ fontSize: 10, color: 'var(--muted)' }}>{expanded ? '▼' : '▶'}</span>
                            <span style={{ fontWeight: 600 }}>{label}</span>
                            <span className="small" style={{ color: 'var(--muted)', marginLeft: 'auto' }}>{relProps.length} properties</span>
                          </div>
                          {expanded && (
                            <div style={{ padding: '0 14px 12px', background: 'rgba(0,0,0,0.15)' }}>
                              {relProps.length === 0 && <div className="small" style={{ color: 'var(--muted)', padding: '8px 0' }}>No properties linked to this asset.</div>}
                              {relProps.map(prop => {
                                const pUri = prop.uri || prop.nodeUri || ''
                                const pLabel = prop.label || prop.propertyName || pUri.split(/[#/:]/).pop() || '?'
                                const val = prop.propertyValue ?? prop.value ?? '—'
                                const unit = prop.propertyUnit || prop.unit || ''
                                const ts = prop.propertyValueTimestamp || prop.timestamp || null
                                return (
                                  <div key={pUri} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '6px 0', borderBottom: '1px solid rgba(128,128,128,0.1)' }}>
                                    <span className="small" style={{ color: 'var(--muted)', minWidth: 130 }}>{pLabel}</span>
                                    <span style={{ fontWeight: 600, fontFamily: 'monospace', fontSize: 13 }}>{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                                    {unit && <span className="small" style={{ color: 'var(--muted)' }}>{unit}</span>}
                                    {ts && <span className="small" style={{ color: 'var(--muted)', marginLeft: 'auto' }}>{ago(ts)}</span>}
                                  </div>
                                )
                              })}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}

                {/* Standalone properties (not linked to an asset) */}
                {stateQ.data.properties.length > stateQ.data.assets.length && (
                  <div>
                    <div className="small" style={{ fontWeight: 700, color: 'var(--ok)', marginBottom: 6 }}>📐 All Properties ({stateQ.data.properties.length})</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 6 }}>
                      {stateQ.data.properties.slice(0, 50).map(prop => {
                        const pUri = prop.uri || prop.nodeUri || ''
                        const pLabel = prop.label || prop.propertyName || pUri.split(/[#/:]/).pop() || '?'
                        const val = prop.propertyValue ?? prop.value ?? '—'
                        const unit = prop.propertyUnit || prop.unit || ''
                        return (
                          <div key={pUri} style={{ padding: '8px 12px', background: 'rgba(158,206,106,0.05)', borderRadius: 6, border: '1px solid rgba(158,206,106,0.1)' }}>
                            <div className="small" style={{ color: 'var(--muted)', fontSize: 10 }}>{pLabel}</div>
                            <div style={{ fontWeight: 600, fontFamily: 'monospace', fontSize: 14 }}>
                              {typeof val === 'number' ? val.toFixed(2) : String(val)}{unit ? ` ${unit}` : ''}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Relationships */}
                {stateQ.data.relationships.length > 0 && (
                  <div>
                    <div className="small" style={{ fontWeight: 700, color: 'var(--warn, #e0af68)', marginBottom: 6 }}>🔗 Relationships ({stateQ.data.relationships.length})</div>
                    <div style={{ display: 'grid', gap: 4 }}>
                      {stateQ.data.relationships.slice(0, 30).map((rel, i) => {
                        // A relationship endpoint may be a plain URI string or a
                        // URI-ref object ({uri}); coerce to a string before split.
                        const asUri = (v: any): string => typeof v === 'string' ? v : (v?.uri || v?.iri || v?.value || '')
                        const src = asUri(rel.relationshipSource || rel.source).split(/[#/:]/).pop()
                        const tgt = asUri(rel.relationshipTarget || rel.target).split(/[#/:]/).pop()
                        const rtype = asUri(rel.relationshipType || rel.type).split(/[#/:]/).pop()
                        return (
                          <div key={i} className="small" style={{ padding: '4px 10px', background: 'rgba(224,175,104,0.05)', borderRadius: 4, border: '1px solid rgba(224,175,104,0.1)', fontFamily: 'monospace', fontSize: 11 }}>
                            {src} <span style={{ color: 'var(--accent)' }}>—[{rtype}]→</span> {tgt}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Node types */}
                {stateQ.data.node_types.length > 0 && (
                  <div>
                    <div className="small" style={{ fontWeight: 700, color: 'var(--muted)', marginBottom: 4 }}>Node Types</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {stateQ.data.node_types.map((nt: any, i: number) => (
                        <span key={i} className="small" style={{ padding: '2px 8px', background: 'rgba(128,128,128,0.08)', borderRadius: 4, fontSize: 10 }}>
                          {typeof nt === 'string' ? nt.split(/[#/]/).pop() : nt.label || nt.uri?.split(/[#/]/).pop() || JSON.stringify(nt)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/* SINDIT API */}
      {subTab === 'sindit-api' && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <div style={{ fontWeight: 700 }}>SINDIT API Explorer</div>
            <StatusBadge ok={sinditOk} label={sinditOk ? 'Connected' : 'Unreachable'} />
            <a href={`${SINDIT_URL}/docs`} target="_blank" rel="noopener noreferrer" className="small" style={{ color: 'var(--accent)', textDecoration: 'underline', marginLeft: 'auto' }}>
              Open in new tab ↗
            </a>
          </div>
          {sinditOk ? (
            <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden' }}>
              <iframe
                src={`${SINDIT_URL}/docs`}
                title="SINDIT API Docs"
                style={{ width: '100%', height: 650, border: 'none', background: '#1a1b26' }}
              />
            </div>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🔌</div>
              <div style={{ fontWeight: 600 }}>SINDIT API is not reachable</div>
              <div className="small" style={{ marginTop: 4 }}>
                Start the SINDIT containers: <code>docker compose --profile sindit up -d</code>
              </div>
            </div>
          )}
        </div>
      )}

      {/* GraphDB */}
      {subTab === 'graphdb' && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <div style={{ fontWeight: 700 }}>GraphDB Workbench</div>
            <StatusBadge ok={graphdbOk} label={graphdbOk ? 'Connected' : 'Unreachable'} />
            <a href={GRAPHDB_URL} target="_blank" rel="noopener noreferrer" className="small" style={{ color: 'var(--accent)', textDecoration: 'underline', marginLeft: 'auto' }}>
              Open in new tab ↗
            </a>
          </div>
          <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
            GraphDB stores SINDIT's RDF knowledge graph — asset models, properties, connections, and SPARQL-queryable data.
          </p>
          {graphdbOk ? (
            <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden' }}>
              <iframe
                src={GRAPHDB_URL}
                title="GraphDB Workbench"
                style={{ width: '100%', height: 650, border: 'none', background: '#1a1b26' }}
              />
            </div>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🗃️</div>
              <div style={{ fontWeight: 600 }}>GraphDB is not reachable</div>
              <div className="small" style={{ marginTop: 4 }}>
                Start the SINDIT containers: <code>docker compose --profile sindit up -d</code>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

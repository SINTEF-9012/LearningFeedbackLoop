import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/http'
import { humanPattern } from '../utils/patternNames'

/* ── Types ─────────────────────────────────────────────── */

type GraphNode = { id: string; weight: number; prior: number }
type GraphEdge = { source: string; target: string; weight: number; strength?: number }
type CoOccurrenceResponse = { nodes: GraphNode[]; edges: GraphEdge[]; source: string }

type KnowledgeGraphProps = {
  /** Inline co-occurrence data from experiment results (pipe-separated keys → weight). */
  coOccurrenceGraph?: Record<string, number>
  /** Pattern priors for node colouring (pattern key → prior value). */
  priors?: Record<string, number>
  /** If true, also fetch live data from the API and merge/prefer it. */
  fetchLive?: boolean
  /** If set, fetch experiment-scoped graph from `/graph/co-occurrence/{runId}`. */
  experimentRunId?: string
  /** SVG width (default 520). */
  width?: number
  /** SVG height (default 380). */
  height?: number
}

/* ── Colours ───────────────────────────────────────────── */

const PAL = [
  '#7aa2f7', '#bb9af7', '#9ece6a', '#e0af68',
  '#f7768e', '#73daca', '#ff9e64', '#2ac3de',
]

function priorColor(prior: number): string {
  // Blue (low) → Yellow (mid) → Red (high)
  const t = Math.max(0, Math.min(1, prior))
  if (t < 0.5) {
    const s = t / 0.5
    const r = Math.round(122 + (224 - 122) * s)
    const g = Math.round(162 + (175 - 162) * s)
    const b = Math.round(247 + (104 - 247) * s)
    return `rgb(${r},${g},${b})`
  }
  const s = (t - 0.5) / 0.5
  const r = Math.round(224 + (247 - 224) * s)
  const g = Math.round(175 + (118 - 175) * s)
  const b = Math.round(104 + (142 - 104) * s)
  return `rgb(${r},${g},${b})`
}

/* ── Helpers ───────────────────────────────────────────── */

function parseInlineGraph(
  coGraph: Record<string, number>,
  priors: Record<string, number>,
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodeMap: Record<string, number> = {}
  const edges: GraphEdge[] = []
  for (const [pairKey, weight] of Object.entries(coGraph)) {
    const parts = pairKey.split('|')
    if (parts.length !== 2) continue
    const [a, b] = parts
    nodeMap[a] = (nodeMap[a] || 0) + weight
    nodeMap[b] = (nodeMap[b] || 0) + weight
    edges.push({ source: a, target: b, weight })
  }
  // Compute client-side strength when missing from response
  for (const e of edges) {
    if (typeof e.strength !== 'number') {
      const maxCount = Math.max(nodeMap[e.source] || 1, nodeMap[e.target] || 1)
      e.strength = maxCount > 0 ? e.weight / maxCount : 0
    }
  }
  const nodes: GraphNode[] = Object.entries(nodeMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([id, weight]) => ({ id, weight, prior: priors[id] ?? 0.5 }))
  return { nodes, edges }
}

/* ── Component ─────────────────────────────────────────── */

export function KnowledgeGraph({
  coOccurrenceGraph,
  priors = {},
  fetchLive = false,
  experimentRunId,
  width = 520,
  height = 380,
}: KnowledgeGraphProps) {
  // Per-experiment scoped query (preferred when experimentRunId is set)
  const experimentQuery = useQuery<CoOccurrenceResponse>({
    queryKey: ['co-occurrence-graph', experimentRunId],
    queryFn: () => api(`/agent/memory/graph/co-occurrence/${encodeURIComponent(experimentRunId!)}`),
    enabled: !!experimentRunId,
    retry: 1,
  })

  const liveQuery = useQuery<CoOccurrenceResponse>({
    queryKey: ['co-occurrence-graph'],
    queryFn: () => api('/agent/memory/graph/co-occurrence'),
    enabled: fetchLive && !experimentRunId,
    refetchInterval: 8000,
    retry: 1,
  })

  const { nodes, edges, source } = useMemo(() => {
    // 1. Prefer experiment-scoped Neo4j data when available
    if (experimentRunId && experimentQuery.data && experimentQuery.data.nodes.length > 0) {
      return { nodes: experimentQuery.data.nodes, edges: experimentQuery.data.edges, source: `neo4j (${experimentRunId})` }
    }
    // 2. Prefer live data when available
    if (fetchLive && liveQuery.data && liveQuery.data.nodes.length > 0) {
      return { nodes: liveQuery.data.nodes, edges: liveQuery.data.edges, source: liveQuery.data.source }
    }
    // 3. Fall back to inline experiment data
    if (coOccurrenceGraph && Object.keys(coOccurrenceGraph).length > 0) {
      const parsed = parseInlineGraph(coOccurrenceGraph, priors)
      return { ...parsed, source: 'experiment_inline' }
    }
    // 4. Try live even if not preferred
    if (liveQuery.data && liveQuery.data.nodes.length > 0) {
      return { nodes: liveQuery.data.nodes, edges: liveQuery.data.edges, source: liveQuery.data.source }
    }
    return { nodes: [] as GraphNode[], edges: [] as GraphEdge[], source: 'empty' }
  }, [coOccurrenceGraph, priors, fetchLive, liveQuery.data, experimentRunId, experimentQuery.data])

  if (nodes.length === 0) {
    return (
      <div className="small" style={{ color: 'var(--muted)', padding: '12px 0' }}>
        No co-occurrence data available. Run the experiment with KG integration enabled.
        {fetchLive && liveQuery.isError && (
          <span> (API error: {String(liveQuery.error)})</span>
        )}
      </div>
    )
  }

  // ── Circular layout ──────────────────────────────────────
  const PAD = { t: 30, r: 20, b: 60, l: 20 }
  const plotW = width - PAD.l - PAD.r
  const plotH = height - PAD.t - PAD.b
  const cx = PAD.l + plotW / 2
  const cy = PAD.t + plotH / 2
  const radius = Math.min(plotW, plotH) * 0.38

  const positions: Record<string, { x: number; y: number }> = {}
  const n = nodes.length
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2
    positions[node.id] = {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    }
  })

  const maxEdgeWeight = Math.max(...edges.map((e) => e.weight), 1)
  const maxNodeWeight = Math.max(...nodes.map((nd) => nd.weight), 1)

  return (
    <div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6 }}
      >
        {/* Edges */}
        {edges.map((e, i) => {
          const from = positions[e.source]
          const to = positions[e.target]
          if (!from || !to) return null
          const strokeW = Math.max(1.5, 7 * (e.weight / maxEdgeWeight))
          return (
            <g key={`e-${i}`}>
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="rgba(122,162,247,0.35)"
                strokeWidth={strokeW}
                strokeLinecap="round"
              >
                <title>{humanPattern(e.source)} ↔ {humanPattern(e.target)}: count={e.weight}{typeof e.strength === 'number' ? `, strength=${(e.strength * 100).toFixed(0)}%` : ''}</title>
              </line>
              {/* Edge weight + strength label at midpoint */}
              <text
                x={(from.x + to.x) / 2}
                y={(from.y + to.y) / 2 - 4}
                textAnchor="middle"
                fill="var(--muted)"
                fontSize={9}
                opacity={0.7}
              >
                {e.weight}{typeof e.strength === 'number' ? ` (${(e.strength * 100).toFixed(0)}%)` : ''}
              </text>
            </g>
          )
        })}

        {/* Nodes */}
        {nodes.map((node, i) => {
          const pos = positions[node.id]
          if (!pos) return null
          const r = 14 + 12 * (node.weight / maxNodeWeight)
          const fill = priorColor(node.prior)
          const label = humanPattern(node.id)
          return (
            <g key={`n-${i}`}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={r}
                fill={fill}
                stroke="rgba(255,255,255,0.2)"
                strokeWidth={2}
                opacity={0.9}
              />
              {/* Prior value inside node */}
              <text
                x={pos.x}
                y={pos.y + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#1a1b26"
                fontSize={10}
                fontWeight={700}
              >
                {node.prior.toFixed(2)}
              </text>
              {/* Label below node */}
              <text
                x={pos.x}
                y={pos.y + r + 12}
                textAnchor="middle"
                fill="var(--fg)"
                fontSize={10}
                fontWeight={500}
              >
                {label}
              </text>
            </g>
          )
        })}

        {/* Title */}
        <text x={PAD.l} y={16} fill="var(--muted)" fontSize={10}>
          Pattern Co-occurrence Network · source: {source}
        </text>
      </svg>

      {/* Edge table */}
      {edges.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>
            Co-occurrence Edges ({edges.length})
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 60px 70px',
              gap: '2px 12px',
              fontSize: 12,
              maxHeight: 180,
              overflowY: 'auto',
            }}
          >
            <div className="small" style={{ fontWeight: 600, color: 'var(--muted)' }}>Pattern A</div>
            <div className="small" style={{ fontWeight: 600, color: 'var(--muted)' }}>Pattern B</div>
            <div className="small" style={{ fontWeight: 600, color: 'var(--muted)', textAlign: 'right' }}>Count</div>
            <div className="small" style={{ fontWeight: 600, color: 'var(--muted)', textAlign: 'right' }}>Strength</div>
            {edges
              .slice()
              .sort((a, b) => b.weight - a.weight)
              .map((e, i) => (
                <React.Fragment key={i}>
                  <div className="small" title={e.source}>{humanPattern(e.source)}</div>
                  <div className="small" title={e.target}>{humanPattern(e.target)}</div>
                  <div className="small" style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{e.weight}</div>
                  <div className="small" style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', color: typeof e.strength === 'number' && e.strength >= 0.5 ? 'var(--accent)' : undefined }}>
                    {typeof e.strength === 'number' ? `${(e.strength * 100).toFixed(0)}%` : '–'}
                  </div>
                </React.Fragment>
              ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="small" style={{ color: 'var(--muted)' }}>Prior scale:</div>
        <div style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <span className="small" style={{ color: 'var(--muted)' }}>0.0</span>
          {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
            <div
              key={v}
              style={{
                width: 20,
                height: 10,
                borderRadius: 2,
                background: priorColor(v),
              }}
            />
          ))}
          <span className="small" style={{ color: 'var(--muted)' }}>1.0</span>
        </div>
        <div className="small" style={{ color: 'var(--muted)' }}>
          Edge width = co-occurrence count · Node size = total weight · (%) = normalized strength
        </div>
      </div>
    </div>
  )
}

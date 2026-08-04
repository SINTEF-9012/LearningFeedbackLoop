/**
 * DiscoveredPatternsTab — Tab 10: Automatically discovered patterns.
 *
 * Shows every pattern that the PatternDiscovery engine has found,
 * with full provenance: which confirmed events contributed, which
 * features deviated and by how much.  Users can tell at a glance
 * which real-world events spawned each discovered pattern.
 */
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/http'
import { PAL } from '../charts'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'

/* ── Types ─────────────────────────────────────────────── */

interface SourceEvent {
  memory_id: string | null
  session_id: string | null
  timestamp: number
  deviations: Record<string, string>   // feature → "high"|"low"
  z_scores: Record<string, number>     // feature → z
}

interface DiscoveredPattern {
  key: string
  features: Record<string, string>
  confirmation_count: number
  first_seen: number
  last_seen: number
  promoted: boolean
  prior: number
  source_events: SourceEvent[]
  source_memory_ids?: string[]
}

interface DiscoveredResponse {
  discovered_patterns: DiscoveredPattern[]
  count: number
}

/* ── Helpers ───────────────────────────────────────────── */

function ts(epoch: number): string {
  if (!epoch || epoch < 1e9) return '—'
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function shortKey(key: string): string {
  return key.replace(/^discovered:/, '').replaceAll('+', ' + ').replaceAll('_', ' ')
}

function dirBadge(dir: string): React.ReactNode {
  const up = dir === 'high'
  return (
    <span style={{
      display: 'inline-block', padding: '1px 5px', borderRadius: 3,
      fontSize: 10, fontWeight: 600, marginLeft: 4,
      background: up ? '#9ece6a22' : '#f7768e22',
      color: up ? '#9ece6a' : '#f7768e',
    }}>
      {up ? '▲ HIGH' : '▼ LOW'}
    </span>
  )
}

/* ── Feature deviation bar ─────────────────────────────── */

function ZBar({ z }: { z: number }) {
  const abs = Math.min(Math.abs(z), 6)
  const pct = (abs / 6) * 100
  const color = z > 0 ? '#9ece6a' : '#f7768e'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 110 }}>
      <div style={{ width: 60, height: 6, borderRadius: 3, background: '#1a1b26' }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: color, transition: 'width .3s' }} />
      </div>
      <span style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>
        {z > 0 ? '+' : ''}{z.toFixed(2)}σ
      </span>
    </div>
  )
}

/* ── Pattern card (expandable) ─────────────────────────── */

function PatternCard({ pat, idx }: { pat: DiscoveredPattern; idx: number }) {
  const [open, setOpen] = useState(false)
  const featureEntries = Object.entries(pat.features)
  const accent = PAL[idx % PAL.length]

  return (
    <div style={{
      border: `1px solid ${pat.promoted ? accent + '66' : '#333'}`,
      borderRadius: 8, marginBottom: 10, overflow: 'hidden',
      background: pat.promoted ? accent + '08' : '#1a1b2611',
    }}>
      {/* Header */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>{open ? '▾' : '▸'}</span>

        {/* Status badge */}
        <span style={{
          padding: '2px 7px', borderRadius: 4, fontSize: 10, fontWeight: 700,
          background: pat.promoted ? '#9ece6a33' : '#e0af6833',
          color: pat.promoted ? '#9ece6a' : '#e0af68',
        }}>
          {pat.promoted ? 'PROMOTED' : 'CANDIDATE'}
        </span>

        {/* Key */}
        <span style={{ fontWeight: 600, fontSize: 13, flex: 1 }}>
          {shortKey(pat.key)}
        </span>

        {/* Summary stats */}
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          {pat.confirmation_count} confirm{pat.confirmation_count !== 1 ? 's' : ''}
        </span>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>
          prior {(pat.prior * 100).toFixed(0)}%
        </span>
        <span style={{ fontSize: 10, color: 'var(--muted)' }}>
          {ts(pat.first_seen)} → {ts(pat.last_seen)}
        </span>
      </div>

      {/* Expanded detail */}
      {open && (
        <div style={{ padding: '0 14px 14px', display: 'grid', gap: 12 }}>
          {/* Feature signature */}
          <div>
            <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6, color: 'var(--muted)' }}>
              Feature Signature ({featureEntries.length} features)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '4px 12px' }}>
              {featureEntries.map(([feat, dir]) => (
                <React.Fragment key={feat}>
                  <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {feat.replaceAll('_', ' ')}
                  </span>
                  {dirBadge(dir)}
                </React.Fragment>
              ))}
            </div>
          </div>

          {/* Source events (provenance) */}
          {pat.source_events.length > 0 && (
            <div>
              <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 6, color: 'var(--muted)' }}>
                Source Events — {pat.source_events.length} confirmed event{pat.source_events.length !== 1 ? 's' : ''} contributed
              </div>
              <div style={{ display: 'grid', gap: 6 }}>
                {pat.source_events.map((se, i) => (
                  <div key={i} style={{
                    background: '#1a1b2699', borderRadius: 6, padding: '8px 10px',
                    border: '1px solid #292e42',
                  }}>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 6 }}>
                      <span style={{ fontSize: 11, color: 'var(--muted)' }}>
                        🕐 {ts(se.timestamp)}
                      </span>
                      {se.memory_id && (
                        <span style={{ fontSize: 10, fontFamily: 'monospace', color: accent }}>
                          mem:{se.memory_id.slice(0, 8)}
                        </span>
                      )}
                      {se.session_id && (
                        <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)' }}>
                          sess:{se.session_id.slice(0, 12)}
                        </span>
                      )}
                    </div>
                    {/* Z-score bars for each deviating feature */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '3px 10px' }}>
                      {Object.entries(se.z_scores || {}).map(([feat, z]) => (
                        <React.Fragment key={feat}>
                          <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                            {feat.replaceAll('_', ' ')}
                          </span>
                          <ZBar z={z} />
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Neo4j graph links (if available) */}
          {pat.source_memory_ids && pat.source_memory_ids.length > 0 && (
            <div>
              <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4, color: 'var(--muted)' }}>
                Graph Provenance — {pat.source_memory_ids.length} linked Memory nodes
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {pat.source_memory_ids.map(mid => (
                  <span key={mid} style={{
                    fontSize: 10, fontFamily: 'monospace', padding: '2px 6px',
                    borderRadius: 4, background: '#7aa2f722', color: '#7aa2f7',
                  }}>
                    {mid.slice(0, 12)}…
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Main tab ──────────────────────────────────────────── */

export function DiscoveredPatternsTab(_props: ExperimentTabProps) {
  const [showAll, setShowAll] = useState(false)

  const q = useQuery<DiscoveredResponse>({
    queryKey: ['discovered-patterns', showAll],
    queryFn: () => api(`/agent/memory/patterns/discovered?promoted_only=${!showAll}`),
    staleTime: 15_000,
    retry: 1,
  })

  const patterns = q.data?.discovered_patterns || []

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <div style={{ fontWeight: 700, fontSize: 15, flex: 1 }}>🔍 Discovered Patterns <HelpIcon text="The PatternDiscovery engine automatically learns new patterns from confirmed events. When an operator confirms a detection, the system extracts which features deviated (z-score > 2σ) and clusters similar deviations into patterns. After enough confirmations (default: 2), a candidate is promoted to a full pattern with its own prior in the scoring pipeline." /></div>
        <button className="small" onClick={() => q.refetch()} style={{ padding: '2px 8px' }}>↻</button>
      </div>

      <p className="small" style={{ color: 'var(--muted)', margin: '0 0 12px' }}>
        Patterns automatically learned from confirmed events. Each pattern
        shows which features deviated and which operator confirmations
        contributed to its discovery. Click a pattern to expand provenance.
      </p>

      {/* Filter toggle */}
      <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showAll}
            onChange={e => setShowAll(e.target.checked)}
          />
          Show candidates (not yet promoted)
        </label>
        <span className="small" style={{ color: 'var(--muted)' }}>
          {patterns.length} pattern{patterns.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Loading / empty states */}
      {q.isLoading && <div className="small" style={{ padding: 10, color: 'var(--muted)' }}>Loading…</div>}
      {q.isError && <div className="small" style={{ color: 'var(--danger)', padding: 10 }}>Failed to load: {String(q.error)}</div>}

      {!q.isLoading && patterns.length === 0 && (
        <div style={{
          padding: '24px 16px', textAlign: 'center', color: 'var(--muted)',
          border: '1px dashed #333', borderRadius: 8,
        }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>🧬</div>
          <div style={{ fontSize: 13, fontWeight: 600 }}>No discovered patterns yet</div>
          <div className="small" style={{ marginTop: 4 }}>
            Patterns emerge when operators confirm events whose feature
            signatures deviate significantly from baseline. Run an experiment
            and confirm some detections to see them here.
          </div>
        </div>
      )}

      {/* Pattern list */}
      {patterns.map((pat, i) => <PatternCard key={pat.key} pat={pat} idx={i} />)}
    </div>
  )
}

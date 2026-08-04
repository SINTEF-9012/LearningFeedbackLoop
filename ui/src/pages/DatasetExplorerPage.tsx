/**
 * DatasetExplorerPage — Interactive Site_a_line2 merged timeseries browser
 * with CNC process annotation regions from the OFs translated xlsx.
 *
 * Features:
 *   • OF selector (filter by manufacturing order)
 *   • Multi-channel selector with curated defaults
 *   • LTTB-downsampled waveform chart with zoom/pan
 *   • Colored process-annotation regions (PGM LINE mapping)
 *   • Rich hover tooltip (process, sub-process, tool, PGM line, breakage)
 *   • OF milestones summary panel (tools, processes, breakage annotations)
 */
import React, { useEffect, useState, useMemo } from 'react'
import { api } from '../api/http'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { DatasetWaveformChart, type WaveformChannel, type ProcessRegion, type AnnotationMarker } from '../components/charts/DatasetWaveformChart'

// ── Types ────────────────────────────────────────────────────────────────────

interface ChannelInfo {
  name: string
  default: boolean
}

interface OfInfo {
  id: string
  sample_count: number
  session: string
  session_label: string
}

interface SessionInfo {
  id: string
  label: string
  date_range: string
  channels: string[]
  rows: number
}

interface WaveformResponse {
  channels: WaveformChannel[]
  regions: ProcessRegion[]
  annotations?: AnnotationMarker[]
  metadata: {
    of_id: string
    side: string
    total_rows: number
    duration_s: number
    duration_h: number
    programs?: string[]
    session?: string
    session_label?: string
  }
  error?: string
}

interface MilestonesResponse {
  side: string
  name: string
  total_milestones: number
  of_ids: string[]
  of_columns: { header: string; of_number: string; col: number }[]
  breakage_summary: Record<string, { pgm_line: number; process: string | null; annotation: string }[]>
  tool_summary: { tool_id: number | string; tool_name: string | null; first_pgm_line: number }[]
  milestones: Record<string, unknown>[]
  error?: string
}

// ── Component ────────────────────────────────────────────────────────────────

export default function DatasetExplorerPage() {
  // Data state
  const [allChannels, setAllChannels] = useState<ChannelInfo[]>([])
  const [defaults, setDefaults] = useState<string[]>([])
  const [ofs, setOfs] = useState<OfInfo[]>([])
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [selectedOf, setSelectedOf] = useState('')
  const [selectedSession, setSelectedSession] = useState('')
  const [selectedChannels, setSelectedChannels] = useState<Set<string>>(new Set())
  const [channelSearch, setChannelSearch] = useState('')

  // Waveform state
  const [waveform, setWaveform] = useState<WaveformResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Milestones state
  const [milestones, setMilestones] = useState<MilestonesResponse | null>(null)
  const [milestonesOpen, setMilestonesOpen] = useState(true)

  // Max points slider
  const [maxPoints, setMaxPoints] = useState(3000)

  // Zoom-driven refetch state
  const [zoomRange, setZoomRange] = useState<[number, number] | null>(null)
  const regionsRef = React.useRef<ProcessRegion[]>([])
  const annotationsRef = React.useRef<AnnotationMarker[]>([])

  // ── Load channel list & OF list on mount ──
  useEffect(() => {
    api<{ channels: ChannelInfo[]; defaults: string[] }>('/dataset/channels')
      .then((data) => {
        setAllChannels(data.channels || [])
        const defs = data.defaults || []
        setDefaults(defs)
        setSelectedChannels(new Set(defs))
      })
      .catch(() => {})

    api<{ ofs: OfInfo[]; sessions: SessionInfo[] }>('/dataset/ofs')
      .then((data) => {
        const ofList = data.ofs || []
        setOfs(ofList)
        setSessions(data.sessions || [])
        if (ofList.length > 0) {
          setSelectedOf(ofList[0].id)
          setSelectedSession(ofList[0].session)
        }
      })
      .catch(() => {})
  }, [])

  // ── Load milestones when OF changes ──
  useEffect(() => {
    if (!selectedOf) return
    // Determine side from waveform metadata or default to A
    const side = waveform?.metadata?.side || 'A'
    api<MilestonesResponse>(`/dataset/milestones?side=${side}`)
      .then(setMilestones)
      .catch(() => {})
  }, [selectedOf, waveform?.metadata?.side])

  // ── Fetch waveform ──
  const fetchWaveform = (timeMin?: number, timeMax?: number) => {
    if (selectedChannels.size === 0) return
    const isZoomRefetch = timeMin != null || timeMax != null
    // Only show loading spinner on initial/full fetches, not zoom refinements
    if (!isZoomRefetch) {
      setLoading(true)
      setError('')
    }
    const chans = Array.from(selectedChannels).join(',')
    let q = `/dataset/waveform?channels=${encodeURIComponent(chans)}&of_id=${encodeURIComponent(selectedOf)}&session=${encodeURIComponent(selectedSession)}&max_points=${maxPoints}`
    if (timeMin != null) q += `&time_min=${timeMin.toFixed(1)}`
    if (timeMax != null) q += `&time_max=${timeMax.toFixed(1)}`
    api<WaveformResponse>(q)
      .then((data) => {
        if (data.error) {
          if (!isZoomRefetch) {
            setError(data.error)
            setWaveform(null)
          }
        } else {
          // Preserve regions and annotations from full-range fetch; zoom fetches don't
          // recompute them (they stay the same across the whole OF).
          if (!isZoomRefetch) {
            regionsRef.current = data.regions
            annotationsRef.current = data.annotations || []
          }
          setWaveform({ ...data, regions: regionsRef.current, annotations: annotationsRef.current })
        }
      })
      .catch((e) => { if (!isZoomRefetch) setError(String(e)) })
      .finally(() => { if (!isZoomRefetch) setLoading(false) })
  }

  // Auto-fetch on OF or channel change (full range)
  useEffect(() => {
    if (selectedChannels.size === 0 || !selectedOf) return
    setZoomRange(null)
    const t = setTimeout(() => fetchWaveform(), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedOf, selectedSession, selectedChannels, maxPoints])

  // Refetch with higher resolution when zoomed (debounced)
  useEffect(() => {
    if (!waveform) return
    const t = setTimeout(() => {
      if (zoomRange) {
        fetchWaveform(zoomRange[0], zoomRange[1])
      } else {
        fetchWaveform()
      }
    }, 400)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoomRange])

  // ── Filtered channels for the picker ──
  const filteredChannels = useMemo(() => {
    if (!channelSearch.trim()) return allChannels
    const q = channelSearch.toLowerCase()
    return allChannels.filter(c => c.name.toLowerCase().includes(q))
  }, [allChannels, channelSearch])

  // ── Channel toggle ──
  const toggleChannel = (name: string) => {
    setSelectedChannels(prev => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })
  }

  const resetToDefaults = () => setSelectedChannels(new Set(defaults))
  const clearChannels = () => setSelectedChannels(new Set())

  // ── Breakage entries for selected OF from milestones ──
  const breakageEntries = useMemo(() => {
    if (!milestones?.breakage_summary || !selectedOf) return []
    // Match OF number (may have different formatting)
    for (const [ofKey, entries] of Object.entries(milestones.breakage_summary)) {
      if (ofKey === selectedOf || ofKey.replace(/\s/g, '') === selectedOf) return entries
    }
    return []
  }, [milestones, selectedOf])

  // ── Process summary from milestones ──
  const processSummary = useMemo(() => {
    if (!milestones?.milestones) return []
    const procs: { process: string; count: number; pgm_range: string }[] = []
    const seen = new Map<string, { count: number; min_pgm: number; max_pgm: number }>()
    for (const m of milestones.milestones) {
      const proc = (m as Record<string, unknown>).process as string | undefined
      if (!proc) continue
      const pgm = (m as Record<string, unknown>).pgm_line as number
      const entry = seen.get(proc)
      if (entry) {
        entry.count++
        entry.min_pgm = Math.min(entry.min_pgm, pgm)
        entry.max_pgm = Math.max(entry.max_pgm, pgm)
      } else {
        seen.set(proc, { count: 1, min_pgm: pgm, max_pgm: pgm })
      }
    }
    for (const [proc, data] of seen) {
      procs.push({
        process: proc,
        count: data.count,
        pgm_range: `${data.min_pgm}–${data.max_pgm}`,
      })
    }
    return procs
  }, [milestones])

  // ── Grouped OFs by session for the picker ──
  const ofsBySession = useMemo(() => {
    const groups = new Map<string, OfInfo[]>()
    for (const of_ of ofs) {
      const list = groups.get(of_.session) || []
      list.push(of_)
      groups.set(of_.session, list)
    }
    return groups
  }, [ofs])

  // When selecting an OF, also set its session
  const handleOfChange = (value: string) => {
    if (!value) {
      setSelectedOf('')
      setSelectedSession('')
      return
    }
    // value format: "session:of_id"
    const [sess, ofId] = value.split(':')
    setSelectedSession(sess)
    setSelectedOf(ofId)
  }

  // Composite key for the select value
  const selectedOfKey = selectedSession && selectedOf ? `${selectedSession}:${selectedOf}` : ''

  return (
    <ErrorBoundary label="Dataset Explorer">
      <div className="panel" style={{ overflow: 'auto', padding: 16 }}>
        <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
          Dataset Explorer
        </div>
        <div className="small" style={{ color: 'var(--muted)', marginBottom: 8 }}>
          Site_a_line2 merged timeseries · CNC process annotations from OFs translated xlsx
        </div>

        {/* ── Session info badges ──────────────────────────── */}
        {sessions.length > 0 && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
            {sessions.map(s => {
              const isActive = s.id === selectedSession
              return (
                <div key={s.id} style={{
                  fontSize: 10, padding: '4px 10px', borderRadius: 4,
                  border: `1px solid ${isActive ? '#7aa2f7' : 'var(--border)'}`,
                  background: isActive ? 'rgba(122,162,247,0.12)' : 'transparent',
                  color: isActive ? '#7aa2f7' : 'var(--muted)',
                }}>
                  <div style={{ fontWeight: 600 }}>{s.label}</div>
                  <div>{s.date_range} · {s.channels} · {s.rows.toLocaleString()} rows</div>
                </div>
              )
            })}
          </div>
        )}

        {/* ── Controls row ──────────────────────────────────── */}
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12, alignItems: 'flex-end' }}>
          {/* OF picker — grouped by session */}
          <div>
            <div className="small" style={{ marginBottom: 2 }}>Manufacturing Order (OF)</div>
            <select
              value={selectedOfKey}
              onChange={(e) => handleOfChange(e.target.value)}
              style={{ fontSize: 12, padding: '4px 10px', minWidth: 280 }}
            >
              <option value="">All data</option>
              {Array.from(ofsBySession.entries()).map(([sessId, sessOfs]) => {
                const sessInfo = sessions.find(s => s.id === sessId)
                return (
                  <optgroup key={sessId} label={sessInfo?.label || sessId}>
                    {sessOfs.map(of => (
                      <option key={`${sessId}:${of.id}`} value={`${sessId}:${of.id}`}>
                        OF {of.id} ({of.sample_count.toLocaleString()} samples)
                      </option>
                    ))}
                  </optgroup>
                )
              })}
            </select>
          </div>

          {/* Max points slider */}
          <div>
            <div className="small" style={{ marginBottom: 2 }}>Resolution: {maxPoints.toLocaleString()} pts</div>
            <input
              type="range" min={500} max={10000} step={500}
              value={maxPoints}
              onChange={(e) => setMaxPoints(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </div>

          {/* Quick actions */}
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button onClick={resetToDefaults}
              style={{ fontSize: 10, padding: '3px 8px', borderRadius: 3, border: '1px solid var(--border)', cursor: 'pointer' }}>
              Reset defaults
            </button>
            <button onClick={clearChannels}
              style={{ fontSize: 10, padding: '3px 8px', borderRadius: 3, border: '1px solid var(--border)', cursor: 'pointer' }}>
              Clear all
            </button>
            <span className="small" style={{ color: 'var(--muted)' }}>
              {selectedChannels.size} channel{selectedChannels.size !== 1 ? 's' : ''} selected
            </span>
          </div>
        </div>

        {/* ── Channel picker ───────────────────────────────── */}
        <details style={{ marginBottom: 12 }}>
          <summary style={{ cursor: 'pointer', fontSize: 12, color: 'var(--muted)' }}>
            Channel selector ({selectedChannels.size}/{allChannels.length})
          </summary>
          <div style={{ marginTop: 6 }}>
            <input
              placeholder="Search channels…"
              value={channelSearch}
              onChange={(e) => setChannelSearch(e.target.value)}
              style={{ fontSize: 11, padding: '3px 8px', width: '100%', maxWidth: 300, marginBottom: 6 }}
            />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxHeight: 200, overflowY: 'auto' }}>
              {filteredChannels.map(ch => {
                const isSel = selectedChannels.has(ch.name)
                return (
                  <button
                    key={ch.name}
                    onClick={() => toggleChannel(ch.name)}
                    style={{
                      fontSize: 9, padding: '2px 6px', borderRadius: 3, cursor: 'pointer',
                      border: `1px solid ${isSel ? '#7aa2f7' : 'var(--border)'}`,
                      background: isSel ? 'rgba(122,162,247,0.15)' : 'transparent',
                      color: isSel ? '#7aa2f7' : 'var(--muted)',
                      fontWeight: ch.default ? 600 : 400,
                    }}
                    title={ch.name}
                  >
                    {ch.name.replace(/^(Monit_chatter_detection_|Cnc_Override_|Axis_FeedRate_)/, '').replace(/_/g, ' ')}
                  </button>
                )
              })}
            </div>
          </div>
        </details>

        {/* ── Waveform chart ───────────────────────────────── */}
        {loading && !waveform && (
          <div style={{ padding: 20, textAlign: 'center', color: 'var(--muted)' }}>
            Loading waveform data…
          </div>
        )}
        {error && (
          <div style={{ padding: 12, color: '#f7768e', background: 'rgba(247,118,142,0.1)', borderRadius: 6, marginBottom: 12, fontSize: 12 }}>
            {error}
          </div>
        )}
        {waveform && (
          <div style={{ marginBottom: 16 }}>
            <div className="small" style={{ color: 'var(--muted)', marginBottom: 4 }}>
              OF {waveform.metadata.of_id || 'all'} · {waveform.metadata.session_label || waveform.metadata.session || '—'} · Side {waveform.metadata.side} ·{' '}
              {waveform.metadata.total_rows.toLocaleString()} rows · {waveform.metadata.duration_h}h duration
              {waveform.metadata.programs?.length ? ` · Program: ${waveform.metadata.programs.join(', ')}` : ''}
            </div>
            <DatasetWaveformChart
              channels={waveform.channels}
              regions={waveform.regions}
              annotations={waveform.annotations}
              durationSeconds={waveform.metadata.duration_s}
              durationHours={waveform.metadata.duration_h}
              title={`OF ${waveform.metadata.of_id || 'all'} — ${waveform.metadata.session_label || ''} — Side ${waveform.metadata.side}`}
              width={960}
              height={440}
              onZoomChange={setZoomRange}
            />
          </div>
        )}

        {/* ── OF Milestones Panel ──────────────────────────── */}
        {milestones && !milestones.error && (
          <div style={{ marginTop: 8, border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden' }}>
            <button
              onClick={() => setMilestonesOpen(!milestonesOpen)}
              style={{
                width: '100%', textAlign: 'left', padding: '8px 12px',
                background: 'rgba(122,162,247,0.05)', border: 'none', cursor: 'pointer',
                fontSize: 13, fontWeight: 600, color: 'var(--fg)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}
            >
              <span>OF Milestones — {milestones.name}</span>
              <span style={{ fontSize: 10, color: 'var(--muted)' }}>
                {milestones.total_milestones} milestones · {milestones.of_ids.length} OFs · {milestones.tool_summary.length} tools
                {milestonesOpen ? ' ▴' : ' ▾'}
              </span>
            </button>

            {milestonesOpen && (
              <div style={{ padding: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                {/* Breakage annotations */}
                <div style={{ flex: '1 1 300px', minWidth: 280 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: '#f7768e' }}>
                    ⚠ Breakage Annotations
                    {breakageEntries.length > 0 ? ` (${breakageEntries.length})` : ''}
                  </div>
                  {breakageEntries.length > 0 ? (
                    <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                      {breakageEntries.map((entry, i) => (
                        <div key={i} style={{
                          fontSize: 10, padding: '4px 8px', marginBottom: 3,
                          background: 'rgba(247,118,142,0.08)', borderRadius: 4,
                          borderLeft: '3px solid #f7768e',
                        }}>
                          <div style={{ fontWeight: 600 }}>{entry.annotation}</div>
                          <div style={{ color: 'var(--muted)' }}>
                            PGM Line {entry.pgm_line} · Process: {entry.process || '—'}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 10, color: 'var(--muted)' }}>
                      {selectedOf ? `No breakage annotations for OF ${selectedOf}` : 'Select an OF to see breakage annotations'}
                    </div>
                  )}
                </div>

                {/* Tools summary */}
                <div style={{ flex: '1 1 250px', minWidth: 230 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>🔧 Tools</div>
                  <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                    <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>ID</th>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>Name</th>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>First PGM</th>
                        </tr>
                      </thead>
                      <tbody>
                        {milestones.tool_summary.map((t, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            <td style={{ padding: '2px 6px', color: '#7aa2f7' }}>{t.tool_id}</td>
                            <td style={{ padding: '2px 6px' }}>{t.tool_name || '—'}</td>
                            <td style={{ padding: '2px 6px', color: 'var(--muted)' }}>{t.first_pgm_line}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Process summary */}
                <div style={{ flex: '1 1 300px', minWidth: 280 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>⚙ Processes ({processSummary.length})</div>
                  <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                    <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>Process</th>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>Count</th>
                          <th style={{ textAlign: 'left', padding: '2px 6px' }}>PGM Range</th>
                        </tr>
                      </thead>
                      <tbody>
                        {processSummary.map((p, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            <td style={{ padding: '2px 6px' }}>{p.process}</td>
                            <td style={{ padding: '2px 6px', color: 'var(--muted)' }}>{p.count}</td>
                            <td style={{ padding: '2px 6px', color: 'var(--muted)' }}>{p.pgm_range}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Available OFs in milestone sheet */}
                <div style={{ flex: '1 1 200px', minWidth: 180 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>📋 OF Columns</div>
                  <div style={{ fontSize: 10 }}>
                    {milestones.of_columns.map((of, i) => {
                      const brks = milestones.breakage_summary[of.of_number] || []
                      return (
                        <div key={i} style={{
                          padding: '3px 6px', marginBottom: 2,
                          background: of.of_number === selectedOf ? 'rgba(122,162,247,0.12)' : 'transparent',
                          borderRadius: 3,
                          cursor: 'pointer',
                        }}
                          onClick={() => {
                            // Find the matching OF entry to get its session
                            const match = ofs.find(o => o.id === of.of_number || o.id.replace(/\s/g, '') === of.of_number)
                            if (match) {
                              setSelectedOf(match.id)
                              setSelectedSession(match.session)
                            } else {
                              setSelectedOf(of.of_number)
                            }
                          }}
                        >
                          <span style={{ fontWeight: of.of_number === selectedOf ? 700 : 400 }}>{of.header}</span>
                          {brks.length > 0 && (
                            <span style={{ color: '#f7768e', marginLeft: 6 }}>⚠ {brks.length} breakage</span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </ErrorBoundary>
  )
}

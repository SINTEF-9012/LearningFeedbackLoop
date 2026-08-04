/**
 * SignalTimelineTab — Tab 7: Raw waveform + feature scatter views.
 *
 * This is the most complex tab — manages sample selection, channel toggling,
 * annotation overlays, and two sub-views (timeseries / features).
 */
import React, { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/http'
import {
  TimeSeriesChart,
  FeatureSignalSVG,
  OperationWaveformChart,
} from '../charts'
import type { ChannelData, SampleAnnotation } from '../charts'
import type { WaveformChannel, EventRegion } from '../charts'
import type { FeatureData } from '../../state/experimentStore'
import type { ExperimentTabProps } from './types'
import { HelpIcon } from '../Tooltip'

/** Extended annotation from the API (may include nulls) */
interface ApiAnnotation {
  sample_id: string | null
  sample_idx: number | null
  phase: string
  true_label: string | null
  predicted: string | null
  combined_score: number | null
  pattern_score: number | null
  model_score: number | null
  event_triggered: boolean
  patterns_detected: string[]
  correct: boolean | null
}

export function SignalTimelineTab({ effectiveRunId }: ExperimentTabProps) {
  const [tsSampleIdx, setTsSampleIdx] = useState(0)
  const [tsChannels, setTsChannels] = useState<Set<string>>(new Set())
  const [tsView, setTsView] = useState<'timeseries' | 'features' | 'operation'>('timeseries')
  const [featureCols, setFeatureCols] = useState<string[]>([])
  const [opId, setOpId] = useState('OF00001')
  const [opChannels, setOpChannels] = useState<Set<string>>(new Set())

  // Feature data
  const featuresQ = useQuery<FeatureData>({
    queryKey: ['experiment-features', effectiveRunId],
    queryFn: () => api(`/agent/memory/experiment/features?run_id=${encodeURIComponent(effectiveRunId)}&limit=500`),
    enabled: tsView === 'features' && Boolean(effectiveRunId),
    staleTime: 120_000,
  })

  // Raw time-series data for selected sample
  const tsQuery = useQuery<{
    sample_idx: number; sample_id: string; label: string
    channels: ChannelData[]; all_channel_names: string[]
    n_timesteps: number; total_samples: number
  }>({
    queryKey: ['experiment-timeseries', tsSampleIdx, Array.from(tsChannels).sort().join(',')],
    queryFn: () => {
      const chParam = tsChannels.size ? `&channels=${Array.from(tsChannels).join(',')}` : ''
      return api(`/experiment/timeseries?sample_idx=${tsSampleIdx}${chParam}`)
    },
    enabled: tsView === 'timeseries',
    staleTime: 300_000,
  })

  // Per-sample annotations
  const annotationsQ = useQuery<{
    run_id: string; threshold: number; total_samples: number
    samples: ApiAnnotation[]; phases: string[]
  }>({
    queryKey: ['experiment-annotations', effectiveRunId],
    queryFn: () => api(`/experiment/annotations?run_id=${encodeURIComponent(effectiveRunId)}`),
    enabled: Boolean(effectiveRunId),
    staleTime: 120_000,
  })

  // Available operations
  const opsQ = useQuery<{ operations: { id: string; n_csv_files: number }[] }>({
    queryKey: ['experiment-operations'],
    queryFn: () => api('/experiment/operations'),
    enabled: tsView === 'operation',
    staleTime: 300_000,
  })

  // Full-operation waveform
  const waveformQ = useQuery<{
    operation_id: string
    channels: WaveformChannel[]
    all_channel_names: string[]
    regions: EventRegion[]
    total_points: number
    displayed_points: number
    duration_seconds: number
    duration_hours: number
    start_time: string
    end_time: string
    error?: string
  }>({
    queryKey: ['operation-waveform', opId, Array.from(opChannels).sort().join(','), effectiveRunId],
    queryFn: () => {
      const chParam = opChannels.size ? `&channels=${Array.from(opChannels).join(',')}` : ''
      const runParam = effectiveRunId ? `&run_id=${encodeURIComponent(effectiveRunId)}` : ''
      return api(`/experiment/operation-waveform?operation_id=${encodeURIComponent(opId)}${chParam}${runParam}&max_points=3000`)
    },
    enabled: tsView === 'operation' && Boolean(opId),
    staleTime: 300_000,
  })

  const annotationByIdx = useMemo(() => {
    const m = new Map<number, ApiAnnotation>()
    if (!annotationsQ.data?.samples) return m
    for (const s of annotationsQ.data.samples) {
      if (s.sample_idx != null) m.set(s.sample_idx, s)
    }
    return m
  }, [annotationsQ.data])

  return (
    <div className="card" style={{ padding: 16 }}>
      {/* View toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <div style={{ fontWeight: 700 }}>Signal Timeline <HelpIcon text="Visualise raw CNC sensor data. Sample View shows one analysis window at a time with annotation overlays. Full Operation shows the entire operation waveform with event regions highlighted. Feature Scatter shows extracted statistical features plotted per-sample." /></div>
        <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border)' }}>
          {([['timeseries', '📈 Sample View'], ['operation', '🔭 Full Operation'], ['features', '📊 Feature Scatter']] as const).map(([k, lbl]) => (
            <button
              key={k}
              title={k === 'timeseries' ? 'View one analysis window at a time, navigate between samples, see channel waveforms with prediction annotations' : k === 'operation' ? 'View the full multi-hour operation waveform with LTTB downsampling and event region overlays' : 'Scatter plot of extracted statistical features (mean, std, slope, etc.) coloured by label'}
              style={{
                padding: '4px 12px', fontSize: 12, cursor: 'pointer',
                border: 'none',
                background: tsView === k ? 'var(--accent)' : 'transparent',
                color: tsView === k ? '#fff' : 'var(--muted)',
              }}
              onClick={() => setTsView(k)}
            >
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* ── Raw Time-Series View ── */}
      {tsView === 'timeseries' && (
        <>
          {tsQuery.isLoading && <div className="small">Loading raw signal data…</div>}
          {tsQuery.isError && <div className="small" style={{ color: 'var(--danger)' }}>Failed to load time-series. Run feature extraction first.</div>}
          {tsQuery.data && (() => {
            const ts = tsQuery.data
            const ann = annotationByIdx.get(tsSampleIdx) || undefined
            const annThreshold = annotationsQ.data?.threshold
            return (
              <>
                {/* Sample strip map */}
                {annotationsQ.data && annotationsQ.data.samples.length > 0 && (
                  <div style={{ marginBottom: 10 }}>
                    <div className="small" style={{ color: 'var(--muted)', marginBottom: 4 }}>
                      Sample map — click to jump <HelpIcon text="Each coloured block is one analysis window. Colour = phase. Red blocks are event-triggered samples. Click any block to jump to that sample." position="right" /> <span style={{ marginLeft: 8 }}>🟦 train 🟨 eval 🟪 test 🔴 event</span>
                    </div>
                    <div style={{ display: 'flex', height: 18, borderRadius: 4, overflow: 'hidden', cursor: 'pointer', border: '1px solid var(--border)' }}>
                      {Array.from({ length: ts.total_samples }, (_, i) => {
                        const a = annotationByIdx.get(i)
                        const isEvent = a?.event_triggered
                        const phase = a?.phase || ''
                        const bg = isEvent ? '#f7768e'
                          : phase === 'train' ? '#7aa2f7'
                          : phase === 'eval' ? '#e0af68'
                          : phase === 'test' ? '#bb9af7'
                          : phase === 'baseline' ? '#73daca'
                          : '#333'
                        return (
                          <div
                            key={i}
                            onClick={() => setTsSampleIdx(i)}
                            style={{
                              flex: 1, background: bg,
                              opacity: i === tsSampleIdx ? 1 : 0.5,
                              borderRight: i === tsSampleIdx ? '2px solid #fff' : 'none',
                              borderLeft: i === tsSampleIdx ? '2px solid #fff' : 'none',
                              minWidth: 1,
                            }}
                            title={`Sample ${i + 1}: ${phase}${isEvent ? ' ⚡ EVENT' : ''}${a?.combined_score != null && Number.isFinite(a.combined_score) ? ` score=${(a.combined_score * 100).toFixed(0)}%` : ''}`}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Sample navigator */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                  <button
                    disabled={tsSampleIdx <= 0}
                    onClick={() => setTsSampleIdx(i => Math.max(0, i - 1))}
                    style={{ padding: '4px 10px', cursor: 'pointer', borderRadius: 4, border: '1px solid var(--border)', background: 'transparent', color: 'var(--fg)' }}
                  >
                    « Prev
                  </button>
                  <span className="small">Sample <strong>{ts.sample_idx + 1}</strong> / {ts.total_samples}</span>
                  <button
                    disabled={tsSampleIdx >= ts.total_samples - 1}
                    onClick={() => setTsSampleIdx(i => Math.min(ts.total_samples - 1, i + 1))}
                    style={{ padding: '4px 10px', cursor: 'pointer', borderRadius: 4, border: '1px solid var(--border)', background: 'transparent', color: 'var(--fg)' }}
                  >
                    Next »
                  </button>
                  <input type="range" min={0} max={ts.total_samples - 1} value={tsSampleIdx} onChange={e => setTsSampleIdx(Number(e.target.value))} style={{ flex: 1, minWidth: 120 }} />
                  <span className="small" style={{ padding: '2px 8px', borderRadius: 4, background: (ts.label === 'pre_stoppage' || ts.label === 'pre_break') ? 'rgba(247,118,142,0.2)' : 'rgba(158,206,106,0.2)', color: (ts.label === 'pre_stoppage' || ts.label === 'pre_break') ? '#f7768e' : '#9ece6a' }}>{(ts.label === 'pre_stoppage' || ts.label === 'pre_break') ? 'pre-stoppage' : ts.label}</span>
                  <span className="small" style={{ color: 'var(--muted)' }}>ID: {ts.sample_id}</span>
                </div>

                {/* Channel selector */}
                {ts.all_channel_names && (
                  <div style={{ marginBottom: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    <button
                      className="small"
                      style={{ padding: '2px 8px', borderRadius: 4, border: '1px solid var(--border)', background: tsChannels.size === 0 ? 'var(--accent)' : 'transparent', color: tsChannels.size === 0 ? '#fff' : 'var(--muted)', cursor: 'pointer', fontSize: 10 }}
                      onClick={() => setTsChannels(new Set())}
                    >
                      Defaults
                    </button>
                    {ts.all_channel_names.map(c => {
                      const active = tsChannels.size === 0 ? ts.channels.some(ch => ch.name === c) : tsChannels.has(c)
                      return (
                        <button
                          key={c}
                          className="small"
                          style={{ padding: '2px 6px', borderRadius: 3, border: '1px solid var(--border)', background: active ? 'rgba(122,162,247,0.15)' : 'transparent', color: active ? 'var(--fg)' : 'var(--muted)', fontSize: 10, cursor: 'pointer' }}
                          onClick={() => {
                            setTsChannels(prev => {
                              const next = new Set(prev.size ? prev : new Set(ts.channels.map(ch => ch.name)))
                              next.has(c) ? next.delete(c) : next.add(c)
                              return next
                            })
                          }}
                        >
                          {c.replace(/^(Machine_State_|Axis_Power_|Vibration_|Energy_)/, '')}
                        </button>
                      )
                    })}
                  </div>
                )}

                {/* Time-series chart */}
                <TimeSeriesChart
                  channels={ts.channels}
                  nTimesteps={ts.n_timesteps}
                  title={`Sample ${ts.sample_idx + 1}: ${ts.sample_id}`}
                  label={ts.label}
                  width={860}
                  height={360}
                  annotation={ann ? {
                    phase: ann.phase,
                    true_label: ann.true_label ?? undefined,
                    predicted: ann.predicted ?? undefined,
                    combined_score: ann.combined_score ?? undefined,
                    pattern_score: ann.pattern_score ?? undefined,
                    model_score: ann.model_score ?? undefined,
                    event_triggered: ann.event_triggered,
                    patterns_detected: ann.patterns_detected,
                    correct: ann.correct ?? undefined,
                    threshold: annThreshold,
                  } : undefined}
                />

                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                  {ts.channels.length} channels · {ts.n_timesteps} timesteps per channel · Click legend items to toggle channels
                </div>
              </>
            )
          })()}
        </>
      )}

      {/* ── Full Operation Waveform View ── */}
      {tsView === 'operation' && (
        <>
          {/* Operation selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
            <label className="small" style={{ fontWeight: 600, color: 'var(--fg)' }}>Operation:</label>
            <select
              value={opId}
              onChange={e => { setOpId(e.target.value); setOpChannels(new Set()) }}
              style={{
                padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)',
                background: 'var(--bg-card, #1a1b26)', color: 'var(--fg)', fontSize: 12,
              }}
            >
              {opsQ.data?.operations.map(op => (
                <option key={op.id} value={op.id}>{op.id}</option>
              )) ?? <option value={opId}>{opId}</option>}
            </select>
            {waveformQ.data && !waveformQ.data.error && (
              <span className="small" style={{ color: 'var(--muted)' }}>
                {waveformQ.data.duration_hours.toFixed(1)}h · {waveformQ.data.total_points.toLocaleString()} points → {waveformQ.data.displayed_points.toLocaleString()} displayed
                · {waveformQ.data.regions.length} event region{waveformQ.data.regions.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {waveformQ.isLoading && <div className="small">Loading full operation data… (this may take a moment)</div>}
          {waveformQ.isError && <div className="small" style={{ color: 'var(--danger)' }}>Failed to load operation data.</div>}
          {waveformQ.data?.error && <div className="small" style={{ color: 'var(--danger)' }}>{waveformQ.data.error}</div>}
          {waveformQ.data && !waveformQ.data.error && (() => {
            const wf = waveformQ.data
            return (
              <>
                {/* Channel selector */}
                {wf.all_channel_names && wf.all_channel_names.length > 0 && (
                  <div style={{ marginBottom: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    <button
                      className="small"
                      style={{ padding: '2px 8px', borderRadius: 4, border: '1px solid var(--border)', background: opChannels.size === 0 ? 'var(--accent)' : 'transparent', color: opChannels.size === 0 ? '#fff' : 'var(--muted)', cursor: 'pointer', fontSize: 10 }}
                      onClick={() => setOpChannels(new Set())}
                    >
                      Defaults
                    </button>
                    {wf.all_channel_names.map(c => {
                      const active = opChannels.size === 0 ? wf.channels.some(ch => ch.name === c) : opChannels.has(c)
                      return (
                        <button
                          key={c}
                          className="small"
                          style={{ padding: '2px 6px', borderRadius: 3, border: '1px solid var(--border)', background: active ? 'rgba(122,162,247,0.15)' : 'transparent', color: active ? 'var(--fg)' : 'var(--muted)', fontSize: 10, cursor: 'pointer' }}
                          onClick={() => {
                            setOpChannels(prev => {
                              const next = new Set(prev.size ? prev : new Set(wf.channels.map(ch => ch.name)))
                              next.has(c) ? next.delete(c) : next.add(c)
                              return next
                            })
                          }}
                        >
                          {c.replace(/^(Machine_State_|Axis_Power_|Vibration_|Energy_)/, '')}
                        </button>
                      )
                    })}
                  </div>
                )}

                {/* Region legend */}
                {wf.regions.length > 0 && (
                  <div className="small" style={{ marginBottom: 8, display: 'flex', gap: 14, alignItems: 'center', color: 'var(--muted)' }}>
                    <span>Region colours:</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: '#f7768e', display: 'inline-block' }} /> Pre-stoppage
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: '#9ece6a', display: 'inline-block' }} /> Confirmed
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: '#7aa2f7', display: 'inline-block' }} /> Dismissed
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: '#e0af68', display: 'inline-block' }} /> Normal/Other
                    </span>
                  </div>
                )}

                <OperationWaveformChart
                  channels={wf.channels}
                  regions={wf.regions}
                  durationSeconds={wf.duration_seconds}
                  durationHours={wf.duration_hours}
                  operationId={wf.operation_id}
                  width={900}
                  height={400}
                />
              </>
            )
          })()}
        </>
      )}

      {/* ── Feature Scatter View ── */}
      {tsView === 'features' && (
        <>
          {featuresQ.isLoading && <div className="small">Loading feature data…</div>}
          {featuresQ.isError && <div className="small" style={{ color: 'var(--danger)' }}>Failed to load features.</div>}
          {featuresQ.data && (() => {
            const data = featuresQ.data
            const availCols = data.feature_columns.slice(0, 40)
            const selected = featureCols.length ? featureCols : availCols.slice(0, 6)
            return (
              <>
                <div className="small" style={{ marginBottom: 8 }}>
                  {data.total_rows} samples · {data.feature_columns.length} features · showing {selected.length} signals
                </div>
                <div style={{ marginBottom: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {availCols.map(c => {
                    const isSel = selected.includes(c)
                    return (
                      <button
                        key={c}
                        className="small"
                        style={{ padding: '1px 6px', borderRadius: 3, border: '1px solid var(--border)', background: isSel ? 'var(--accent)' : 'transparent', color: isSel ? '#fff' : 'var(--muted)', fontSize: 10, cursor: 'pointer' }}
                        onClick={() => {
                          if (isSel) setFeatureCols(selected.filter(x => x !== c))
                          else setFeatureCols([...selected, c])
                        }}
                      >
                        {c.replace(/^(power_|vib_|chatter_|feed_|spindle_|temp_)/, '')}
                      </button>
                    )
                  })}
                </div>
                {selected.map(col => (
                  <FeatureSignalSVG key={col} rows={data.rows} column={col} labelKey="label" />
                ))}
              </>
            )
          })()}
        </>
      )}
    </div>
  )
}

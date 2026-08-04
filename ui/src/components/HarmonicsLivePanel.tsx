import React, { useMemo } from 'react'

import { useInferenceStore, type InferencePoint } from '../state/inferenceStore'
import { HarmonicScoreChart } from './HarmonicScoreChart'

type NamedValue = {
  label: string
  value: number | null | undefined
}

type HarmonicsLivePanelProps = {
  sessionInfo?: Record<string, unknown> | null
}

function formatNumber(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1000) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(2)
  if (abs >= 1) return value.toFixed(3)
  if (abs === 0) return '0.000'
  return value.toExponential(2)
}

function formatScore(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toFixed(4)
}

function harmonicDatasetLabel(dataset?: string | null): string {
  if (!dataset) return 'Auto'
  if (dataset === 'pair_lfl') return 'Pair LFL'
  if (dataset === 'pair_casedata') return 'Pair casedata'
  if (dataset === 'pair_raw') return 'Pair raw'
  if (dataset === 'casedata') return 'Casedata'
  if (dataset === 'raw_accelerometer') return 'Raw accelerometer'
  return dataset.replace(/_/g, ' ')
}

function resolveActivePairDataset(sessionInfo?: Record<string, unknown> | null): string | null {
  const cfg = sessionInfo?.config && typeof sessionInfo.config === 'object'
    ? sessionInfo.config as Record<string, unknown>
    : {}
  const metadata = sessionInfo?.metadata && typeof sessionInfo.metadata === 'object'
    ? sessionInfo.metadata as Record<string, unknown>
    : {}
  const explicit = [cfg.harmonic_dataset, metadata.harmonic_dataset, metadata.harmonic_dataset_name]
    .find((value): value is string => typeof value === 'string' && value.trim().length > 0)

  if (explicit && explicit.startsWith('pair_')) return explicit
  if (explicit === 'casedata') return 'pair_lfl'

  const casedata = metadata.casedata && typeof metadata.casedata === 'object'
    ? metadata.casedata as Record<string, unknown>
    : {}
  const sourceHints = [
    typeof metadata.source === 'string' ? metadata.source.toLowerCase() : '',
    typeof casedata.root === 'string' ? casedata.root.toLowerCase() : '',
    typeof casedata.case_dir === 'string' ? casedata.case_dir.toLowerCase() : '',
    typeof metadata.machine_id === 'string' ? metadata.machine_id.toLowerCase() : '',
  ].filter(Boolean).join(' ')

  if (sourceHints.includes('site_c') || sourceHints.includes('site_b') || sourceHints.includes('casedata')) {
    return 'pair_lfl'
  }

  return null
}

function statusLabel(status?: string | null): string {
  if (!status) return 'streaming'
  if (status === 'zero_input') return 'zero input'
  if (status === 'warming_up') return 'warming up'
  if (status === 'no_pair_columns') return 'no pair columns'
  if (status === 'no_harmonic_columns') return 'no harmonic columns'
  if (status === 'nan_logit') return 'invalid'
  if (status === 'invalid_score') return 'invalid'
  return status.replace(/_/g, ' ')
}

function statusColor(status?: string | null): string {
  if (!status) return 'var(--ok)'
  if (status === 'zero_input') return 'var(--accent)'
  if (status === 'warming_up') return 'var(--muted)'
  return 'var(--danger)'
}

function formatTime(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (value > 1_000_000_000) {
    return new Date(value * 1000).toLocaleTimeString()
  }
  return `${value.toFixed(2)} s`
}

function buildFeatureRows(point: InferencePoint | null): NamedValue[] {
  const features = point?.features ?? {}
  return [
    { label: 'Spindle speed (mean)', value: features.spindle_speed_mean },
    { label: 'Feed rate (mean)', value: features.feed_rate_mean },
    { label: 'Spindle power (mean)', value: features.power_spindle_mean },
    { label: 'Active power (mean)', value: features.power_active_mean },
    { label: 'Vibration severity (mean)', value: features.vib_severity_mean },
    { label: 'Chatter ratio', value: features.chatter_ratio },
    { label: 'Z-score', value: point?.scores.z_score },
  ]
}

function buildPairRows(point: InferencePoint | null): Array<{ label: string; value: number }> {
  const labels = point?.harmonic_pair_feature_labels ?? []
  const values = point?.harmonic_pair_values ?? []
  return labels
    .map((label, index) => ({ label, value: values[index] }))
    .filter((row): row is { label: string; value: number } => typeof row.value === 'number' && Number.isFinite(row.value))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
}

function SummaryCard({ label, value, tone = 'var(--text)', detail }: { label: string; value: string; tone?: string; detail?: string }) {
  return (
    <div className="panelCard" style={{ minWidth: 180, flex: '1 1 180px' }}>
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: tone, marginTop: 6, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {detail && <div style={{ marginTop: 4, fontSize: 11, color: 'var(--muted)' }}>{detail}</div>}
    </div>
  )
}

function ValueTable({ title, rows }: { title: string; rows: NamedValue[] }) {
  return (
    <div className="panelCard">
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>{title}</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                <td style={{ padding: '6px 8px', color: 'var(--muted)' }}>{row.label}</td>
                <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{formatNumber(row.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function HarmonicsLivePanel({ sessionInfo = null }: HarmonicsLivePanelProps) {
  const points = useInferenceStore((s) => s.points)
  const latest = points.length > 0 ? points[points.length - 1] : null

  const featureRows = useMemo(() => buildFeatureRows(latest), [latest])
  const pairRows = useMemo(() => buildPairRows(latest).slice(0, 10), [latest])
  const recentRows = useMemo(() => [...points].slice(-12).reverse(), [points])

  const pairScore = latest?.scores.harmonic_pair_score
  const contextScore = latest?.scores.harmonic_context_score
  const pairStatus = latest?.harmonic_status?.pair
  const contextStatus = latest?.harmonic_status?.context
  const pairThreshold = latest?.harmonic_thresholds?.pair
  const contextThreshold = latest?.harmonic_thresholds?.context
  const pairTriggered = typeof pairScore === 'number' && typeof pairThreshold === 'number' && pairScore >= pairThreshold
  const activePairDataset = useMemo(() => resolveActivePairDataset(sessionInfo), [sessionInfo])
  const pairDatasetText = harmonicDatasetLabel(activePairDataset)

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <SummaryCard
          label={`Pair score${activePairDataset ? ` (${pairDatasetText})` : ''}`}
          value={formatScore(pairScore)}
          tone={typeof pairScore === 'number' && typeof pairThreshold === 'number' && pairScore >= pairThreshold ? 'var(--danger)' : 'var(--ok)'}
          detail={`status ${statusLabel(pairStatus)}`}
        />
        <SummaryCard
          label="Pair threshold"
          value={formatScore(pairThreshold)}
          tone="var(--accent)"
          detail={pairTriggered ? 'triggered on latest window' : 'latest window below threshold'}
        />
        <SummaryCard
          label="Pair model"
          value={pairDatasetText}
          tone="var(--accent)"
          detail={activePairDataset ? 'active pair dataset from session config' : 'session did not expose an explicit pair dataset'}
        />
        <SummaryCard
          label="Context score"
          value={formatScore(contextScore)}
          tone={statusColor(contextStatus)}
          detail={`status ${statusLabel(contextStatus)}`}
        />
        <SummaryCard
          label="Inference windows"
          value={String(points.length)}
          tone="var(--text)"
          detail={latest ? `latest window ${latest.window[0]}-${latest.window[1]}` : 'waiting for first window'}
        />
      </div>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div className="panelCard" style={{ flex: '2 1 520px', minWidth: 360 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', marginBottom: 8, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>Live harmonic score graph</div>
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>
                {activePairDataset
                  ? `${pairDatasetText} and context scores from the live inference stream for the selected session.`
                  : 'Pair and context scores from the live inference stream for the selected session.'}
              </div>
            </div>
            {latest && (
              <div style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'monospace' }}>
                t1 {formatTime(latest.t1 ?? latest.t)}
              </div>
            )}
          </div>
          <HarmonicScoreChart height={320} />
        </div>

        <div style={{ flex: '1 1 340px', minWidth: 300, display: 'grid', gap: 12 }}>
          <div className="panelCard">
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Latest harmonic window</div>
            {latest ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <tbody>
                    {[
                      { label: 'Window', value: `${latest.window[0]}-${latest.window[1]}` },
                      { label: 'Window seconds', value: typeof latest.window_seconds === 'number' ? latest.window_seconds.toFixed(2) : '—' },
                      { label: 'Window samples', value: typeof latest.window_entries === 'number' ? String(latest.window_entries) : '—' },
                      { label: 'Sample rate', value: formatNumber(latest.sample_rate_hz ?? latest.fs) },
                      { label: 'Pair model', value: pairDatasetText },
                      { label: 'Pair status', value: statusLabel(pairStatus) },
                      { label: 'Context status', value: statusLabel(contextStatus) },
                    ].map((row) => (
                      <tr key={row.label} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        <td style={{ padding: '6px 8px', color: 'var(--muted)' }}>{row.label}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{row.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>Waiting for the first inference window.</div>
            )}
          </div>

          <ValueTable title="Live model variables" rows={featureRows} />

          <div className="panelCard">
            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Latest pair inputs</div>
            {pairRows.length === 0 ? (
              <div style={{ fontSize: 12, color: 'var(--muted)' }}>
                Pair amplitudes will appear here when the backend emits `harmonic_pair_values`.
              </div>
            ) : (
              <div style={{ overflowX: 'auto', maxHeight: 260, overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px' }}>Feature</th>
                      <th style={{ textAlign: 'right', padding: '6px 8px' }}>Amplitude</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairRows.map((row) => (
                      <tr key={row.label} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        <td style={{ padding: '6px 8px' }}>{row.label}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{formatNumber(row.value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="panelCard">
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', flexWrap: 'wrap', marginBottom: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 700 }}>Recent harmonic windows</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>
            Latest 12 inference windows from the live websocket store.
          </div>
        </div>
        {recentRows.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>No inference windows received yet.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>t1</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px' }}>Pair</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Pair status</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px' }}>Threshold</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px' }}>Context</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Context status</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Window</th>
                </tr>
              </thead>
              <tbody>
                {recentRows.map((point) => (
                  <tr key={`${point.t}-${point.i_center}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{formatTime(point.t1 ?? point.t)}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{formatScore(point.scores.harmonic_pair_score)}</td>
                    <td style={{ padding: '6px 8px', color: statusColor(point.harmonic_status?.pair) }}>{statusLabel(point.harmonic_status?.pair)}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{formatScore(point.harmonic_thresholds?.pair)}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{formatScore(point.scores.harmonic_context_score)}</td>
                    <td style={{ padding: '6px 8px', color: statusColor(point.harmonic_status?.context) }}>{statusLabel(point.harmonic_status?.context)}</td>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{point.window[0]}-{point.window[1]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
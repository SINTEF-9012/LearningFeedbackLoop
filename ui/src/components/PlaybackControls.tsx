/**
 * PlaybackControls — pause/resume/replay/speed/batch controls.
 *
 * Extracted from App.tsx's `<details>🏛️ Playback Controls</details>` section.
 */
import React, { useState, useEffect } from 'react'
import { api } from '../api/http'
import { useStreamStore } from '../state/streamStore'
import { useQuery, type UseQueryResult } from '@tanstack/react-query'

type HarmonicScorerKind = 'context' | 'pair'
type HarmonicDataset = 'auto' | 'casedata' | 'site_a_line2' | 'raw_accelerometer' | 'pair_raw' | 'pair_casedata' | 'pair_lfl'
type HarmonicStatusDataset = Exclude<HarmonicDataset, 'auto'> | 'default'

type HarmonicStatusResponse = {
  available: boolean
  torch_installed: boolean
  model_loaded: boolean
  dataset_name: string
  model_save_path: string
}

const CONTEXT_DATASET_OPTIONS: Array<{ value: HarmonicDataset; label: string }> = [
  { value: 'auto', label: 'Auto' },
  { value: 'casedata', label: 'Casedata' },
  { value: 'site_a_line2', label: 'Site_a_line2' },
  { value: 'raw_accelerometer', label: 'Raw accelerometer' },
]

const PAIR_DATASET_OPTIONS: Array<{ value: HarmonicDataset; label: string }> = [
  { value: 'auto', label: 'Auto' },
  { value: 'pair_lfl', label: 'Pair LFL' },
  { value: 'pair_casedata', label: 'Pair casedata' },
  { value: 'pair_raw', label: 'Pair raw' },
]

function resolveHarmonicDataset(
  scorerKind: HarmonicScorerKind,
  dataset: HarmonicDataset,
  sessionInfo: Record<string, unknown> | null | undefined,
): HarmonicStatusDataset {
  if (scorerKind === 'pair') {
    if (dataset === 'pair_casedata' || dataset === 'pair_raw' || dataset === 'pair_lfl') return dataset
    const metadata = (sessionInfo?.metadata as Record<string, unknown> | undefined) ?? {}
    const casedata = metadata.casedata && typeof metadata.casedata === 'object'
      ? metadata.casedata as Record<string, unknown>
      : {}
    const sourceHints = [
      typeof metadata.source === 'string' ? metadata.source.toLowerCase() : '',
      typeof casedata.root === 'string' ? casedata.root.toLowerCase() : '',
      typeof casedata.case_dir === 'string' ? casedata.case_dir.toLowerCase() : '',
      typeof metadata.machine_id === 'string' ? metadata.machine_id.toLowerCase() : '',
    ].filter(Boolean).join(' ')
    return casedata.case_dir || sourceHints.includes('casedata') || sourceHints.includes('site_b') || sourceHints.includes('site_c')
      ? 'pair_lfl'
      : 'pair_raw'
  }
  return dataset === 'auto' ? 'default' : dataset
}

interface Props {
  sessionInfoQuery: UseQueryResult<Record<string, unknown> | null>
  freezeOnAlert: boolean
  setFreezeOnAlert: (v: boolean) => void
  pausedByAlert: { at: number; memoryId: string } | null
  onPause: () => Promise<void>
  onResume: () => Promise<void>
  onReplay: () => Promise<void>
}

export function PlaybackControls({
  sessionInfoQuery,
  freezeOnAlert,
  setFreezeOnAlert,
  pausedByAlert,
  onPause,
  onResume,
  onReplay,
}: Props) {
  const streamSessionId = useStreamStore((s) => s.sessionId)
  const streamDownsample = useStreamStore((s) => s.streamDownsample)
  const setStreamDownsample = useStreamStore((s) => s.setStreamDownsample)

  const [replaySpeed, setReplaySpeed] = useState(1.0)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [samplesPerTick, setSamplesPerTick] = useState(32)
  const [harmonicScorerKind, setHarmonicScorerKind] = useState<HarmonicScorerKind>('context')
  const [harmonicDataset, setHarmonicDataset] = useState<HarmonicDataset>('auto')
  const [tuningSeedFor, setTuningSeedFor] = useState('')
  const info = sessionInfoQuery.data as Record<string, unknown> | null
  const resolvedHarmonicDataset = resolveHarmonicDataset(harmonicScorerKind, harmonicDataset, info)
  const harmonicDatasetOptions = harmonicScorerKind === 'pair'
    ? PAIR_DATASET_OPTIONS
    : CONTEXT_DATASET_OPTIONS

  const harmonicStatusQuery = useQuery<HarmonicStatusResponse>({
    queryKey: ['harmonic-status', streamSessionId, harmonicScorerKind, resolvedHarmonicDataset],
    queryFn: () => api(`/harmonic/status?dataset=${encodeURIComponent(resolvedHarmonicDataset)}`),
    enabled: Boolean(streamSessionId),
    refetchInterval: harmonicScorerKind === 'pair' ? 5000 : false,
    staleTime: 2000,
  })

  // Seed playback params from session config when switching sessions.
  useEffect(() => {
    if (!streamSessionId) return
    if (tuningSeedFor === streamSessionId) return
    const cfg = (sessionInfoQuery.data as Record<string, unknown>)?.config as
      | Record<string, unknown>
      | undefined
    if (!cfg) return

    if (typeof cfg.speed === 'number') setPlaybackSpeed(cfg.speed)
    if (typeof cfg.samples_per_tick === 'number') setSamplesPerTick(cfg.samples_per_tick)
    if (typeof cfg.harmonic_scorer_kind === 'string') {
      setHarmonicScorerKind(cfg.harmonic_scorer_kind === 'pair' ? 'pair' : 'context')
    }
    if (typeof cfg.harmonic_dataset === 'string' && cfg.harmonic_dataset.trim()) {
      const dataset = cfg.harmonic_dataset as HarmonicDataset
      setHarmonicDataset(dataset)
    } else {
      setHarmonicDataset('auto')
    }
    setTuningSeedFor(streamSessionId)
  }, [streamSessionId, tuningSeedFor, sessionInfoQuery.data])

  const applyPlayback = async () => {
    if (!streamSessionId) return
    await api(
      `/sessions/${encodeURIComponent(streamSessionId)}/playback`,
      'POST',
      {
        speed: Number(playbackSpeed) || 1.0,
        samples_per_tick: Math.max(1, Number(samplesPerTick) || 1),
        harmonic_scorer_kind: harmonicScorerKind,
        harmonic_dataset: harmonicDataset,
      },
    )
    await sessionInfoQuery.refetch()
  }

  const pairModelWarning = streamSessionId && harmonicScorerKind === 'pair'
    ? harmonicStatusQuery.data && !harmonicStatusQuery.data.available
    : false
  const pairDatasetLabel = resolvedHarmonicDataset === 'default' ? 'pair model' : resolvedHarmonicDataset

  return (
    <details open style={{ margin: '4px 0' }}>
      <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: 13, padding: '4px 0' }}>
        🏛️ Playback Controls
      </summary>

      <div className="hrow" style={{ justifyContent: 'space-between' }}>
        <div style={{ fontWeight: 700 }}>Playback controls</div>
        <div className="small">
          {streamSessionId
            ? `running=${String(info?.running)} paused=${String(info?.paused)} position=${String(info?.position)}${info?.last_error ? ` error=${String(info.last_error)}` : ''}`
            : 'select a session'}
        </div>
      </div>

      <div className="hrow" style={{ justifyContent: 'space-between', marginTop: 8 }}>
        <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={freezeOnAlert}
            onChange={(e) => setFreezeOnAlert(e.target.checked)}
          />
          Pause on alert
        </label>
        {pausedByAlert ? (
          <div className="small">
            paused for {pausedByAlert.memoryId.slice(0, 10)} at{' '}
            {new Date(pausedByAlert.at).toLocaleTimeString()}
          </div>
        ) : (
          <div className="small">&nbsp;</div>
        )}
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <div>
          <div className="small">Replay speed (1.0=real-time, 0.02=50x slower)</div>
          <input
            value={String(replaySpeed)}
            onChange={(e) => setReplaySpeed(Number(e.target.value))}
            placeholder="0.02"
          />
        </div>
        <div>
          <div className="small">Live speed</div>
          <input
            value={String(playbackSpeed)}
            onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
            placeholder="1.0"
          />
        </div>
        <div>
          <div className="small">Samples per tick</div>
          <input
            value={String(samplesPerTick)}
            onChange={(e) => setSamplesPerTick(Number(e.target.value))}
            placeholder="32"
          />
        </div>
        <div>
          <div className="small">Harmonic model</div>
          <select
            value={harmonicScorerKind}
            onChange={(e) => {
              const nextKind: HarmonicScorerKind = e.target.value === 'pair' ? 'pair' : 'context'
              setHarmonicScorerKind(nextKind)
              if (nextKind === 'pair') {
                setHarmonicDataset('auto')
              } else if (harmonicDataset === 'pair_raw' || harmonicDataset === 'pair_casedata' || harmonicDataset === 'pair_lfl') {
                setHarmonicDataset('auto')
              }
            }}
          >
            <option value="context">Context weights</option>
            <option value="pair">Pair peaks</option>
          </select>
        </div>
        <div>
          <div className="small">Harmonic dataset</div>
          <select
            value={harmonicDataset}
            onChange={(e) => setHarmonicDataset(e.target.value as HarmonicDataset)}
          >
            {harmonicDatasetOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <div className="small" title="Server-side LTTB target points per chunk (0=off, min effective=3)">
            Stream downsample
          </div>
          <select
            value={String(streamDownsample)}
            onChange={(e) => setStreamDownsample(Number(e.target.value))}
          >
            <option value="0">off</option>
            <option value="250">250 pts</option>
            <option value="500">500 pts</option>
            <option value="1000">1000 pts</option>
            <option value="2000">2000 pts</option>
          </select>
        </div>
        <div>
          <div className="small">Actions</div>
          <div className="hrow">
            <button onClick={onPause} disabled={!streamSessionId}>
              Pause
            </button>
            <button onClick={onResume} disabled={!streamSessionId}>
              Resume
            </button>
            <button onClick={applyPlayback} disabled={!streamSessionId}>
              Apply playback
            </button>
            <button className="primary" onClick={onReplay} disabled={!streamSessionId}>
              Replay
            </button>
          </div>
        </div>
      </div>

      {streamSessionId && harmonicScorerKind === 'pair' && (
        <div
          className="small"
          style={{
            marginTop: 8,
            color: pairModelWarning ? 'var(--danger)' : 'var(--muted)',
            fontSize: 11,
          }}
        >
          {harmonicStatusQuery.isPending
            ? 'Checking pair model availability…'
            : harmonicStatusQuery.isError
            ? `Could not verify ${pairDatasetLabel} model availability.`
            : harmonicStatusQuery.data?.available
            ? 'Pair model ready for live scoring.'
            : harmonicStatusQuery.data?.torch_installed === false
            ? 'Pair model unavailable: PyTorch is not installed on the backend.'
            : `Pair model unavailable. Train the ${pairDatasetLabel} checkpoint before enabling pair mode${harmonicStatusQuery.data?.model_save_path ? ` (${harmonicStatusQuery.data.model_save_path})` : ''}.`}
        </div>
      )}
    </details>
  )
}

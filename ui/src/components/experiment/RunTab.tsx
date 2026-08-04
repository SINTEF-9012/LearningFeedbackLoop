/**
 * RunTab — Tab 8: Feature extraction, run experiment (subprocess or live),
 *                 live progress dashboard, LLM analysis,
 *                 experiment → live bridge, model retrain status.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api, wsUrl } from '../../api/http'
import type { ExperimentTabProps } from './types'
import { PriorDiffSVG } from '../charts/PriorDiffSVG'
import { HelpIcon } from '../Tooltip'
import { ExperimentScorePanel } from './ExperimentScorePanel'
import { useExperimentScoreStore } from '../../state/experimentScoreStore'

// ── Progress event from WebSocket ──
interface ProgressEvent {
  run_id: string
  phase: string
  status: string
  message: string
  pct: number
  elapsed_s: number
  detail?: Record<string, unknown>
}

// ── Model info ──
interface ModelInfo {
  filename: string
  size_bytes: number
  modified: string
  type: string
}

// ── Config schema field from /experiment/config/schema ──
interface ConfigField {
  name: string
  type: string
  default: unknown
  group: string
}

// ── Threshold recommendation from LLM/heuristic analysis ──
interface ThresholdRec {
  parameter: string
  current_value: number | null
  recommended_value: number
  reason: string
}

const LIVE_BREAKAGE_SERVER_CONFIG_FIELDS = new Set([
  'api_mode',
  'api_mode_strict',
  'api_base_url',
  'experiment_fast_path',
  'api_use_server_patterns',
  'api_batch_size',
  'persist_shared_priors',
  'feedback_user_id',
])

function buildLiveBreakagePayload(
  dataset: 'site_a_line2' | 'casedata',
  labelScheme: 'original' | 'conservative' | 'conservative_3class' | 'v2' | 'v3',
  configValues: Record<string, unknown>,
  experimentGenerateExplanations: boolean,
) {
  const filteredConfigValues = Object.fromEntries(
    Object.entries(configValues).filter(([key]) => !LIVE_BREAKAGE_SERVER_CONFIG_FIELDS.has(key)),
  )

  return {
    dataset,
    sandbox_priors: true,
    label_scheme: labelScheme,
    api_generate_explanations: experimentGenerateExplanations,
    ...filteredConfigValues,
  }
}

function latestProgressEvent(events: ProgressEvent[], phase: string, status: string): ProgressEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event.phase === phase && event.status === status) {
      return event
    }
  }
  return undefined
}

export function RunTab({ effectiveRunId, evalPhase, fullResultsQ, runs, setSelectedRunId, runsRefetch }: ExperimentTabProps & {
  runsRefetch: () => void
}) {
  // ── Extraction state
  const [runGap, setRunGap] = useState(10)
  const [runWindow, setRunWindow] = useState(60)
  const [runHz, setRunHz] = useState(1.0)
  const [runTrainOps, setRunTrainOps] = useState('OF00001,OF00002')
  const [runTestOp, setRunTestOp] = useState('OF00003')
  const [runEvalOp, setRunEvalOp] = useState('OF00004')
  const [runApiMode, setRunApiMode] = useState(false)
  const [extractGaps, setExtractGaps] = useState('0, 5, 10, 30')

  // ── Dynamic config from schema endpoint ──
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({})
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false)

  // ── Backend config (LLM status, flags) ──
  interface BackendConfig {
    generate_explanations: boolean
    dispatch_alerts: boolean
    llm_available: boolean
    llm_provider: string
    ollama_model: string
    ollama_url: string
    groq_model: string
    [k: string]: unknown
  }
  const backendConfigQ = useQuery<BackendConfig>({
    queryKey: ['backend-config'],
    queryFn: () => api('/agent/memory/config'),
    staleTime: 10_000,
    retry: 1,
  })
  const [experimentLlmEnabled, setExperimentLlmEnabled] = useState<boolean | null>(null)
  const llmEnabled = experimentLlmEnabled ?? (backendConfigQ.data?.generate_explanations ?? false)
  const llmAvailable = backendConfigQ.data?.llm_available ?? false
  useEffect(() => {
    if (experimentLlmEnabled == null && backendConfigQ.data?.generate_explanations != null) {
      setExperimentLlmEnabled(Boolean(backendConfigQ.data.generate_explanations))
    }
  }, [experimentLlmEnabled, backendConfigQ.data])

  const configSchemaQ = useQuery<{ fields: ConfigField[] }>({
    queryKey: ['config-schema'],
    queryFn: () => api('/agent/memory/experiment/config/schema'),
    staleTime: 120_000,
  })

  // Populate defaults once schema loads
  useEffect(() => {
    if (configSchemaQ.data?.fields && Object.keys(configValues).length === 0) {
      const defaults: Record<string, unknown> = {}
      for (const f of configSchemaQ.data.fields) {
        defaults[f.name] = f.default
      }
      setConfigValues(defaults)
    }
  }, [configSchemaQ.data])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Prior diff state (after live experiment) ──
  const [priorDiff, setPriorDiff] = useState<Record<string, { before: number; after: number }> | null>(null)

  // ── Run mode: "subprocess" (original) or "live" (in-process with WS)
  const [runMode, setRunMode] = useState<'subprocess' | 'live'>('live')

  // ── Experiment type: "stoppage" or "breakage"
  const [experimentType, setExperimentType] = useState<'stoppage' | 'breakage'>('stoppage')
  const [breakageDataset, setBreakageDataset] = useState<'site_a_line2' | 'casedata'>('site_a_line2')

  // ── Label scheme: original broad labels vs conservative signal-anchored labels
  const [labelScheme, setLabelScheme] = useState<'original' | 'conservative' | 'conservative_3class' | 'v2' | 'v3'>('conservative')

  // Default breakage to 'live' mode (previously restricted to subprocess)
  const liveBreakageUsesApi = runMode === 'live' && experimentType === 'breakage'
  const effectiveApiMode = runApiMode || liveBreakageUsesApi

  // ── Live progress state
  const [liveRunId, setLiveRunId] = useState<string | null>(null)
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([])
  const [liveStatus, setLiveStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [llmRunWarning, setLlmRunWarning] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // ── WS connection for live progress
  const connectProgressWs = useCallback((runId: string) => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    setProgressEvents([])
    useExperimentScoreStore.getState().clear()
    setLiveStatus('running')

    const ws = new WebSocket(wsUrl(`/agent/memory/experiment/progress/${runId}`))
    wsRef.current = ws

    ws.onmessage = (evt) => {
      try {
        const data: ProgressEvent = JSON.parse(evt.data)
        // Score streaming events — feed to experiment score store
        if (data.phase === 'scores' && data.detail?.samples) {
          const samples = data.detail.samples as any[]
          const phase = (data.detail.phase as 'test' | 'eval') || 'test'
          const fold = (data.detail.fold as number) || 0
          useExperimentScoreStore.getState().push(phase, samples, fold)
          return
        }
        setProgressEvents(prev => [...prev, data])
        if (data.status === 'completed' && data.phase === 'done') {
          setLiveStatus('done')
          // Fetch prior diff after live experiment completes
          api<Record<string, { before: number; after: number }>>(`/agent/memory/experiment/sandbox/diff/${runId}`)
            .then(d => setPriorDiff(d))
            .catch(() => {/* non-critical */})
          // Refetch run list and auto-select the saved run
          runsRefetch()
          const diskRunId = (data.detail?.disk_run_id as string) || ''
          if (diskRunId) {
            // Small delay so the refetch has time to complete
            setTimeout(() => setSelectedRunId(diskRunId), 500)
          }
        } else if (data.status === 'error') {
          setLiveStatus('error')
        }
      } catch { /* ignore parse errors */ }
    }

    ws.onerror = () => {
      setLiveStatus('error')
      // Inject a synthetic progress event so the dashboard shows context
      setProgressEvents(prev => [...prev, {
        run_id: runId, phase: 'connection', status: 'error',
        message: 'WebSocket connection failed — the backend may have crashed or the run errored before the connection was established.',
        pct: prev.length > 0 ? prev[prev.length - 1].pct : 0, elapsed_s: 0,
      }])
    }
    ws.onclose = () => {
      wsRef.current = null
      // If the WS closed without receiving a terminal event (done/error),
      // the experiment may have finished but progress was lost.
      // Transition out of 'running' so we don't appear stuck.
      setLiveStatus(prev => {
        if (prev === 'running') {
          // Refetch runs in case the experiment completed on the backend
          runsRefetch()
          return 'done'
        }
        return prev
      })
    }
  }, [runsRefetch, setSelectedRunId])

  // ── Auto-reconnect to active experiment on mount / page reload ──
  const [reconnectedRunId, setReconnectedRunId] = useState<string | null>(null)
  useEffect(() => {
    // Only attempt reconnect when idle (no local experiment running)
    if (liveStatus !== 'idle') return
    let cancelled = false
    api<{ runs: { run_id: string; done: boolean; experiment_type: string }[] }>('/agent/memory/experiment/live-runs')
      .then(resp => {
        if (cancelled) return
        const active = resp.runs.filter(r => !r.done)
        if (active.length > 0) {
          // Active experiment found — reconnect WebSocket
          const run = active[0]
          setLiveRunId(run.run_id)
          setReconnectedRunId(run.run_id)
          setLiveStatus('running')
          const ws = new WebSocket(wsUrl(`/agent/memory/experiment/progress/${run.run_id}`))
          wsRef.current = ws
          ws.onmessage = (evt) => {
            try {
              const data: ProgressEvent = JSON.parse(evt.data)
              // Score streaming events — feed to experiment score store
              if (data.phase === 'scores' && data.detail?.samples) {
                const samples = data.detail.samples as any[]
                const phase = (data.detail.phase as 'test' | 'eval') || 'test'
                const fold = (data.detail.fold as number) || 0
                useExperimentScoreStore.getState().push(phase, samples, fold)
                return
              }
              setProgressEvents(prev => [...prev, data])
              if (data.status === 'completed' && data.phase === 'done') {
                setLiveStatus('done')
                setReconnectedRunId(null)
                api<Record<string, { before: number; after: number }>>(`/agent/memory/experiment/sandbox/diff/${run.run_id}`)
                  .then(d => setPriorDiff(d)).catch(() => {})
                runsRefetch()
                const diskRunId = (data.detail?.disk_run_id as string) || ''
                if (diskRunId) setTimeout(() => setSelectedRunId(diskRunId), 500)
              } else if (data.status === 'error') {
                setLiveStatus('error')
                setReconnectedRunId(null)
              }
            } catch { /* ignore */ }
          }
          ws.onerror = () => { setLiveStatus('error'); setReconnectedRunId(null) }
          ws.onclose = () => {
            wsRef.current = null
            setReconnectedRunId(null)
            setLiveStatus(prev => {
              if (prev === 'running') { runsRefetch(); return 'done' }
              return prev
            })
          }
        } else {
          // No active run — check if the newest completed run finished
          // very recently (within 5 min).  If so, auto-select it and
          // show a "completed while away" banner so the user doesn't
          // lose track of results.
          api<{ runs: { run_id: string; timestamp?: number | string; error?: boolean }[] }>('/agent/memory/experiment/runs')
            .then(runsResp => {
              if (cancelled) return
              const newest = runsResp.runs?.[0]
              if (!newest || newest.error) return
              const ts = typeof newest.timestamp === 'number'
                ? newest.timestamp * 1000
                : typeof newest.timestamp === 'string'
                  ? new Date(newest.timestamp).getTime()
                  : 0
              const ageMs = Date.now() - ts
              if (ts > 0 && ageMs < 5 * 60 * 1000) {
                // Finished within last 5 minutes — auto-select
                setReconnectedRunId(newest.run_id)
                setLiveStatus('done')
                setSelectedRunId(newest.run_id)
                runsRefetch()
              }
            })
            .catch(() => {})
        }
      })
      .catch(() => { /* backend unreachable — no reconnect */ })
    return () => { cancelled = true }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up WS on unmount
  useEffect(() => () => { wsRef.current?.close() }, [])

  // ── Mutations
  const runExperimentMut = useMutation<{ success: boolean; stdout: string; stderr: string }>({
    mutationFn: () => api('/agent/memory/experiment/run', 'POST', {
      gap: runGap, window: runWindow, hz: runHz,
      train_ops: runTrainOps.split(',').map(s => s.trim()).filter(Boolean),
      test_op: runTestOp, eval_op: runEvalOp, api_mode: runApiMode,
    }),
  })

  // ── Cancel mutation
  const cancelMut = useMutation<{ run_id: string; status: string }>({
    mutationFn: () => api(`/agent/memory/experiment/cancel/${encodeURIComponent(liveRunId!)}`, 'POST'),
    onSuccess: () => {
      setLiveStatus('error')
      setProgressEvents(prev => [...prev, {
        run_id: liveRunId ?? '', phase: 'cancel', status: 'error',
        message: 'Experiment cancelled by user.', pct: prev.length > 0 ? prev[prev.length - 1].pct : 0, elapsed_s: 0,
      }])
    },
  })

  const [liveRunMeta, setLiveRunMeta] = useState<Record<string, unknown> | null>(null)
  const runLiveMut = useMutation<{ run_id: string; ws_url: string; status: string; llm_warning?: string; generate_explanations?: boolean; llm_available?: boolean; ollama_ok?: boolean; ollama_model?: string; features_csv?: string }>({
    mutationFn: () => api('/agent/memory/experiment/run-live', 'POST', {
      prediction_gap_s: runGap, window_size_s: runWindow, sample_rate_hz: runHz,
      train_ops: runTrainOps.split(',').map(s => s.trim()).filter(Boolean),
      test_op: runTestOp, eval_op: runEvalOp,
      sandbox_priors: true,
      api_generate_explanations: llmEnabled,
      ...configValues,
      api_mode: runApiMode,
    }),
    onSuccess: (data) => {
      setLiveRunId(data.run_id)
      setLlmRunWarning(data.llm_warning ?? null)
      setLiveRunMeta(data as unknown as Record<string, unknown>)
      connectProgressWs(data.run_id)
    },
  })

  // ── Feature extraction status (does a CSV exist already?) ──
  interface FeaturesFileMeta {
    exists: boolean; file: string; size_bytes?: number; modified?: string
    age_hours?: number; rows?: number; columns?: number
    labels?: Record<string, number>; operations?: string[]
  }
  interface FeaturesStatusResponse {
    site_a_line2?: FeaturesFileMeta
    site_a_line2_conservative?: FeaturesFileMeta & { label_binary?: Record<string, number>; label_3class?: Record<string, number> }
    stoppage?: FeaturesFileMeta[]
  }
  const featuresStatusQ = useQuery<FeaturesStatusResponse>({
    queryKey: ['features-status', experimentType, breakageDataset],
    queryFn: () => {
      const ds = experimentType === 'breakage' && breakageDataset === 'site_a_line2' ? 'site_a_line2' : 'stoppage'
      return api(`/agent/memory/experiment/features-status?dataset=${ds}`)
    },
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  })

  const extractSite_a_line2Mut = useMutation<{ success: boolean; stdout: string; stderr: string }>({
    mutationFn: () => api('/agent/memory/experiment/extract-site_a_line2', 'POST', {
      window: runWindow, hz: runHz,
    }),
    onSuccess: () => { featuresStatusQ.refetch() },
  })

  const extractMut = useMutation<{ success: boolean; stdout: string; stderr: string }>({
    mutationFn: () => api('/agent/memory/experiment/extract', 'POST', {
      gaps: extractGaps.split(',').map(s => Number(s.trim())).filter(n => !isNaN(n)),
      window: runWindow, hz: runHz,
    }),
    onSuccess: () => { featuresStatusQ.refetch() },
  })

  // ── Breakage experiment mutation (subprocess)
  const runBreakageMut = useMutation<{ success: boolean; stdout: string; stderr: string }>({
    mutationFn: () => api('/agent/memory/experiment/run-breakage', 'POST', {
      test_op: runTestOp, api_mode: runApiMode, dataset: breakageDataset,
      label_scheme: labelScheme,
      neo4j: Boolean(configValues['neo4j']),
    }),
  })

  // ── Live breakage experiment mutation
  const runLiveBreakageMut = useMutation<{ run_id: string; ws_url: string; status: string; llm_warning?: string }>({
    mutationFn: () => api(
      '/agent/memory/experiment/run-live-breakage',
      'POST',
      buildLiveBreakagePayload(breakageDataset, labelScheme, configValues, llmEnabled),
    ),
    onSuccess: (data) => {
      setLiveRunId(data.run_id)
      setLlmRunWarning(data.llm_warning ?? null)
      connectProgressWs(data.run_id)
    },
  })

  // ── Models query
  const modelsQ = useQuery<{ models: ModelInfo[] }>({
    queryKey: ['models'],
    queryFn: () => api('/agent/memory/models'),
    staleTime: 30_000,
  })

  // ── LLM analysis
  const [llmSummary, setLlmSummary] = useState<{
    summary: string; recommendations: string[];
    threshold_recommendations?: ThresholdRec[];
    key_metrics: Record<string, unknown>; source: string
  } | null>(null)
  const [llmSummaryPending, setLlmSummaryPending] = useState(false)

  // ── Review bridge
  const [reviewResult, setReviewResult] = useState<{ updated: number; failed: number; prior_changes: Record<string, unknown>; retrain_triggered: boolean } | null>(null)
  const [reviewPending, setReviewPending] = useState(false)

  // ── Retrain status
  const retrainStatusQ = useQuery<{
    total_feedback: number; since_last_retrain: number; should_retrain: boolean
    threshold: number; last_retrain_at: string | null
  }>({ queryKey: ['retrain-status'], queryFn: () => api('/agent/memory/retrain/status'), staleTime: 15_000 })

  const retrainMut = useMutation<{ success: boolean; message: string; n_samples_used?: number }>({
    mutationFn: () => api('/agent/memory/retrain', 'POST'),
    onSuccess: () => { retrainStatusQ.refetch() },
  })

  // ── Derived: last progress event
  const lastProgress = progressEvents[progressEvents.length - 1] ?? null

  return (
    <>
      {/* ── Feature Extraction ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 12 }}>🔧 Feature Extraction <HelpIcon text="Extracts statistical features from raw sensor data. Stoppage extraction produces one CSV per (window, Hz, gap) combination from CNC casedata. Site_a_line2 extraction produces a single features CSV from the Site_a_line2 milling dataset for tool condition detection." /></div>
        <p className="small" style={{ color: 'var(--muted)', marginBottom: 12 }}>
          {experimentType === 'breakage' && breakageDataset === 'site_a_line2'
            ? 'Extract features from the Site_a_line2 milling dataset (tool wear labels: unworn / chipped / broken).'
            : 'Extract statistical features from raw CNC channel data. Produces one CSV per (window, Hz, gap) combination.'}
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 12 }}>
          <label className="small" style={{ display: 'grid', gap: 2 }}>
            Window (seconds) <HelpIcon text="Duration in seconds of each analysis window. A 60s window at 1Hz = 60 data points per channel. Larger windows capture more context but reduce sample count." />
            <input type="number" value={runWindow} onChange={e => setRunWindow(Number(e.target.value))} min={1} style={{ fontSize: 12, padding: '4px 8px' }} title="Analysis window duration in seconds" />
          </label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>
            Sample Rate (Hz) <HelpIcon text="Resampling frequency. Raw data is ~1Hz. Higher rates give more points per window but may introduce interpolation artifacts." />
            <input type="number" value={runHz} onChange={e => setRunHz(Number(e.target.value))} min={0.1} step={0.1} style={{ fontSize: 12, padding: '4px 8px' }} title="Data resampling frequency" />
          </label>
          {/* Gaps only apply to stoppage / casedata extraction */}
          {!(experimentType === 'breakage' && breakageDataset === 'site_a_line2') && (
            <label className="small" style={{ display: 'grid', gap: 2 }}>
              Gaps (comma-sep.) <HelpIcon text="Prediction gap values in seconds. The gap is how far before a stoppage event the analysis window ends. Gap=0 means the window ends at the event. Gap=30 means predicting 30s ahead. Multiple gaps produce multiple feature CSVs for comparison." />
              <input type="text" value={extractGaps} onChange={e => setExtractGaps(e.target.value)} placeholder="0, 5, 10, 30" style={{ fontSize: 12, padding: '4px 8px' }} title="Comma-separated prediction gap values" />
            </label>
          )}
          <div className="small" style={{ color: 'var(--muted)', alignSelf: 'end', padding: '4px 0' }}>= {runWindow * runHz} entries/window</div>
        </div>

        {/* ── Extraction status banner ── */}
        {(() => {
          const isSite_a_line2 = experimentType === 'breakage' && breakageDataset === 'site_a_line2'
          const status = featuresStatusQ.data
          const meta = isSite_a_line2
            ? status?.site_a_line2
            : status?.stoppage?.[0]
          const consMeta = isSite_a_line2 ? status?.site_a_line2_conservative : undefined

          if (featuresStatusQ.isLoading) return (
            <div className="small" style={{ color: 'var(--muted)', marginBottom: 10, padding: '6px 10px', background: 'rgba(255,255,255,0.04)', borderRadius: 6 }}>
              ⏳ Checking extraction status…
            </div>
          )

          if (!meta || !meta.exists) return (
            <div style={{ marginBottom: 10, padding: '8px 12px', background: 'rgba(243,156,18,0.10)', borderRadius: 6, border: '1px solid rgba(243,156,18,0.3)' }}>
              <div className="small" style={{ color: '#f39c12', fontWeight: 700 }}>⚠ No features CSV found — extraction required</div>
              <div className="small" style={{ color: 'var(--muted)', marginTop: 2 }}>
                {isSite_a_line2
                  ? 'Run Site_a_line2 extraction to generate site_a_line2_features.csv before running the experiment.'
                  : 'Run extraction to generate stoppage feature CSVs before running the experiment.'}
              </div>
            </div>
          )

          // Determine which label distribution to show based on selected scheme
          const isConsScheme = isSite_a_line2 && labelScheme.startsWith('conservative')
          const activeMeta = isConsScheme && consMeta?.exists ? consMeta : meta

          // For conservative schemes, show the appropriate label column distribution
          let activeLabels: Record<string, number> = {}
          if (isConsScheme && consMeta?.exists) {
            if (labelScheme === 'conservative' && consMeta.label_binary) {
              activeLabels = consMeta.label_binary
            } else if (labelScheme === 'conservative_3class' && consMeta.label_3class) {
              activeLabels = consMeta.label_3class
            } else {
              activeLabels = consMeta.labels || {}
            }
          } else {
            activeLabels = meta.labels || {}
          }

          // CSV exists — show metadata
          const ageText = activeMeta.age_hours !== undefined
            ? activeMeta.age_hours < 1 ? `${Math.round(activeMeta.age_hours * 60)}m ago`
              : activeMeta.age_hours < 24 ? `${Math.round(activeMeta.age_hours)}h ago`
              : `${Math.round(activeMeta.age_hours / 24)}d ago`
            : ''
          const labelEntries = Object.entries(activeLabels)
          const preBreakCount = activeLabels['pre_break'] ?? activeLabels['pre_stoppage'] ?? activeLabels['anomalous'] ?? 0
          const hasLabels = preBreakCount > 0

          return (
            <>
            <div style={{ marginBottom: isConsScheme && consMeta?.exists ? 4 : 10, padding: '8px 12px', background: hasLabels ? 'rgba(39,174,96,0.08)' : 'rgba(243,156,18,0.08)', borderRadius: 6, border: `1px solid ${hasLabels ? 'rgba(39,174,96,0.25)' : 'rgba(243,156,18,0.25)'}` }}>
              <div className="small" style={{ fontWeight: 700, color: hasLabels ? 'var(--ok)' : '#f39c12' }}>
                {hasLabels ? '✓' : '⚠'} {activeMeta.file} — {activeMeta.rows?.toLocaleString()} rows{ageText ? ` · extracted ${ageText}` : ''}
                {isConsScheme && consMeta?.exists && (
                  <span style={{ fontWeight: 400, color: '#3498db', marginLeft: 8, fontSize: 10 }}>
                    🎯 conservative labels ({labelScheme === 'conservative' ? 'binary' : '3-class'})
                  </span>
                )}
              </div>
              <div className="small" style={{ color: 'var(--muted)', marginTop: 3, display: 'flex', flexWrap: 'wrap', gap: '6px 14px' }}>
                {labelEntries.map(([lbl, cnt]) => (
                  <span key={lbl}>
                    <span style={{ fontWeight: 600, color: lbl.includes('pre_') || lbl === 'worn' || lbl === 'chipped' || lbl === 'broken' || lbl === 'anomalous' || lbl === 'break_event' || lbl === 'degraded' ? '#e67e22' : lbl === 'suspect' ? '#f39c12' : lbl === 'post_change' ? '#3498db' : 'var(--muted)' }}>
                      {cnt as number}
                    </span>{' '}{lbl}
                  </span>
                ))}
                {activeMeta.operations && activeMeta.operations.length > 0 && (
                  <span>· {activeMeta.operations.length} operation{activeMeta.operations.length > 1 ? 's' : ''}: {activeMeta.operations.join(', ')}</span>
                )}
                {activeMeta.columns !== undefined && <span>· {activeMeta.columns} features</span>}
              </div>
              {!hasLabels && (
                <div className="small" style={{ color: '#e67e22', marginTop: 4 }}>
                  ⚠ No labelled positive samples detected. Check ground truth data or re-extract.
                </div>
              )}
            </div>
            {/* Conservative CSV missing warning */}
            {isConsScheme && !consMeta?.exists && (
              <div style={{ marginBottom: 10, padding: '6px 12px', background: 'rgba(243,156,18,0.08)', borderRadius: 6, border: '1px solid rgba(243,156,18,0.25)' }}>
                <div className="small" style={{ color: '#f39c12' }}>
                  ⚠ Conservative labels CSV not found. Run: <code style={{ fontSize: 10 }}>python scripts/label_site_a_line2_conservative.py</code>
                </div>
              </div>
            )}
            </>
          )
        })()}

        {experimentType === 'breakage' && breakageDataset === 'site_a_line2' ? (
          <button
            className="primary"
            onClick={() => extractSite_a_line2Mut.mutate()}
            disabled={extractSite_a_line2Mut.isPending}
            style={{ padding: '6px 16px', background: '#27ae60' }}
          >
            {extractSite_a_line2Mut.isPending
              ? 'Extracting Site_a_line2…'
              : featuresStatusQ.data?.site_a_line2?.exists
                ? `🔄 Re-extract Site_a_line2 (${runWindow}s @ ${runHz}Hz)`
                : `🟢 Extract Site_a_line2 (${runWindow}s @ ${runHz}Hz)`}
          </button>
        ) : (
          <button className="primary" onClick={() => extractMut.mutate()} disabled={extractMut.isPending} style={{ padding: '6px 16px' }}>
            {extractMut.isPending
              ? 'Extracting…'
              : (featuresStatusQ.data?.stoppage?.[0]?.exists)
                ? `🔄 Re-extract (${runWindow}s @ ${runHz}Hz, gaps: ${extractGaps})`
                : `🔧 Extract (${runWindow}s @ ${runHz}Hz, gaps: ${extractGaps})`}
          </button>
        )}
        {/* Site_a_line2 extraction result */}
        {extractSite_a_line2Mut.isError && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.3)' }}>
            <div className="small" style={{ color: 'var(--danger)', fontWeight: 700 }}>✗ Site_a_line2 extraction failed</div>
            <pre style={{ fontSize: 10, color: 'var(--danger)', marginTop: 4, whiteSpace: 'pre-wrap' }}>
              {String((extractSite_a_line2Mut.error as Error)?.message || extractSite_a_line2Mut.error)}
            </pre>
          </div>
        )}
        {extractSite_a_line2Mut.data && (
          <div style={{ marginTop: 12 }}>
            <div className="small" style={{ color: extractSite_a_line2Mut.data.success ? 'var(--ok)' : 'var(--danger)', fontWeight: 700 }}>
              {extractSite_a_line2Mut.data.success ? '✓ Site_a_line2 extraction complete — site_a_line2_features.csv updated' : '✗ Site_a_line2 extraction failed'}
            </div>
            <pre style={{ fontSize: 10, maxHeight: 200, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4, marginTop: 4 }}>
              {extractSite_a_line2Mut.data.stdout || extractSite_a_line2Mut.data.stderr}
            </pre>
          </div>
        )}
        {/* Stoppage/casedata extraction result */}
        {extractMut.data && (
          <div style={{ marginTop: 12 }}>
            <div className="small" style={{ color: extractMut.data.success ? 'var(--ok)' : 'var(--danger)', fontWeight: 700 }}>
              {extractMut.data.success ? '✓ Extraction complete' : '✗ Extraction failed'}
            </div>
            <pre style={{ fontSize: 10, maxHeight: 200, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4, marginTop: 4 }}>
              {extractMut.data.stdout || extractMut.data.stderr}
            </pre>
          </div>
        )}
      </div>

      {/* ── Run Experiment ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 12 }}>▶️ Run Experiment <HelpIcon text="Launch a stoppage prediction or tool condition experiment. Both use a 3-phase pipeline (train → test/baseline → eval/feedback) with LOOCV. Results appear in the run list above." /></div>

        {/* Experiment type selector */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          {(['stoppage', 'breakage'] as const).map(t => (
            <button
              key={t}
              onClick={() => setExperimentType(t)}
              style={{
                padding: '4px 14px', fontSize: 12, borderRadius: 6, cursor: 'pointer',
                background: experimentType === t ? 'rgba(155,89,182,0.2)' : 'rgba(255,255,255,0.06)',
                color: experimentType === t ? '#bb86fc' : 'var(--muted)',
                border: experimentType === t ? '1px solid rgba(155,89,182,0.4)' : '1px solid rgba(255,255,255,0.1)',
                fontWeight: experimentType === t ? 700 : 400,
              }}
            >
              {t === 'stoppage' ? '🔴 Stoppage Prediction' : '🔧 Tool Condition Detection'}
              {t === 'stoppage' && <HelpIcon text="Predicts premature machine stoppages using a 3-phase pipeline: Phase 1 trains on normal data, Phase 2 tests without feedback (baseline), Phase 3 evaluates with feedback. Supports LOOCV rotations across operations." />}
              {t === 'breakage' && <HelpIcon text="Detects tool condition patterns using the Learning Feedback Loop. Uses Site_a_line2 dataset (real tool wear data from milling experiments) or legacy casedata. Runs the same 3-phase pipeline as stoppage (train → test-baseline → eval-feedback) with LOOCV across operations." />}
            </button>
          ))}
        </div>

        {/* Dataset selector for breakage mode */}
        {experimentType === 'breakage' && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
            <span className="small" style={{ color: 'var(--muted)', fontWeight: 600 }}>Dataset: <HelpIcon text="Site_a_line2: Real tool wear dataset from milling experiments with labeled wear states (unworn, chipped, broken). Auto-generated from raw data if not found. Casedata: Legacy features extracted from CNC operation logs." /></span>
            {(['site_a_line2', 'casedata'] as const).map(ds => (
              <button
                key={ds}
                onClick={() => setBreakageDataset(ds)}
                style={{
                  padding: '4px 14px', fontSize: 12, borderRadius: 6, cursor: 'pointer',
                  background: breakageDataset === ds ? 'rgba(46,204,113,0.2)' : 'rgba(255,255,255,0.06)',
                  color: breakageDataset === ds ? 'var(--ok)' : 'var(--muted)',
                  border: breakageDataset === ds ? '1px solid rgba(46,204,113,0.4)' : '1px solid rgba(255,255,255,0.1)',
                  fontWeight: breakageDataset === ds ? 700 : 400,
                }}
              >
                {ds === 'site_a_line2' ? '🟢 Site_a_line2 (default)' : '📁 Casedata (legacy)'}
              </button>
            ))}
          </div>
        )}

        {/* Label scheme selector for Site_a_line2 */}
        {experimentType === 'breakage' && breakageDataset === 'site_a_line2' && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <span className="small" style={{ color: 'var(--muted)', fontWeight: 600 }}>Labels: <HelpIcon text="Original: broad labels from operator notes — marks entire operations as pre_break (38% positive). Conservative: signal-anchored labels using CNC alarm timestamps — only ±10min around confirmed FStop alarms count as break_event (5% positive). Much higher label quality." /></span>
            {([
              { key: 'original' as const, label: '📋 Original (broad)', desc: '38% positive — entire OF as pre_break' },
              { key: 'conservative' as const, label: '🎯 Conservative (binary)', desc: '5% positive — alarm-anchored, insert-change-aware' },
              { key: 'conservative_3class' as const, label: '🔬 Conservative (3-class)', desc: 'anomalous + suspect + normal' },
              { key: 'v2' as const, label: '🧩 v2 tool-segmented', desc: '(OF, tool) labels, leakage-controlled — 266 pre_break / 13 991 normal' },
              { key: 'v3' as const, label: '🎚️ v3 sub-pass', desc: 'inspection-aligned break-pass tail — 78 pre_break, graded chipped/worn' },
            ]).map(({ key, label, desc }) => (
              <button
                key={key}
                onClick={() => setLabelScheme(key)}
                title={desc}
                style={{
                  padding: '4px 12px', fontSize: 11, borderRadius: 6, cursor: 'pointer',
                  background: labelScheme === key ? 'rgba(52,152,219,0.2)' : 'rgba(255,255,255,0.06)',
                  color: labelScheme === key ? '#3498db' : 'var(--muted)',
                  border: labelScheme === key ? '1px solid rgba(52,152,219,0.4)' : '1px solid rgba(255,255,255,0.1)',
                  fontWeight: labelScheme === key ? 700 : 400,
                }}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Run mode selector */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          {(['subprocess', 'live'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setRunMode(mode)}
              style={{
                padding: '4px 14px', fontSize: 12, borderRadius: 6,
                cursor: 'pointer',
                background: runMode === mode ? 'var(--accent)' : 'rgba(255,255,255,0.06)',
                color: runMode === mode ? '#fff' : 'var(--muted)',
                border: runMode === mode ? '1px solid var(--accent)' : '1px solid rgba(255,255,255,0.1)',
                fontWeight: runMode === mode ? 700 : 400,
              }}
            >
              {mode === 'subprocess' ? '🖥️ Subprocess' : '⚡ Live (in-process)'}
            </button>
          ))}
          <span className="small" style={{ color: 'var(--muted)', alignSelf: 'center', marginLeft: 4 }}>
            {runMode === 'live'
              ? 'Runs in-process with real-time progress streaming and sandboxed priors'
              : 'Spawns a subprocess — no live progress, priors may be contaminated'}
          </span>
        </div>

        {/* Stoppage-specific config fields (not applicable to breakage LOOCV) */}
        {experimentType === 'stoppage' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 12 }}>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Gap (seconds) <HelpIcon text="Prediction gap: how many seconds before a stoppage the model must predict. Gap=0 uses data up to the event. Gap=30 predicts 30s ahead (harder but more useful)." /><input type="number" value={runGap} onChange={e => setRunGap(Number(e.target.value))} style={{ fontSize: 12, padding: '4px 8px' }} title="Prediction gap in seconds before event" /></label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Window (seconds) <HelpIcon text="Size of each analysis window. Must match the extraction window." /><input type="number" value={runWindow} onChange={e => setRunWindow(Number(e.target.value))} min={1} style={{ fontSize: 12, padding: '4px 8px' }} title="Analysis window duration" /></label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Sample Rate (Hz) <HelpIcon text="Resampling rate. Must match the extraction sample rate." /><input type="number" value={runHz} onChange={e => setRunHz(Number(e.target.value))} min={0.1} step={0.1} style={{ fontSize: 12, padding: '4px 8px' }} title="Data resampling frequency" /></label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Train Operations <HelpIcon text="Comma-separated operation IDs used for training (Phase 1). The model learns normal behaviour from these operations. Typically OF00001, OF00002." /><input type="text" value={runTrainOps} onChange={e => setRunTrainOps(e.target.value)} style={{ fontSize: 12, padding: '4px 8px' }} title="Comma-separated operation IDs for training" /></label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Test Operation <HelpIcon text="Operation used for Phase 2 (baseline/no feedback). Evaluates raw model performance before any feedback." /><input type="text" value={runTestOp} onChange={e => setRunTestOp(e.target.value)} style={{ fontSize: 12, padding: '4px 8px' }} title="Operation ID for test/baseline phase" /></label>
          <label className="small" style={{ display: 'grid', gap: 2 }}>Eval Operation <HelpIcon text="Operation used for Phase 3 (with feedback). The improvement over the test phase shows the feedback loop's learning effect." /><input type="text" value={runEvalOp} onChange={e => setRunEvalOp(e.target.value)} style={{ fontSize: 12, padding: '4px 8px' }} title="Operation ID for eval/feedback phase" /></label>
        </div>
        )}

        {/* Advanced config from schema */}
        {configSchemaQ.data?.fields && (
          <div style={{ marginBottom: 12 }}>
            {/* ── Feedback Realism quick-access panel ── */}
            {(() => {
              const realismFields = (configSchemaQ.data?.fields ?? []).filter(f => f.group === 'feedback_realism')
              if (realismFields.length === 0) return null

              // Check if any realism field differs from default (i.e. non-oracle mode)
              const isNonOracle = realismFields.some(f => {
                const val = configValues[f.name]
                return val !== undefined && val !== f.default
              })

              return (
                <div style={{
                  marginBottom: 12, padding: '10px 14px', borderRadius: 8,
                  background: isNonOracle ? 'rgba(243,156,18,0.06)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${isNonOracle ? 'rgba(243,156,18,0.25)' : 'rgba(255,255,255,0.08)'}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ fontWeight: 700, fontSize: 12 }}>
                      🎭 Feedback Realism
                      <HelpIcon text="Controls how realistic the simulated operator feedback is. Default values (all zeros / 1.0) give a perfect oracle baseline — every event gets immediate, perfect feedback. Increase these values to simulate real-world conditions: delayed responses, alert fatigue, noisy labels, and pattern-dependent importance." />
                      {isNonOracle && (
                        <span style={{ marginLeft: 8, fontSize: 10, color: '#f39c12', fontWeight: 600 }}>
                          ⚠ Non-oracle mode
                        </span>
                      )}
                    </div>
                    {isNonOracle && (
                      <button
                        style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 10, color: 'var(--accent)', textDecoration: 'underline' }}
                        onClick={() => {
                          const reset: Record<string, unknown> = { ...configValues }
                          for (const f of realismFields) reset[f.name] = f.default
                          setConfigValues(reset)
                        }}
                      >
                        Reset to oracle defaults
                      </button>
                    )}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 8 }}>
                    {realismFields.map(f => {
                      const val = configValues[f.name] ?? f.default
                      const isSlider = f.type === 'float' && (f.name.includes('rate') || f.name.includes('decay') || f.name.includes('ambiguity'))
                      const label = f.name.replace(/^feedback_/, '').replace(/_/g, ' ')
                      const helpTexts: Record<string, string> = {
                        feedback_delay_samples: 'Number of samples before feedback takes effect. 0 = instant oracle feedback (default).',
                        feedback_response_rate: 'Probability that an operator responds to a flagged event. 1.0 = always responds (default).',
                        feedback_fatigue_decay: 'Exponential decay rate for response probability over successive alerts. 0.0 = no fatigue (default).',
                        noise_rate_base: 'Base probability of flipping a feedback label (incorrect confirmation/dismissal). 0.0 = perfect labels (default).',
                        noise_rate_ambiguity: 'Extra noise scaled by prediction confidence ambiguity. 0.0 = no ambiguity effect (default).',
                        feedback_per_pattern_weighting: 'Weight feedback updates by pattern severity (critical patterns get stronger updates). Off = uniform weighting (default).',
                      }
                      return (
                        <label key={f.name} className="small" style={{ display: 'grid', gap: 2 }}>
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 4 }} title={f.name}>
                            {label}
                            {helpTexts[f.name] && <HelpIcon text={helpTexts[f.name]} />}
                          </span>
                          {f.type === 'bool' ? (
                            <input
                              type="checkbox"
                              checked={Boolean(val)}
                              onChange={e => setConfigValues(v => ({ ...v, [f.name]: e.target.checked }))}
                            />
                          ) : isSlider ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <input
                                type="range" min={0} max={1} step={0.01}
                                value={Number(val)}
                                onChange={e => setConfigValues(v => ({ ...v, [f.name]: Number(e.target.value) }))}
                                style={{ flex: 1, accentColor: 'var(--accent)' }}
                              />
                              <span style={{ minWidth: 32, textAlign: 'right', fontSize: 11, fontWeight: 600 }}>
                                {Number(val).toFixed(2)}
                              </span>
                            </div>
                          ) : (
                            <input
                              type="number"
                              value={String(val ?? '')}
                              step={f.type === 'float' ? 0.01 : 1}
                              min={0}
                              onChange={e => setConfigValues(v => ({ ...v, [f.name]: Number(e.target.value) }))}
                              style={{ fontSize: 12, padding: '4px 8px' }}
                            />
                          )}
                        </label>
                      )
                    })}
                  </div>
                  <div className="small" style={{ color: 'var(--muted)', marginTop: 6, fontSize: 10, lineHeight: 1.4 }}>
                    Defaults give a <strong>perfect oracle</strong> upper-bound (instant, complete, noiseless feedback).
                    Adjust to simulate real operator conditions.
                  </div>
                </div>
              )
            })()}
            <button
              onClick={() => setShowAdvancedConfig(v => !v)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', fontSize: 12,
                color: 'var(--accent)', fontWeight: 600, padding: '4px 0',
              }}
            >
              {showAdvancedConfig ? '▾ Hide' : '▸ Show'} Advanced Config ({configSchemaQ.data.fields.length} fields)
            </button>
            {showAdvancedConfig && (() => {
              const groups: Record<string, ConfigField[]> = {}
              for (const f of configSchemaQ.data!.fields) {
                (groups[f.group] ??= []).push(f)
              }
              return Object.entries(groups).map(([group, fields]) => (
                <details key={group} style={{ marginTop: 8 }}>
                  <summary className="small" style={{ cursor: 'pointer', fontWeight: 600, color: '#bb86fc' }}>
                    {group} ({fields.length})
                  </summary>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8, padding: '8px 0' }}>
                    {fields.map(f => (
                      <label key={f.name} className="small" style={{ display: 'grid', gap: 2 }}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={f.name}>
                          {f.name.replace(/_/g, ' ')}
                        </span>
                        {f.type === 'bool' ? (
                          <input
                            type="checkbox"
                            checked={Boolean(configValues[f.name] ?? f.default)}
                            onChange={e => setConfigValues(v => ({ ...v, [f.name]: e.target.checked }))}
                          />
                        ) : f.type === 'float' || f.type === 'int' ? (
                          <input
                            type="number"
                            value={String(configValues[f.name] ?? f.default ?? '')}
                            step={f.type === 'float' ? 0.01 : 1}
                            onChange={e => setConfigValues(v => ({ ...v, [f.name]: Number(e.target.value) }))}
                            style={{ fontSize: 12, padding: '4px 8px' }}
                          />
                        ) : (
                          <input
                            type="text"
                            value={String(configValues[f.name] ?? f.default ?? '')}
                            onChange={e => setConfigValues(v => ({ ...v, [f.name]: e.target.value }))}
                            style={{ fontSize: 12, padding: '4px 8px' }}
                          />
                        )}
                      </label>
                    ))}
                  </div>
                </details>
              ))
            })()}
          </div>
        )}
        {/* ── LLM Status Banner ── */}
        <div style={{
          marginBottom: 12,
          padding: '8px 12px',
          background: llmEnabled
            ? (llmAvailable ? 'rgba(46,204,113,0.1)' : 'rgba(241,196,15,0.12)')
            : 'rgba(255,255,255,0.03)',
          borderRadius: 6,
          border: llmEnabled
            ? (llmAvailable ? '1px solid rgba(46,204,113,0.3)' : '1px solid rgba(241,196,15,0.35)')
            : '1px solid rgba(255,255,255,0.08)',
        }}>
          {backendConfigQ.isError && (
            <p className="small" style={{ color: 'var(--danger)', marginBottom: 6 }}>
              ⚠ Cannot reach backend config: {String((backendConfigQ.error as Error)?.message || backendConfigQ.error)}
            </p>
          )}
          <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <input type="checkbox" checked={llmEnabled} onChange={e => setExperimentLlmEnabled(e.target.checked)} />
            <span style={{ fontWeight: 600 }}>🧠 Experiment LLM Explanations</span>
            <>
              <span style={{
                display: 'inline-block',
                width: 8, height: 8,
                borderRadius: '50%',
                background: llmEnabled
                  ? (llmAvailable ? '#2ecc71' : '#f1c40f')
                  : '#555',
                marginLeft: 2,
              }} />
              <span style={{ color: 'var(--muted)' }}>
                {!llmEnabled
                  ? '— off for runs launched here'
                  : llmAvailable
                    ? `— on for runs launched here, connected to ${backendConfigQ.data?.ollama_model ?? 'Ollama'}`
                    : '— on for runs launched here, but Ollama is not reachable'}
              </span>
            </>
            <HelpIcon text={
              'Controls whether experiments launched from this tab request LLM-generated explanations. It does not change the detailed live playback tab.\n\n' +
              'OFF: Experiment runs use heuristic fallbacks only. No external service calls, no timeouts.\n\n' +
              'ON: Experiment events request LLM explanations. If the service is unreachable you\'ll see fallback text.\n\n' +
              'Providers:\n' +
              '• Ollama (local): install Ollama, pull a model, set OLLAMA_URL and OLLAMA_MODEL.\n' +
              '• Groq (cloud, fast): set LLM_PROVIDER=groq, GROQ_API_KEY, and optionally GROQ_MODEL in your .env.'
            } />
          </label>
          <p className="small" style={{ color: 'var(--muted)', marginTop: 4, marginLeft: 24 }}>
            Applies only to new experiment runs from this tab. It does not change the live detailed playback view.
          </p>
          {llmEnabled && !llmAvailable && (
            <p className="small" style={{ color: '#f1c40f', marginTop: 4, marginLeft: 24 }}>
              ⚠ Ollama is not reachable at {backendConfigQ.data?.ollama_url ?? 'localhost:11434'}.
              LLM calls will timeout after 60s and fall back to heuristics.
              Turn this off to eliminate the timeouts.
            </p>
          )}
        </div>

        <div style={{ marginBottom: 12, padding: '8px 12px', background: effectiveApiMode ? 'rgba(46,204,113,0.1)' : 'transparent', borderRadius: 6, border: effectiveApiMode ? '1px solid rgba(46,204,113,0.3)' : '1px solid transparent' }}>
          <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={effectiveApiMode}
              disabled={liveBreakageUsesApi}
              onChange={e => setRunApiMode(e.target.checked)}
            />
            <span style={{ fontWeight: 600 }}>🔗 API Mode</span>
            <span style={{ color: 'var(--muted)' }}>— route events through the live backend API (Neo4j + SINDIT integration)</span>
            <HelpIcon text="When enabled, experiment events are POSTed to the running backend API instead of being processed in the subprocess. This enables Neo4j persistence, SINDIT enrichment, co-occurrence tracking, and live prior propagation. Requires the backend to be running (uvicorn backend.app:app)." />
          </label>
          {liveBreakageUsesApi ? (
            <p className="small" style={{ color: 'var(--ok)', marginTop: 4, marginLeft: 24 }}>
              ✓ Live breakage runs always use the backend API. The backend forces strict API mode, server-side pattern derivation, isolated feedback scope, and no shared-prior persistence for this path.
            </p>
          ) : effectiveApiMode && (
            <p className="small" style={{ color: 'var(--ok)', marginTop: 4, marginLeft: 24 }}>
              ✓ Events will be POSTed to the backend API. Co-occurrence tracking, prior propagation, and SINDIT enrichment will happen server-side.
            </p>
          )}
        </div>

        {/* Run button */}
        {runMode === 'subprocess' ? (
          experimentType === 'stoppage' ? (
            <button className="primary" onClick={() => runExperimentMut.mutate()} disabled={runExperimentMut.isPending} style={{ padding: '6px 16px' }}>
              {runExperimentMut.isPending ? 'Running…' : (runApiMode ? '🔗 Run Stoppage via API' : '▶️ Run Stoppage Experiment')}
            </button>
          ) : (
            <button className="primary" onClick={() => runBreakageMut.mutate()} disabled={runBreakageMut.isPending} style={{ padding: '6px 16px', background: '#9b59b6' }}>
              {runBreakageMut.isPending ? 'Running…' : '🔧 Run Breakage Experiment'}
            </button>
          )
        ) : experimentType === 'breakage' ? (
          <button
            className="primary"
            onClick={() => runLiveBreakageMut.mutate()}
            disabled={runLiveBreakageMut.isPending || liveStatus === 'running'}
            style={{ padding: '6px 16px', background: '#9b59b6' }}
          >
            {liveStatus === 'running' ? '⏳ Running…' : '⚡ Run Live Breakage Experiment'}
          </button>
        ) : (
          <button
            className="primary"
            onClick={() => runLiveMut.mutate()}
            disabled={runLiveMut.isPending || liveStatus === 'running'}
            style={{ padding: '6px 16px', background: 'var(--accent)' }}
          >
            {liveStatus === 'running' ? '⏳ Running…' : '⚡ Run Live Experiment'}
          </button>
        )}

        {/* Subprocess results */}
        {runMode === 'subprocess' && experimentType === 'stoppage' && runExperimentMut.isError && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.3)' }}>
            <div className="small" style={{ color: 'var(--danger)', fontWeight: 700 }}>✗ Failed to start experiment</div>
            <pre style={{ fontSize: 10, color: 'var(--danger)', marginTop: 4, whiteSpace: 'pre-wrap' }}>
              {String((runExperimentMut.error as Error)?.message || runExperimentMut.error)}
            </pre>
          </div>
        )}
        {runMode === 'subprocess' && experimentType === 'stoppage' && runExperimentMut.data && (
          <div style={{ marginTop: 12 }}>
            <div className="small" style={{ color: runExperimentMut.data.success ? 'var(--ok)' : 'var(--danger)', fontWeight: 700 }}>
              {runExperimentMut.data.success ? (runApiMode ? '✓ API-mode experiment complete — events persisted to knowledge graph' : '✓ Experiment complete — refresh to see results') : '✗ Experiment failed'}
            </div>
            <pre style={{ fontSize: 10, maxHeight: 300, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4, marginTop: 4 }}>
              {runExperimentMut.data.stdout || runExperimentMut.data.stderr}
            </pre>
          </div>
        )}

        {/* Live run HTTP error */}
        {runMode === 'live' && runLiveMut.isError && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.3)' }}>
            <div className="small" style={{ color: 'var(--danger)', fontWeight: 700 }}>✗ Failed to start live experiment</div>
            <pre style={{ fontSize: 10, color: 'var(--danger)', marginTop: 4, whiteSpace: 'pre-wrap' }}>
              {(() => {
                const msg = String((runLiveMut.error as Error)?.message || runLiveMut.error)
                // Extract JSON detail from HTTP error response if present
                try {
                  const jsonMatch = msg.match(/\{.*\}/)
                  if (jsonMatch) {
                    const parsed = JSON.parse(jsonMatch[0])
                    if (parsed.detail) return parsed.detail
                  }
                } catch { /* use raw message */ }
                return msg
              })()}
            </pre>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 6 }}>Troubleshooting:
              • Is the backend running? (uvicorn backend.app:app)
              • Check the terminal for Python import errors or missing dependencies
              • Ensure feature CSVs exist (run Feature Extraction first)
            </div>
          </div>
        )}

        {/* Breakage subprocess results */}
        {runMode === 'subprocess' && experimentType === 'breakage' && runBreakageMut.isError && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.3)' }}>
            <div className="small" style={{ color: 'var(--danger)', fontWeight: 700 }}>✗ Breakage experiment failed to start</div>
            <pre style={{ fontSize: 10, color: 'var(--danger)', marginTop: 4, whiteSpace: 'pre-wrap' }}>
              {String((runBreakageMut.error as Error)?.message || runBreakageMut.error)}
            </pre>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.5 }}>
              <strong>Troubleshooting:</strong><br/>
              • Is the backend running? (<code>uvicorn backend.app:app</code>)<br/>
              • Check the terminal for Python import errors or missing dependencies<br/>
              • Ensure feature CSVs exist (run Feature Extraction first)
            </div>
          </div>
        )}
        {runMode === 'subprocess' && experimentType === 'breakage' && runBreakageMut.data && (() => {
          const d = runBreakageMut.data
          // Parse key metrics from stdout if successful
          const stdout = d.stdout || ''
          const stderr = d.stderr || ''
          const f1Match = stdout.match(/F1=(\d+\.\d+)/)
          const precMatch = stdout.match(/Precision=(\d+\.\d+)/)
          const recMatch = stdout.match(/Recall=(\d+\.\d+)/)
          const tpMatch = stdout.match(/TP=(\d+)/)
          const fpMatch = stdout.match(/FP=(\d+)/)
          // Extract Python error if failed
          const errLines = stderr.split('\n').filter(l => l.includes('Error') || l.includes('raise '))
          const lastErr = errLines.length > 0 ? errLines[errLines.length - 1].trim() : ''
          return (
            <div style={{ marginTop: 12 }}>
              <div className="small" style={{ color: d.success ? 'var(--ok)' : 'var(--danger)', fontWeight: 700 }}>
                {d.success ? '✓ Breakage experiment complete' : '✗ Breakage experiment failed'}
              </div>
              {/* Quick metrics summary on success */}
              {d.success && f1Match && (
                <div style={{ display: 'flex', gap: 12, marginTop: 6, flexWrap: 'wrap' }}>
                  {[{ lbl: 'F1', val: f1Match?.[1] }, { lbl: 'Precision', val: precMatch?.[1] }, { lbl: 'Recall', val: recMatch?.[1] }, { lbl: 'TP', val: tpMatch?.[1] }, { lbl: 'FP', val: fpMatch?.[1] }].filter(x => x.val).map(x => (
                    <div key={x.lbl} style={{ background: 'rgba(46,204,113,0.1)', padding: '4px 10px', borderRadius: 4, fontSize: 11 }}>
                      <span style={{ color: 'var(--muted)' }}>{x.lbl}: </span><strong>{x.val}</strong>
                    </div>
                  ))}
                </div>
              )}
              {/* Parsed error summary on failure */}
              {!d.success && lastErr && (
                <div style={{ marginTop: 6, padding: '6px 10px', background: 'rgba(231,76,60,0.08)', borderRadius: 4, fontSize: 11, color: 'var(--danger)', fontWeight: 600 }}>
                  {lastErr}
                </div>
              )}
              {!d.success && (
                <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.5 }}>
                  <strong>Troubleshooting:</strong><br/>
                  • Check that feature CSVs have labelled positive samples (pre_break &gt; 0)<br/>
                  • Ensure Site_a_line2 data is extracted: run Feature Extraction above<br/>
                  • With a single operation, self-training mode is used automatically
                </div>
              )}
              <details style={{ marginTop: 6 }}>
                <summary className="small" style={{ cursor: 'pointer', color: 'var(--accent)' }}>Show full output</summary>
                {stdout && (
                  <>
                    <div className="small" style={{ fontWeight: 600, marginTop: 4 }}>stdout:</div>
                    <pre style={{ fontSize: 10, maxHeight: 200, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4 }}>{stdout}</pre>
                  </>
                )}
                {stderr && (
                  <>
                    <div className="small" style={{ fontWeight: 600, marginTop: 4, color: 'var(--danger)' }}>stderr:</div>
                    <pre style={{ fontSize: 10, maxHeight: 200, overflow: 'auto', background: 'rgba(231,76,60,0.06)', padding: 8, borderRadius: 4, color: 'var(--danger)' }}>{stderr}</pre>
                  </>
                )}
              </details>
            </div>
          )
        })()}

        {/* Live breakage HTTP error */}
        {runMode === 'live' && experimentType === 'breakage' && runLiveBreakageMut.isError && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(231,76,60,0.1)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.3)' }}>
            <div className="small" style={{ color: 'var(--danger)', fontWeight: 700 }}>✗ Failed to start live breakage experiment</div>
            <pre style={{ fontSize: 10, color: 'var(--danger)', marginTop: 4, whiteSpace: 'pre-wrap' }}>
              {String((runLiveBreakageMut.error as Error)?.message || runLiveBreakageMut.error)}
            </pre>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.5 }}>
              <strong>Troubleshooting:</strong><br/>
              • Is the backend running? (<code>uvicorn backend.app:app</code>)<br/>
              • Ensure Site_a_line2 features are extracted (run Feature Extraction first)<br/>
              • Check the backend terminal for Python tracebacks
            </div>
          </div>
        )}

        {/* Recently-completed banner — shown when experiment finished while UI was away (standalone, outside progress dashboard) */}
        {reconnectedRunId && liveStatus === 'done' && !liveRunId && (
          <div style={{
            marginTop: 12, marginBottom: 4, padding: '8px 12px', borderRadius: 6, fontSize: 12,
            background: 'rgba(52,152,219,0.10)', border: '1px solid rgba(52,152,219,0.30)',
            color: '#3498db', display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <span style={{ fontSize: 16 }}>✅</span>
            <span>
              Experiment <strong>{reconnectedRunId}</strong> completed while you were away.
              It has been auto-selected — switch to the <strong>Overview</strong> tab to see results.
            </span>
            <button
              onClick={() => { setReconnectedRunId(null); setLiveStatus('idle') }}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#3498db', cursor: 'pointer', fontSize: 14, padding: '0 4px' }}
              title="Dismiss"
            >✕</button>
          </div>
        )}

        {/* Live progress dashboard */}
        {runMode === 'live' && liveRunId && (
          <div style={{ marginTop: 16 }}>
            {/* Reconnect banner — shown when UI re-attached to a running experiment */}
            {reconnectedRunId && liveStatus === 'running' && (
              <div style={{
                marginBottom: 10, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                background: 'rgba(46,204,113,0.10)', border: '1px solid rgba(46,204,113,0.30)',
                color: '#2ecc71', display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span style={{ fontSize: 16 }}>🔄</span>
                <span>
                  Reconnected to active experiment <strong>{reconnectedRunId}</strong>.
                  Earlier progress events before reload are not available — new events will appear below.
                </span>
              </div>
            )}
            {/* LLM warning banner — surfaces errors when explanations won't work */}
            {llmRunWarning && (
              <div style={{
                marginBottom: 10, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                background: 'rgba(241,196,15,0.10)', border: '1px solid rgba(241,196,15,0.30)',
                color: '#f1c40f', display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span style={{ fontSize: 16 }}>⚠</span>
                <span>{llmRunWarning}</span>
              </div>
            )}
            {/* LLM status from setup progress event */}
            {(() => {
              const setupEvt = progressEvents.find(e => e.phase === 'setup' && e.status === 'completed')
              const llmStatus = (setupEvt?.detail as Record<string, unknown> | undefined)?.llm_status as Record<string, unknown> | undefined
              const warning = llmStatus?.llm_warning as string | undefined
              if (!warning || warning === llmRunWarning) return null
              return (
                <div style={{
                  marginBottom: 10, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                  background: 'rgba(241,196,15,0.10)', border: '1px solid rgba(241,196,15,0.30)',
                  color: '#f1c40f', display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <span style={{ fontSize: 16 }}>🧠</span>
                  <span>{warning}</span>
                </div>
              )
            })()}
            {/* Explanation stats from eval completed event */}
            {(() => {
              const evalEvt = progressEvents.find(e => e.phase === 'eval' && e.status === 'completed')
              const ed = evalEvt?.detail as Record<string, unknown> | undefined
              if (!ed || typeof ed.n_explained !== 'number') return null
              const nExplained = ed.n_explained as number
              const nLlm = (ed.n_llm as number) ?? 0
              const nFallback = (ed.n_fallback as number) ?? 0
              const nSamples = (ed.n_samples as number) ?? 0
              if (nExplained === 0 && effectiveApiMode) {
                return (
                  <div style={{
                    marginBottom: 10, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                    background: 'rgba(231,76,60,0.10)', border: '1px solid rgba(231,76,60,0.30)',
                    color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    <span style={{ fontSize: 16 }}>❌</span>
                    <span>
                      No LLM explanations generated for {nSamples} eval samples.
                      {!llmEnabled && ' The 🧠 LLM toggle is OFF — enable it before running.'}
                      {llmEnabled && !llmAvailable && ' Ollama is not reachable — check that it\'s running.'}
                      {llmEnabled && llmAvailable && ' This is unexpected — check backend logs for errors.'}
                    </span>
                  </div>
                )
              }
              if (nExplained > 0) {
                return (
                  <div style={{
                    marginBottom: 10, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                    background: 'rgba(46,204,113,0.08)', border: '1px solid rgba(46,204,113,0.25)',
                    color: 'var(--ok)', display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    <span style={{ fontSize: 16 }}>💬</span>
                    <span>
                      {nExplained}/{nSamples} eval samples have explanations
                      {nLlm > 0 && <span> ({nLlm} 🧠 LLM</span>}
                      {nLlm > 0 && nFallback > 0 && ', '}
                      {nFallback > 0 && <span>{nLlm > 0 ? '' : '('}{nFallback} 📖 fallback</span>}
                      {(nLlm > 0 || nFallback > 0) && ')'}
                      . View them in the Sample Inspector tab.
                    </span>
                  </div>
                )
              }
              return null
            })()}
            {/* Progress bar */}
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, marginBottom: 4 }}>
                <span style={{ fontWeight: 600 }}>
                  {liveStatus === 'running' ? `⏳ ${lastProgress?.phase ?? '…'}` :
                   liveStatus === 'done' ? '✅ Complete' :
                   liveStatus === 'error' ? '❌ Error' : ''}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: 'var(--muted)' }}>
                    {lastProgress ? `${(lastProgress.pct ?? 0).toFixed(0)}% · ${(lastProgress.elapsed_s ?? 0).toFixed(1)}s` : ''}
                  </span>
                  {liveStatus === 'running' && liveRunId && (
                    <button
                      onClick={() => { if (confirm('Cancel the running experiment? Partial results will be lost.')) cancelMut.mutate() }}
                      disabled={cancelMut.isPending}
                      style={{
                        padding: '2px 10px', fontSize: 11, fontWeight: 600, borderRadius: 4,
                        cursor: cancelMut.isPending ? 'wait' : 'pointer',
                        background: 'rgba(231,76,60,0.12)', color: '#e74c3c',
                        border: '1px solid rgba(231,76,60,0.3)',
                        opacity: cancelMut.isPending ? 0.5 : 1,
                      }}
                    >
                      {cancelMut.isPending ? '…' : '■'} Cancel
                    </button>
                  )}
                </span>
              </div>
              <div style={{ height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', borderRadius: 3, transition: 'width 0.3s',
                  width: `${lastProgress?.pct ?? 0}%`,
                  background: liveStatus === 'error' ? 'var(--danger)' :
                              liveStatus === 'done' ? 'var(--ok)' : 'var(--accent)',
                }} />
              </div>
            </div>

            {/* Real-time score charts */}
            <ExperimentScorePanel />

            {/* Phase timeline */}
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
              {['setup', 'split', 'train', 'test', 'eval', 'report', 'sindit', 'neo4j'].map(phase => {
                const evts = progressEvents.filter(e => e.phase === phase)
                const last = evts[evts.length - 1]
                const isDone = last?.status === 'completed'
                const isWarn = last?.status === 'warning'
                const isActive = last?.status === 'started' || last?.status === 'progress'
                return (
                  <div key={phase} style={{
                    padding: '3px 10px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                    background: isDone ? 'rgba(46,204,113,0.15)' :
                                isWarn ? 'rgba(243,156,18,0.15)' :
                                isActive ? 'rgba(52,152,219,0.2)' : 'rgba(255,255,255,0.04)',
                    color: isDone ? 'var(--ok)' : isWarn ? '#f39c12' : isActive ? 'var(--accent)' : 'var(--muted)',
                    border: isActive ? '1px solid var(--accent)' : '1px solid transparent',
                  }}>
                    {isDone ? '✓' : isWarn ? '⚠' : isActive ? '▸' : '○'} {phase}
                  </div>
                )
              })}
            </div>

            {/* Detail of last phase */}
            {lastProgress?.detail && !Array.isArray(lastProgress.detail.prior_history) && (
              <div style={{ background: 'rgba(0,0,0,0.15)', padding: 8, borderRadius: 4, fontSize: 11, marginBottom: 8 }}>
                {Object.entries(lastProgress.detail).map(([k, v]) => (
                  <span key={k} style={{ marginRight: 12 }}>
                    <span style={{ color: 'var(--muted)' }}>{k}: </span>
                    <strong>{typeof v === 'number' ? (Number.isInteger(v) ? v : (v as number).toFixed(3)) : JSON.stringify(v)}</strong>
                  </span>
                ))}
              </div>
            )}

            {/* ── Live Progress Charts: F1 / Precision / Recall + Prior Evolution ── */}
            {(() => {
              // Gather phase metrics from completed events
              const testEvt = latestProgressEvent(progressEvents, 'test', 'completed')
              const evalEvt = latestProgressEvent(progressEvents, 'eval', 'completed')
              const reportEvt = latestProgressEvent(progressEvents, 'report', 'progress')
              const td = testEvt?.detail || {} as Record<string, unknown>
              const ed = evalEvt?.detail || {} as Record<string, unknown>
              const rd = reportEvt?.detail || {} as Record<string, unknown>

              const hasMetrics = typeof td.f1 === 'number' || typeof ed.f1 === 'number'
              const priorHist = (Array.isArray(ed.prior_history) ? ed.prior_history : []) as Record<string, number>[]

              if (!hasMetrics && priorHist.length === 0) return null

              // Metric bars helper
              const MetricBar = ({ label, testVal, evalVal }: { label: string; testVal?: number; evalVal?: number }) => {
                const tv = testVal ?? 0, ev = evalVal ?? 0
                const delta = ev - tv
                const deltaColor = delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--danger)' : 'var(--muted)'
                return (
                  <div style={{ marginBottom: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}>
                      <span style={{ fontWeight: 600 }}>{label}</span>
                      {typeof evalVal === 'number' && typeof testVal === 'number' && (
                        <span style={{ color: deltaColor, fontWeight: 600 }}>
                          {delta > 0 ? '+' : ''}{(delta * 100).toFixed(1)}pp
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                      {/* Test bar */}
                      <div style={{ flex: 1, position: 'relative', height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 3, overflow: 'hidden' }}>
                        {typeof testVal === 'number' && (
                          <div style={{ height: '100%', width: `${tv * 100}%`, background: 'rgba(52,152,219,0.45)', borderRadius: 3 }} />
                        )}
                        <span style={{ position: 'absolute', left: 4, top: 1, fontSize: 9, fontWeight: 600, color: 'rgba(255,255,255,0.7)' }}>
                          test {typeof testVal === 'number' ? (tv * 100).toFixed(1) + '%' : '—'}
                        </span>
                      </div>
                      {/* Eval bar */}
                      <div style={{ flex: 1, position: 'relative', height: 14, background: 'rgba(255,255,255,0.04)', borderRadius: 3, overflow: 'hidden' }}>
                        {typeof evalVal === 'number' && (
                          <div style={{ height: '100%', width: `${ev * 100}%`, background: 'rgba(46,204,113,0.45)', borderRadius: 3 }} />
                        )}
                        <span style={{ position: 'absolute', left: 4, top: 1, fontSize: 9, fontWeight: 600, color: 'rgba(255,255,255,0.7)' }}>
                          eval {typeof evalVal === 'number' ? (ev * 100).toFixed(1) + '%' : '—'}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              }

              return (
                <div style={{ display: 'grid', gridTemplateColumns: priorHist.length > 0 ? '1fr 1fr' : '1fr', gap: 12, marginBottom: 8 }}>
                  {/* F1 / Precision / Recall / AUC comparison */}
                  {hasMetrics && (
                    <div style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6, padding: 10 }}>
                      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--muted)', marginBottom: 6 }}>
                        Metrics: <span style={{ color: 'rgba(52,152,219,0.8)' }}>Test (baseline)</span> vs <span style={{ color: 'rgba(46,204,113,0.8)' }}>Eval (feedback)</span>
                      </div>
                      <MetricBar label="F1 Score" testVal={td.f1 as number | undefined} evalVal={ed.f1 as number | undefined} />
                      <MetricBar label="Precision" testVal={td.precision as number | undefined} evalVal={ed.precision as number | undefined} />
                      <MetricBar label="Recall" testVal={td.recall as number | undefined} evalVal={ed.recall as number | undefined} />
                      <MetricBar label="AUC-ROC" testVal={td.auc_roc as number | undefined} evalVal={ed.auc_roc as number | undefined} />
                      {/* Comparison deltas summary */}
                      {rd.pct_f1_improvement != null && (
                        <div style={{ marginTop: 6, padding: '4px 8px', background: 'rgba(255,255,255,0.04)', borderRadius: 4, fontSize: 10 }}>
                          <span style={{ color: 'var(--muted)' }}>F1 improvement: </span>
                          <strong style={{ color: (rd.pct_f1_improvement as number) > 0 ? 'var(--ok)' : 'var(--danger)' }}>
                            {(rd.pct_f1_improvement as number) > 0 ? '+' : ''}{(rd.pct_f1_improvement as number).toFixed(1)}%
                          </strong>
                          {rd.n_feedback_events != null && (
                            <span style={{ marginLeft: 12, color: 'var(--muted)' }}>
                              ({rd.n_feedback_events as number} feedback events)
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Prior Evolution sparklines */}
                  {priorHist.length > 0 && (() => {
                    // Get all pattern keys across all snapshots
                    const allKeys = [...new Set(priorHist.flatMap(s => Object.keys(s)))]
                    const topKeys = allKeys
                      .map(k => ({ k, range: Math.abs((priorHist[priorHist.length - 1]?.[k] ?? 0.5) - (priorHist[0]?.[k] ?? 0.5)) }))
                      .sort((a, b) => b.range - a.range)
                      .slice(0, 8)
                      .map(o => o.k)

                    // Compute the actual value range to zoom the Y axis
                    const allVals = topKeys.flatMap(k => priorHist.map(s => s[k] ?? 0.5))
                    const yMin = Math.max(0, Math.min(...allVals) - 0.05)
                    const yMax = Math.min(1, Math.max(...allVals) + 0.05)
                    const yRange = Math.max(yMax - yMin, 0.1)

                    const W = 280, H = 140, pad = 8, padL = 24
                    const toX = (i: number) => padL + (i / Math.max(priorHist.length - 1, 1)) * (W - padL - pad)
                    const toY = (v: number) => H - pad - Math.max(0, Math.min(1, (v - yMin) / yRange)) * (H - 2 * pad)

                    // High-contrast palette — distinct hues, all bright on dark bg
                    const colors = ['#7aa2f7', '#9ece6a', '#f7768e', '#e0af68', '#bb9af7', '#73daca', '#ff9e64', '#2ac3de']

                    return (
                      <div style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6, padding: 10 }}>
                        <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--muted)', marginBottom: 4 }}>
                          Prior Evolution ({priorHist.length} snapshots, top {topKeys.length} patterns)
                        </div>
                        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
                          {/* Y-axis grid lines — adaptive to zoomed range */}
                          {Array.from({ length: 5 }, (_, i) => yMin + (i / 4) * yRange).map(v => (
                            <g key={v}>
                              <line x1={padL} x2={W - pad} y1={toY(v)} y2={toY(v)} stroke="rgba(255,255,255,0.08)" strokeWidth={0.5} />
                              <text x={padL - 2} y={toY(v) + 3} textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize={6}>{v.toFixed(2)}</text>
                            </g>
                          ))}
                          {/* 0.5 reference line if in range */}
                          {yMin < 0.5 && yMax > 0.5 && (
                            <line x1={padL} x2={W - pad} y1={toY(0.5)} y2={toY(0.5)} stroke="rgba(255,255,255,0.15)" strokeWidth={0.5} strokeDasharray="3,3" />
                          )}
                          {/* Lines per pattern — thicker strokes, distinct dash patterns for overlaps */}
                          {topKeys.map((key, ki) => {
                            const d = priorHist.map((snap, i) => {
                              const v = snap[key] ?? 0.5
                              return `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`
                            }).join(' ')
                            const lastVal = priorHist[priorHist.length - 1]?.[key] ?? 0.5
                            // Alternate dash patterns for overlapping lines
                            const dashPatterns = ['', '6,2', '2,2', '8,3,2,3', '', '4,4', '1,3', '10,2']
                            return (
                              <g key={key}>
                                <path d={d} fill="none" stroke={colors[ki % colors.length]}
                                  strokeWidth={ki < 4 ? 1.8 : 1.4}
                                  strokeDasharray={dashPatterns[ki] || ''}
                                  opacity={0.9}
                                />
                                <circle cx={toX(priorHist.length - 1)} cy={toY(lastVal)} r={2.5} fill={colors[ki % colors.length]} />
                                {/* End-of-line value label for top patterns */}
                                {ki < 4 && (
                                  <text
                                    x={toX(priorHist.length - 1) + 4}
                                    y={toY(lastVal) + 3 + ki * 7}
                                    fill={colors[ki % colors.length]}
                                    fontSize={5.5}
                                    fontWeight={600}
                                  >
                                    {lastVal.toFixed(2)}
                                  </text>
                                )}
                              </g>
                            )
                          })}
                        </svg>
                        {/* Legend — 2-column grid for readability */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 12px', marginTop: 6 }}>
                          {topKeys.map((key, ki) => {
                            const firstVal = priorHist[0]?.[key] ?? 0.5
                            const lastVal = priorHist[priorHist.length - 1]?.[key] ?? 0.5
                            const delta = lastVal - firstVal
                            const shortKey = key.length > 22 ? key.slice(0, 20) + '…' : key
                            return (
                              <span key={key} style={{ fontSize: 9, display: 'flex', alignItems: 'center', gap: 3 }}>
                                <span style={{
                                  width: 10, height: 3, borderRadius: 1,
                                  background: colors[ki % colors.length],
                                  display: 'inline-block',
                                  borderBottom: (ki >= 1 && ki <= 3) ? `1px dashed ${colors[ki % colors.length]}` : undefined,
                                }} />
                                <span style={{ color: '#bbb', maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={key}>{shortKey}</span>
                                <span style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{lastVal.toFixed(2)}</span>
                                <span style={{ fontSize: 8, color: delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--danger)' : 'var(--muted)', fontWeight: 600 }}>
                                  {delta > 0 ? '+' : ''}{delta.toFixed(3)}
                                </span>
                              </span>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })()}
                </div>
              )
            })()}

            {/* Error detail panel */}
            {liveStatus === 'error' && (() => {
              const errorEvts = progressEvents.filter(e => e.status === 'error')
              const lastErr = errorEvts[errorEvts.length - 1]
              return (
                <div style={{ marginTop: 8, padding: 12, background: 'rgba(231,76,60,0.08)', borderRadius: 6, border: '1px solid rgba(231,76,60,0.25)' }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: 'var(--danger)', marginBottom: 6 }}>❌ Experiment Failed</div>
                  {lastErr ? (
                    <>
                      <div style={{ fontSize: 12, marginBottom: 4 }}>
                        <span style={{ color: 'var(--muted)' }}>Phase: </span>
                        <strong>{lastErr.phase}</strong>
                        <span style={{ color: 'var(--muted)', marginLeft: 12 }}>at {(lastErr.elapsed_s ?? 0).toFixed(1)}s</span>
                      </div>
                      <div style={{ fontSize: 12, marginBottom: 6, color: 'var(--danger)' }}>{lastErr.message}</div>
                      {lastErr.detail?.traceback && (
                        <details style={{ marginBottom: 8 }}>
                          <summary className="small" style={{ cursor: 'pointer', color: 'var(--muted)' }}>Show traceback</summary>
                          <pre style={{ fontSize: 10, maxHeight: 200, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4, marginTop: 4, whiteSpace: 'pre-wrap' }}>
                            {String(lastErr.detail.traceback)}
                          </pre>
                        </details>
                      )}
                    </>
                  ) : (
                    <div style={{ fontSize: 12, color: 'var(--muted)' }}>No error details received — the connection may have dropped before error data was sent.</div>
                  )}
                  <div className="small" style={{ color: 'var(--muted)', marginTop: 6, lineHeight: 1.6 }}>
                    Troubleshooting:<br/>
                    • Check the backend terminal for Python tracebacks<br/>
                    • Ensure feature CSV files exist (run extraction first)<br/>
                    • Verify operation IDs match your data (e.g. OF00001, OF00002)<br/>
                    {liveRunMeta && !liveRunMeta.ollama_ok && (
                      <>• <strong style={{ color: 'var(--danger)' }}>Ollama is unreachable</strong> — LLM calls will hang. Turn off LLM explanations or start Ollama<br/></>
                    )}
                    • If the error persists, try subprocess mode as a fallback
                  </div>
                  <button
                    style={{ marginTop: 8, padding: '4px 14px', fontSize: 12, borderRadius: 6, cursor: 'pointer', background: 'rgba(255,255,255,0.08)', color: 'var(--muted)', border: '1px solid rgba(255,255,255,0.15)' }}
                    onClick={() => { setLiveStatus('idle'); setLiveRunId(null); setProgressEvents([]); useExperimentScoreStore.getState().clear(); setLlmRunWarning(null); setLiveRunMeta(null); setReconnectedRunId(null) }}
                  >
                    ↩ Reset &amp; Try Again
                  </button>
                </div>
              )
            })()}

            {/* Run metadata banner */}
            {liveRunMeta && (
              <div style={{ marginBottom: 8, padding: '6px 10px', background: 'rgba(255,255,255,0.04)', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', display: 'flex', flexWrap: 'wrap', gap: '4px 16px' }}>
                <span>run: <strong>{String(liveRunMeta.run_id ?? '')}</strong></span>
                {liveRunMeta.features_csv ? <span>csv: <strong>{String(liveRunMeta.features_csv)}</strong></span> : null}
                <span>ollama:
                  <strong style={{ color: liveRunMeta.ollama_ok ? 'var(--ok)' : 'var(--danger)' }}>
                    {liveRunMeta.ollama_ok ? ' \u2713 connected' : ' \u2717 unreachable'}
                  </strong>
                  {liveRunMeta.ollama_model ? <span> ({String(liveRunMeta.ollama_model)})</span> : null}
                </span>
                {liveRunMeta.generate_explanations != null && (
                  <span>LLM explanations: <strong>{liveRunMeta.generate_explanations ? 'on' : 'off'}</strong></span>
                )}
              </div>
            )}

            {/* Event log — open by default while running */}
            <details open={liveStatus === 'running' || liveStatus === 'error'}>
              <summary className="small" style={{ cursor: 'pointer', color: 'var(--muted)' }}>
                Event log ({progressEvents.length} events)
              </summary>
              <div style={{ maxHeight: 300, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 4, marginTop: 4 }}>
                {progressEvents.length === 0 && liveStatus === 'running' && (
                  <div style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--muted)' }}>
                    Waiting for first event from runner… (idle timeout: 90s)
                  </div>
                )}
                {progressEvents.map((evt, i) => (
                  <div key={i} style={{ fontSize: 10, fontFamily: 'monospace', padding: '2px 0', color: evt.status === 'error' ? 'var(--danger)' : evt.status === 'warning' ? '#f39c12' : evt.status === 'completed' ? 'var(--ok)' : 'inherit' }}>
                    <span style={{ color: 'var(--muted)' }}>[{(evt.elapsed_s ?? 0).toFixed(1)}s]</span>{' '}
                    <strong>{evt.phase}</strong>.{evt.status}: {evt.message}
                    {evt.detail && Object.keys(evt.detail).length > 0 && evt.detail.traceback == null && (
                      <span style={{ color: 'var(--muted)', marginLeft: 8 }}>
                        {Object.entries(evt.detail).slice(0, 4).map(([k, v]) =>
                          `${k}=${typeof v === 'number' ? (Number.isInteger(v) ? v : (v as number).toFixed(3)) : typeof v === 'string' ? v.slice(0, 60) : JSON.stringify(v)?.slice(0, 40)}`
                        ).join(', ')}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>

      {/* ── Model Registry ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>📦 Model Registry <HelpIcon text="Lists trained model files (IsolationForest + LOF seed models, RL agent states) saved in data/models/. Each experiment fold trains a new model. Models are used by the scorer pipeline for anomaly detection." /></div>
        {modelsQ.data?.models && modelsQ.data.models.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 8 }}>
            {modelsQ.data.models.map(m => (
              <div key={m.filename} className="card" style={{ padding: 8, fontSize: 11 }}>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{m.filename}</div>
                <div style={{ color: 'var(--muted)' }}>
                  {(m.size_bytes / 1024).toFixed(1)} KB · {m.type}
                </div>
                <div style={{ color: 'var(--muted)', fontSize: 10 }}>
                  {new Date(m.modified).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        ) : modelsQ.isLoading ? (
          <div className="small">Loading models…</div>
        ) : (
          <div className="small" style={{ color: 'var(--muted)' }}>No models found in data/models/</div>
        )}
      </div>

      {/* ── LLM Analysis ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>🤖 LLM Experiment Analysis <HelpIcon text="Uses an LLM (or heuristic fallback) to analyse experiment results and generate actionable recommendations: which thresholds to adjust, which patterns are under/over-performing, and where false positives concentrate." /></div>
        <p className="small" style={{ color: 'var(--muted)', marginBottom: 8 }}>
          Generate an AI-powered review of experiment results — pattern performance, prior recommendations, and false-positive analysis.
        </p>
        <button
          className="primary"
          disabled={llmSummaryPending || !effectiveRunId}
          style={{ padding: '6px 16px', marginBottom: 8 }}
          onClick={async () => {
            setLlmSummaryPending(true)
            try {
              const body = {
                results: fullResultsQ.data || { run_id: effectiveRunId },
                focus: 'pattern performance and prior recommendations',
                include_threshold_recommendations: true,
              }
              const res = await api<{
                summary: string; recommendations: string[];
                threshold_recommendations?: ThresholdRec[];
                key_metrics: Record<string, unknown>; source: string
              }>('/agent/memory/experiment/summary', 'POST', body)
              setLlmSummary(res)
            } catch (e: unknown) {
              const err = e as Error
              setLlmSummary({ summary: `Error: ${err.message || err}`, recommendations: [], key_metrics: {}, source: 'error' })
            } finally {
              setLlmSummaryPending(false)
            }
          }}
        >
          {llmSummaryPending ? 'Analyzing…' : '🧠 Generate Analysis'}
        </button>
        {llmSummary && (
          <div style={{ marginTop: 8 }}>
            <div style={{ background: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 6, fontSize: 12, whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
              {llmSummary.summary}
            </div>
            {llmSummary.recommendations.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>📋 Recommendations</div>
                <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12 }}>
                  {llmSummary.recommendations.map((r, i) => <li key={i} style={{ marginBottom: 2 }}>{r}</li>)}
                </ul>
              </div>
            )}
            {Object.keys(llmSummary.key_metrics).length > 0 && (
              <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {Object.entries(llmSummary.key_metrics).map(([k, v]) => (
                  <div key={k} style={{ background: 'rgba(52,152,219,0.1)', padding: '4px 10px', borderRadius: 4, fontSize: 11 }}>
                    <span style={{ color: 'var(--muted)' }}>{k}: </span>
                    <strong>{typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : String(v ?? '–')}</strong>
                  </div>
                ))}
              </div>
            )}
            <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>Source: {llmSummary.source}</div>

            {/* Threshold Recommendations — actionable chips */}
            {llmSummary.threshold_recommendations && llmSummary.threshold_recommendations.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 6 }}>🎯 Threshold Recommendations</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {llmSummary.threshold_recommendations.map((rec, i) => (
                    <div
                      key={i}
                      style={{
                        padding: '6px 12px', borderRadius: 6, fontSize: 11,
                        background: 'rgba(155,89,182,0.1)', border: '1px solid rgba(155,89,182,0.3)',
                        cursor: 'pointer', maxWidth: 320,
                      }}
                      title={rec.reason}
                      onClick={() => {
                        // Apply recommendation to config if it's a known config parameter
                        if (!rec.parameter.startsWith('prior:')) {
                          setConfigValues(v => ({ ...v, [rec.parameter]: rec.recommended_value }))
                          setShowAdvancedConfig(true)
                        }
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: 2 }}>
                        {rec.parameter}
                        {rec.current_value != null && (
                          <span style={{ color: 'var(--muted)', fontWeight: 400 }}>
                            {' '}{Number.isFinite(rec.current_value) ? rec.current_value.toFixed(3) : String(rec.current_value)} →{' '}
                          </span>
                        )}
                        <span style={{ color: 'var(--ok)' }}>{rec.recommended_value != null && Number.isFinite(rec.recommended_value) ? rec.recommended_value.toFixed(3) : String(rec.recommended_value ?? '–')}</span>
                      </div>
                      <div style={{ color: 'var(--muted)', fontSize: 10, lineHeight: 1.3 }}>{rec.reason}</div>
                    </div>
                  ))}
                </div>
                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                  Click a recommendation to apply it to the config form
                </div>
              </div>
            )}
          </div>
        )}

        {/* Prior diff chart (after live experiment) */}
        {priorDiff && Object.keys(priorDiff).length > 0 && (
          <div style={{ marginTop: 12 }}>
            <PriorDiffSVG diff={priorDiff} title="Prior Changes (sandbox diff)" />
          </div>
        )}
      </div>

      {/* ── Experiment → Live Feedback Bridge ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>🔁 Experiment → Live Feedback Bridge <HelpIcon text="Transfers experiment findings into the live system's memory store. 'Confirm All True Positives' bulk-submits confirm feedback for correctly flagged pre-stoppage samples, strengthening their pattern priors. 'Dismiss All False Positives' submits dismiss feedback for falsely flagged normal samples, weakening their priors. This bridges offline experiment results into the live monitoring system." /></div>
        <p className="small" style={{ color: 'var(--muted)', marginBottom: 8 }}>
          Bridge experiment findings into the live system's memory. Bulk-confirm true positives and dismiss false positives.
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            className="primary"
            disabled={reviewPending || !effectiveRunId}
            style={{ padding: '6px 16px', background: 'var(--ok)' }}
            onClick={async () => {
              setReviewPending(true)
              try {
                const reviews = (evalPhase?.samples || [])
                  .filter(s => s.label === 'fault' && s.predicted_positive)
                  .map(s => ({ sample_id: s.sample_id, action: 'confirm' as const, pattern_keys: s.detected_patterns || [], reason: 'Batch confirm from experiment — true positive' }))
                if (reviews.length === 0) { setReviewResult({ updated: 0, failed: 0, prior_changes: {}, retrain_triggered: false }); return }
                const res = await api<{ updated: number; failed: number; prior_changes: Record<string, unknown>; retrain_triggered: boolean }>('/agent/memory/experiment/review', 'POST', { reviews })
                setReviewResult(res)
                retrainStatusQ.refetch()
              } catch {
                setReviewResult({ updated: 0, failed: 1, prior_changes: {}, retrain_triggered: false })
              } finally {
                setReviewPending(false)
              }
            }}
          >
            {reviewPending ? 'Processing…' : '✅ Confirm All True Positives'}
          </button>
          <button
            className="primary"
            disabled={reviewPending || !effectiveRunId}
            style={{ padding: '6px 16px', background: 'var(--danger)' }}
            onClick={async () => {
              setReviewPending(true)
              try {
                const reviews = (evalPhase?.samples || [])
                  .filter(s => s.label === 'normal' && s.predicted_positive)
                  .map(s => ({ sample_id: s.sample_id, action: 'dismiss' as const, pattern_keys: s.detected_patterns || [], reason: 'Batch dismiss from experiment — false positive' }))
                if (reviews.length === 0) { setReviewResult({ updated: 0, failed: 0, prior_changes: {}, retrain_triggered: false }); return }
                const res = await api<{ updated: number; failed: number; prior_changes: Record<string, unknown>; retrain_triggered: boolean }>('/agent/memory/experiment/review', 'POST', { reviews })
                setReviewResult(res)
                retrainStatusQ.refetch()
              } catch {
                setReviewResult({ updated: 0, failed: 1, prior_changes: {}, retrain_triggered: false })
              } finally {
                setReviewPending(false)
              }
            }}
          >
            ❌ Dismiss All False Positives
          </button>
        </div>
        {reviewResult && (
          <div style={{ marginTop: 8, background: 'rgba(0,0,0,0.15)', padding: 10, borderRadius: 6, fontSize: 12 }}>
            <div><strong>{reviewResult.updated}</strong> priors updated, <strong>{reviewResult.failed}</strong> failed</div>
            {reviewResult.retrain_triggered && <div style={{ color: 'var(--ok)', marginTop: 4 }}>🔄 Model retraining was triggered by this batch!</div>}
            {Object.keys(reviewResult.prior_changes).length > 0 && (
              <div style={{ marginTop: 6 }}>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>Prior Changes:</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {Object.entries(reviewResult.prior_changes).map(([k, v]) => (
                    <span key={k} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: 'rgba(52,152,219,0.1)' }}>{k}: {typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Model Retrain Status ── */}
      <div className="card" style={{ padding: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>🏋️ Model Retrain Status <HelpIcon text="Shows how much feedback has been received since the last model retrain. When enough new feedback accumulates (exceeds the threshold), the system recommends retraining the anomaly model to incorporate the new data. You can also trigger a manual retrain." /></div>
        {retrainStatusQ.data ? (
          <div style={{ fontSize: 12 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, marginBottom: 8 }}>
              {[
                { val: retrainStatusQ.data.total_feedback, lbl: 'Total Feedback' },
                { val: retrainStatusQ.data.since_last_retrain, lbl: 'Since Retrain' },
                { val: retrainStatusQ.data.threshold, lbl: 'Threshold' },
              ].map(item => (
                <div key={item.lbl} className="card" style={{ padding: 8, textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>{item.val}</div>
                  <div className="small" style={{ color: 'var(--muted)' }}>{item.lbl}</div>
                </div>
              ))}
              <div className="card" style={{ padding: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: retrainStatusQ.data.should_retrain ? 'var(--ok)' : 'var(--muted)' }}>
                  {retrainStatusQ.data.should_retrain ? 'Yes' : 'No'}
                </div>
                <div className="small" style={{ color: 'var(--muted)' }}>Should Retrain</div>
              </div>
            </div>
            {retrainStatusQ.data.last_retrain_at && (
              <div className="small" style={{ color: 'var(--muted)' }}>Last retrain: {retrainStatusQ.data.last_retrain_at}</div>
            )}
            <button className="primary" disabled={retrainMut.isPending} style={{ padding: '6px 16px', marginTop: 8 }} onClick={() => retrainMut.mutate()}>
              {retrainMut.isPending ? 'Retraining…' : '🔄 Trigger Retrain Now'}
            </button>
            {retrainMut.data && (
              <div className="small" style={{ marginTop: 4, color: retrainMut.data.success ? 'var(--ok)' : 'var(--danger)' }}>
                {retrainMut.data.message}{retrainMut.data.n_samples_used != null && ` (${retrainMut.data.n_samples_used} samples)`}
              </div>
            )}
          </div>
        ) : retrainStatusQ.isLoading ? (
          <div className="small">Loading retrain status…</div>
        ) : (
          <div className="small" style={{ color: 'var(--muted)' }}>Retrain status unavailable. Ensure the backend is running.</div>
        )}
      </div>
    </>
  )
}

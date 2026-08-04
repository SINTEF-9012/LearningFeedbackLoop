/**
 * HarmonicsTab — Offline view of harmonic model checkpoints and training.
 *
 * Shows persisted status, training metrics, and checkpoint config for both
 * harmonic context and harmonic pair presets. NOT tied to the live demo
 * stream — this reads `/harmonic/status`, starts preset training via
 * `/harmonic/train`, and can retrain from feedback via `/harmonic/retrain`.
 *
 * Tag: [HARMONIC_CONTEXT_V1]
 */
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../../api/http'
import { f3 } from '../charts'

const DATASET_OPTIONS = [
  { value: 'casedata', label: 'casedata' },
  { value: 'stoppage_1hz', label: 'stoppage_1hz' },
  { value: 'site_a_line2', label: 'site_a_line2' },
  { value: 'raw_accelerometer', label: 'raw_accelerometer' },
  { value: 'pair_raw', label: 'pair_raw' },
  { value: 'pair_casedata', label: 'pair_casedata' },
  { value: 'pair_lfl', label: 'pair_lfl' },
] as const

type HarmonicDataset = (typeof DATASET_OPTIONS)[number]['value']
type HarmonicScorerKind = 'context' | 'pair'

interface CheckpointStatus {
  available: boolean
  torch_installed: boolean
  model_loaded: boolean
  dataset_name: string
  scorer_kind: string
  harmonic_mode?: string
  cnn_window?: number
  trained_at?: string | null
  model_save_path: string
  model_path_exists: boolean
}

interface StatusResponse {
  available: boolean
  torch_installed: boolean
  model_loaded: boolean
  dataset_name: string
  scorer_kind: string
  n_harm_features: number
  n_params: number
  harmonic_mode: string
  cnn_window: number
  trained_at: string | null
  training_metrics: Record<string, unknown>
  model_save_path: string
  model_path_exists: boolean
  checkpoint_statuses: Record<string, CheckpointStatus>
}

interface TrainResponse {
  status: string
  message: string
  task_id?: string
}

interface TrainRequestBody {
  dataset: HarmonicDataset
  random_seed?: number
  model_save_path?: string
  checkpoint_suffix?: string
  replace_checkpoint?: boolean
}

interface RetrainBucketStatus {
  dataset_name: string
  scorer_kind: string
  total_feedback: number
  since_last_retrain: number
  retrain_threshold: number
  buffer_size: number
  confirmed_in_buffer: number
  dismissed_in_buffer: number
  should_retrain: boolean
  retrain_count: number
  last_retrain?: string | null
  model_save_path: string
}

interface RetrainStatusResponse {
  total_feedback: number
  active_bucket?: string | null
  buckets: Record<string, RetrainBucketStatus>
}

interface RetrainRequestBody {
  dataset: HarmonicDataset
  scorer_kind: HarmonicScorerKind
  random_seed?: number
  model_save_path?: string
  checkpoint_suffix?: string
  replace_checkpoint?: boolean
}

interface RetrainResponse {
  success: boolean
  message: string
  bucket_key: string
  dataset_name: string
  scorer_kind: string
  n_samples_used: number
  n_confirmed: number
  n_dismissed: number
  model_path: string
  best_val_loss?: number | null
  best_val_acc?: number | null
  duration_s: number
  training_result: Record<string, unknown>
}

interface HarmonicFeedbackSeedRequestBody {
  dataset: HarmonicDataset
  scorer_kind: HarmonicScorerKind
  confirmed?: number
  dismissed?: number
  clear_existing?: boolean
}

interface HarmonicFeedbackSeedResponse {
  enabled: boolean
  bucket_key: string
  dataset_name: string
  scorer_kind: string
  added_confirmed: number
  added_dismissed: number
  cleared_existing: boolean
  removed_buffer_size: number
  removed_total_feedback: number
  total_feedback: number
  buffer_size: number
  confirmed_in_buffer: number
  dismissed_in_buffer: number
  should_retrain: boolean
}

interface SharedCheckpointOptions {
  random_seed?: number
  model_save_path?: string
  checkpoint_suffix?: string
  replace_checkpoint?: boolean
}

function MetricCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ padding: 8, background: 'var(--bg-alt, #1a1a1a)', borderRadius: 4 }}>
      <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</div>
      <div style={{ fontSize: 14, fontFamily: 'monospace', marginTop: 2 }}>{value}</div>
    </div>
  )
}

function fmtMetric(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'number') return Number.isFinite(v) ? f3(v) : '—'
  if (typeof v === 'string') return v
  return String(v)
}

function datasetScorerKind(dataset: HarmonicDataset): HarmonicScorerKind {
  return dataset === 'pair_raw' || dataset === 'pair_casedata' || dataset === 'pair_lfl'
    ? 'pair'
    : 'context'
}

export function HarmonicsTab() {
  const qc = useQueryClient()
  const [dataset, setDataset] = useState<HarmonicDataset>('pair_lfl')
  const [randomSeed, setRandomSeed] = useState('')
  const [checkpointSuffix, setCheckpointSuffix] = useState('')
  const [modelSavePath, setModelSavePath] = useState('')
  const [replaceCheckpoint, setReplaceCheckpoint] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const selectedScorerKind = datasetScorerKind(dataset)

  const statusQ = useQuery<StatusResponse>({
    queryKey: ['harmonic-status', dataset],
    queryFn: () => api(`/harmonic/status?dataset=${encodeURIComponent(dataset)}`),
    refetchInterval: 5000,
    staleTime: 2000,
  })

  const lastTrainQ = useQuery<Record<string, unknown>>({
    queryKey: ['harmonic-train-result'],
    queryFn: () => api('/harmonic/train/result'),
    refetchInterval: 5000,
    staleTime: 5000,
  })

  const retrainStatusQ = useQuery<RetrainStatusResponse>({
    queryKey: ['harmonic-retrain-status', dataset, selectedScorerKind],
    queryFn: () => api(`/harmonic/retrain/status?dataset=${encodeURIComponent(dataset)}&scorer_kind=${encodeURIComponent(selectedScorerKind)}`),
    refetchInterval: 5000,
    staleTime: 5000,
  })

  const trainMut = useMutation<TrainResponse, Error, TrainRequestBody>({
    mutationFn: body => api('/harmonic/train', 'POST', body),
    onSuccess: () => {
      setFormError(null)
      qc.invalidateQueries({ queryKey: ['harmonic-status'] })
      qc.invalidateQueries({ queryKey: ['harmonic-train-result'] })
    },
  })

  const retrainMut = useMutation<RetrainResponse, Error, RetrainRequestBody>({
    mutationFn: body => api('/harmonic/retrain', 'POST', body),
    onSuccess: () => {
      setFormError(null)
      qc.invalidateQueries({ queryKey: ['harmonic-status'] })
      qc.invalidateQueries({ queryKey: ['harmonic-retrain-status'] })
    },
  })

  const seedMut = useMutation<HarmonicFeedbackSeedResponse, Error, HarmonicFeedbackSeedRequestBody>({
    mutationFn: body => api('/harmonic/dev/seed-feedback', 'POST', body),
    onSuccess: () => {
      setFormError(null)
      qc.invalidateQueries({ queryKey: ['harmonic-retrain-status'] })
    },
  })

  const status = statusQ.data
  const metrics = status?.training_metrics || {}
  const metricKeys = Object.keys(metrics)
  const checkpointEntries = Object.entries(status?.checkpoint_statuses || {})
  const lastTrainStatus = typeof lastTrainQ.data?.status === 'string' ? lastTrainQ.data.status : null
  const runtimeCheckpointActivated = typeof lastTrainQ.data?.runtime_checkpoint_activated === 'boolean'
    ? lastTrainQ.data.runtime_checkpoint_activated
    : null
  const retrainBucketKey = retrainStatusQ.data?.active_bucket || `${selectedScorerKind}:${dataset}`
  const retrainBucket = retrainStatusQ.data?.buckets?.[retrainBucketKey]

  function buildCheckpointOptions(): SharedCheckpointOptions | null {
    const payload: SharedCheckpointOptions = {}
    const seedText = randomSeed.trim()
    const suffixText = checkpointSuffix.trim()
    const modelPathText = modelSavePath.trim()

    if (seedText) {
      if (!/^-?\d+$/.test(seedText)) {
        setFormError('Random seed must be an integer.')
        return null
      }
      payload.random_seed = Number.parseInt(seedText, 10)
    }
    if (suffixText) payload.checkpoint_suffix = suffixText
    if (modelPathText) payload.model_save_path = modelPathText
    if (replaceCheckpoint) payload.replace_checkpoint = true

    setFormError(null)
    return payload
  }

  function buildTrainRequest(): TrainRequestBody | null {
    const options = buildCheckpointOptions()
    if (!options) return null
    return { dataset, ...options }
  }

  function buildRetrainRequest(): RetrainRequestBody | null {
    const options = buildCheckpointOptions()
    if (!options) return null
    return { dataset, scorer_kind: selectedScorerKind, ...options }
  }

  function handleTrain() {
    const payload = buildTrainRequest()
    if (!payload) return
    trainMut.mutate(payload)
  }

  function handleFeedbackRetrain() {
    const payload = buildRetrainRequest()
    if (!payload) return
    retrainMut.mutate(payload)
  }

  function handleSeedFeedbackBucket() {
    seedMut.mutate({
      dataset,
      scorer_kind: selectedScorerKind,
      confirmed: 12,
      dismissed: 8,
      clear_existing: true,
    })
  }

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {/* Header card */}
      <div className="card" style={{ padding: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Harmonic Model Trainer</div>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 2 }}>
              Offline checkpoint status and training controls for harmonic context and pair presets.
            </div>
          </div>
          <label className="small" style={{ color: 'var(--muted)' }}>
            Preset:
            <select
              value={dataset}
              onChange={e => setDataset(e.target.value as HarmonicDataset)}
              disabled={trainMut.isPending}
              style={{ marginLeft: 6, padding: '2px 6px', fontSize: 12 }}
            >
              {DATASET_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <span
              style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: status?.model_loaded ? 'var(--ok)' : status?.available ? '#f0a050' : 'var(--danger)',
              }}
            />
            {status?.model_loaded
              ? 'Model loaded'
              : status?.available
                ? 'Available (not loaded)'
                : status?.torch_installed === false
                  ? 'PyTorch not installed'
                  : 'Unavailable'}
          </div>
        </div>
      </div>

      {/* Config grid */}
      <div className="card" style={{ padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Model Configuration</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
          <MetricCell label="Dataset" value={status?.dataset_name || '—'} />
          <MetricCell label="Scorer kind" value={status?.scorer_kind || '—'} />
          <MetricCell label="Harmonic features" value={status?.n_harm_features ?? '—'} />
          <MetricCell label="Parameters" value={status?.n_params?.toLocaleString() ?? '—'} />
          <MetricCell label="Mode" value={status?.harmonic_mode || '—'} />
          <MetricCell label="CNN window" value={status?.cnn_window ?? '—'} />
          <MetricCell label="Trained at" value={status?.trained_at || '—'} />
        </div>
        {status?.model_save_path && (
          <div className="small" style={{ marginTop: 8, fontFamily: 'monospace', color: 'var(--muted)', fontSize: 10 }}>
            {status.model_save_path}
          </div>
        )}
      </div>

      {/* Preset availability */}
      <div className="card" style={{ padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Preset Checkpoints</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
          {checkpointEntries.map(([name, checkpoint]) => {
            const isSelected = name === dataset
            const statusColor = checkpoint.model_loaded
              ? 'var(--ok)'
              : checkpoint.available
                ? '#f0a050'
                : checkpoint.torch_installed === false
                  ? 'var(--danger)'
                  : 'var(--muted)'
            return (
              <div
                key={name}
                style={{
                  padding: 8,
                  borderRadius: 6,
                  border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border)',
                  background: 'var(--bg-alt, #1a1a1a)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600 }}>{name}</div>
                  <span style={{ color: statusColor, fontSize: 11, fontWeight: 600 }}>
                    {checkpoint.model_loaded ? 'loaded' : checkpoint.available ? 'available' : 'missing'}
                  </span>
                </div>
                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                  {checkpoint.scorer_kind || '—'} / {checkpoint.harmonic_mode || '—'} / window {checkpoint.cnn_window ?? '—'}
                </div>
                <div className="small" style={{ color: 'var(--muted)', marginTop: 4 }}>
                  {checkpoint.trained_at || 'untrained'}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Training metrics */}
      <div className="card" style={{ padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Training Metrics</div>
        {metricKeys.length === 0 ? (
          <div className="small" style={{ color: 'var(--muted)' }}>No training metrics recorded — train the model to populate.</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
            {metricKeys.map(k => (
              <MetricCell key={k} label={k} value={fmtMetric(metrics[k])} />
            ))}
          </div>
        )}
      </div>

      {/* Training controls */}
      <div className="card" style={{ padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Train / Retrain</div>
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
            <label className="small" style={{ color: 'var(--muted)', display: 'grid', gap: 4 }}>
              <span>Dataset</span>
              <select
                value={dataset}
                onChange={e => setDataset(e.target.value as HarmonicDataset)}
                disabled={trainMut.isPending}
                style={{ padding: '6px 8px', fontSize: 12 }}
              >
                {DATASET_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label className="small" style={{ color: 'var(--muted)', display: 'grid', gap: 4 }}>
              <span>Random seed</span>
              <input
                type="number"
                inputMode="numeric"
                value={randomSeed}
                onChange={e => setRandomSeed(e.target.value)}
                disabled={trainMut.isPending}
                placeholder="optional"
                style={{ padding: '6px 8px', fontSize: 12 }}
              />
            </label>
            <label className="small" style={{ color: 'var(--muted)', display: 'grid', gap: 4 }}>
              <span>Checkpoint suffix</span>
              <input
                type="text"
                value={checkpointSuffix}
                onChange={e => setCheckpointSuffix(e.target.value)}
                disabled={trainMut.isPending}
                placeholder="seed17"
                style={{ padding: '6px 8px', fontSize: 12 }}
              />
            </label>
            <label className="small" style={{ color: 'var(--muted)', display: 'grid', gap: 4 }}>
              <span>Checkpoint output path</span>
              <input
                type="text"
                value={modelSavePath}
                onChange={e => setModelSavePath(e.target.value)}
                disabled={trainMut.isPending}
                placeholder="optional explicit path"
                style={{ padding: '6px 8px', fontSize: 12 }}
              />
            </label>
          </div>
          <label className="small" style={{ color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <input
              type="checkbox"
              checked={replaceCheckpoint}
              onChange={e => setReplaceCheckpoint(e.target.checked)}
              disabled={trainMut.isPending || retrainMut.isPending}
            />
            Replace canonical checkpoint
          </label>
          <div className="small" style={{ color: 'var(--muted)' }}>
            These checkpoint fields apply to both preset training and feedback retraining. Leave the output path empty to use the preset checkpoint location. A suffix or explicit seed writes to an experiment checkpoint by default; an explicit output path overrides the suffix.
          </div>
          <div style={{ display: 'grid', gap: 8, paddingTop: 4 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>Preset Training</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <button
                className="primary"
                onClick={handleTrain}
                disabled={trainMut.isPending || retrainMut.isPending || status?.torch_installed === false}
                style={{ padding: '4px 16px' }}
              >
                {trainMut.isPending ? 'Training…' : 'Train'}
              </button>
              {trainMut.isError && (
                <span className="small" style={{ color: 'var(--danger)' }}>
                  {trainMut.error?.message || 'Training failed'}
                </span>
              )}
              {trainMut.isSuccess && trainMut.data && (
                <span className="small" style={{ color: 'var(--ok)' }}>
                  {trainMut.data.status}: {trainMut.data.message}
                </span>
              )}
            </div>
          </div>
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, display: 'grid', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ fontSize: 12, fontWeight: 600 }}>Feedback Retrain</div>
              <div className="small" style={{ color: 'var(--muted)' }}>
                Bucket {retrainBucketKey}
              </div>
            </div>
            <div
              style={{
                display: 'grid',
                gap: 8,
                padding: 10,
                borderRadius: 6,
                border: '1px solid var(--border)',
                background: 'var(--bg-alt, #1a1a1a)',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600 }}>Demo warmup</div>
              <div className="small" style={{ color: 'var(--muted)' }}>
                Seed this feedback bucket with synthetic confirmed and dismissed samples so retrain demos are ready immediately. This uses the backend dev-only route and requires HARMONIC_ENABLE_DEV_SEED=1.
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <button
                  onClick={handleSeedFeedbackBucket}
                  disabled={trainMut.isPending || retrainMut.isPending || seedMut.isPending}
                  style={{ padding: '4px 16px' }}
                >
                  {seedMut.isPending ? 'Seeding…' : 'Seed demo feedback bucket'}
                </button>
                {seedMut.isError && (
                  <span className="small" style={{ color: 'var(--danger)' }}>
                    {seedMut.error?.message || 'Feedback seed failed'}
                  </span>
                )}
                {seedMut.isSuccess && seedMut.data && (
                  <span className="small" style={{ color: 'var(--ok)' }}>
                    Seeded {seedMut.data.bucket_key} with {seedMut.data.added_confirmed} confirmed / {seedMut.data.added_dismissed} dismissed samples.
                  </span>
                )}
              </div>
            </div>
            {retrainBucket ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
                <MetricCell label="Scorer kind" value={retrainBucket.scorer_kind || selectedScorerKind} />
                <MetricCell label="Buffered samples" value={retrainBucket.buffer_size} />
                <MetricCell label="Confirmed" value={retrainBucket.confirmed_in_buffer} />
                <MetricCell label="Dismissed" value={retrainBucket.dismissed_in_buffer} />
                <MetricCell label="Since retrain" value={retrainBucket.since_last_retrain} />
                <MetricCell label="Threshold" value={retrainBucket.retrain_threshold} />
              </div>
            ) : (
              <div className="small" style={{ color: 'var(--muted)' }}>
                No buffered feedback bucket exists yet for this preset. The retrain endpoint will fail until operator feedback has been collected for {selectedScorerKind}:{dataset}.
              </div>
            )}
            {retrainBucket?.model_save_path && (
              <div className="small" style={{ color: 'var(--muted)', fontFamily: 'monospace', fontSize: 10 }}>
                Canonical checkpoint: {retrainBucket.model_save_path}
              </div>
            )}
            {retrainBucket?.last_retrain && (
              <div className="small" style={{ color: 'var(--muted)' }}>
                Last retrain: {retrainBucket.last_retrain}
              </div>
            )}
            {retrainBucket && (
              <div className="small" style={{ color: retrainBucket.should_retrain ? 'var(--ok)' : '#f0a050' }}>
                {retrainBucket.should_retrain
                  ? 'Feedback bucket meets the automatic retrain threshold.'
                  : 'Feedback bucket is below the automatic retrain threshold or class-balance minimums. Manual retrain is still available and the backend will validate sample counts.'}
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <button
                className="primary"
                onClick={handleFeedbackRetrain}
                disabled={trainMut.isPending || retrainMut.isPending || status?.torch_installed === false}
                style={{ padding: '4px 16px' }}
              >
                {retrainMut.isPending ? 'Retraining…' : 'Retrain from Feedback'}
              </button>
              {formError && (
                <span className="small" style={{ color: 'var(--danger)' }}>
                  {formError}
                </span>
              )}
              {retrainMut.isError && (
                <span className="small" style={{ color: 'var(--danger)' }}>
                  {retrainMut.error?.message || 'Feedback retrain failed'}
                </span>
              )}
              {retrainMut.isSuccess && retrainMut.data && (
                <span className="small" style={{ color: retrainMut.data.success ? 'var(--ok)' : '#f0a050' }}>
                  {retrainMut.data.message}
                </span>
              )}
            </div>
            <div className="small" style={{ color: 'var(--muted)' }}>
              Feedback retrains use the selected preset and scorer kind. Experiment-style checkpoint targets do not refresh runtime scoring unless you intentionally replace the canonical checkpoint.
            </div>
          </div>
          {lastTrainStatus === 'completed' && runtimeCheckpointActivated === false && (
            <div className="small" style={{ color: '#f0a050' }}>
              The last completed run wrote to an experiment checkpoint. Runtime scoring stayed on the canonical preset checkpoint.
            </div>
          )}
          {lastTrainStatus === 'completed' && runtimeCheckpointActivated === true && (
            <div className="small" style={{ color: 'var(--ok)' }}>
              The last completed run refreshed the runtime checkpoint for this preset.
            </div>
          )}
        </div>
        {status?.torch_installed === false && (
          <div className="small" style={{ color: 'var(--danger)', marginTop: 6 }}>
            PyTorch is not installed in this environment — install torch to enable training.
          </div>
        )}
      </div>

      {retrainMut.data && (
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Last Feedback Retrain</div>
          <pre style={{ fontSize: 11, fontFamily: 'monospace', margin: 0, overflow: 'auto', maxHeight: 220, color: 'var(--muted)' }}>
            {JSON.stringify(retrainMut.data, null, 2)}
          </pre>
        </div>
      )}

      {/* Last training result */}
      {lastTrainQ.data && lastTrainQ.data.status !== 'no_training_run' && (
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Last Training Run</div>
          <pre style={{ fontSize: 11, fontFamily: 'monospace', margin: 0, overflow: 'auto', maxHeight: 200, color: 'var(--muted)' }}>
            {JSON.stringify(lastTrainQ.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

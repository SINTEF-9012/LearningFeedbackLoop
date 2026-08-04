/**
 * SessionControls — API URL configuration + session selector + demo launcher.
 *
 * Extracted from App.tsx's `<details>⚙️ Configuration</details>` section.
 */
import React, { useEffect, useMemo, useState } from 'react'
import { api, baseUrl, setBaseUrl } from '../api/http'
import { useStreamStore } from '../state/streamStore'
import { useQuery, type UseQueryResult } from '@tanstack/react-query'
import type { SessionSummary } from '../contexts/AppContext'

const DEMO_MODE_KEY = 'demoMode'
const DEMO_SOURCE_KEY = 'demoSource'
const MQTT_BROKER_HOST_KEY = 'mqttBrokerHost'
const MQTT_BROKER_PORT_KEY = 'mqttBrokerPort'
const MQTT_TOPIC_KEY = 'mqttTopic'
const MQTT_SAMPLE_FREQUENCY_KEY = 'mqttSampleFrequency'
const MQTT_USERNAME_KEY = 'mqttUsername'
const DEMO_SESSION_FILE_KEY = 'demoSessionFile'
const CASEDATA_CASE_DIR_KEY = 'casedataCaseDir'
const CASEDATA_OPERATION_ID_KEY = 'casedataOperationId'
const CASEDATA_VALID_TOOLS_ONLY_KEY = 'casedataValidToolsOnly'
const CASEDATA_START_AT_FIRST_CUTTING_ROW_KEY = 'casedataStartAtFirstCuttingRow'
const DEMO_START_POSITION_KEY = 'demoStartPosition'

function getStoredValue(key: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const raw = window.localStorage.getItem(key)
  return raw == null ? fallback : raw
}

type DemoPreflightTone = 'ready' | 'warning' | 'optional' | 'error' | 'loading'

interface DemoPreflightCheck {
  label: string
  tone: DemoPreflightTone
  detail: string
}

interface DemoPreflightSnapshot {
  api: DemoPreflightCheck
  harmonic: DemoPreflightCheck
  llm: DemoPreflightCheck
  sindit: DemoPreflightCheck
  checkedAt: string
}

interface Props {
  sessionsQuery: UseQueryResult<{ sessions: string[] }>
  priorsQuery: UseQueryResult<unknown>
  sessionOptions: string[]
  sessionSummaries: SessionSummary[]
}

interface DemoResponse {
  session_id: string
  ws_url: string
  mode: string
  n_events: number
  source?: string
  status: string
}

interface CasedataCatalogOperation {
  operation_id: string
  tool_id: string
  tool_label?: string
  tool_number?: number | null
  n_channels: number
  harmonic_ready?: boolean
  missing_fields?: string[]
}

interface CasedataCatalogCase {
  case_dir: string
  label: string
  default_operation_id?: string | null
  default_valid_operation_id?: string | null
  operations: CasedataCatalogOperation[]
}

interface CasedataCatalogResponse {
  root: string
  cases: CasedataCatalogCase[]
}

interface HealthResponse {
  status?: string
}

interface LlmStatusResponse {
  available?: boolean
  provider?: string | null
  model?: string | null
}

interface HarmonicStatusResponse {
  available: boolean
  torch_installed: boolean
  model_loaded: boolean
  dataset_name: string
  model_path_exists?: boolean
  model_save_path?: string
}

interface SinditHealthResponse {
  sindit: boolean
  graphdb: boolean
  sindit_url?: string
  sindit_enabled: boolean
}

function pendingPreflightCheck(label: string, detail = 'Checking…'): DemoPreflightCheck {
  return { label, tone: 'loading', detail }
}

function preflightToneColor(tone: DemoPreflightTone): string {
  switch (tone) {
    case 'ready':
      return 'var(--ok)'
    case 'warning':
      return '#f0a050'
    case 'optional':
      return 'var(--accent)'
    case 'error':
      return 'var(--danger)'
    default:
      return 'var(--muted)'
  }
}

function preflightToneLabel(tone: DemoPreflightTone): string {
  switch (tone) {
    case 'ready':
      return 'Ready'
    case 'warning':
      return 'Caution'
    case 'optional':
      return 'Optional'
    case 'error':
      return 'Unavailable'
    default:
      return 'Checking'
  }
}

function errorDetail(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function pickRecommendedCasedataCase(cases: CasedataCatalogCase[]): CasedataCatalogCase | null {
  return cases.find((item) => /site_c|MACHINE_C1|CASE_C1/i.test(`${item.case_dir} ${item.label}`)) || cases[0] || null
}

function formatSessionSummaryLabel(session: SessionSummary): string {
  const sourceLabel = session.source_label || [session.case_dir, session.operation_id].filter(Boolean).join(' / ') || session.source || ''
  const progress =
    typeof session.progress === 'number' && Number.isFinite(session.progress) && session.total_samples > 0
      ? session.progress <= 0
        ? '0%'
        : session.progress < 0.001
          ? '<0.1%'
          : session.progress < 0.01
            ? `${(session.progress * 100).toFixed(1)}%`
            : `${Math.round(session.progress * 100)}%`
      : ''
  return [sourceLabel, session.session_id, session.status_label, progress].filter(Boolean).join(' · ')
}

export function SessionControls({ sessionsQuery, priorsQuery, sessionOptions, sessionSummaries }: Props) {
  const streamSessionId = useStreamStore((s) => s.sessionId)
  const setStreamSessionId = useStreamStore((s) => s.setSessionId)
  const [apiUrlDraft, setApiUrlDraft] = useState(baseUrl())
  const [apiUrlApplied, setApiUrlApplied] = useState(baseUrl())
  const [demoLoading, setDemoLoading] = useState(false)
  const [demoError, setDemoError] = useState<string | null>(null)
  const [demoMode, setDemoMode] = useState<string>(() => getStoredValue(DEMO_MODE_KEY, 'labeled'))
  const [demoSource, setDemoSource] = useState<string>(() => getStoredValue(DEMO_SOURCE_KEY, 'simulated_casedata'))
  const [mqttBrokerHost, setMqttBrokerHost] = useState<string>(() => getStoredValue(MQTT_BROKER_HOST_KEY, 'localhost'))
  const [mqttBrokerPort, setMqttBrokerPort] = useState<string>(() => getStoredValue(MQTT_BROKER_PORT_KEY, '1883'))
  const [mqttTopic, setMqttTopic] = useState<string>(() => getStoredValue(MQTT_TOPIC_KEY, ''))
  const [mqttSampleFrequency, setMqttSampleFrequency] = useState<string>(() => getStoredValue(MQTT_SAMPLE_FREQUENCY_KEY, '1'))
  const [mqttUsername, setMqttUsername] = useState<string>(() => getStoredValue(MQTT_USERNAME_KEY, ''))
  const [mqttPassword, setMqttPassword] = useState<string>('')
  const [demoSessionFile, setDemoSessionFile] = useState<string>(() => getStoredValue(DEMO_SESSION_FILE_KEY, ''))
  const [casedataCaseDir, setCasedataCaseDir] = useState<string>(() => getStoredValue(CASEDATA_CASE_DIR_KEY, ''))
  const [casedataOperationId, setCasedataOperationId] = useState<string>(() => getStoredValue(CASEDATA_OPERATION_ID_KEY, ''))
  const [casedataValidToolsOnly, setCasedataValidToolsOnly] = useState<boolean>(() => getStoredValue(CASEDATA_VALID_TOOLS_ONLY_KEY, 'true') === 'true')
  const [casedataStartAtFirstCuttingRow, setCasedataStartAtFirstCuttingRow] = useState<boolean>(() => getStoredValue(CASEDATA_START_AT_FIRST_CUTTING_ROW_KEY, 'true') === 'true')
  const [demoStartPosition, setDemoStartPosition] = useState<string>(() => getStoredValue(DEMO_START_POSITION_KEY, '0'))
  const [casedataCatalog, setCasedataCatalog] = useState<CasedataCatalogResponse | null>(null)
  const [casedataCatalogLoading, setCasedataCatalogLoading] = useState(false)
  const [casedataCatalogError, setCasedataCatalogError] = useState<string | null>(null)
  const [catalogRefreshNonce, setCatalogRefreshNonce] = useState(0)

  const isFileSource = demoSource === 'simulated_file'
  const isMqttSource = demoSource === 'mqtt'
  const isCasedataSource = demoSource === 'simulated_casedata'
  const recommendedCasedataCase = useMemo(
    () => pickRecommendedCasedataCase(casedataCatalog?.cases || []),
    [casedataCatalog],
  )
  const selectedCasedataCase = casedataCatalog?.cases.find((item) => item.case_dir === casedataCaseDir) ?? recommendedCasedataCase ?? null
  const visibleCasedataOperations = useMemo(() => {
    const operations = selectedCasedataCase?.operations || []
    if (!casedataValidToolsOnly) return operations
    return operations.filter((item) => item.harmonic_ready)
  }, [selectedCasedataCase, casedataValidToolsOnly])
  const harmonicReadyCount = useMemo(
    () => (selectedCasedataCase?.operations || []).filter((item) => item.harmonic_ready).length,
    [selectedCasedataCase],
  )
  const noValidCasedataOperations = Boolean(
    isCasedataSource
    && selectedCasedataCase
    && casedataValidToolsOnly
    && selectedCasedataCase.operations.length > 0
    && visibleCasedataOperations.length === 0,
  )
  const casedataStreamSummaries = useMemo(
    () => sessionSummaries.filter((session) => session.source === 'simulated_casedata'),
    [sessionSummaries],
  )
  const selectedSessionKnown = sessionSummaries.some((session) => session.session_id === streamSessionId)
  const selectedCasedataStreamId = casedataStreamSummaries.some((session) => session.session_id === streamSessionId)
    ? streamSessionId
    : ''
  const activeCasedataStreamSummary = useMemo(
    () => casedataStreamSummaries.find((session) => session.session_id === selectedCasedataStreamId) || null,
    [casedataStreamSummaries, selectedCasedataStreamId],
  )
  const casedataResolvedStartHint = useMemo(() => {
    const session = activeCasedataStreamSummary
    if (!session || !session.start_at_first_cutting_row) return null

    const targetLabel = session.source_label || session.operation_id || 'Current casedata stream'
    if (session.loading) {
      return `${targetLabel}: resolving first cutting row…`
    }

    const resolvedStart = session.resolved_start_position
    if (typeof resolvedStart !== 'number' || !Number.isFinite(resolvedStart)) return null

    const extraSkip = Math.max(0, Number(session.requested_start_position) || 0)
    if (extraSkip > 0) {
      return `${targetLabel}: starts at sample ${resolvedStart} (+${extraSkip} extra skip)`
    }
    return `${targetLabel}: starts at sample ${resolvedStart}`
  }, [activeCasedataStreamSummary])

  const preflightQ = useQuery<DemoPreflightSnapshot>({
    queryKey: ['demo-preflight', apiUrlApplied],
    queryFn: async (): Promise<DemoPreflightSnapshot> => {
      const [healthResult, harmonicResult, llmResult, sinditResult] = await Promise.allSettled([
        api<HealthResponse>('/health'),
        api<HarmonicStatusResponse>('/harmonic/status?dataset=pair_lfl'),
        api<LlmStatusResponse>('/agent/memory/llm/status'),
        api<SinditHealthResponse>('/health/sindit'),
      ])

      const apiCheck: DemoPreflightCheck = healthResult.status === 'fulfilled'
        ? {
            label: 'Backend API',
            tone: healthResult.value.status === 'ok' ? 'ready' : 'warning',
            detail: healthResult.value.status === 'ok' ? 'Backend responded to /health.' : `Unexpected health status: ${healthResult.value.status || 'unknown'}`,
          }
        : {
            label: 'Backend API',
            tone: 'error',
            detail: errorDetail(healthResult.reason),
          }

      const harmonicCheck: DemoPreflightCheck = harmonicResult.status === 'fulfilled'
        ? harmonicResult.value.model_loaded
          ? {
              label: 'Harmonic pair_lfl',
              tone: 'ready',
              detail: harmonicResult.value.model_save_path
                ? `Loaded checkpoint ${harmonicResult.value.model_save_path}.`
                : 'pair_lfl checkpoint is loaded.',
            }
          : harmonicResult.value.available
            ? {
                label: 'Harmonic pair_lfl',
                tone: 'warning',
                detail: harmonicResult.value.model_path_exists
                  ? 'Checkpoint exists but is not loaded yet.'
                  : 'Runtime is available, but no pair_lfl checkpoint is loaded.',
              }
            : {
                label: 'Harmonic pair_lfl',
                tone: harmonicResult.value.torch_installed === false ? 'error' : 'warning',
                detail: harmonicResult.value.torch_installed === false
                  ? 'PyTorch is not installed in this backend environment.'
                  : 'pair_lfl status is unavailable.',
              }
        : {
            label: 'Harmonic pair_lfl',
            tone: 'error',
            detail: errorDetail(harmonicResult.reason),
          }

      const llmCheck: DemoPreflightCheck = llmResult.status === 'fulfilled'
        ? llmResult.value.available
          ? {
              label: 'LLM explanations',
              tone: 'ready',
              detail: [llmResult.value.provider, llmResult.value.model].filter(Boolean).join(' / ') || 'Explanation service is available.',
            }
          : {
              label: 'LLM explanations',
              tone: 'optional',
              detail: 'Memory explanations are unavailable; alerting still works without them.',
            }
        : {
            label: 'LLM explanations',
            tone: 'optional',
            detail: errorDetail(llmResult.reason),
          }

      const sinditCheck: DemoPreflightCheck = sinditResult.status === 'fulfilled'
        ? !sinditResult.value.sindit_enabled
          ? {
              label: 'SINDIT / GraphDB',
              tone: 'optional',
              detail: 'External digital-twin services are disabled for this backend.',
            }
          : sinditResult.value.sindit && sinditResult.value.graphdb
            ? {
                label: 'SINDIT / GraphDB',
                tone: 'ready',
                detail: 'SINDIT and GraphDB are reachable.',
              }
            : {
                label: 'SINDIT / GraphDB',
                tone: 'warning',
                detail: `SINDIT=${sinditResult.value.sindit ? 'up' : 'down'}, GraphDB=${sinditResult.value.graphdb ? 'up' : 'down'}`,
              }
        : {
            label: 'SINDIT / GraphDB',
            tone: 'optional',
            detail: errorDetail(sinditResult.reason),
          }

      return {
        api: apiCheck,
        harmonic: harmonicCheck,
        llm: llmCheck,
        sindit: sinditCheck,
        checkedAt: new Date().toISOString(),
      }
    },
    staleTime: 5_000,
    // Re-check periodically so the readiness banner self-heals: services like
    // SINDIT/GraphDB can take ~30 s to become reachable after a backend restart,
    // and without this the panel would keep showing the (stale) startup state.
    refetchInterval: 10_000,
    refetchOnWindowFocus: true,
  })

  const casedataPreflightCheck = useMemo<DemoPreflightCheck>(() => {
    if (casedataCatalogLoading && !casedataCatalog) {
      return pendingPreflightCheck('Casedata catalog')
    }
    if (casedataCatalogError) {
      return {
        label: 'Casedata catalog',
        tone: 'error',
        detail: casedataCatalogError,
      }
    }
    if (!casedataCatalog?.cases.length) {
      return {
        label: 'Casedata catalog',
        tone: 'error',
        detail: `No casedata cases were found under ${casedataCatalog?.root || 'the configured casedata root'}.`,
      }
    }
    const recommendedLabel = recommendedCasedataCase?.label || recommendedCasedataCase?.case_dir || 'first available case'
    return {
      label: 'Casedata catalog',
      tone: 'ready',
      detail: `${casedataCatalog.cases.length} case${casedataCatalog.cases.length === 1 ? '' : 's'} available. Defaulting to ${recommendedLabel}.`,
    }
  }, [casedataCatalog, casedataCatalogError, casedataCatalogLoading, recommendedCasedataCase])

  const preflightChecks = useMemo(
    () => [
      preflightQ.data?.api ?? pendingPreflightCheck('Backend API'),
      casedataPreflightCheck,
      preflightQ.data?.harmonic ?? pendingPreflightCheck('Harmonic pair_lfl'),
      preflightQ.data?.llm ?? pendingPreflightCheck('LLM explanations'),
      preflightQ.data?.sindit ?? pendingPreflightCheck('SINDIT / GraphDB'),
    ],
    [casedataPreflightCheck, preflightQ.data],
  )
  const preflightBlockingCount = preflightChecks.filter((check) => check.tone === 'error').length
  const preflightCautionCount = preflightChecks.filter((check) => check.tone === 'warning').length
  const preflightSummary = preflightBlockingCount > 0
    ? `${preflightBlockingCount} blocking issue${preflightBlockingCount === 1 ? '' : 's'} need attention before the recommended demo flow.`
    : preflightCautionCount > 0
      ? `${preflightCautionCount} caution${preflightCautionCount === 1 ? '' : 's'} detected; the core demo can still run.`
      : 'Ready for the recommended operator demo.'

  useEffect(() => {
    let cancelled = false

    const loadCasedataCatalog = async () => {
      setCasedataCatalogLoading(true)
      try {
        const nextCatalog = await api<CasedataCatalogResponse>('/sessions/casedata/catalog')
        if (cancelled) return
        setCasedataCatalog(nextCatalog)
        setCasedataCatalogError(null)

        const storedCaseStillExists = nextCatalog.cases.some((item) => item.case_dir === casedataCaseDir)
        const nextCaseDir = storedCaseStillExists
          ? casedataCaseDir
          : (pickRecommendedCasedataCase(nextCatalog.cases)?.case_dir ?? '')
        if (nextCaseDir !== casedataCaseDir) {
          setCasedataCaseDir(nextCaseDir)
          if (typeof window !== 'undefined') {
            window.localStorage.setItem(CASEDATA_CASE_DIR_KEY, nextCaseDir)
          }
        }
      } catch (e: any) {
        if (cancelled) return
        setCasedataCatalog(null)
        setCasedataCatalogError(e?.message || String(e))
      } finally {
        if (!cancelled) setCasedataCatalogLoading(false)
      }
    }

    void loadCasedataCatalog()
    return () => {
      cancelled = true
    }
  }, [apiUrlApplied, casedataCaseDir, catalogRefreshNonce])

  useEffect(() => {
    if (!selectedCasedataCase) return
    const operationExists = visibleCasedataOperations.some((item) => item.operation_id === casedataOperationId)
    if (operationExists) return
    const nextOperationId = (
      casedataValidToolsOnly
        ? selectedCasedataCase.default_valid_operation_id
        : selectedCasedataCase.default_operation_id
    ) || visibleCasedataOperations[0]?.operation_id || ''
    setCasedataOperationId(nextOperationId)
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(CASEDATA_OPERATION_ID_KEY, nextOperationId)
    }
  }, [selectedCasedataCase, visibleCasedataOperations, casedataOperationId, casedataValidToolsOnly])

  const applyApiBaseUrl = () => {
    const v = (apiUrlDraft || '').trim()
    if (!v) return
    setBaseUrl(v)
    setApiUrlApplied(baseUrl())
  }

  const refreshPreflight = () => {
    setCatalogRefreshNonce((current) => current + 1)
    void preflightQ.refetch()
  }

  const startDemo = async () => {
    setDemoLoading(true)
    setDemoError(null)
    try {
      const requestedMode = isFileSource ? demoMode : isCasedataSource ? 'casedata' : 'live'
      const body: Record<string, unknown> = {
        mode: requestedMode,
        source: demoSource,
        reset_priors: demoSource === 'simulated_file',
      }
      const startPosition = Math.max(0, Math.floor(Number(demoStartPosition) || 0))
      if (startPosition > 0) body.start_position = startPosition

      if (isFileSource) {
        body.speed = demoMode === 'casedata' ? 1 : demoMode === 'site_a_line2' ? 8 : 0.02
        body.sleep = demoMode === 'casedata' ? 5 : demoMode === 'site_a_line2' ? 8 : 0.8
        if (demoSessionFile.trim()) body.session_file = demoSessionFile.trim()
      } else if (isCasedataSource) {
        body.speed = 1
        body.samples_per_tick = 1
        if (casedataValidToolsOnly) body.valid_tools_only = true
        if (casedataStartAtFirstCuttingRow) body.start_at_first_cutting_row = true
        if (selectedCasedataCase?.case_dir) body.case_dir = selectedCasedataCase.case_dir
        if (casedataOperationId.trim()) body.operation_id = casedataOperationId.trim()
      } else {
        body.speed = 1
      }

      if (isMqttSource) {
        const topic = mqttTopic.trim()
        if (!topic) {
          throw new Error('MQTT topic is required')
        }
        body.broker_host = mqttBrokerHost.trim() || 'localhost'
        body.broker_port = Math.max(1, Number(mqttBrokerPort) || 1883)
        body.topic = topic
        body.sample_frequency = Math.max(0.001, Number(mqttSampleFrequency) || 1)
        if (mqttUsername.trim()) body.username = mqttUsername.trim()
        if (mqttPassword) body.password = mqttPassword
      }

      const res = await api<DemoResponse>('/sessions/start-demo', 'POST', body)
      setStreamSessionId(res.session_id)
      void sessionsQuery.refetch()
    } catch (e: any) {
      setDemoError(e?.message || String(e))
    } finally {
      setDemoLoading(false)
    }
  }

  return (
    <details style={{ margin: '4px 0' }}>
      <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: 13, padding: '4px 0' }}>
        ⚙️ Configuration
      </summary>

      <div className="row">
        <div>
          <div className="small">API base URL</div>
          <input
            value={apiUrlDraft}
            onChange={(e) => setApiUrlDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') applyApiBaseUrl()
            }}
          />
          <div className="hrow" style={{ marginTop: 8 }}>
            <button className="primary" onClick={applyApiBaseUrl}>
              Apply
            </button>
            <div className="small">active: {apiUrlApplied}</div>
          </div>
          <div className="small" style={{ marginTop: 6 }}>
            example: http://localhost:8000
          </div>
        </div>
        <div>
          <div className="small">Session</div>
          <div className="panel" style={{ marginBottom: 10, borderColor: 'rgba(122, 162, 247, 0.25)' }}>
            <div className="hrow" style={{ justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontWeight: 700 }}>Demo preflight</div>
                <div className="small" style={{ color: 'var(--muted)', marginTop: 2 }}>
                  Checks the recommended SITE_C operator path against the current backend.
                </div>
              </div>
              <div className="hrow" style={{ gap: 8 }}>
                {preflightQ.data?.checkedAt && (
                  <div className="small" style={{ color: 'var(--muted)' }}>
                    Last checked {new Date(preflightQ.data.checkedAt).toLocaleTimeString()}
                  </div>
                )}
                <button onClick={refreshPreflight} disabled={preflightQ.isFetching || casedataCatalogLoading}>
                  {preflightQ.isFetching || casedataCatalogLoading ? 'Rechecking…' : 'Recheck'}
                </button>
              </div>
            </div>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 8 }}>
              {preflightSummary}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8, marginTop: 10 }}>
              {preflightChecks.map((check) => (
                <div
                  key={check.label}
                  style={{
                    padding: 8,
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                    background: 'var(--bg-alt, rgba(255, 255, 255, 0.02))',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 600 }}>{check.label}</div>
                    <span style={{ color: preflightToneColor(check.tone), fontSize: 11, fontWeight: 700 }}>
                      {preflightToneLabel(check.tone)}
                    </span>
                  </div>
                  <div className="small" style={{ color: 'var(--muted)', marginTop: 6 }}>
                    {check.detail}
                  </div>
                </div>
              ))}
            </div>
            <div className="small" style={{ color: 'var(--muted)', marginTop: 10 }}>
              Recommended defaults are already selected: casedata stream, harmonic-ready tools only, and first cutting row.
            </div>
          </div>
          <select aria-label="Active session" value={streamSessionId} onChange={(e) => setStreamSessionId(e.target.value)}>
            <option value="">(select)</option>
            {streamSessionId && !selectedSessionKnown && (
              <option value={streamSessionId}>
                {['Starting demo', streamSessionId].join(' · ')}
              </option>
            )}
            {sessionSummaries.map((session) => (
              <option key={session.session_id} value={session.session_id}>
                {formatSessionSummaryLabel(session)}
              </option>
            ))}
          </select>
          {casedataStreamSummaries.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div className="small">Casedata streams</div>
              <select
                value={selectedCasedataStreamId}
                onChange={(e) => setStreamSessionId(e.target.value)}
              >
                <option value="">(select casedata stream)</option>
                {casedataStreamSummaries.map((session) => (
                  <option key={`casedata-${session.session_id}`} value={session.session_id}>
                    {formatSessionSummaryLabel(session)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="hrow" style={{ marginTop: 8, gap: 6 }}>
            <select
              aria-label="Demo source"
              value={demoSource}
              onChange={(e) => {
                const value = e.target.value
                setDemoSource(value)
                window.localStorage.setItem(DEMO_SOURCE_KEY, value)
              }}
              style={{ fontSize: 12, padding: '2px 4px', width: 140 }}
            >
              <option value="simulated_file">Demo file</option>
              <option value="simulated_casedata">Casedata stream</option>
              <option value="mqtt">MQTT live</option>
            </select>
            <select
              aria-label="Demo mode"
              value={demoMode}
              onChange={(e) => {
                const value = e.target.value
                setDemoMode(value)
                window.localStorage.setItem(DEMO_MODE_KEY, value)
              }}
              disabled={demoSource !== 'simulated_file'}
              style={{ fontSize: 12, padding: '2px 4px', width: 90 }}
            >
              <option value="labeled">Labeled</option>
              <option value="default">Default</option>
              <option value="casedata">Casedata</option>
              <option value="site_a_line2">Site_a_line2</option>
            </select>
            <button
              className="primary"
              aria-label={isMqttSource ? 'Start MQTT' : 'Start Demo'}
              disabled={demoLoading || noValidCasedataOperations}
              onClick={startDemo}
              title={isMqttSource ? 'Create a live MQTT-backed session' : 'Create a demo session with auto-injected events and feedback'}
            >
              {demoLoading ? '⏳ Starting…' : isMqttSource ? '▶ Start MQTT' : '▶ Start Demo'}
            </button>
          </div>
          {isMqttSource && (
            <div className="row" style={{ marginTop: 10 }}>
              <div>
                <div className="small">Broker host</div>
                <input
                  value={mqttBrokerHost}
                  onChange={(e) => {
                    const value = e.target.value
                    setMqttBrokerHost(value)
                    window.localStorage.setItem(MQTT_BROKER_HOST_KEY, value)
                  }}
                  placeholder="localhost"
                />
              </div>
              <div>
                <div className="small">Broker port</div>
                <input
                  value={mqttBrokerPort}
                  onChange={(e) => {
                    const value = e.target.value
                    setMqttBrokerPort(value)
                    window.localStorage.setItem(MQTT_BROKER_PORT_KEY, value)
                  }}
                  placeholder="1883"
                />
              </div>
              <div>
                <div className="small">Topic</div>
                <input
                  value={mqttTopic}
                  onChange={(e) => {
                    const value = e.target.value
                    setMqttTopic(value)
                    window.localStorage.setItem(MQTT_TOPIC_KEY, value)
                  }}
                  placeholder="machine/live"
                />
              </div>
              <div>
                <div className="small">Sample frequency (Hz)</div>
                <input
                  value={mqttSampleFrequency}
                  onChange={(e) => {
                    const value = e.target.value
                    setMqttSampleFrequency(value)
                    window.localStorage.setItem(MQTT_SAMPLE_FREQUENCY_KEY, value)
                  }}
                  placeholder="1"
                />
              </div>
              <div>
                <div className="small">Username (optional)</div>
                <input
                  value={mqttUsername}
                  onChange={(e) => {
                    const value = e.target.value
                    setMqttUsername(value)
                    window.localStorage.setItem(MQTT_USERNAME_KEY, value)
                  }}
                  placeholder="mqtt-user"
                />
              </div>
              <div>
                <div className="small">Password (optional)</div>
                <input
                  type="password"
                  value={mqttPassword}
                  onChange={(e) => setMqttPassword(e.target.value)}
                  placeholder="mqtt-password"
                />
              </div>
            </div>
          )}
          {isFileSource && (
            <div className="row" style={{ marginTop: 10 }}>
              <div>
                <div className="small">Demo file (optional)</div>
                <input
                  aria-label="Demo file"
                  value={demoSessionFile}
                  onChange={(e) => {
                    const value = e.target.value
                    setDemoSessionFile(value)
                    window.localStorage.setItem(DEMO_SESSION_FILE_KEY, value)
                  }}
                  placeholder={demoMode === 'casedata' ? 'casedata_session.json' : 'cnc_session.json'}
                />
              </div>
              <div>
                <div className="small">Skip ahead (samples)</div>
                <input
                  aria-label="Demo skip ahead"
                  value={demoStartPosition}
                  onChange={(e) => {
                    const value = e.target.value
                    setDemoStartPosition(value)
                    window.localStorage.setItem(DEMO_START_POSITION_KEY, value)
                  }}
                  placeholder="0"
                />
              </div>
            </div>
          )}
          {isCasedataSource && (
            <div className="row" style={{ marginTop: 10 }}>
              <div>
                <div className="small">Machine</div>
                {casedataCatalog?.cases.length ? (
                  <select
                    aria-label="Casedata machine"
                    value={selectedCasedataCase?.case_dir || ''}
                    onChange={(e) => {
                      const value = e.target.value
                      setCasedataCaseDir(value)
                      window.localStorage.setItem(CASEDATA_CASE_DIR_KEY, value)
                    }}
                  >
                    {casedataCatalog.cases.map((item) => (
                      <option key={item.case_dir} value={item.case_dir}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    aria-label="Casedata machine"
                    value={casedataCaseDir}
                    onChange={(e) => {
                      const value = e.target.value
                      setCasedataCaseDir(value)
                      window.localStorage.setItem(CASEDATA_CASE_DIR_KEY, value)
                    }}
                    placeholder="Site_b - MACHINE_B1 - CASE_B1"
                  />
                )}
              </div>
              <div>
                <div className="small">Mode</div>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, minHeight: 32 }}>
                  <input
                    aria-label="Harmonic-ready tools only"
                    type="checkbox"
                    checked={casedataValidToolsOnly}
                    onChange={(e) => {
                      const checked = e.target.checked
                      setCasedataValidToolsOnly(checked)
                      window.localStorage.setItem(CASEDATA_VALID_TOOLS_ONLY_KEY, String(checked))
                    }}
                  />
                  Harmonic-ready tools only
                </label>
                <div className="small" style={{ marginTop: 4 }}>
                  {harmonicReadyCount}/{selectedCasedataCase?.operations.length || 0} harmonic-ready
                </div>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, minHeight: 32, marginTop: 8 }}>
                  <input
                    aria-label="Start at first cutting row"
                    type="checkbox"
                    checked={casedataStartAtFirstCuttingRow}
                    onChange={(e) => {
                      const checked = e.target.checked
                      setCasedataStartAtFirstCuttingRow(checked)
                      window.localStorage.setItem(CASEDATA_START_AT_FIRST_CUTTING_ROW_KEY, String(checked))
                    }}
                  />
                  Start at first cutting row
                </label>
                {casedataResolvedStartHint && (
                  <div className="small" style={{ marginTop: 4 }}>
                    {casedataResolvedStartHint}
                  </div>
                )}
              </div>
              <div>
                <div className="small">Start operation</div>
                {visibleCasedataOperations.length ? (
                  <select
                    aria-label="Casedata start operation"
                    value={casedataOperationId}
                    onChange={(e) => {
                      const value = e.target.value
                      setCasedataOperationId(value)
                      window.localStorage.setItem(CASEDATA_OPERATION_ID_KEY, value)
                    }}
                  >
                    {visibleCasedataOperations.map((item) => (
                      <option key={`${selectedCasedataCase?.case_dir || casedataCaseDir || 'case'}:${item.operation_id}`} value={item.operation_id}>
                        {item.operation_id} · {item.tool_label || item.tool_id}
                        {item.harmonic_ready ? ' · ready' : ` · missing ${(item.missing_fields || []).join(', ')}`}
                      </option>
                    ))}
                  </select>
                ) : selectedCasedataCase?.operations.length ? (
                  <input
                    aria-label="Casedata start operation"
                    value=""
                    readOnly
                    placeholder="No harmonic-ready operations"
                  />
                ) : (
                  <input
                    aria-label="Casedata start operation"
                    value={casedataOperationId}
                    onChange={(e) => {
                      const value = e.target.value
                      setCasedataOperationId(value)
                      window.localStorage.setItem(CASEDATA_OPERATION_ID_KEY, value)
                    }}
                    placeholder="OF00003"
                  />
                )}
              </div>
              <div>
                <div className="small">{casedataStartAtFirstCuttingRow ? 'Extra skip after cutting start (samples)' : 'Skip ahead (samples)'}</div>
                <input
                  aria-label="Casedata skip ahead"
                  value={demoStartPosition}
                  onChange={(e) => {
                    const value = e.target.value
                    setDemoStartPosition(value)
                    window.localStorage.setItem(DEMO_START_POSITION_KEY, value)
                  }}
                  placeholder="0"
                />
              </div>
            </div>
          )}
          {noValidCasedataOperations && (
            <div className="small" style={{ marginTop: 4, color: '#f7768e' }}>
              No harmonic-ready operations are available for the selected machine.
            </div>
          )}
          {isMqttSource && (
            <div className="small" style={{ marginTop: 6 }}>
              Target: {(mqttBrokerHost.trim() || 'localhost')}:{Math.max(1, Number(mqttBrokerPort) || 1883)}
              {mqttTopic.trim() ? ` · ${mqttTopic.trim()}` : ''}
              {mqttUsername.trim() ? ` · user ${mqttUsername.trim()}` : ''}
              {mqttPassword ? ' · password set' : ' · no password'}
            </div>
          )}
          {demoError && (
            <div className="small" style={{ color: '#f7768e', marginTop: 4 }}>
              Demo error: {demoError}
            </div>
          )}
          {isCasedataSource && casedataCatalogError && (
            <div className="small" style={{ color: '#f7768e', marginTop: 4 }}>
              Casedata catalog error: {casedataCatalogError}
            </div>
          )}
          <div className="small" style={{ marginTop: 6 }}>
            {isCasedataSource
              ? `casedata stream uses ${selectedCasedataCase?.case_dir || casedataCaseDir || 'the selected machine'} / ${casedataOperationId.trim() || (casedataValidToolsOnly ? selectedCasedataCase?.default_valid_operation_id : selectedCasedataCase?.default_operation_id) || 'the first available OF operation'}${casedataValidToolsOnly ? ' with harmonic-ready tools only' : ''}${casedataStartAtFirstCuttingRow ? ', starting at the first detected cutting row' : ''} unless overridden server-side`
              : isMqttSource
                ? 'MQTT live uses the broker/topic above and starts a source-aware session through /sessions/start-demo'
                : isFileSource && demoMode === 'casedata' && !demoSessionFile.trim()
                  ? 'file-backed casedata mode uses test_data/casedata_session.json and falls back to cnc_session.json when that file is absent'
                : demoSessionFile.trim()
                  ? `demo file override: ${demoSessionFile.trim()}`
                  : `sessions endpoint: ${sessionsQuery.isError ? String(sessionsQuery.error) : '/sessions'}`}
          </div>
        </div>
      </div>

      {(sessionsQuery.isError || priorsQuery.isError) && (
        <div className="panel" style={{ marginTop: 12, borderColor: 'rgba(247, 118, 142, 0.35)' }}>
          <div style={{ fontWeight: 700 }}>Data load error</div>
          <div className="small">
            {sessionsQuery.isError ? `sessions: ${String(sessionsQuery.error)}` : ''}
          </div>
          <div className="small">
            {priorsQuery.isError ? `priors: ${String(priorsQuery.error)}` : ''}
          </div>
          <div className="hrow" style={{ marginTop: 8 }}>
            <button onClick={() => sessionsQuery.refetch()} disabled={!sessionsQuery.isError}>
              Retry sessions
            </button>
            <button onClick={() => priorsQuery.refetch()} disabled={!priorsQuery.isError}>
              Retry priors
            </button>
          </div>
        </div>
      )}
    </details>
  )
}

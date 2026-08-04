/**
 * LandingPage — system-introduction dashboard (T1.1 scaffold).
 *
 * The opening surface: a short intro to what the system is and what data it
 * uses, a live "process pulse" (anomaly score + abstracted overview), and
 * clickable tiles into each main feature with a one-line context blurb.
 *
 * First-pass scaffold — intended to be reviewed and rebuilt. The live pulse
 * reads whatever is already streaming on the alerts WebSocket (liveScoreStore).
 */

import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAppContext } from '../contexts/AppContext'
import { useLiveScoreStore } from '../state/liveScoreStore'
import { colors, fontSize, radii, shadows, spacing } from '../styles/tokens'

type FeatureTile = {
  route: string
  title: string
  blurb: string
  emoji: string
  /** Shown in the operator-focused view; the rest appear only in Full view. */
  operatorVisible: boolean
}

// The "main features" surfaced on the intro dashboard, each with a one-line
// context blurb. Operator-facing tiles first; "how it works" surfaces are
// Full-view only (this is primarily the operator's landing page).
const FEATURE_TILES: FeatureTile[] = [
  { route: '/operator', emoji: '📟', title: 'Monitoring', blurb: 'Watch a live process and get anomaly alerts you can act on.', operatorVisible: true },
  { route: '/detailed', emoji: '🛠️', title: 'Detailed', blurb: 'The full operator workspace: stream, alerts, feedback and priors.', operatorVisible: true },
  { route: '/learnings', emoji: '🧠', title: 'Learnings', blurb: 'What your feedback taught the system — and what propagates to the fleet / MaaS.', operatorVisible: true },
  { route: '/harmonics', emoji: '〰️', title: 'Harmonics', blurb: 'The harmonic context-weighted model behind anomaly scoring.', operatorVisible: false },
  { route: '/graph', emoji: '🕸️', title: 'Knowledge Graph', blurb: 'How events, patterns and documentation connect.', operatorVisible: false },
  { route: '/sindit', emoji: '⚙️', title: 'Digital Twin', blurb: 'Live asset and tool context from the SINDIT digital twin.', operatorVisible: false },
  { route: '/documents', emoji: '📄', title: 'Documents', blurb: 'Machine documentation that grounds the explanations.', operatorVisible: false },
  { route: '/experiment', emoji: '🧪', title: 'Experiment', blurb: 'Offline experiment results and model evaluation.', operatorVisible: false },
  { route: '/dataset', emoji: '📊', title: 'Dataset', blurb: 'Explore the raw machining sensor datasets behind the demo.', operatorVisible: false },
]

/**
 * Abstracted process-overview fields pulled from the live cutting context.
 * Each field lists candidate keys (first non-empty wins) so it stays populated
 * across process phases (e.g. machine_type may be blank but machine_id set).
 */
const PROCESS_FIELDS: { keys: string[]; label: string; unit?: string }[] = [
  { keys: ['machine_type', 'machine_id'], label: 'Machine' },
  { keys: ['tool_type', 'tool_id'], label: 'Tool' },
  { keys: ['workpiece_material'], label: 'Material' },
  { keys: ['operating_regime'], label: 'Regime' },
  { keys: ['spindle_speed'], label: 'Spindle', unit: 'rpm' },
  { keys: ['feed_rate'], label: 'Feed', unit: 'mm/min' },
]

function pickValue(context: Record<string, unknown>, keys: string[]): unknown {
  for (const k of keys) {
    const v = context[k]
    if (v != null && v !== '') return v
  }
  return null
}

function formatProcessValue(raw: unknown, unit?: string): string {
  if (raw == null || raw === '') return '—'
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    const n = Number.isInteger(raw) ? String(raw) : raw.toFixed(1)
    return unit ? `${n} ${unit}` : n
  }
  return String(raw)
}

const pageStyle: React.CSSProperties = {
  background: colors.bg,
  color: colors.text,
  minHeight: '100%',
  padding: spacing.xl,
}

const panelStyle: React.CSSProperties = {
  background: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: radii.lg,
  boxShadow: shadows.panel,
  padding: spacing.lg,
}

function scoreColor(score: number): string {
  if (score >= 0.9) return colors.bad
  if (score >= 0.75) return colors.warn
  return colors.good
}

/** Minimal dependency-free sparkline of recent significance scores. */
function Sparkline({ values }: { values: number[] }) {
  const width = 160
  const height = 36
  if (values.length < 2) return null
  const max = 1
  const min = 0
  const span = max - min || 1
  const step = width / (values.length - 1)
  const points = values
    .map((v, i) => {
      const x = i * step
      const y = height - ((Math.max(min, Math.min(max, v)) - min) / span) * height
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  return (
    <svg width={width} height={height} style={{ display: 'block' }} aria-hidden>
      <polyline points={points} fill="none" stroke={colors.accent} strokeWidth={1.5} />
    </svg>
  )
}

export default function LandingPage() {
  const ctx = useAppContext()
  const navigate = useNavigate()
  const points = useLiveScoreStore((s) => s.points)
  const latestSnapshot = useLiveScoreStore((s) => s.latest)

  const latest = points.length ? points[points.length - 1] : null
  const recentScores = useMemo(() => points.slice(-40).map((p) => p.significance_score), [points])
  const hasSession = Boolean(ctx.streamSessionId)
  const processContext = latestSnapshot?.context ?? {}

  // Operator's landing page: show operator-facing tiles by default; Full view
  // additionally reveals the "how it works" surfaces.
  const tiles = ctx.operatorMode ? FEATURE_TILES.filter((t) => t.operatorVisible) : FEATURE_TILES

  return (
    <div style={pageStyle}>
      {/* ── Hero ── */}
      <div style={{ marginBottom: spacing.xl }}>
        <h1 style={{ fontSize: fontSize.xxl, fontWeight: 700, margin: 0 }}>Learning Feedback Loop</h1>
        <p style={{ color: colors.text, fontSize: fontSize.lg, margin: `${spacing.sm}px 0 0`, maxWidth: 780 }}>
          Operator-facing anomaly detection and feedback for CNC manufacturing.
        </p>
        <p style={{ color: colors.textMuted, fontSize: fontSize.md, margin: `${spacing.sm}px 0 0`, maxWidth: 780 }}>
          The system streams real machining sensor data (vibration, power, spindle &amp; feed),
          detects anomalies in the process, explains them to the operator, and learns from every
          confirm / dismiss so the alerts get sharper over time — and, at the end of a process,
          proposes what to reconfigure next. It is machine-agnostic: new machines and shop floors
          are onboarded through configuration, not new code.
        </p>
      </div>

      {/* ── Live process pulse ── */}
      <section style={{ ...panelStyle, marginBottom: spacing.xl }}>
        <div style={{ alignItems: 'baseline', display: 'flex', gap: spacing.md, justifyContent: 'space-between', marginBottom: spacing.md }}>
          <h2 style={{ fontSize: fontSize.lg, fontWeight: 600, margin: 0 }}>Live process pulse</h2>
          <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
            {hasSession ? `session ${ctx.streamSessionId}` : 'no active session'}
          </span>
        </div>
        {latest ? (
          <div style={{ display: 'grid', gap: spacing.lg }}>
            {/* Raw scores */}
            <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.xl }}>
              <div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Anomaly score</div>
                <div style={{ color: scoreColor(latest.significance_score), fontSize: 34, fontWeight: 700, lineHeight: 1 }}>
                  {latest.significance_score.toFixed(2)}
                </div>
              </div>
              <div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: spacing.xs }}>Recent trend</div>
                <Sparkline values={recentScores} />
              </div>
              <div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Detector score</div>
                <div style={{ color: colors.text, fontSize: fontSize.xl, fontWeight: 600 }}>
                  {latest.anomaly_detector_score.toFixed(2)}
                </div>
              </div>
              <div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Prior boost</div>
                <div style={{ color: colors.text, fontSize: fontSize.xl, fontWeight: 600 }}>
                  {latest.prior_boost.toFixed(2)}
                </div>
              </div>
              <div>
                <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Rules fired</div>
                <div style={{ color: colors.text, fontSize: fontSize.xl, fontWeight: 600 }}>
                  {latest.n_rules_triggered}
                </div>
              </div>
            </div>

            {/* Abstracted process overview */}
            <div>
              <div style={{ color: colors.textMuted, fontSize: fontSize.xs, letterSpacing: '0.06em', marginBottom: spacing.sm, textTransform: 'uppercase' }}>
                Process overview
              </div>
              <div style={{ display: 'grid', gap: spacing.md, gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))' }}>
                {PROCESS_FIELDS.map((f) => (
                  <div key={f.label}>
                    <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>{f.label}</div>
                    <div style={{ color: colors.text, fontSize: fontSize.md, fontWeight: 600 }}>
                      {formatProcessValue(pickValue(processContext, f.keys), f.unit)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ color: colors.textMuted, fontSize: fontSize.sm }}>
            No live scores yet.{' '}
            <button
              type="button"
              onClick={() => navigate('/operator')}
              style={{ background: 'none', border: 'none', color: colors.accent, cursor: 'pointer', fontSize: fontSize.sm, padding: 0, textDecoration: 'underline' }}
            >
              Start a session in Monitoring
            </button>{' '}
            to see the anomaly score and process overview here.
          </div>
        )}
        <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginTop: spacing.md }}>
          Abstracted overview of the running process — a first look before diving into a feature below.
        </div>
      </section>

      {/* ── Feature tiles ── */}
      <div style={{ marginBottom: spacing.md }}>
        <h2 style={{ fontSize: fontSize.lg, fontWeight: 600, margin: 0 }}>Explore the system</h2>
        <p style={{ color: colors.textMuted, fontSize: fontSize.sm, margin: `${spacing.xs}px 0 0` }}>
          Each area below is a main feature — open one to see it in context.
        </p>
      </div>
      <div
        style={{
          display: 'grid',
          gap: spacing.lg,
          gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
        }}
      >
        {tiles.map((tile) => (
          <button
            key={tile.route}
            type="button"
            onClick={() => navigate(tile.route)}
            style={{
              ...panelStyle,
              cursor: 'pointer',
              display: 'grid',
              gap: spacing.xs,
              padding: spacing.lg,
              textAlign: 'left',
            }}
          >
            <div style={{ alignItems: 'center', display: 'flex', gap: spacing.sm }}>
              <span style={{ fontSize: fontSize.xl }} aria-hidden>{tile.emoji}</span>
              <span style={{ color: colors.text, fontSize: fontSize.md, fontWeight: 600 }}>{tile.title}</span>
            </div>
            <span style={{ color: colors.textMuted, fontSize: fontSize.sm }}>{tile.blurb}</span>
            <span style={{ color: colors.accent, fontSize: fontSize.xs, marginTop: spacing.xs }}>Open →</span>
          </button>
        ))}
      </div>
    </div>
  )
}

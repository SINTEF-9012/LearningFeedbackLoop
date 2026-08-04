/**
 * FeedbackImpactTab — Tab 3: before/after comparison chart, prediction flips,
 *                     score shift scatter, pattern feedback breakdown,
 *                     feedback stats + weight evolution charts.
 */
import React from 'react'
import { WeightHistorySVG, PAL } from '../charts'
import { HelpIcon } from '../Tooltip'
import { humanPattern } from '../../utils/patternNames'
import type { ExperimentTabProps } from './types'
import type { SampleResult, PhaseDetail } from '../../state/experimentStore'

/* ── Prediction Flip Summary ─────────────────────────────────────────── */
function PredictionFlipCard({ evalPhase, testPhase }: { evalPhase: PhaseDetail; testPhase: PhaseDetail }) {
  // Build lookup: sample_id → test prediction
  const testMap = new Map<string, { pred: boolean; label: string }>()
  for (const s of testPhase.samples) {
    testMap.set(s.sample_id, { pred: s.predicted_positive, label: s.label })
  }

  let wrongToRight = 0, rightToWrong = 0, stayedCorrect = 0, stayedWrong = 0
  for (const es of evalPhase.samples) {
    const ts = testMap.get(es.sample_id)
    if (!ts) continue
    const isPos = es.label === 'pre_stoppage' || es.label === 'pre_break'
    const testCorrect = ts.pred === isPos
    const evalCorrect = es.predicted_positive === isPos
    if (!testCorrect && evalCorrect) wrongToRight++
    else if (testCorrect && !evalCorrect) rightToWrong++
    else if (testCorrect && evalCorrect) stayedCorrect++
    else stayedWrong++
  }
  const total = wrongToRight + rightToWrong + stayedCorrect + stayedWrong
  if (total === 0) return null

  const netFixed = wrongToRight - rightToWrong
  const netColor = netFixed > 0 ? 'var(--ok, #4caf50)' : netFixed < 0 ? 'var(--danger, #e74c3c)' : 'var(--muted)'

  // Mini horizontal stacked bar
  const W = 400, H = 28, PAD = 4
  const barW = W - PAD * 2
  const segments = [
    { n: stayedCorrect, color: '#4caf50', label: 'Stayed correct' },
    { n: wrongToRight,  color: '#2196f3', label: 'Fixed (wrong→right)' },
    { n: rightToWrong,  color: '#f44336', label: 'Broken (right→wrong)' },
    { n: stayedWrong,   color: '#666',    label: 'Stayed wrong' },
  ]

  let xOff = PAD
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>
        Prediction Flips
        <HelpIcon text="How many individual predictions changed between Test (no feedback) and Eval (with feedback). 'Fixed' = feedback corrected a wrong prediction. 'Broken' = feedback caused a previously-correct prediction to become wrong. Net gain = Fixed − Broken." />
      </div>
      <div className="small" style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 10 }}>
        <span style={{ color: '#2196f3', fontWeight: 600 }}>Fixed: {wrongToRight}</span>
        <span style={{ color: '#f44336', fontWeight: 600 }}>Broken: {rightToWrong}</span>
        <span style={{ color: netColor, fontWeight: 700 }}>Net: {netFixed > 0 ? '+' : ''}{netFixed}</span>
        <span style={{ color: 'var(--muted)' }}>Unchanged: {stayedCorrect + stayedWrong} ({stayedCorrect} correct, {stayedWrong} wrong)</span>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ borderRadius: 4, display: 'block', maxWidth: '100%' }}>
        {segments.map((seg, i) => {
          const w = total > 0 ? (seg.n / total) * barW : 0
          const el = (
            <g key={i}>
              {w > 0 && <rect x={xOff} y={PAD} width={w} height={H - PAD * 2} fill={seg.color} rx={i === 0 ? 3 : 0} opacity={0.8}>
                <title>{seg.label}: {seg.n}</title>
              </rect>}
              {w > 20 && (
                <text x={xOff + w / 2} y={H / 2 + 4} textAnchor="middle" fill="#fff" fontSize={10} fontWeight={600}>
                  {seg.n}
                </text>
              )}
            </g>
          )
          xOff += w
          return el
        })}
      </svg>
      <div style={{ display: 'flex', gap: 14, marginTop: 4, fontSize: 10 }}>
        {segments.filter(s => s.n > 0).map(s => (
          <span key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 10, height: 8, background: s.color, borderRadius: 2, opacity: 0.8 }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  )
}

/* ── Score Shift: Test vs Eval scatter ───────────────────────────────── */
function ScoreShiftChart({ evalPhase, testPhase }: { evalPhase: PhaseDetail; testPhase: PhaseDetail }) {
  // Pair samples by ID — plot test score (x) vs eval score (y)
  const testMap = new Map<string, SampleResult>()
  for (const s of testPhase.samples) testMap.set(s.sample_id, s)

  const points: { x: number; y: number; label: string; fb: string }[] = []
  for (const es of evalPhase.samples) {
    const ts = testMap.get(es.sample_id)
    if (!ts) continue
    points.push({
      x: ts.significance_score,
      y: es.significance_score,
      label: es.label,
      fb: es.feedback_action || '',
    })
  }
  if (points.length === 0) return null

  const W = 340, H = 340
  const PAD = { t: 16, r: 16, b: 32, l: 40 }
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b

  const xScale = (v: number) => PAD.l + Math.min(Math.max(v, 0), 1) * plotW
  const yScale = (v: number) => PAD.t + plotH - Math.min(Math.max(v, 0), 1) * plotH

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>
        Score Shift: Test → Eval
        <HelpIcon text="Each dot is one sample. X = significance score in the Test phase (no feedback). Y = score in the Eval phase (with feedback). The diagonal grey line = no change. Dots above the line were scored higher after feedback; below = lower. Red = pre_stoppage, blue = normal. Ideally, red dots move up-right and blue dots move down-left." />
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.12)', borderRadius: 6, display: 'block', maxWidth: '100%' }}>
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map(v => (
          <g key={v}>
            <line x1={xScale(v)} y1={PAD.t} x2={xScale(v)} y2={PAD.t + plotH} stroke="rgba(255,255,255,0.05)" />
            <line x1={PAD.l} y1={yScale(v)} x2={PAD.l + plotW} y2={yScale(v)} stroke="rgba(255,255,255,0.05)" />
            <text x={xScale(v)} y={PAD.t + plotH + 14} textAnchor="middle" fill="var(--muted)" fontSize={9}>{v.toFixed(2)}</text>
            <text x={PAD.l - 6} y={yScale(v) + 3} textAnchor="end" fill="var(--muted)" fontSize={9}>{v.toFixed(2)}</text>
          </g>
        ))}
        {/* Diagonal (no-change line) */}
        <line x1={xScale(0)} y1={yScale(0)} x2={xScale(1)} y2={yScale(1)} stroke="rgba(255,255,255,0.15)" strokeDasharray="4,3" />
        {/* Points */}
        {points.map((p, i) => {
          const isPos = p.label === 'pre_stoppage' || p.label === 'pre_break'
          return (
            <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={3}
              fill={isPos ? 'rgba(247,118,142,0.7)' : 'rgba(122,162,247,0.6)'}
              stroke={isPos ? '#f7768e' : '#7aa2f7'} strokeWidth={0.5}>
              <title>{p.label} — test: {p.x.toFixed(3)}, eval: {p.y.toFixed(3)}{p.fb ? `, fb: ${p.fb}` : ''}</title>
            </circle>
          )
        })}
        {/* Axis labels */}
        <text x={PAD.l + plotW / 2} y={H - 4} textAnchor="middle" fill="var(--muted)" fontSize={10}>Test Score</text>
        <text x={12} y={PAD.t + plotH / 2} textAnchor="middle" fill="var(--muted)" fontSize={10} transform={`rotate(-90, 12, ${PAD.t + plotH / 2})`}>Eval Score</text>
      </svg>
      <div style={{ display: 'flex', gap: 16, marginTop: 4, fontSize: 10 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: 'rgba(247,118,142,0.7)' }} />
          Pre-stoppage
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: 'rgba(122,162,247,0.6)' }} />
          Normal
        </span>
        <span style={{ color: 'var(--muted)' }}>┈ No change line</span>
      </div>
    </div>
  )
}

/* ── Pattern × Feedback breakdown ────────────────────────────────────── */
function PatternFeedbackChart({ phase }: { phase: PhaseDetail }) {
  // Count confirms / dismissals per pattern
  const patternStats = new Map<string, { confirms: number; dismissals: number; total: number }>()
  for (const s of phase.samples) {
    if (!s.feedback_given || !s.detected_patterns?.length) continue
    const isConfirm = s.feedback_action.toUpperCase() === 'CONFIRM'
    for (const pk of s.detected_patterns) {
      const cur = patternStats.get(pk) ?? { confirms: 0, dismissals: 0, total: 0 }
      if (isConfirm) cur.confirms++
      else cur.dismissals++
      cur.total++
      patternStats.set(pk, cur)
    }
  }

  const sorted = [...patternStats.entries()]
    .sort((a, b) => b[1].total - a[1].total)
    .slice(0, 12)  // top 12
  if (sorted.length === 0) return null

  const maxCount = Math.max(1, ...sorted.map(([, s]) => s.total))
  const W = 540, rowH = 22, gap = 4
  const PAD = { t: 8, r: 16, b: 8, l: 160 }
  const barMaxW = W - PAD.l - PAD.r
  const H = PAD.t + sorted.length * (rowH + gap) - gap + PAD.b

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>
        Pattern × Feedback
        <HelpIcon text="For each pattern that triggered during the Eval phase, how many operator confirms (green) vs dismissals (red) it received. Patterns with high dismiss rates may be generating false positives and should be tuned." />
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.12)', borderRadius: 6, display: 'block', maxWidth: '100%' }}>
        {sorted.map(([pk, stats], i) => {
          const y = PAD.t + i * (rowH + gap)
          const cw = (stats.confirms / maxCount) * barMaxW
          const dw = (stats.dismissals / maxCount) * barMaxW
          const confirmRate = stats.total > 0 ? Math.round((stats.confirms / stats.total) * 100) : 0
          return (
            <g key={pk}>
              <text x={PAD.l - 6} y={y + rowH / 2 + 4} textAnchor="end" fill="#cdd6f4" fontSize={10}>
                {humanPattern(pk)}
              </text>
              {/* Confirm bar */}
              <rect x={PAD.l} y={y} width={Math.max(0, cw)} height={rowH / 2 - 1} rx={2}
                fill="var(--ok, #4caf50)" opacity={0.8}>
                <title>Confirms: {stats.confirms}</title>
              </rect>
              {/* Dismiss bar */}
              <rect x={PAD.l} y={y + rowH / 2 + 1} width={Math.max(0, dw)} height={rowH / 2 - 1} rx={2}
                fill="var(--danger, #e74c3c)" opacity={0.7}>
                <title>Dismissals: {stats.dismissals}</title>
              </rect>
              {/* Count + rate annotation */}
              <text x={PAD.l + Math.max(cw, dw) + 6} y={y + rowH / 2 + 4}
                fill="var(--muted)" fontSize={9}>
                {stats.total} ({confirmRate}% ✓)
              </text>
            </g>
          )
        })}
      </svg>
      <div style={{ display: 'flex', gap: 14, marginTop: 4, fontSize: 10 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 8, background: 'var(--ok, #4caf50)', borderRadius: 2, opacity: 0.8 }} />
          Confirms
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ display: 'inline-block', width: 10, height: 8, background: 'var(--danger, #e74c3c)', borderRadius: 2, opacity: 0.7 }} />
          Dismissals
        </span>
      </div>
    </div>
  )
}

/* ── Before / After grouped horizontal bar chart ─────────────────────── */
function ImpactChart({ selectedRun }: { selectedRun?: ExperimentTabProps['selectedRun'] }) {
  const tm = selectedRun?.test_metrics ?? {}
  const em = selectedRun?.eval_metrics ?? {}

  const metrics = [
    { label: 'Precision', before: tm.precision ?? 0, after: em.precision ?? 0 },
    { label: 'Recall',    before: tm.recall ?? 0,    after: em.recall ?? 0 },
    { label: 'F1 Score',  before: tm.f1 ?? 0,        after: em.f1 ?? 0 },
    { label: 'AUC-ROC',   before: tm.auc_roc ?? 0,   after: em.auc_roc ?? 0 },
  ]
  if (!metrics.some(m => m.before > 0 || m.after > 0)) return null

  const W = 540, barH = 14, gapInGroup = 3, groupGap = 18
  const groupH = barH * 2 + gapInGroup
  const PAD = { t: 14, r: 72, b: 26, l: 76 }
  const plotW = W - PAD.l - PAD.r
  const plotH = metrics.length * (groupH + groupGap) - groupGap
  const H = PAD.t + plotH + PAD.b

  const maxVal = Math.max(1, ...metrics.flatMap(m => [m.before, m.after]))
  const x = (v: number) => PAD.l + Math.max(0, Math.min(v / maxVal, 1)) * plotW

  const testColor = '#7aa2f7'
  const evalColor = '#9ece6a'

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 10 }}>
        Feedback Impact: Before vs After
        <HelpIcon text="Grouped bar chart comparing model performance before (Test, blue) and after (Eval, green) the feedback loop. Deltas on the right show the absolute change — green ▲ = improvement, red ▼ = degradation." />
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.12)', borderRadius: 6, display: 'block', maxWidth: '100%' }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
          const xPos = PAD.l + frac * plotW
          return (
            <g key={frac}>
              <line x1={xPos} y1={PAD.t - 4} x2={xPos} y2={PAD.t + plotH} stroke="rgba(255,255,255,0.06)" />
              <text x={xPos} y={PAD.t + plotH + 14} textAnchor="middle" fill="var(--muted)" fontSize={9}>
                {(frac * maxVal).toFixed(2)}
              </text>
            </g>
          )
        })}
        {metrics.map((m, i) => {
          const yBase = PAD.t + i * (groupH + groupGap)
          const d = m.after - m.before
          const dSign = d > 0 ? '+' : ''
          const dColor = Math.abs(d) < 0.001 ? 'var(--muted)' : d > 0 ? 'var(--ok, #4caf50)' : 'var(--danger, #e74c3c)'
          const arrow = Math.abs(d) < 0.001 ? '' : d > 0 ? '▲ ' : '▼ '
          return (
            <g key={m.label}>
              <text x={PAD.l - 8} y={yBase + groupH / 2 + 4} textAnchor="end" fill="#cdd6f4" fontSize={11} fontWeight={500}>
                {m.label}
              </text>
              {/* Test bar (before) */}
              <rect x={PAD.l} y={yBase} width={Math.max(2, x(m.before) - PAD.l)} height={barH} rx={3}
                fill={testColor} opacity={0.75} />
              <text x={x(m.before) + 4} y={yBase + barH - 2} fill={testColor} fontSize={9} fontWeight={600}>
                {m.before.toFixed(3)}
              </text>
              {/* Eval bar (after) */}
              <rect x={PAD.l} y={yBase + barH + gapInGroup} width={Math.max(2, x(m.after) - PAD.l)} height={barH} rx={3}
                fill={evalColor} opacity={0.85} />
              <text x={x(m.after) + 4} y={yBase + barH + gapInGroup + barH - 2} fill={evalColor} fontSize={9} fontWeight={600}>
                {m.after.toFixed(3)}
              </text>
              {/* Delta */}
              <text x={W - PAD.r + 8} y={yBase + groupH / 2 + 4} textAnchor="start"
                fill={dColor} fontSize={10} fontWeight={700}>
                {arrow}{dSign}{d.toFixed(3)}
              </text>
            </g>
          )
        })}
      </svg>
      <div style={{ display: 'flex', gap: 20, marginTop: 6, fontSize: 11 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ display: 'inline-block', width: 14, height: 10, background: testColor, borderRadius: 3, opacity: 0.75 }} />
          Test (before feedback)
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ display: 'inline-block', width: 14, height: 10, background: evalColor, borderRadius: 3, opacity: 0.85 }} />
          Eval (after feedback)
        </span>
      </div>
    </div>
  )
}

export function FeedbackImpactTab({ evalPhase, testPhase, selectedRun }: ExperimentTabProps) {
  if (!evalPhase && !testPhase && !selectedRun) return null

  return (
    <>
      {/* Before / After comparison chart */}
      <ImpactChart selectedRun={selectedRun} />

      {/* Prediction Flip Summary */}
      {evalPhase && testPhase && <PredictionFlipCard evalPhase={evalPhase} testPhase={testPhase} />}

      {/* Score Shift: Test → Eval scatter */}
      {evalPhase && testPhase && <ScoreShiftChart evalPhase={evalPhase} testPhase={testPhase} />}

      {/* Pattern × Feedback breakdown */}
      {evalPhase && <PatternFeedbackChart phase={evalPhase} />}

      {/* Per-phase feedback breakdown + weight charts */}
      {[evalPhase, testPhase].filter(Boolean).map(phase => {
        const fb = phase!.samples.filter(s => s.feedback_given)
        const confirms = fb.filter(s => s.feedback_action.toUpperCase() === 'CONFIRM').length
        const dismissals = fb.filter(s => s.feedback_action.toUpperCase() === 'DISMISS').length
        const negSampled = fb.filter(s => (s.feedback_source || '').toLowerCase() === 'negative_sample').length
        const missed = fb.filter(s => (s.feedback_source || '').toLowerCase() === 'missed_event').length

        return (
          <div key={phase!.phase} className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, marginBottom: 12 }}>
              {phase!.phase === 'eval' ? 'Evaluation' : 'Test'}: {phase!.operation}
              <HelpIcon text="Feedback breakdown for this phase. Confirms = operator verified the detection was correct. Dismissals = operator marked it as a false alarm. Missed-event = a real event that the model missed (false negative recovered by operator). Negative-sampled = a normal sample selected for feedback to balance the dataset." />
            </div>
            <div className="small" style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 12 }}>
              <span>Total feedback: {fb.length}</span>
              <span style={{ color: 'var(--ok)' }}>✓ Confirms: {confirms}</span>
              <span style={{ color: 'var(--danger)' }}>✗ Dismissals: {dismissals}</span>
              <span style={{ color: 'var(--accent)' }}>↩ Missed-event: {missed}</span>
              <span style={{ color: PAL[3] }}>⊖ Negative-sampled: {negSampled}</span>
            </div>
            {phase!.weight_history && Object.keys(phase!.weight_history).length > 0 && (
              <WeightHistorySVG
                history={phase!.weight_history}
                title="Model Weight Evolution"
                keys={['supervised', 'unsupervised']}
              />
            )}
            {phase!.tool_prior_history && Object.keys(phase!.tool_prior_history).length > 0 && (
              <div style={{ marginTop: 12 }}>
                <WeightHistorySVG
                  history={phase!.tool_prior_history}
                  title="Tool Prior Evolution"
                />
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}

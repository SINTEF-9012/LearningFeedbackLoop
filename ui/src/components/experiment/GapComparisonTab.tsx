/**
 * GapComparisonTab — Tab 6: SVG bar chart + table comparing all runs sorted by gap.
 */
import React from 'react'
import { f3, pct, PAL, clamp } from '../charts'
import { HelpIcon } from '../Tooltip'
import type { ExperimentTabProps } from './types'

function GapBarChart({ runs }: { runs: ExperimentTabProps['runs'] }) {
  const sorted = [...runs].sort((a, b) => (a.gap_s ?? 999) - (b.gap_s ?? 999))
  if (sorted.length === 0) return null

  const W = 520, barH = 18, gap = 6
  const PAD = { t: 20, r: 60, b: 10, l: 70 }
  const H = PAD.t + sorted.length * (barH * 2 + gap + 10) + PAD.b
  const plotW = W - PAD.l - PAD.r

  const maxF1 = Math.max(0.01, ...sorted.flatMap(r => [r.eval_metrics?.f1 ?? 0, r.test_metrics?.f1 ?? 0]))

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ background: 'rgba(0,0,0,0.15)', borderRadius: 6, marginBottom: 12 }}>
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1.0].map(frac => {
        const x = PAD.l + frac * plotW
        return (
          <g key={frac}>
            <line x1={x} y1={PAD.t} x2={x} y2={H - PAD.b} stroke="rgba(255,255,255,0.06)" />
            <text x={x} y={PAD.t - 6} textAnchor="middle" fill="var(--muted)" fontSize={9}>
              {(frac * maxF1).toFixed(2)}
            </text>
          </g>
        )
      })}
      {sorted.map((r, i) => {
        const yBase = PAD.t + i * (barH * 2 + gap + 10)
        const evalF1 = r.eval_metrics?.f1 ?? 0
        const testF1 = r.test_metrics?.f1 ?? 0
        const ew = clamp(evalF1 / maxF1, 0, 1) * plotW
        const tw = clamp(testF1 / maxF1, 0, 1) * plotW
        return (
          <g key={r.run_id}>
            <text x={PAD.l - 6} y={yBase + barH + 2} textAnchor="end" fill="#ccc" fontSize={10}>
              gap={r.gap_s ?? '?'}
            </text>
            {/* Eval F1 */}
            <rect x={PAD.l} y={yBase} width={Math.max(1, ew)} height={barH} rx={2} fill={PAL[0]} opacity={0.8} />
            <text x={PAD.l + ew + 4} y={yBase + barH - 4} fill={PAL[0]} fontSize={9}>{f3(evalF1)}</text>
            {/* Test F1 */}
            <rect x={PAD.l} y={yBase + barH + 2} width={Math.max(1, tw)} height={barH} rx={2} fill={PAL[1]} opacity={0.7} />
            <text x={PAD.l + tw + 4} y={yBase + barH * 2} fill={PAL[1]} fontSize={9}>{f3(testF1)}</text>
          </g>
        )
      })}
    </svg>
  )
}

export function GapComparisonTab({ runs, effectiveRunId, setSelectedRunId }: ExperimentTabProps) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 8 }}>Gap Comparison ({runs.length} runs) <HelpIcon text="Compare experiment runs across different prediction gaps. A larger gap (e.g. 30s) means predicting further ahead — harder but more operationally useful. The bar chart and table show how F1, precision, recall, and AUC change as the gap increases. Click a row to select that run." /></div>

      {/* SVG bar chart */}
      {runs.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <GapBarChart runs={runs} />
          <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ display: 'inline-block', width: 12, height: 8, background: PAL[0], borderRadius: 2, opacity: 0.8 }} />
              Eval F1
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ display: 'inline-block', width: 12, height: 8, background: PAL[1], borderRadius: 2, opacity: 0.7 }} />
              Test F1
            </span>
          </div>
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '4px 8px' }}>Run</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Gap</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Eval F1</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Eval Prec</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Eval Rec</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Eval AUC</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Test F1</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Confirms</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>Dismiss</th>
              <th style={{ textAlign: 'right', padding: '4px 8px' }}>N</th>
            </tr>
          </thead>
          <tbody>
            {[...runs]
              .sort((a, b) => (a.gap_s ?? 999) - (b.gap_s ?? 999))
              .map(r => {
                const isSel = r.run_id === effectiveRunId
                return (
                  <tr
                    key={r.run_id}
                    style={{
                      borderBottom: '1px solid var(--border)',
                      background: isSel ? 'rgba(122,162,247,0.1)' : undefined,
                      cursor: 'pointer',
                    }}
                    onClick={() => setSelectedRunId(r.run_id)}
                  >
                    <td style={{ padding: '4px 8px', fontWeight: isSel ? 700 : 400, maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.run_id}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px' }}>{r.gap_s ?? '?'}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{f3(r.eval_metrics?.f1)}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{pct(r.eval_metrics?.precision)}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{pct(r.eval_metrics?.recall)}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{f3(r.eval_metrics?.auc_roc)}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px', fontVariantNumeric: 'tabular-nums' }}>{f3(r.test_metrics?.f1)}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px' }}>{r.feedback_stats?.n_confirms ?? '–'}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px' }}>{r.feedback_stats?.n_dismissals ?? '–'}</td>
                    <td style={{ textAlign: 'right', padding: '4px 8px' }}>{r.eval_metrics?.n_samples ?? '–'}</td>
                  </tr>
                )
              })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

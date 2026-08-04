import React, { useMemo, useRef } from 'react'
import { humanPattern, patternDescription, patternOrigin } from '../utils/patternNames'

export type PriorSeverityCalibration = {
  average_delta?: number
  weight_total?: number
  targets?: Partial<Record<'info' | 'warning' | 'critical', number>>
}

export type PriorRow = {
  pattern: string
  prior: number
  effective_weight_total?: number
  passive_outcome_count?: number
  severity_correction_count?: number
  severity_calibration?: PriorSeverityCalibration
}

export type PriorsChartProps = {
  priors: PriorRow[]
  maxRows?: number
  showDiagnostics?: boolean
}

const diagnosticChipStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  padding: '2px 8px',
  borderRadius: 999,
  border: '1px solid var(--border)',
  background: 'rgba(255,255,255,0.03)',
  fontSize: 11,
  lineHeight: 1.3,
}

function clamp01(x: number) {
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

function finiteNumber(value: unknown) {
  return Number.isFinite(Number(value)) ? Number(value) : 0
}

function formatWeight(value: number) {
  if (value >= 10) return value.toFixed(1)
  if (value >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

export function PriorsChart(props: PriorsChartProps) {
  const maxRows = props.maxRows ?? 30
  const showDiagnostics = Boolean(props.showDiagnostics)

  const rows = useMemo(() => {
    const inRows = (props.priors || []).filter((p) => p && typeof p.pattern === 'string')
    const sorted = [...inRows].sort((a, b) => Number(b.prior) - Number(a.prior))
    return sorted.slice(0, maxRows)
  }, [props.priors, maxRows])

  const maxPrior = useMemo(() => {
    let m = 0
    for (const r of rows) m = Math.max(m, Number(r.prior) || 0)
    return m || 1
  }, [rows])

  const prevRef = useRef<Record<string, number>>({})
  const prev = prevRef.current

  // Update previous priors map after render computation.
  const deltas = useMemo(() => {
    const out: Record<string, number> = {}
    for (const r of rows) {
      const p = Number(r.prior) || 0
      const old = prev[r.pattern]
      out[r.pattern] = typeof old === 'number' ? p - old : 0
    }
    // mutate after we computed deltas so current render can still read old
    for (const r of rows) prev[r.pattern] = Number(r.prior) || 0
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, maxPrior])

  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {rows.map((r) => {
        const prior = Number(r.prior) || 0
        const widthPct = clamp01(prior / maxPrior) * 100
        const d = deltas[r.pattern] || 0
        const dText = Math.abs(d) < 1e-6 ? '' : (d > 0 ? `+${d.toFixed(3)}` : d.toFixed(3))
        const dColor = d > 0 ? 'var(--ok)' : d < 0 ? 'var(--danger)' : 'var(--muted)'
        const effectiveWeight = finiteNumber(r.effective_weight_total)
        const passiveOutcomes = Math.max(0, Math.trunc(finiteNumber(r.passive_outcome_count)))
        const severityCorrections = Math.max(0, Math.trunc(finiteNumber(r.severity_correction_count)))
        const severityCalibration = r.severity_calibration || {}
        const severityWeight = finiteNumber(severityCalibration.weight_total)
        const severityDelta = finiteNumber(severityCalibration.average_delta)
        const severityTargets = Object.entries(severityCalibration.targets || {})
          .filter(([, value]) => finiteNumber(value) > 0)
          .sort((a, b) => finiteNumber(b[1]) - finiteNumber(a[1]))
        const severityDeltaColor = severityDelta > 0 ? 'var(--ok)' : severityDelta < 0 ? 'var(--danger)' : 'var(--muted)'

        return (
          <div key={r.pattern} style={{ display: 'grid', gridTemplateColumns: '1fr 80px', gap: 8, alignItems: 'start' }}>
            <div style={{ display: 'grid', gap: 4 }}>
              <div className="small" style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                <span title={patternDescription(r.pattern)} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {patternOrigin(r.pattern) === 'detected' ? '📊 ' : patternOrigin(r.pattern) === 'live' ? '🧬 ' : ''}{humanPattern(r.pattern)}
                </span>
                <span style={{ color: dColor }}>{dText}</span>
              </div>
              <div
                style={{
                  height: 10,
                  borderRadius: 999,
                  border: '1px solid var(--border)',
                  background: 'rgba(0,0,0,0.2)',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${widthPct}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, rgba(122,162,247,0.9), rgba(187,154,247,0.9))',
                  }}
                />
              </div>
              {showDiagnostics && (effectiveWeight > 0 || passiveOutcomes > 0 || severityCorrections > 0) && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, color: 'var(--muted)' }}>
                  {effectiveWeight > 0 && (
                    <span style={diagnosticChipStyle}>Weight {formatWeight(effectiveWeight)}</span>
                  )}
                  {passiveOutcomes > 0 && (
                    <span style={diagnosticChipStyle}>Passive {passiveOutcomes}</span>
                  )}
                  {severityCorrections > 0 && (
                    <span style={diagnosticChipStyle}>Severity {severityCorrections}</span>
                  )}
                  {severityWeight > 0 && (
                    <span style={{ ...diagnosticChipStyle, color: severityDeltaColor }}>
                      Delta {severityDelta > 0 ? '+' : ''}{severityDelta.toFixed(3)}
                    </span>
                  )}
                  {severityTargets.map(([label, value]) => (
                    <span key={`${r.pattern}-${label}`} style={diagnosticChipStyle}>
                      {label} {formatWeight(finiteNumber(value))}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{prior.toFixed(3)}</div>
          </div>
        )
      })}

      {!rows.length && <div className="small">No priors yet (send events + feedback).</div>}
    </div>
  )
}

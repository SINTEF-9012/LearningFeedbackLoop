import React, { useMemo } from 'react'
import { HarmonicWeightsChart } from './HarmonicWeightsChart'

type HarmonicContextSnapshotProps = {
  score?: number
  weights?: number[] | null
  labels?: string[] | null
  values?: number[] | null
  title?: string
  subtitle?: string
  compact?: boolean
}

type HarmonicRow = {
  label: string
  weight?: number
  value?: number
  contribution?: number
}

function finiteNumbers(values?: number[] | null): number[] {
  if (!Array.isArray(values)) return []
  return values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
}

function normalizeLabels(labels: string[] | null | undefined, count: number): string[] {
  const safeLabels = Array.isArray(labels) ? labels.filter((label): label is string => typeof label === 'string') : []
  return Array.from({ length: count }, (_, index) => safeLabels[index] || `F${index + 1}`)
}

function fmtNumber(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1000) return value.toFixed(0)
  if (abs >= 10) return value.toFixed(2)
  if (abs >= 1) return value.toFixed(3)
  if (abs === 0) return '0.000'
  return value.toExponential(2)
}

export function HarmonicContextSnapshot({
  score,
  weights,
  labels,
  values,
  title = 'Harmonic context',
  subtitle,
  compact = false,
}: HarmonicContextSnapshotProps) {
  const safeWeights = finiteNumbers(weights)
  const safeValues = finiteNumbers(values)
  const hasWeights = safeWeights.length > 0
  const featureCount = Math.max(safeWeights.length, safeValues.length)
  const safeLabels = normalizeLabels(labels, featureCount)

  const rows = useMemo<HarmonicRow[]>(() => {
    return Array.from({ length: featureCount }, (_, index) => {
      const weight = index < safeWeights.length ? safeWeights[index] : undefined
      const value = index < safeValues.length ? safeValues[index] : undefined
      const contribution =
        typeof weight === 'number' && typeof value === 'number'
          ? weight * value
          : undefined
      return {
        label: safeLabels[index],
        weight,
        value,
        contribution,
      }
    })
  }, [featureCount, safeLabels, safeValues, safeWeights])

  if (safeWeights.length === 0 && safeValues.length === 0) return null

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 700 }}>{title}</div>
          <div style={{ fontSize: 10, color: 'var(--muted)' }}>
            {subtitle || (hasWeights
              ? 'Live harmonic feature values aligned to the current weight vector.'
              : 'Live harmonic outputs for the current inference window.')}
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>
          {featureCount} features{typeof score === 'number' ? ` · score ${score.toFixed(3)}` : ''}
        </div>
      </div>

      {(safeWeights.length > 0 || safeValues.length > 0) && (
        <HarmonicWeightsChart
          weights={safeWeights.length > 0 ? safeWeights : safeValues}
          labels={safeLabels}
          score={safeWeights.length > 0 ? score : undefined}
          kind={safeWeights.length > 0 ? 'weights' : 'outputs'}
          height={compact ? 130 : 150}
        />
      )}

      {rows.length > 0 && (
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
            {hasWeights ? 'Harmonic outputs and weights' : 'Live harmonic outputs'}
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 6 }}>
            {hasWeights
              ? 'Weighted contribution is the live output multiplied by the learned context weight.'
              : 'Pair-input mode exposes the live peak amplitudes used by the scorer for this window.'}
          </div>
          <div style={{ overflowX: 'auto', maxHeight: compact ? 240 : 320, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th style={{ textAlign: 'left', padding: '4px 6px' }}>Feature</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px' }}>{hasWeights ? 'Output' : 'Amplitude'}</th>
                  {hasWeights && <th style={{ textAlign: 'right', padding: '4px 6px' }}>Weight</th>}
                  {hasWeights && <th style={{ textAlign: 'right', padding: '4px 6px' }}>Weighted</th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const tone = typeof row.contribution === 'number'
                    ? row.contribution > 0
                      ? 'var(--ok)'
                      : row.contribution < 0
                      ? 'var(--danger)'
                      : 'var(--muted)'
                    : 'var(--muted)'
                  return (
                    <tr key={row.label} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.06)' }}>
                      <td style={{ padding: '4px 6px' }}>{row.label}</td>
                      <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace' }}>{fmtNumber(row.value)}</td>
                      {hasWeights && <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace' }}>{fmtNumber(row.weight)}</td>}
                      {hasWeights && <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace', color: tone }}>{fmtNumber(row.contribution)}</td>}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
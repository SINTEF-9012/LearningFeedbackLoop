import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/http'
import { HarmonicWeightsChart } from '../HarmonicWeightsChart'
import type {
  HarmonicExplainResponse,
  ModelBreakdown,
  ModelBreakdownSection,
  SampleResult,
} from '../../state/experimentStore'

const panelStyle: React.CSSProperties = {
  padding: 12,
  border: '1px solid var(--border)',
  borderRadius: 8,
  background: 'rgba(122, 162, 247, 0.05)',
}

const metricGridStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: 8,
}

function fmtNumber(value: unknown, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-'
}

function fmtPercent(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : '-'
}

function hasValues(section?: ModelBreakdownSection | null): boolean {
  return !!section && Object.values(section).some(value => value !== null && value !== undefined && value !== '')
}

function compactItems(items: Array<[string, string]>): Array<[string, string]> {
  return items.filter(([, value]) => value !== '-')
}

function MetricCard({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div style={{ padding: 8, borderRadius: 6, background: 'rgba(0, 0, 0, 0.16)' }}>
      <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</div>
      <div style={{ marginTop: 4, fontFamily: 'monospace', fontSize: 13, color: tone || 'var(--fg)' }}>{value}</div>
    </div>
  )
}

function BreakdownCard({ title, items }: { title: string; items: Array<[string, string]> }) {
  if (items.length === 0) return null
  return (
    <div style={{ padding: 10, borderRadius: 6, background: 'rgba(0, 0, 0, 0.16)' }}>
      <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      <div style={{ display: 'grid', gap: 4 }}>
        {items.map(([label, value]) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11 }}>
            <span style={{ color: 'var(--muted)' }}>{label}</span>
            <span style={{ fontFamily: 'monospace' }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function breakdownItems(breakdown: ModelBreakdown | null | undefined) {
  const cards: Array<{ title: string; items: Array<[string, string]> }> = []

  if (hasValues(breakdown?.classical)) {
    cards.push({
      title: 'Classical Models',
      items: compactItems([
        ['Anomaly', fmtNumber(breakdown?.classical?.anomaly_score)],
        ['Confidence', fmtNumber(breakdown?.classical?.model_confidence)],
        ['Isolation Forest', fmtNumber(breakdown?.classical?.isolation_forest)],
        ['LOF', fmtNumber(breakdown?.classical?.lof)],
        ['Ensemble', fmtNumber(breakdown?.classical?.ensemble)],
        ['Breakage Prob.', fmtNumber(breakdown?.classical?.breakage_probability)],
      ]),
    })
  }

  if (hasValues(breakdown?.harmonic)) {
    cards.push({
      title: 'Harmonic Model',
      items: compactItems([
        ['Score', fmtNumber(breakdown?.harmonic?.score)],
        ['Source', typeof breakdown?.harmonic?.source === 'string' ? breakdown.harmonic.source : '-'],
      ]),
    })
  }

  if (hasValues(breakdown?.stoppage)) {
    cards.push({
      title: 'Stoppage Model',
      items: compactItems([
        ['Probability', fmtNumber(breakdown?.stoppage?.probability)],
        ['ETA (s)', fmtNumber(breakdown?.stoppage?.eta_s, 1)],
        ['Label', typeof breakdown?.stoppage?.label === 'string' ? breakdown.stoppage.label : '-'],
      ]),
    })
  }

  if (hasValues(breakdown?.online)) {
    cards.push({
      title: 'Online Model',
      items: compactItems([
        ['Probability', fmtNumber(breakdown?.online?.probability)],
        ['Running', typeof breakdown?.online?.running === 'boolean' ? (breakdown.online.running ? 'yes' : 'no') : '-'],
      ]),
    })
  }

  return cards
}

export function SampleEvidencePanel({ sample }: { sample: SampleResult }) {
  const harmonicQ = useQuery<HarmonicExplainResponse>({
    queryKey: ['harmonic-explain', sample.memory_id, 8],
    queryFn: () => api(`/harmonic/explain/${encodeURIComponent(sample.memory_id || '')}?top_k=8`),
    enabled: !!sample.memory_id,
    staleTime: 60_000,
  })

  const breakdownCards = breakdownItems(sample.model_breakdown)
  const harmonic = harmonicQ.data

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {(sample.alert_line || sample.explanation) && (
        <div style={panelStyle}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Narrative</div>
          {sample.alert_line && (
            <div style={{ fontSize: 12, marginBottom: sample.explanation ? 8 : 0 }}>
              <span style={{ color: 'var(--muted)' }}>Alert line:</span>{' '}
              <span>{sample.alert_line}</span>
              {sample.alert_line_source && (
                <span style={{ color: 'var(--muted)' }}> ({sample.alert_line_source})</span>
              )}
            </div>
          )}
          {sample.explanation && (
            <div style={{ fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
              <span style={{ color: 'var(--muted)' }}>Explanation:</span>{' '}
              <span>{sample.explanation}</span>
              {sample.explanation_source && (
                <span style={{ color: 'var(--muted)' }}> ({sample.explanation_source})</span>
              )}
            </div>
          )}
        </div>
      )}

      <div style={panelStyle}>
        <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Supervised and Unsupervised Fusion</div>
        <div style={metricGridStyle}>
          <MetricCard label="Final Significance" value={fmtNumber(sample.significance_score)} tone={sample.significance_score >= 0.8 ? 'var(--danger)' : undefined} />
          <MetricCard label="Combined Model" value={fmtNumber(sample.combined_score)} />
          <MetricCard label="Supervised Score" value={fmtNumber(sample.supervised_score)} />
          <MetricCard label="Unsupervised Score" value={fmtNumber(sample.unsupervised_score)} />
          <MetricCard label="Supervised Weight" value={fmtPercent(sample.weight_supervised)} />
          <MetricCard label="Unsupervised Weight" value={fmtPercent(sample.weight_unsupervised)} />
          <MetricCard label="Pattern Score" value={fmtNumber(sample.pattern_rule_score)} />
          <MetricCard label="Prior Boost" value={fmtNumber(sample.prior_boost)} />
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
          Model evidence is separated from symbolic pattern matches so anomaly-model outputs do not masquerade as named patterns.
        </div>
      </div>

      {breakdownCards.length > 0 && (
        <div style={panelStyle}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Runtime Model Breakdown</div>
          <div style={metricGridStyle}>
            {breakdownCards.map(card => (
              <BreakdownCard key={card.title} title={card.title} items={card.items} />
            ))}
          </div>
        </div>
      )}

      <div style={panelStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', marginBottom: 8, flexWrap: 'wrap' }}>
          <div style={{ fontSize: 12, fontWeight: 700 }}>Harmonic Context Weights</div>
          {sample.memory_id && (
            <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>
              memory {sample.memory_id.slice(0, 12)}
            </div>
          )}
        </div>
        {!sample.memory_id ? (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            This sample has no stored memory link, so per-harmonic weights are unavailable.
          </div>
        ) : harmonicQ.isLoading ? (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>Loading harmonic explanation...</div>
        ) : harmonicQ.isError ? (
          <div style={{ fontSize: 11, color: 'var(--danger)' }}>
            Failed to load harmonic explanation: {harmonicQ.error.message}
          </div>
        ) : harmonic?.available ? (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={metricGridStyle}>
              <MetricCard label="Harmonic Score" value={fmtNumber(harmonic.score)} tone={typeof harmonic.score === 'number' && harmonic.score >= 0.7 ? 'var(--danger)' : undefined} />
              <MetricCard label="Dataset" value={harmonic.dataset || '-'} />
              <MetricCard label="Model Source" value={harmonic.model_source || '-'} />
              <MetricCard label="Top Features" value={String(harmonic.top_weighted.length)} />
            </div>
            {harmonic.context_weights.length > 0 && (
              <HarmonicWeightsChart
                weights={harmonic.context_weights}
                labels={harmonic.feature_labels}
                score={typeof harmonic.score === 'number' ? harmonic.score : undefined}
                height={140}
              />
            )}
            {harmonic.top_weighted.length > 0 && (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      <th style={{ textAlign: 'left', padding: '4px 6px' }}>Feature</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Weight</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Value</th>
                      <th style={{ textAlign: 'right', padding: '4px 6px' }}>Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {harmonic.top_weighted.map(row => (
                      <tr key={row.label} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.06)' }}>
                        <td style={{ padding: '4px 6px' }}>{row.label}</td>
                        <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace' }}>{fmtNumber(row.weight)}</td>
                        <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace' }}>{fmtNumber(row.value)}</td>
                        <td style={{ textAlign: 'right', padding: '4px 6px', fontFamily: 'monospace' }}>{fmtNumber(row.contribution)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            {harmonic?.reason || 'Harmonic explanation unavailable for this sample.'}
          </div>
        )}
      </div>
    </div>
  )
}
import React, { useState } from 'react'

import type { SignificantEventAlert } from '../state/alertsStore'
import {
  alertCuttingContextEntries,
  alertEvidenceBadges,
  alertHeadline,
  alertHistoryDetails,
  alertHistorySummary,
  alertIndicatorDetails,
  alertModelSourceExplanation,
  alertOperatorExplanation,
  alertPatternLabels,
  severityLabelFor,
  severityToneColor,
} from '../utils/alerts'
import { AlertContextChart } from './AlertContextChart'
import { DocLinksSection } from './DocLinksSection'
import { MemoryGraphLink } from './MemoryGraphLink'

function sevLabel(sev?: string) {
  if (sev === 'CRITICAL') return 'CRITICAL'
  if (sev === 'WARNING') return 'WARNING'
  return 'INFO'
}

function severityThresholdHint(sev?: string) {
  if (sev === 'CRITICAL') return '\u2265 0.85'
  if (sev === 'WARNING') return '\u2265 0.60'
  return '< 0.60'
}

function normalizeText(value: string | null | undefined) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function normalizeSource(value?: string | null) {
  return String(value || '').trim().toLowerCase()
}

function isLlmSource(value?: string | null) {
  const source = normalizeSource(value)
  return source === 'llm'
    || source.includes('groq')
    || source.includes('ollama')
    || source.includes('openai')
}

function sourcePillLabel(value: string | null | undefined, kind: 'summary' | 'explanation') {
  const source = normalizeSource(value)
  if (!source) return ''
  if (isLlmSource(source)) return `LLM ${kind}`
  if (source.includes('signature')) return `Rule ${kind}`
  if (source.includes('fallback')) return `Fallback ${kind}`
  if (source.includes('demo')) return `Template ${kind}`

  const readable = source.replace(/[_-]+/g, ' ').trim()
  return `${readable.charAt(0).toUpperCase()}${readable.slice(1)} ${kind}`
}

function sourcePillToneClass(value: string | null | undefined) {
  return isLlmSource(value) ? 'alertSourcePillLlm' : 'alertSourcePillFallback'
}

type AspectFeedback = (action: 'confirm' | 'dismiss', aspect: 'explanation' | 'recommendation') => void

/** Compact 👍/👎 rating for one facet of an alert (explanation | recommendation). */
function AspectRating({
  aspect,
  optional,
  pending,
  onFeedback,
}: {
  aspect: 'explanation' | 'recommendation'
  optional?: boolean
  pending?: boolean
  onFeedback: AspectFeedback
}) {
  return (
    <div className="alertAspectRating">
      <span className="alertAspectRatingLabel">
        {optional ? 'Helpful? (optional)' : 'Helpful?'}
      </span>
      <button
        type="button"
        className="alertAspectBtn"
        disabled={pending}
        title={`Mark the ${aspect} helpful`}
        onClick={() => onFeedback('confirm', aspect)}
      >
        👍
      </button>
      <button
        type="button"
        className="alertAspectBtn"
        disabled={pending}
        title={`Mark the ${aspect} not helpful`}
        onClick={() => onFeedback('dismiss', aspect)}
      >
        👎
      </button>
    </div>
  )
}

type AlertDetailContentProps = {
  alert: SignificantEventAlert
  isMuted?: boolean
  headerAction?: React.ReactNode
  controls?: React.ReactNode
  /** When provided, renders per-aspect (explanation / recommendation) feedback. */
  onAspectFeedback?: AspectFeedback
  aspectPending?: boolean
  /** Operator-focused layout: keep only headline + what's-happening + action up
   *  front; drop the meta pills and the redundant severity chip (they stay
   *  reachable via the collapsible "Why this score?" section). */
  compact?: boolean
}

export function AlertDetailContent({ alert, isMuted = false, headerAction, controls, onAspectFeedback, aspectPending, compact = false }: AlertDetailContentProps) {
  const headline = alertHeadline(alert)
  const explanation = alertOperatorExplanation(alert)
  const badges = alertEvidenceBadges(alert)
  const indicatorDetails = alertIndicatorDetails(alert)
  const historySummary = alertHistorySummary(alert)
  const historyDetails = alertHistoryDetails(alert)
  const modelSource = alertModelSourceExplanation(alert)
  const patterns = alertPatternLabels(alert)
  const contextEntries = alertCuttingContextEntries(alert)
  const docLinks = Array.isArray(alert.doc_links) ? alert.doc_links : []
  const supportingExplanation = explanation && explanation !== headline ? explanation : ''
  const modalExplanation = normalizeText(alert.explanation)
  const summarySourceLabel = sourcePillLabel(alert.summary_source, 'summary')
  const explanationSourceLabel = sourcePillLabel(alert.explanation_source, 'explanation')
  const contextLead = contextEntries[0]
    ? `${contextEntries[0].label}: ${contextEntries[0].value}`
    : ''
  const sevLabelStr = severityLabelFor(alert.severity, alert.significance?.score)
  const sevColor = severityToneColor(sevLabelStr)
  // Two-tier recommendation model: the immediate breakage-avoidance action,
  // shown as a distinct block from the explanation.
  const recommendation = normalizeText(alert.recommendation)

  // In compact (operator modal) mode the technical sections — score breakdown,
  // evidence, docs, history, signal context, controls — stay hidden behind a
  // single toggle. In full mode they render as before.
  const [techOpen, setTechOpen] = useState(false)
  const showTechnical = !compact || techOpen

  return (
    <>
      <div className="hrow" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div className="alertDetailEyebrow">
            <span style={{ color: sevColor, fontWeight: 700 }}>{sevLabelStr}</span>
            {alert.category ? ` · ${alert.category}` : ''}
            {alert.timestamp ? ` · ${new Date(alert.timestamp).toLocaleTimeString()}` : ''}
          </div>
          <div style={{ fontWeight: 750, fontSize: 18 }}>{headline}</div>
          {!compact && (summarySourceLabel || explanationSourceLabel) && (
            <div className="alertSourcePills">
              {summarySourceLabel && (
                <span className={`alertSourcePill ${sourcePillToneClass(alert.summary_source)}`}>
                  {summarySourceLabel}
                </span>
              )}
              {explanationSourceLabel && explanationSourceLabel !== summarySourceLabel && (
                <span className={`alertSourcePill ${sourcePillToneClass(alert.explanation_source)}`}>
                  {explanationSourceLabel}
                </span>
              )}
            </div>
          )}
        </div>
        {headerAction || null}
      </div>

      {(modalExplanation || supportingExplanation) && (
        <div className="alertAspectBlock">
          <div className="alertAspectHeader">
            <span className="alertAspectEyebrow">What's happening</span>
            {onAspectFeedback && (
              <AspectRating aspect="explanation" pending={aspectPending} onFeedback={onAspectFeedback} />
            )}
          </div>
          <div className="alertDetailLead">{modalExplanation || supportingExplanation}</div>
        </div>
      )}

      {recommendation && (
        <div className="alertAspectBlock alertRecommendation">
          <div className="alertAspectHeader">
            <span className="alertAspectEyebrow" style={{ color: 'var(--warning)' }}>Recommended action</span>
            {onAspectFeedback && (
              <AspectRating aspect="recommendation" optional pending={aspectPending} onFeedback={onAspectFeedback} />
            )}
          </div>
          <div className="alertDetailLead">{recommendation}</div>
        </div>
      )}

      {isMuted && (
        <div className="alertMutedBanner">
          This recurring signature is muted for the rest of the session.
        </div>
      )}

      {/* Traceability: this event's neighbourhood (patterns, feedback, context)
          in the memory graph. Always visible so any opened event is traceable. */}
      <div style={{ margin: '2px 0 12px' }}>
        <MemoryGraphLink
          memoryId={alert.event_id}
          label="View this event in the memory graph"
          style={{ color: 'var(--accent)', fontSize: 12, textDecoration: 'none' }}
        />
      </div>

      {compact && (
        <button
          type="button"
          className="alertTechToggle"
          aria-expanded={techOpen}
          onClick={() => setTechOpen((v) => !v)}
        >
          {techOpen ? '− Hide technical details' : '+ Show technical details (score, evidence, context)'}
        </button>
      )}

      {showTechnical && (
      <>
      <div className="alertDetailKpis" style={{ marginBottom: 12 }}>
        {!compact && alert.significance?.score != null && (
          <span className="alertMetaChip" title={`Severity bands: CRITICAL \u2265 0.85 · WARNING \u2265 0.60 · else INFO`}>
            <span style={{ color: sevColor, fontWeight: 700 }}>{sevLabelStr}</span>
          </span>
        )}
        {alert.recurrence && (alert.recurrence.occurrences ?? 0) > 1 && (
          <span
            className="alertMetaChip"
            title="One episode — a run of windows sharing this fault signature. Your confirm/dismiss adjudicates the whole episode once."
          >
            1 episode · {alert.recurrence.occurrences} windows
          </span>
        )}
        {isMuted && (
          <span className="alertMetaChip">Muted for session</span>
        )}
      </div>

      {alert.significance?.score != null && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Why this score?</span>
            <span className="alertDisclosureHint">
              {alert.significance.score.toFixed(3)} → {sevLabel(alert.severity)}
            </span>
          </summary>
          <div className="alertDisclosureBody">
            <div className="alertDetailText">
              <span title={`Severity bands: CRITICAL \u2265 0.85 · WARNING \u2265 0.60 · else INFO`}>
                Score: {alert.significance.score.toFixed(3)} ({sevLabel(alert.severity)} {severityThresholdHint(alert.severity)})
              </span>
              {alert.metrics && typeof (alert.metrics as Record<string, unknown>).anomaly_detector_score === 'number' && (
                <span title="Classical fault-model score fed into rule fusion as external signal">
                  {' · Classical fault model: '}{((alert.metrics as Record<string, number>).anomaly_detector_score).toFixed(3)}
                </span>
              )}
              {alert.significance.prior_factor != null ? (
                <span title="Historical pattern prior multiplier (×1 = neutral, >1 boosts, <1 dampens)">
                  {' · Prior ×'}{alert.significance.prior_factor.toFixed(2)}
                </span>
              ) : (
                alert.significance.prior_boost != null && alert.significance.prior_boost > 0 && (
                  <> · Prior boost: +{alert.significance.prior_boost.toFixed(3)}</>
                )
              )}
            </div>
            {modelSource && (
              <div className="alertDetailText" style={{ marginTop: 10 }}>
                Model context: {modelSource}
              </div>
            )}
            {alert.significance?.score_trace && alert.significance.score_trace.length > 0 && (
              <div className="alertScoreTrace" style={{ marginTop: 10 }}>
                {alert.significance.score_trace.map((entry, index) => (
                  <div key={`${entry.component}-${index}`} className="alertScoreTraceRow">
                    <span className="alertScoreTraceValue">
                      {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(3)}
                    </span>
                    <span>{entry.component}</span>
                    {entry.source && <span style={{ opacity: 0.7 }}>({entry.source})</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>
      )}

      {(badges.length > 0 || patterns.length > 0 || indicatorDetails.length > 0) && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Evidence</span>
            <span className="alertDisclosureHint">
              {indicatorDetails.length > 0
                ? `${indicatorDetails.length} indicators`
                : `${badges.length + patterns.length} items`}
            </span>
          </summary>
          <div className="alertDisclosureBody">
            {indicatorDetails.length > 0 && (
              <ul className="alertDetailList">
                {indicatorDetails.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
            )}
            {(badges.length > 0 || patterns.length > 0) && (
              <div className="alertStripBadges" style={{ marginTop: indicatorDetails.length > 0 ? 10 : 0 }}>
                {badges.map((badge) => (
                  <span key={badge} className="alertStripBadge">{badge}</span>
                ))}
                {patterns.map((pattern) => (
                  <span key={pattern} className="alertStripBadge alertStripBadgePattern">{pattern}</span>
                ))}
              </div>
            )}
          </div>
        </details>
      )}

      {docLinks.length > 0 && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Documentation</span>
            <span className="alertDisclosureHint">{`${docLinks.length} citations`}</span>
          </summary>
          <div className="alertDisclosureBody">
            <div className="alertDetailText" style={{ marginBottom: 10 }}>
              Persisted document matches proposed for this alert.
            </div>
            <DocLinksSection docLinks={docLinks} limit={5} memoryId={alert.event_id} />
          </div>
        </details>
      )}

      {(historySummary || historyDetails.length > 0) && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Similar past events</span>
            <span className="alertDisclosureHint">
              {historyDetails.length > 0 ? `${historyDetails.length} notes` : 'History available'}
            </span>
          </summary>
          <div className="alertDisclosureBody">
            {historySummary && (
              <div className="alertDetailText">{historySummary}</div>
            )}
            {historyDetails.length > 0 && (
              <ul className="alertDetailList" style={{ marginTop: historySummary ? 10 : 0 }}>
                {historyDetails.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
            )}
          </div>
        </details>
      )}

      {contextEntries.length > 0 && (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Signal context</span>
            <span className="alertDisclosureHint">{contextLead || `${contextEntries.length} fields`}</span>
          </summary>
          <div className="alertDisclosureBody">
            <div className="alertStripContextGrid" style={{ marginTop: 0 }}>
              {contextEntries.map((entry) => (
                <div key={entry.label} className="alertStripContextCard">
                  <div className="alertStripContextLabel">{entry.label}</div>
                  <div className="alertStripContextValue">{entry.value}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 12 }}>
              <AlertContextChart alert={alert} compact />
            </div>
          </div>
        </details>
      )}

      {controls ? (
        <details className="alertDisclosure">
          <summary className="alertDisclosureSummary">
            <span>Operator controls</span>
            <span className="alertDisclosureHint">Mute or rate the explanation</span>
          </summary>
          <div className="alertDisclosureBody">{controls}</div>
        </details>
      ) : null}
      </>
      )}
    </>
  )
}
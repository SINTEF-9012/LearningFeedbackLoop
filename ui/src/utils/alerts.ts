import type { SignificantEventAlert } from '../state/alertsStore'
import { humanPatterns, humanReason } from './patternNames'

export type AlertContextEntry = {
  label: string
  value: string
}

export type AlertHistoryEntry = NonNullable<SignificantEventAlert['similar_history']>[number]

function normalizeText(value: string | null | undefined): string {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function sentenceSlice(value: string, maxSentences = 2): string {
  const text = normalizeText(value)
  if (!text) return ''
  const matches = text.match(/[^.!?]+[.!?]?/g)?.map((part) => normalizeText(part)).filter(Boolean) || []
  return matches.length > 0 ? matches.slice(0, maxSentences).join(' ') : text
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function numberField(record: Record<string, unknown> | null, key: string): number | null {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringField(record: Record<string, unknown> | null, key: string): string {
  const value = record?.[key]
  return typeof value === 'string' ? value.trim() : ''
}

function scoreTrace(alert: SignificantEventAlert): Array<Record<string, unknown>> {
  const significance = asRecord(alert.significance)
  const trace = significance?.score_trace
  if (!Array.isArray(trace)) return []
  return trace.map(asRecord).filter((entry): entry is Record<string, unknown> => Boolean(entry))
}

function findTraceEntry(alert: SignificantEventAlert, component: string): Record<string, unknown> | null {
  return scoreTrace(alert).find((entry) => stringField(entry, 'component') === component) ?? null
}

function titleCase(value: string): string {
  return value
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatFixed(value: number, digits = 0): string {
  if (Number.isInteger(value)) return value.toFixed(0)
  return value.toFixed(digits)
}

function similarHistory(alert: SignificantEventAlert): AlertHistoryEntry[] {
  return Array.isArray(alert.similar_history) ? alert.similar_history.filter(Boolean) : []
}

function formatHistoricalLabel(value: string): string {
  return value.replace(/[_-]+/g, ' ').trim()
}

/**
 * Severity label from an explicit server severity, else a score-band fallback.
 * Bands mirror the operator UI: CRITICAL ≥ 0.9, WARNING ≥ 0.75, else INFO.
 */
export function severityLabelFor(severity?: string | null, score?: number): 'CRITICAL' | 'WARNING' | 'INFO' {
  const explicit = String(severity || '').trim().toUpperCase()
  if (explicit === 'CRITICAL' || explicit === 'WARNING' || explicit === 'INFO') return explicit
  const value = typeof score === 'number' && Number.isFinite(score) ? score : 0
  if (value >= 0.9) return 'CRITICAL'
  if (value >= 0.75) return 'WARNING'
  return 'INFO'
}

/** Single source of truth for severity → colour (CSS var). */
export function severityToneColor(severityLabel?: string | null): string {
  const label = String(severityLabel || '').trim().toUpperCase()
  if (label === 'CRITICAL') return 'var(--danger)'
  if (label === 'WARNING') return 'var(--warning)'
  return 'var(--muted)'
}

export function alertHeadline(alert: SignificantEventAlert): string {
  if (
    alert.primary_observation_label
    && typeof alert.indicators_present === 'number'
    && typeof alert.indicators_required === 'number'
    && alert.indicators_required > 0
  ) {
    return `Possible ${alert.primary_observation_label} — ${alert.indicators_present}/${alert.indicators_required} indicators present`
  }
  const summary = normalizeText(alert.summary)
  if (summary) return summary
  const reasons = Array.isArray(alert.significance?.reasons) ? alert.significance.reasons : []
  if (reasons.length > 0) {
    const first = humanReason(String(reasons[0] || ''))
    if (first) return first
  }
  return humanPatterns(alert.patterns || []).join(', ') || 'Anomaly detected'
}

export function alertOperatorExplanation(alert: SignificantEventAlert): string {
  const explanation = sentenceSlice(String(alert.explanation || ''))
  if (explanation) return explanation
  const reasons = Array.isArray(alert.significance?.reasons) ? alert.significance.reasons : []
  const rewritten = reasons
    .map((reason) => sentenceSlice(humanReason(String(reason || '')), 1))
    .filter(Boolean)
    .slice(0, 2)
  return rewritten.join(' ')
}

export function alertPatternLabels(alert: SignificantEventAlert, maxCount = 3): string[] {
  return humanPatterns(alert.patterns || []).slice(0, maxCount)
}

export function alertIndicatorDetails(alert: SignificantEventAlert, maxCount = 4): string[] {
  if (!Array.isArray(alert.indicator_details) || alert.indicator_details.length === 0) return []

  return alert.indicator_details
    .map((detail) => normalizeText(String(detail?.label || humanPatterns([String(detail?.key || '')])[0] || '')))
    .filter(Boolean)
    .slice(0, maxCount)
}

export function alertModelSourceLabel(alert: SignificantEventAlert, includeConfidence = false): string {
  const metrics = asRecord(alert.metrics)
  const raw = stringField(metrics, 'model_source') || stringField(metrics, 'harmonic_context_source')
  const label = raw ? titleCase(raw) : ''
  if (!includeConfidence) return label

  const confidence = numberField(metrics, 'model_confidence')
  if (!label) return confidence !== null ? `Confidence ${confidence.toFixed(2)}` : ''
  return confidence !== null ? `${label} (${confidence.toFixed(2)})` : label
}

export function alertModelSourceExplanation(alert: SignificantEventAlert): string {
  const label = alertModelSourceLabel(alert)
  const metrics = asRecord(alert.metrics)
  const confidence = numberField(metrics, 'model_confidence')
  if (!label) return confidence !== null ? `Model confidence ${confidence.toFixed(2)}` : ''
  if (confidence !== null) return `${label} (confidence ${confidence.toFixed(2)})`
  return label
}

export function alertHistoryBadge(alert: SignificantEventAlert): string {
  const history = similarHistory(alert)
  if (history.length > 0) return `${history.length} similar`

  const evidenceCount = typeof alert.significance?.prior_evidence_count === 'number'
    ? alert.significance.prior_evidence_count
    : null
  if (evidenceCount !== null && evidenceCount > 0) return `History x${evidenceCount}`

  const similarCount = Array.isArray(alert.similar_memories) ? alert.similar_memories.filter(Boolean).length : 0
  if (similarCount > 0) return `${similarCount} similar`

  return ''
}

export function alertHistorySummary(alert: SignificantEventAlert): string {
  const history = similarHistory(alert)
  if (history.length > 0) {
    const labelled = history.filter((entry) => typeof entry.label === 'string' && entry.label.trim())
    const mostSupported = history.reduce((best, entry) => {
      const confirmCount = typeof entry.feedback?.confirm_count === 'number' ? entry.feedback.confirm_count : 0
      return confirmCount > best ? confirmCount : best
    }, 0)

    if (labelled.length > 0) {
      const rendered = formatHistoricalLabel(String(labelled[0].label || ''))
      return `${history.length} similar past events found. Historical labels include "${rendered}".`
    }

    if (mostSupported > 0) {
      return `${history.length} similar past events found. At least one was previously confirmed by an operator.`
    }

    return `${history.length} similar past events found.`
  }

  const evidenceCount = typeof alert.significance?.prior_evidence_count === 'number'
    ? alert.significance.prior_evidence_count
    : null
  const damping = typeof alert.significance?.prior_damping_factor === 'number'
    ? alert.significance.prior_damping_factor
    : null

  if (evidenceCount !== null && evidenceCount > 0) {
    return damping !== null && damping < 0.999
      ? `Historical prior based on ${evidenceCount} reviewed events (damped ${damping.toFixed(2)}).`
      : `Historical prior based on ${evidenceCount} reviewed events.`
  }

  const similarCount = Array.isArray(alert.similar_memories) ? alert.similar_memories.filter(Boolean).length : 0
  if (similarCount > 0) return `${similarCount} similar past events found.`

  return ''
}

export function alertHistoryDetails(alert: SignificantEventAlert, maxCount = 2): string[] {
  return similarHistory(alert)
    .slice(0, maxCount)
    .map((entry) => {
      const parts: string[] = []
      const shared = Array.isArray(entry.shared_pattern_keys) ? humanPatterns(entry.shared_pattern_keys).slice(0, 2) : []
      if (shared.length > 0) parts.push(`Shared observations: ${shared.join(', ')}`)

      if (typeof entry.label === 'string' && entry.label.trim()) {
        parts.push(`Historical label: "${formatHistoricalLabel(entry.label)}"`)
      }

      const lastAction = typeof entry.feedback?.last_action === 'string' ? entry.feedback.last_action : ''
      const lastComment = typeof entry.feedback?.last_comment === 'string' ? entry.feedback.last_comment.trim() : ''
      if (lastAction) {
        const actionLabel = lastAction === 'confirm'
          ? 'Past review confirmed'
          : lastAction === 'dismiss'
            ? 'Past review dismissed'
            : `Past review: ${lastAction}`
        parts.push(lastComment ? `${actionLabel} (${lastComment})` : actionLabel)
      }

      const strongest = Array.isArray(entry.shared_pattern_details)
        ? entry.shared_pattern_details.find((detail) => typeof detail?.candidate_strength === 'number')
        : null
      if (strongest && typeof strongest.candidate_strength === 'number') {
        parts.push(`Past observation strength ${strongest.candidate_strength.toFixed(2)}`)
      }

      return parts.join(' • ')
    })
    .filter(Boolean)
}

export function alertEvidenceBadges(alert: SignificantEventAlert): string[] {
  const metrics = asRecord(alert.metrics)
  const badges: string[] = []
  const modelParts: string[] = []
  const modelSource = alertModelSourceLabel(alert)

  if (alert.persistence_label === 'candidate') badges.push('Candidate')
  if (alert.persistence_label === 'recurring') badges.push('Recurring')

  const anomalyScore = numberField(metrics, 'anomaly_detector_score')
  if (anomalyScore !== null) modelParts.push(`Model ${anomalyScore.toFixed(2)}`)

  const modelConfidence = numberField(metrics, 'model_confidence')
  if (modelConfidence !== null) modelParts.push(`Conf ${modelConfidence.toFixed(2)}`)

  const breakagePrediction = numberField(metrics, 'breakage_prediction')
  if (breakagePrediction !== null) modelParts.push(`Risk ${breakagePrediction.toFixed(2)}`)

  if (modelSource) badges.push(`Source ${titleCase(modelSource)}`)
  if (modelParts.length > 0) badges.push(modelParts.join(', '))

  const trustEntry = findTraceEntry(alert, 'model_trust')
  const trustValue = numberField(trustEntry, 'value')
  if (trustValue !== null && trustValue < 0.999) {
    badges.push(`Trust ${trustValue.toFixed(2)}`)
  }

  const historyBadge = alertHistoryBadge(alert)
  if (historyBadge) badges.push(historyBadge)

  const priorBoost = typeof alert.significance?.prior_boost === 'number' && Number.isFinite(alert.significance.prior_boost)
    ? alert.significance.prior_boost
    : null
  if (priorBoost !== null && Math.abs(priorBoost) >= 0.005) {
    badges.push(`History boost ${priorBoost >= 0 ? '+' : ''}${priorBoost.toFixed(2)}`)
  }

  const suppressionEntry = findTraceEntry(alert, 'suppression_pattern_match')
  const suppressionValue = numberField(suppressionEntry, 'value')
  if (suppressionValue !== null && suppressionValue < 0) {
    badges.push(`Suppression ${suppressionValue.toFixed(2)}`)
  }

  const protectiveEntry = findTraceEntry(alert, 'protective_pattern_match')
  const protectiveValue = numberField(protectiveEntry, 'value')
  if (protectiveValue !== null && protectiveValue < 0) {
    badges.push(`Protective ${protectiveValue.toFixed(2)}`)
  }

  return badges
}

export function alertCuttingContextEntries(alert: SignificantEventAlert): AlertContextEntry[] {
  const context = asRecord(alert.context)
  if (!context) return []

  const entries: AlertContextEntry[] = []
  const groundTruth = stringField(context, 'ground_truth_label')
  if (groundTruth) entries.push({ label: 'Ground truth', value: titleCase(groundTruth) })

  const regime = stringField(context, 'operating_regime')
  // Skip a meaningless "unknown" — the machine reports a mode we can't map to a
  // known regime, so showing "Unknown" is noise rather than information.
  if (regime && !['unknown', 'unspecified', 'none', 'n/a'].includes(regime.trim().toLowerCase())) {
    entries.push({ label: 'Regime', value: titleCase(regime) })
  }

  const material = stringField(context, 'workpiece_material')
  if (material) entries.push({ label: 'Material', value: material })

  const tool = stringField(context, 'tool_type')
  if (tool) entries.push({ label: 'Tool', value: titleCase(tool) })

  const spindle = numberField(context, 'spindle_speed')
  if (spindle !== null) entries.push({ label: 'Spindle', value: `${formatFixed(spindle, 0)} RPM` })

  const feed = numberField(context, 'feed_rate')
  if (feed !== null) entries.push({ label: 'Feed', value: `${formatFixed(feed, 0)} mm/min` })

  const axialDepth = numberField(context, 'axial_depth')
  const radialDepth = numberField(context, 'radial_depth')
  if (axialDepth !== null || radialDepth !== null) {
    const parts = [
      axialDepth !== null ? `${formatFixed(axialDepth, 2)} mm axial` : '',
      radialDepth !== null ? `${formatFixed(radialDepth, 2)} mm radial` : '',
    ].filter(Boolean)
    if (parts.length > 0) entries.push({ label: 'Depth', value: parts.join(' / ') })
  }

  const coolant = stringField(context, 'coolant')
  if (coolant) entries.push({ label: 'Coolant', value: titleCase(coolant) })

  return entries.slice(0, 6)
}
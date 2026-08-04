import React from 'react'

import type { SignificantEventAlert } from '../state/alertsStore'
import { alertHeadline, alertHistoryBadge, alertOperatorExplanation, severityLabelFor, severityToneColor } from '../utils/alerts'
import { MemoryGraphLink } from './MemoryGraphLink'

function fmtTime(ts?: string, receivedAt?: number): string {
  if (ts) {
    const date = new Date(ts)
    if (!Number.isNaN(date.getTime())) return date.toLocaleTimeString()
  }
  if (receivedAt) return new Date(receivedAt).toLocaleTimeString()
  return ''
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

function truncateLine(value: string, maxChars = 140) {
  const text = value.replace(/\s+/g, ' ').trim()
  if (!text) return ''
  if (text.length <= maxChars) return text
  return `${text.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`
}

type StripProps = {
  variant: 'strip'
  alert: SignificantEventAlert
  actions?: React.ReactNode
}

type ListProps = {
  variant: 'list'
  alert: SignificantEventAlert
  onClick: () => void
  selected?: boolean
  unread?: boolean
}

type MiniProps = {
  variant: 'mini'
  alert: SignificantEventAlert
  onClick: () => void
  selected?: boolean
  unread?: boolean
}

type AlertSummaryCardProps = StripProps | ListProps | MiniProps

export function AlertSummaryCard(props: AlertSummaryCardProps) {
  const { alert } = props
  const scoreNum = typeof alert.significance?.score === 'number' ? alert.significance.score : undefined
  const severityLabel = severityLabelFor(alert.severity, scoreNum)
  const severityColor = severityToneColor(severityLabel)
  const headline = alertHeadline(alert)
  const explanation = alertOperatorExplanation(alert)
  const llmSupportLine = isLlmSource(alert.explanation_source) && explanation && explanation !== headline
    ? truncateLine(explanation)
    : ''
  const historyBadge = alertHistoryBadge(alert)
  const score = typeof scoreNum === 'number' ? scoreNum.toFixed(2) : ''
  const category = alert.category || ''
  const summarySource = typeof alert.summary_source === 'string' ? alert.summary_source.trim() : ''
  const time = fmtTime(alert.timestamp, alert._received_at)

  if (props.variant === 'strip') {
    return (
      <div
        className="alertStrip alertStripCompact"
        style={{ borderLeftColor: severityColor }}
      >
        <div className="alertStripMain">
          <div className="alertStripCompactTop">
            <div className="alertStripHeader">
              <span
                className="alertStripSevPill"
                style={{
                  color: severityColor,
                  borderColor: severityColor,
                }}
              >
                {severityLabel}
              </span>
            </div>
            <div className="alertStripCompactMeta">
              <span className="alertStripTime">{time}</span>
            </div>
          </div>
          <div className="alertStripBody">{headline}</div>
          {llmSupportLine && (
            <div className="alertStripSummaryLine">{llmSupportLine}</div>
          )}
          <MemoryGraphLink
            memoryId={alert.event_id}
            style={{ fontSize: 11, color: severityColor, textDecoration: 'none', marginTop: 2, display: 'inline-block' }}
          />
        </div>
        {props.actions ? (
          <div className="alertStripActions alertStripActionsCompact">{props.actions}</div>
        ) : null}
      </div>
    )
  }

  if (props.variant === 'mini') {
    return (
      <div
        className={`operatorAlertItem ${props.selected ? 'active' : ''} ${props.unread ? 'unread' : ''}`}
        onClick={props.onClick}
      >
        <span className="miniSev" style={{ background: severityColor }} />
        <span className="operatorAlertText">{headline}</span>
        <span style={{ fontSize: 10, color: 'var(--muted)', whiteSpace: 'nowrap' }}>{time}</span>
      </div>
    )
  }

  return (
    <button
      className={`alertItem${props.unread ? ' unread' : ''}${props.selected ? ' selected' : ''}`}
      onClick={props.onClick}
      title={headline}
    >
      <div style={{ fontWeight: 650, lineHeight: 1.35 }}>{headline}</div>
      <div className="hrow" style={{ justifyContent: 'space-between', marginTop: 4 }}>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          <span className="badge" style={{ color: severityColor, borderColor: 'rgba(255,255,255,0.10)' }}>{severityLabel}</span>
          {category ? <span className="badge">{category}</span> : null}
          {score ? <span className="badge">{score}</span> : null}
          {historyBadge ? <span className="badge">{historyBadge}</span> : null}
          {summarySource ? (
            <span className="badge" style={{ opacity: 0.7 }}>
              {summarySource}
            </span>
          ) : null}
        </div>
        <div className="small" style={{ whiteSpace: 'nowrap' }}>{time}</div>
      </div>
    </button>
  )
}
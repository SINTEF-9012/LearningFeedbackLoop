import React from 'react'

import { AlertSummaryCard } from './AlertSummaryCard'
import type { SignificantEventAlert } from '../state/alertsStore'

type Props = {
  alerts: SignificantEventAlert[]
  selectedMemoryId?: string
  onSelectMemoryId: (memoryId: string) => void
  onClear?: () => void
  onMarkAllRead?: () => void
  flash?: boolean
}

export function AlertsPanel({ alerts, selectedMemoryId, onSelectMemoryId, onClear, onMarkAllRead, flash }: Props) {
  const unread = alerts.reduce((acc, a) => acc + (a._unread ? 1 : 0), 0)

  return (
    <div>
      <div className={`hrow${flash ? ' flashHeader' : ''}`} style={{ justifyContent: 'space-between' }}>
        <div style={{ fontWeight: 700 }}>Alerts {unread > 0 ? <span className="badge" style={{ color: 'var(--danger)' }}>{unread} new</span> : null}</div>
        <div className="hrow">
          {onMarkAllRead && (
            <button onClick={onMarkAllRead} disabled={alerts.length === 0}>
              Mark read
            </button>
          )}
          {onClear && (
            <button onClick={onClear} disabled={alerts.length === 0}>
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="small">Click an alert to inspect &amp; review the memory.</div>

      <div className="hr" />

      {alerts.length === 0 ? (
        <div className="small">No alerts yet.</div>
      ) : (
        <div className="alertList">
          {[...alerts].reverse().map((a) => {
            const memoryId = a.event_id
            const isSelected = Boolean(selectedMemoryId && memoryId === selectedMemoryId)

            return (
              <AlertSummaryCard
                key={`${a.event_id}-${a._received_at || a.timestamp || ''}`}
                variant="list"
                alert={a}
                selected={isSelected}
                unread={Boolean(a._unread)}
                onClick={() => onSelectMemoryId(memoryId)}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

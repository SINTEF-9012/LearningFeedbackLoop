/**
 * Tooltip — Reusable tooltip component for UI help text.
 *
 * Wraps any element and shows a styled tooltip on hover.
 * Uses pure CSS positioning (no portal) for simplicity.
 */
import React, { type CSSProperties, type ReactNode } from 'react'

interface Props {
  text: string
  children: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  maxWidth?: number
}

const baseStyle: CSSProperties = {
  position: 'relative',
  display: 'inline-flex',
  alignItems: 'center',
}

export function Tooltip({ text, children, position = 'top', maxWidth = 300 }: Props) {
  return (
    <span className={`ui-tooltip ui-tooltip-${position}`} style={baseStyle} data-tip={text}>
      {children}
      <span
        className="ui-tooltip-text"
        style={{ maxWidth }}
      >
        {text}
      </span>
    </span>
  )
}

/** Small circled "?" icon that shows a tooltip on hover. */
export function HelpIcon({ text, position = 'top', maxWidth = 320 }: { text: string; position?: Props['position']; maxWidth?: number }) {
  return (
    <Tooltip text={text} position={position} maxWidth={maxWidth}>
      <span
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 15, height: 15, borderRadius: '50%', fontSize: 9, fontWeight: 700,
          background: 'rgba(122,162,247,0.15)', color: 'var(--accent, #7aa2f7)',
          cursor: 'help', marginLeft: 4, flexShrink: 0,
          border: '1px solid rgba(122,162,247,0.3)',
        }}
      >
        ?
      </span>
    </Tooltip>
  )
}

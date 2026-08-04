import React from 'react'

export type ConfusionMatrix = { tp: number; fp: number; tn: number; fn: number }

/**
 * Renders a 2×2 confusion‐matrix heatmap with TP / FP / FN / TN cells.
 */
export function ConfusionHeatmap({ cm, label }: { cm: ConfusionMatrix; label: string }) {
  const cells = [
    { v: cm.tp, lbl: 'TP', bg: 'rgba(158,206,106,0.45)' },
    { v: cm.fp, lbl: 'FP', bg: 'rgba(247,118,142,0.35)' },
    { v: cm.fn, lbl: 'FN', bg: 'rgba(247,118,142,0.25)' },
    { v: cm.tn, lbl: 'TN', bg: 'rgba(122,162,247,0.35)' },
  ]
  return (
    <div style={{ display: 'inline-grid', gap: 8 }}>
      <div className="small" style={{ fontWeight: 600 }}>{label}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '60px 60px', gridTemplateRows: '48px 48px', gap: 2 }}>
        {cells.map(({ v, lbl, bg }) => (
          <div key={lbl} style={{ background: bg, borderRadius: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', fontVariantNumeric: 'tabular-nums' }}>
            <span style={{ fontSize: 18, fontWeight: 700 }}>{v ?? 0}</span>
            <span className="small" style={{ color: 'var(--muted)', fontSize: 10 }}>{lbl}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

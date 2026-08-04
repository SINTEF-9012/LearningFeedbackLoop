/**
 * DemoDirector — a presenter-only floating control for the video/presentation
 * demo. Fires curated scripted events (backend /demo-director) into the active
 * session so the sequence (event A → feedback → similar event B → …) can be
 * driven deterministically on stage.
 *
 * Hidden by default; toggle with Ctrl+Shift+D. Never part of the operator story.
 */

import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { api } from '../api/http'
import { useAppContext } from '../contexts/AppContext'
import { colors, fontSize, radii, shadows, spacing } from '../styles/tokens'

interface DemoEvent { key: string; file: string; label: string }
interface FireResult { event: string; label?: string; significant?: boolean; action?: string | null; memory_id?: string | null }
interface SeedResult { requested: number; fired: number; confirmed: number; dismissed: number; skipped: number }

const STORAGE_KEY = 'demoDirector'
const SEED_COUNTS = [12, 24, 48]
const SEED_RATES = [0.6, 0.75, 0.9]

export function DemoDirector() {
  const ctx = useAppContext()
  const [visible, setVisible] = useState(() => (typeof window !== 'undefined' && localStorage.getItem(STORAGE_KEY) === '1'))
  const [events, setEvents] = useState<DemoEvent[]>([])
  const [collapsed, setCollapsed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [seedCount, setSeedCount] = useState(24)
  const [seedRate, setSeedRate] = useState(0.75)
  const [fillCount, setFillCount] = useState(5)
  const [last, setLast] = useState<{ kind: 'ok' | 'err' | 'info'; text: string } | null>(null)

  // Ctrl+Shift+D toggles the panel (and persists the choice).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
        e.preventDefault()
        setVisible((v) => {
          const nv = !v
          localStorage.setItem(STORAGE_KEY, nv ? '1' : '0')
          return nv
        })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!visible || events.length) return
    api<{ events: DemoEvent[] }>('/demo-director/events')
      .then((r) => setEvents(r.events || []))
      .catch(() => setLast({ kind: 'err', text: 'demo endpoint unavailable' }))
  }, [visible, events.length])

  if (!visible) return null

  const sessionId = ctx.streamSessionId

  const fire = async (ev: DemoEvent) => {
    if (busy) return
    if (!sessionId) {
      setLast({ kind: 'err', text: 'No active session — select one first.' })
      return
    }
    setBusy(true)
    setLast({ kind: 'info', text: `Firing ${ev.label}…` })
    try {
      const r = await api<FireResult>('/demo-director/fire-event', 'POST', { session_id: sessionId, event: ev.key })
      setLast({
        kind: 'ok',
        text: `${ev.label} → ${r.significant ? 'significant' : 'not significant'}${r.action ? ` (${r.action})` : ''}`,
      })
    } catch (e) {
      setLast({ kind: 'err', text: `failed: ${String(e)}` })
    } finally {
      setBusy(false)
    }
  }

  const fill = async () => {
    if (busy) return
    if (!sessionId) {
      setLast({ kind: 'err', text: 'No active session — select one first.' })
      return
    }
    setBusy(true)
    setLast({ kind: 'info', text: `Firing ${fillCount} background events…` })
    try {
      const r = await api<SeedResult>('/demo-director/fill', 'POST', {
        session_id: sessionId,
        count: fillCount,
        confirm_rate: 0.7,
      })
      setLast({
        kind: 'ok',
        text: `Background: ${r.fired} events · ${r.confirmed}✓ / ${r.dismissed}✗${r.skipped ? ` · ${r.skipped} skipped` : ''}`,
      })
    } catch (e) {
      setLast({ kind: 'err', text: `fill failed: ${String(e)}` })
    } finally {
      setBusy(false)
    }
  }

  const seed = async () => {
    if (busy) return
    if (!sessionId) {
      setLast({ kind: 'err', text: 'No active session — select one first.' })
      return
    }
    setBusy(true)
    setLast({ kind: 'info', text: `Seeding ${seedCount} events + feedback…` })
    try {
      const r = await api<SeedResult>('/demo-director/seed', 'POST', {
        session_id: sessionId,
        count: seedCount,
        confirm_rate: seedRate,
      })
      setLast({
        kind: 'ok',
        text: `Seeded ${r.fired} events · ${r.confirmed}✓ / ${r.dismissed}✗${r.skipped ? ` · ${r.skipped} skipped` : ''}`,
      })
    } catch (e) {
      setLast({ kind: 'err', text: `seed failed: ${String(e)}` })
    } finally {
      setBusy(false)
    }
  }

  const pillStyle = (active: boolean): CSSProperties => ({
    background: active ? colors.accent : 'transparent',
    border: `1px solid ${active ? colors.accent : colors.border}`,
    borderRadius: radii.sm,
    color: active ? colors.bg : colors.textMuted,
    cursor: 'pointer',
    fontSize: fontSize.xs,
    fontWeight: active ? 700 : 400,
    padding: '1px 8px',
  })

  return (
    <div
      style={{
        position: 'fixed',
        left: spacing.lg,
        bottom: spacing.lg,
        zIndex: 300,
        width: collapsed ? 'auto' : 320,
        background: colors.surface,
        border: `1px solid ${colors.accent}`,
        borderRadius: radii.lg,
        boxShadow: shadows.raised,
        padding: spacing.md,
        fontSize: fontSize.sm,
      }}
    >
      <div style={{ alignItems: 'center', display: 'flex', justifyContent: 'space-between', gap: spacing.sm }}>
        <span style={{ color: colors.accent, fontWeight: 700 }}>🎬 Demo Director</span>
        <div style={{ display: 'flex', gap: spacing.xs }}>
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? 'Expand' : 'Collapse'}
            style={{ background: 'transparent', border: `1px solid ${colors.border}`, borderRadius: radii.sm, color: colors.textMuted, cursor: 'pointer', fontSize: fontSize.xs, padding: '1px 8px' }}
          >
            {collapsed ? '▸' : '▾'}
          </button>
          <button
            type="button"
            onClick={() => { localStorage.setItem(STORAGE_KEY, '0'); setVisible(false) }}
            title="Hide (Ctrl+Shift+D to reopen)"
            style={{ background: 'transparent', border: `1px solid ${colors.border}`, borderRadius: radii.sm, color: colors.textMuted, cursor: 'pointer', fontSize: fontSize.xs, padding: '1px 8px' }}
          >
            ✕
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <div style={{ color: colors.textMuted, fontSize: fontSize.xs, margin: `${spacing.xs}px 0 ${spacing.sm}px` }}>
            {sessionId ? `session ${sessionId}` : 'no active session'}
          </div>
          <div style={{ display: 'grid', gap: spacing.xs }}>
            {events.length === 0 ? (
              <div style={{ color: colors.textMuted, fontSize: fontSize.xs }}>Loading events…</div>
            ) : (
              events.map((ev) => (
                <button
                  key={ev.key}
                  type="button"
                  disabled={busy || !sessionId}
                  onClick={() => fire(ev)}
                  style={{
                    background: colors.surfaceAlt,
                    border: `1px solid ${colors.border}`,
                    borderRadius: radii.sm,
                    color: sessionId ? colors.text : colors.textDim,
                    cursor: busy || !sessionId ? 'not-allowed' : 'pointer',
                    fontSize: fontSize.xs,
                    padding: '6px 10px',
                    textAlign: 'left',
                  }}
                >
                  {ev.label}
                </button>
              ))
            )}
          </div>

          {/* Background filler — several synthetic events between A and B */}
          <div style={{ borderTop: `1px solid ${colors.border}`, marginTop: spacing.sm, paddingTop: spacing.sm }}>
            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginBottom: spacing.xs }}>
              Between A and B — background events (not chatter)
            </div>
            <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.xs }}>
              {[3, 5, 8].map((c) => (
                <button key={c} type="button" onClick={() => setFillCount(c)} style={pillStyle(fillCount === c)}>{c}</button>
              ))}
              <button
                type="button"
                disabled={busy || !sessionId}
                onClick={fill}
                style={{
                  background: colors.surfaceAlt,
                  border: `1px solid ${colors.border}`,
                  borderRadius: radii.sm,
                  color: sessionId ? colors.text : colors.textDim,
                  cursor: busy || !sessionId ? 'not-allowed' : 'pointer',
                  fontSize: fontSize.xs,
                  fontWeight: 700,
                  opacity: busy || !sessionId ? 0.5 : 1,
                  padding: '6px 10px',
                }}
              >
                {busy ? 'Working…' : `Fire ${fillCount} background`}
              </button>
            </div>
          </div>

          {/* Populate graph — fire a batch of events + synthetic feedback */}
          <div style={{ borderTop: `1px solid ${colors.border}`, marginTop: spacing.sm, paddingTop: spacing.sm }}>
            <div style={{ color: colors.textMuted, fontSize: fontSize.xs, marginBottom: spacing.xs }}>
              Populate graph — events + synthetic feedback
            </div>
            <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.xs, marginBottom: spacing.xs }}>
              <span style={{ color: colors.textDim, fontSize: fontSize.xs, width: 48 }}>events</span>
              {SEED_COUNTS.map((c) => (
                <button key={c} type="button" onClick={() => setSeedCount(c)} style={pillStyle(seedCount === c)}>{c}</button>
              ))}
            </div>
            <div style={{ alignItems: 'center', display: 'flex', flexWrap: 'wrap', gap: spacing.xs, marginBottom: spacing.xs }}>
              <span style={{ color: colors.textDim, fontSize: fontSize.xs, width: 48 }}>confirm</span>
              {SEED_RATES.map((r) => (
                <button key={r} type="button" onClick={() => setSeedRate(r)} style={pillStyle(seedRate === r)}>{Math.round(r * 100)}%</button>
              ))}
            </div>
            <button
              type="button"
              disabled={busy || !sessionId}
              onClick={seed}
              style={{
                background: colors.accent,
                border: `1px solid ${colors.accent}`,
                borderRadius: radii.sm,
                color: colors.bg,
                cursor: busy || !sessionId ? 'not-allowed' : 'pointer',
                fontSize: fontSize.xs,
                fontWeight: 700,
                opacity: busy || !sessionId ? 0.5 : 1,
                padding: '6px 10px',
                width: '100%',
              }}
            >
              {busy ? 'Working…' : `Seed ${seedCount} events + feedback`}
            </button>
          </div>

          {last && (
            <div style={{ color: last.kind === 'ok' ? colors.good : last.kind === 'err' ? colors.bad : colors.textMuted, fontSize: fontSize.xs, marginTop: spacing.sm }}>
              {last.text}
            </div>
          )}
          <div style={{ color: colors.textDim, fontSize: fontSize.xs, marginTop: spacing.sm }}>
            Presenter-only · Ctrl+Shift+D to toggle
          </div>
        </>
      )}
    </div>
  )
}

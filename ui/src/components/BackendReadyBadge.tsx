/**
 * BackendReadyBadge — a subtle header indicator of backend warm-up state.
 *
 * `/health` returns 200 as soon as the server binds, but the classical seed
 * model can still be training in the background (cold start), during which live
 * scoring is slower. This polls `/health/ready` and shows an unobtrusive dot:
 *   • green  = ready (warm-up finished, no extra latency expected)
 *   • amber  = warming up (give it a moment before the live demo)
 *   • grey   = backend unreachable
 * Not in-your-face; details on hover.
 */
import { useEffect, useState } from 'react'
import { api } from '../api/http'

type Ready = { ready: boolean; classical_models?: string; store?: string | null; error?: string }

export function BackendReadyBadge() {
  const [state, setState] = useState<Ready | null>(null)
  const [reachable, setReachable] = useState(true)

  useEffect(() => {
    let alive = true
    let ready = false
    const tick = async () => {
      try {
        const r = await api<Ready>('/health/ready')
        if (!alive) return
        setState(r)
        setReachable(true)
        ready = r.ready
      } catch {
        if (!alive) return
        setReachable(false)
      }
    }
    tick()
    // Poll quickly while warming, then back off once ready.
    const id = setInterval(() => { if (!ready) tick(); else tick() }, 5000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const color = !reachable ? '#6b7385' : state?.ready ? '#57c785' : '#f0b429'
  const label = !reachable ? 'Backend: offline' : state?.ready ? 'Backend: ready' : 'Backend: warming up'
  const tip = !reachable
    ? 'Backend not reachable'
    : state?.ready
      ? 'Warm-up finished — no extra scoring latency expected.'
      : `Warming up (models ${state?.classical_models ?? '…'}). Live scoring may lag until ready.`

  return (
    <span
      className="small"
      title={tip}
      style={{ alignItems: 'center', color: 'var(--muted)', display: 'inline-flex', gap: 5, whiteSpace: 'nowrap' }}
    >
      <span
        aria-hidden
        style={{
          background: color,
          borderRadius: '50%',
          display: 'inline-block',
          height: 7,
          width: 7,
          boxShadow: state?.ready ? 'none' : `0 0 0 2px ${color}22`,
          animation: state?.ready || !reachable ? 'none' : 'pulse 1.4s ease-in-out infinite',
        }}
      />
      {label}
    </span>
  )
}

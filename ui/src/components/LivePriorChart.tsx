/**
 * LivePriorChart — Real-time pattern prior evolution chart for live playback.
 *
 * Renders the top-N pattern priors over time from scored_event messages.
 * Each prior key becomes a separate coloured line.
 */
import React, { useMemo } from 'react'
import { useLiveScoreStore, type PriorSnapshot } from '../state/liveScoreStore'
import { RealtimeScoreChart, type SeriesDef } from './charts/RealtimeScoreChart'
import { PAL } from './charts/chartUtils'

/** Max prior keys to chart (keep legend readable). */
const MAX_KEYS = 8

/**
 * Derive the union of all prior keys across all snapshots,
 * sorted by the most recent value (descending), limited to MAX_KEYS.
 */
function topKeys(snapshots: PriorSnapshot[]): string[] {
  if (snapshots.length === 0) return []
  const last = snapshots[snapshots.length - 1].priors
  return Object.entries(last)
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_KEYS)
    .map(([k]) => k)
}

export function LivePriorChart({ height = 170 }: { height?: number }) {
  const snapshots = useLiveScoreStore((s) => s.priorSnapshots)

  const keys = useMemo(() => topKeys(snapshots), [snapshots])

  const series: SeriesDef[] = useMemo(
    () =>
      keys.map((k, i) => ({
        key: k,
        label: k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        color: PAL[i % PAL.length],
        width: 1.5,
      })),
    [keys],
  )

  const data = useMemo(() => {
    const n = snapshots.length
    if (n === 0 || keys.length === 0) return []
    const t = new Float64Array(n)
    const arrs = keys.map(() => new Float64Array(n))
    for (let i = 0; i < n; i++) {
      const s = snapshots[i]
      t[i] = i
      for (let j = 0; j < keys.length; j++) {
        arrs[j][i] = s.priors[keys[j]] ?? 0
      }
    }
    return [t, ...arrs]
  }, [snapshots, keys])

  if (snapshots.length === 0) return null

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--fg)', marginBottom: 4 }}>
        Live Prior Evolution
      </div>
      <RealtimeScoreChart
        data={data}
        series={series}
        xLabel="Event"
        yLabel="Prior Weight"
        height={height}
        yMax={3}
      />
    </div>
  )
}

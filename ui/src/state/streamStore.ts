import { create } from 'zustand'
import { resolveVisiblePlotChannels } from '../utils/plotChannels'

export type TimeFrame = Record<string, unknown>

export type RingSeries = {
  x: number[]
  xTime: number[]
  yByChannel: Record<string, number[]>
  fs: number
}

function parseTimestampSeconds(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 1_000_000_000_000 ? value / 1000 : value
  }
  if (typeof value === 'string' && value.trim()) {
    const asNumber = Number(value)
    if (Number.isFinite(asNumber)) return asNumber > 1_000_000_000_000 ? asNumber / 1000 : asNumber
    const parsed = Date.parse(value)
    if (Number.isFinite(parsed)) return parsed / 1000
  }
  return null
}

function fallbackTimeFromIndex(index: number, fs: number): number {
  return fs > 0 ? index / fs : index
}

function frameTimeValue(frame: TimeFrame, index: number, fs: number): number {
  const record = frame as Record<string, unknown>
  const timestamp = parseTimestampSeconds(record.ts_unix ?? record.timestamp)
  if (timestamp !== null) return timestamp
  const t = record.t
  if (typeof t === 'number' && Number.isFinite(t)) return t
  return fallbackTimeFromIndex(index, fs)
}

function frameTimeSeries(frame: TimeFrame, i0: number, n: number, fs: number): number[] {
  const record = frame as Record<string, unknown>
  const tsDownsampled = record.ts_unix_downsampled
  if (Array.isArray(tsDownsampled) && tsDownsampled.length === n) {
    return tsDownsampled.map((value, idx) => parseTimestampSeconds(value) ?? fallbackTimeFromIndex(i0 + idx, fs))
  }

  const tDownsampled = record.t_downsampled
  if (Array.isArray(tDownsampled) && tDownsampled.length === n) {
    return tDownsampled.map((value, idx) => (typeof value === 'number' && Number.isFinite(value) ? value : fallbackTimeFromIndex(i0 + idx, fs)))
  }

  const startTs = parseTimestampSeconds(record.ts_unix0 ?? record.timestamp0)
  const endTs = parseTimestampSeconds(record.ts_unix1 ?? record.timestamp1)
  if (startTs !== null && endTs !== null && n > 0) {
    if (n === 1) return [startTs]
    const step = (endTs - startTs) / Math.max(1, n - 1)
    return Array.from({ length: n }, (_, idx) => startTs + idx * step)
  }

  const t0 = typeof record.t0 === 'number' && Number.isFinite(record.t0) ? record.t0 : null
  const t1 = typeof record.t1 === 'number' && Number.isFinite(record.t1) ? record.t1 : null
  if (t0 !== null && t1 !== null && n > 0) {
    if (n === 1) return [t0]
    const step = (t1 - t0) / Math.max(1, n - 1)
    return Array.from({ length: n }, (_, idx) => t0 + idx * step)
  }

  return Array.from({ length: n }, (_, idx) => fallbackTimeFromIndex(i0 + idx, fs))
}

function lowerBound(arr: number[], x: number): number {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] < x) lo = mid + 1
    else hi = mid
  }
  return lo
}

export type StreamState = {
  sessionId: string
  channels: string[]
  maxPoints: number
  followTail: boolean
  windowSeconds: number
  /**
   * Agent Q Round 23: when > 2, the UI asks the server to LTTB-downsample
   * time-chunk frames to this many points via the `?downsample=N` query
   * parameter on the stream WebSocket. 0 disables (server passthrough).
   * See `backend/routers/_stream_downsample.py` for the server contract.
   */
  streamDownsample: number
  series: RingSeries
  seriesVersion: number
  wsStatus: string

  setSessionId: (sessionId: string) => void
  setChannels: (channels: string[]) => void
  setMaxPoints: (n: number) => void
  setFollowTail: (v: boolean) => void
  setWindowSeconds: (s: number) => void
  setStreamDownsample: (n: number) => void
  setWsStatus: (s: string) => void

  reset: () => void
  appendFrame: (frame: TimeFrame) => void
  getWindowSamples: (
    i0: number,
    i1: number,
  ) => { channels: string[]; fs: number; i0: number; i1: number; samples: Record<string, number[]> }
}

function initSeries(): RingSeries {
  return { x: [], xTime: [], yByChannel: {}, fs: 1000 }
}

export const useStreamStore = create<StreamState>((set, get) => ({
  sessionId: localStorage.getItem('sessionId') || '',
  channels: [],
  maxPoints: 50_000,
  followTail: true,
  windowSeconds: 5,
  streamDownsample: (() => {
    const raw = localStorage.getItem('streamDownsample')
    if (raw == null) return 0
    const n = Number(raw)
    return Number.isFinite(n) && n >= 0 ? n | 0 : 0
  })(),
  series: initSeries(),
  seriesVersion: 0,
  wsStatus: 'idle',

  setSessionId: (sessionId) => {
    localStorage.setItem('sessionId', sessionId)
    set({ sessionId })
  },
  setChannels: (channels) => set({ channels }),
  setMaxPoints: (n) => set({ maxPoints: Math.max(10_000, n | 0) }),
  setFollowTail: (v) => set({ followTail: !!v }),
  setWindowSeconds: (s) => set({ windowSeconds: Math.max(0.1, Number(s) || 0.1) }),
  setStreamDownsample: (n) => {
    const v = Math.max(0, Number.isFinite(n) ? n | 0 : 0)
    localStorage.setItem('streamDownsample', String(v))
    set({ streamDownsample: v })
  },
  setWsStatus: (s) => set({ wsStatus: s }),

  reset: () => set({ series: initSeries(), seriesVersion: 0, channels: [] }),

  appendFrame: (frame) => {
    const { series, maxPoints, seriesVersion } = get()

    if (frame && (frame as any).eos) return

    const fs = typeof (frame as any).fs === 'number' ? ((frame as any).fs as number) : series.fs

    // Auto-adjust window for low-frequency data on first frame
    const { windowSeconds } = get()
    if (fs !== series.fs && fs <= 10 && windowSeconds <= 5) {
      // Low-freq data (e.g. Site_a_line2 1 Hz): widen the visible window
      // so the operator sees meaningful trends instead of 5 points.
      set({ windowSeconds: Math.min(300, Math.max(60, 120 / fs)) })
    }

    // Determine per-sample vs chunk frame.
    const isChunk = typeof (frame as any).i0 === 'number' && typeof (frame as any).i1 === 'number'

    // Agent Q Round 20: server may emit LTTB-downsampled chunk frames.
    // Shape: channel arrays have length `downsample_threshold` (< i1-i0),
    // and the frame carries `downsampled: true`, `downsample_threshold: N`,
    // and a shared `t_downsampled: number[]` (seconds). We keep the store's
    // sample-index x-axis contract by distributing N points linearly across
    // [i0, i1); LTTB preserves endpoints so the first/last x values are
    // exact.
    const isDownsampled = (frame as any).downsampled === true

    // Collect channel keys if not set.
    const reserved = new Set([
      't', 'i', 'fs', 't0', 't1', 'i0', 'i1',
      'ts_unix', 'timestamp', 'ts_unix0', 'ts_unix1', 'timestamp0', 'timestamp1',
      'ts_unix_downsampled', 't_downsampled', 'downsampled', 'downsample_threshold',
    ])
    const keys = Object.keys(frame || {}).filter((k) => !reserved.has(k))
    const useChannels = keys

    // Mutate buffers in-place for performance, but keep a new yByChannel object
    // so channels can be added without losing referential updates.
    const yByChannel = { ...series.yByChannel }
    for (const ch of useChannels) {
      if (!yByChannel[ch]) yByChannel[ch] = []
    }

    const x = series.x
    const xTime = series.xTime

    if (!isChunk) {
      const i = typeof (frame as any).i === 'number' ? ((frame as any).i as number) : x.length
      x.push(i)
      xTime.push(frameTimeValue(frame, i, fs))
      for (const ch of useChannels) {
        const v = (frame as any)[ch]
        yByChannel[ch].push(typeof v === 'number' ? v : Number.NaN)
      }
    } else {
      const i0 = (frame as any).i0 as number
      const i1 = (frame as any).i1 as number
      const span = Math.max(0, i1 - i0)

      // When downsampled, the per-channel array length is the true point
      // count; otherwise expect one sample per original index.
      let n = span
      if (isDownsampled) {
        for (const ch of useChannels) {
          const vals = (frame as any)[ch]
          if (Array.isArray(vals)) {
            n = vals.length
            break
          }
        }
      }

      if (isDownsampled && n > 0 && n !== span) {
        // Map N downsampled points linearly into [i0, i1-1]. LTTB preserves
        // endpoints so x[0] = i0 and x[n-1] = i1-1 are exact.
        const step = span > 1 && n > 1 ? (span - 1) / (n - 1) : 0
        for (let k = 0; k < n; k++) x.push(Math.round(i0 + k * step))
      } else {
        for (let k = 0; k < n; k++) x.push(i0 + k)
      }

      const timeValues = frameTimeSeries(frame, i0, n, fs)
      for (let k = 0; k < n; k++) xTime.push(timeValues[k])

      for (const ch of useChannels) {
        const vals = (frame as any)[ch]
        if (Array.isArray(vals)) {
          for (let k = 0; k < n; k++) yByChannel[ch].push(typeof vals[k] === 'number' ? vals[k] : Number.NaN)
        } else {
          for (let k = 0; k < n; k++) yByChannel[ch].push(Number.NaN)
        }
      }
    }

    // Trim ring buffer.
    const excess = x.length - maxPoints
    if (excess > 0) {
      x.splice(0, excess)
      xTime.splice(0, excess)
      for (const ch of Object.keys(yByChannel)) yByChannel[ch].splice(0, excess)
    }

    set({ series: { x, xTime, yByChannel, fs }, seriesVersion: seriesVersion + 1 })
  },

  getWindowSamples: (i0, i1) => {
    const { series, channels } = get()
    const x = series.x
    const start = lowerBound(x, i0)
    const end = lowerBound(x, i1)

    // Clamp to the portion of the selection actually present in the ring buffer.
    const actualI0 = start < x.length ? x[start] : i0
    const actualI1 = end > 0 && end <= x.length ? x[end - 1] + 1 : i1

    const outCh = resolveVisiblePlotChannels(series.yByChannel, channels)
    const samples: Record<string, number[]> = {}
    for (const ch of outCh) {
      const arr = series.yByChannel[ch] || []
      samples[ch] = arr.slice(start, end).map((v) => (Number.isFinite(v) ? v : 0))
    }

    return { channels: outCh, fs: series.fs, i0: actualI0, i1: actualI1, samples }
  },
}))

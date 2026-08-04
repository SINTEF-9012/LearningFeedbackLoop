export type WsHandler<T> = (msg: T) => void

export function connectWs<T>(url: string, onMsg: WsHandler<T>, onStatus?: (s: string) => void) {
  let ws: WebSocket | null = null
  let stopped = false
  let retry = 0

  const connect = () => {
    if (stopped) return

    onStatus?.('connecting')
    ws = new WebSocket(url)
    let openedAt = 0

    ws.onopen = () => {
      openedAt = Date.now()
      // Only reset retry if the *previous* connection stayed alive for
      // a meaningful duration, otherwise keep backing off (avoids the
      // open-then-immediate-close spam when the server has no session).
      onStatus?.('connected')
    }

    ws.onmessage = (ev) => {
      // A real message means the connection is healthy — reset backoff.
      retry = 0
      try {
        const parsed = JSON.parse(ev.data)
        onMsg(parsed as T)
      } catch {
        // Surface schema/encoding mismatches during development.
        try {
          console.warn('[ws] failed to parse message', ev.data)
        } catch {
          // ignore
        }
      }
    }

    ws.onclose = (ev) => {
      ws = null
      // 4404 = session not found on the server (e.g. after restart).
      // Stop reconnecting — the session is gone.
      if (ev.code === 4404 || ev.code === 4403) {
        onStatus?.('rejected')
        stopped = true
        return
      }
      onStatus?.('closed')
      if (stopped) return
      // If connection lasted < 3 seconds without receiving a message,
      // it was likely rejected (stale session, no data). Keep backing off.
      const aliveMs = openedAt ? Date.now() - openedAt : 0
      if (aliveMs > 3000) retry = 0
      retry = Math.min(retry + 1, 8)
      const delay = Math.min(1000 * 2 ** retry, 30000)
      onStatus?.(`reconnecting in ${(delay / 1000).toFixed(0)}s`)
      setTimeout(connect, delay)
    }

    ws.onerror = () => {
      // close triggers reconnect
      try {
        ws?.close()
      } catch {
        // ignore
      }
    }
  }

  connect()

  return {
    stop() {
      stopped = true
      try {
        ws?.close()
      } catch {
        // ignore
      }
      ws = null
      onStatus?.('stopped')
    },
  }
}

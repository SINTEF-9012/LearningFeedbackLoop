"""Visualize a streaming session in real time (plots the first numeric channel).

Usage: python scripts/visualize_stream.py <session_id>

This script connects to the time-domain WebSocket and plots a rolling buffer
of values from the first available channel. For chunk frames it uses the
mean value of the chunk.
"""
import sys
import json
import asyncio
import threading
import collections
import numpy as np
import websockets
import matplotlib.pyplot as plt

BUF_MAX = 1000

def start_ws_reader(sid, buffer):
    url = f"ws://localhost:8000/streams/{sid}"

    async def reader():
        async with websockets.connect(url) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                # pick first non-meta key as channel
                keys = [k for k in data.keys() if k not in ("t","i","fs","t0","t1","i0","i1")]
                if not keys:
                    continue
                ch = keys[0]
                val = data.get(ch)
                # handle arrays or scalars
                try:
                    if isinstance(val, list):
                        arr = np.asarray(val)
                        if arr.size:
                            buffer.append(float(arr.mean()))
                    else:
                        buffer.append(float(val))
                except Exception:
                    continue

    def run():
        asyncio.run(reader())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_stream.py <session_id>")
        raise SystemExit(1)
    sid = sys.argv[1]
    buf = collections.deque(maxlen=BUF_MAX)
    start_ws_reader(sid, buf)

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, BUF_MAX)
    ax.set_xlabel('samples')
    ax.set_ylabel('value')
    ax.set_title(f'Session {sid} stream')

    try:
        while True:
            data = list(buf)
            if data:
                line.set_data(range(len(data)), data)
                ax.set_xlim(0, max(len(data), BUF_MAX))
                ymin = min(data)
                ymax = max(data)
                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5
                ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)
    except KeyboardInterrupt:
        print('Exiting')

if __name__ == '__main__':
    main()

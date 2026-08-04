"""Visualize FFT stream in real time.

Connects to `ws://localhost:8000/sessions/{session_id}/fft`, receives FFT payloads and
renders a live spectrum (line) and a rolling spectrogram (imshow) for one channel.

Usage: python scripts/visualize_fft.py <session_id> [channel_name]
"""
import sys
import asyncio
import json
import threading
from collections import deque

import numpy as np
import websockets
import matplotlib.pyplot as plt


def start_fft_ws_reader(sid, buffer, freqs_holder, channel_name=None):
    url = f"ws://localhost:8000/sessions/{sid}/fft"

    async def reader():
        # Keep trying to connect; when the server restarts or the
        # playback is replayed the WS may close — reconnect and
        # reset buffers so the visualizer can pick up the new stream.
        while True:
            try:
                # clear buffers before (re)connecting so old data doesn't mix
                try:
                    buffer.clear()
                except Exception:
                    pass
                try:
                    freqs_holder.clear()
                except Exception:
                    pass

                async with websockets.connect(url) as ws:
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        # data expected to include 'freqs' and 'channels' mapping
                        freqs = np.array(data.get("freqs", []))
                        channels = data.get("channels", {})
                        # choose channel
                        if channel_name is None:
                            # pick first key
                            keys = list(channels.keys())
                            if not keys:
                                continue
                            ch = keys[0]
                        else:
                            ch = channel_name
                        spec = np.array(channels.get(ch, []), dtype=float)
                        if spec.size == 0:
                            continue
                        # update freq axis and buffer
                        freqs_holder.clear()
                        freqs_holder.extend(freqs.tolist())
                        buffer.append(spec)
            except Exception as exc:
                # connection dropped or failed; wait and reconnect
                print(f"FFT WS disconnected or failed: {exc}; reconnecting in 1s...")
                await asyncio.sleep(1)
                continue

    def run():
        asyncio.run(reader())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_fft.py <session_id> [channel_name]")
        raise SystemExit(1)
    sid = sys.argv[1]
    channel = sys.argv[2] if len(sys.argv) > 2 else None

    BUF = 200  # number of frames to keep for spectrogram
    speclist = deque(maxlen=BUF)
    freqs_holder = deque()

    start_fft_ws_reader(sid, speclist, freqs_holder, channel_name=channel)

    plt.ion()
    fig, (ax_spec, ax_img) = plt.subplots(2, 1, figsize=(10, 6))

    line, = ax_spec.plot([], [])
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Amplitude')
    ax_spec.set_title(f'Live spectrum (session {sid})')

    img = None
    prev_freq_len = 0
    try:
        while True:
            if len(speclist) == 0 or len(freqs_holder) == 0:
                plt.pause(0.1)
                continue

            freqs = np.array(list(freqs_holder))
            # If the frequency axis changed (playback restarted or FFT params changed)
            # reset the spectrogram buffer and image so axes reinitialize cleanly.
            if prev_freq_len != 0 and len(freqs) != prev_freq_len:
                speclist.clear()
                img = None
                try:
                    line.set_data([], [])
                except Exception:
                    pass
                prev_freq_len = len(freqs)
                # skip updating until we have new frames
                plt.pause(0.05)
                continue
            prev_freq_len = len(freqs)
            latest = np.array(speclist[-1])
            # update line
            line.set_data(freqs, latest)
            ax_spec.set_xlim(freqs[0], freqs[-1])
            ymin = latest.min()
            ymax = latest.max()
            if ymin == ymax:
                ymin -= 1e-6
                ymax += 1e-6
            ax_spec.set_ylim(ymin, ymax)

            # build spectrogram matrix: time x freq
            S = np.vstack(list(speclist))
            S = S.T  # freq x time for imshow
            if img is None:
                img = ax_img.imshow(S, aspect='auto', origin='lower', extent=[0, S.shape[1], freqs[0], freqs[-1]])
                ax_img.set_ylabel('Frequency (Hz)')
                ax_img.set_xlabel('Frames (time)')
                ax_img.set_title('Rolling spectrogram')
            else:
                img.set_data(S)
                img.set_extent([0, S.shape[1], freqs[0], freqs[-1]])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)
    except KeyboardInterrupt:
        print('Exiting')


if __name__ == '__main__':
    main()

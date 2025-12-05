# fft_streamer.py
import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional
from computation import compute_rfft_multichannel  # your helper

async def _safe_put(q: asyncio.Queue, item, timeout: float = 1.0):
    try:
        await asyncio.wait_for(q.put(item), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass

async def fft_stream_task(session: any):
    s = session
    print("[fft_stream_task] starting", flush=True)
    data: Dict[str, np.ndarray] = s["data"]
    metadata: Dict[str, Any] = s["metadata"]
    fs = float(metadata["file_header"]["SampleFrequency"])

    # Channels from selection -> config -> all
    sel = s.get("selection", {})
    channels = sel.get("channels") or s["config"].get("channels") or list(data.keys())

    # FFT config
    cfg_fft = s.get("fft_config", {})
    nfft = int(cfg_fft.get("nfft", 1026))
    overlap = float(cfg_fft.get("overlap", 0.75))
    hop = max(1, int(round(nfft * (1.0 - overlap))))
    window_type = cfg_fft.get("window_type", "hann")
    detrend = bool(cfg_fft.get("detrend", True))
    output = cfg_fft.get("output", "amplitude")
    db = bool(cfg_fft.get("db", False))
    bin_stride = int(cfg_fft.get("bin_stride", 1))
    max_freq_hz = cfg_fft.get("max_freq_hz", None)
    inherit_speed = bool(cfg_fft.get("inherit_speed", True))

    # Playback speed (optional inheritance)
    speed = float(s["config"].get("speed", 1.0)) if inherit_speed else 1.0
    speed = max(speed, 1e-9)

    # Limits
    try:
        n_max = min(len(data[ch]) for ch in channels)
    except Exception:
        s["running_fft"] = False
        s["fft_task"] = None
        return

    # Absolute frame scheduling
    start_wall = time.perf_counter()
    frames_sent = 0
    last_i1 = 0  # last right-edge sample used
    print("IN FFT STREAMER TASK")
    # Diagnostic: report key session values so we can understand early exits
    try:
        print(f"[fft_stream_task] diag session_id={s.get('session_id')} running_fft={s.get('running_fft')} position={s.get('position')} n_max={n_max} nfft={nfft} channels={channels}", flush=True)
        # also print per-channel lengths to detect mismatches
        try:
            lengths = {ch: len(s['data'].get(ch, [])) for ch in channels}
            print(f"[fft_stream_task] channel_lengths={lengths}", flush=True)
        except Exception:
            pass
    except Exception:
        pass
    try:
        # Main loop - will only run while `s['running_fft']` is True. If this
        # is False at task start the loop will not execute and the task will
        # exit immediately; the diagnostics above help pinpoint that state.
        if not s.get("running_fft", False):
            print(f"[fft_stream_task] not starting main loop because running_fft={s.get('running_fft')}", flush=True)
        while s.get("running_fft", False):
            try:
                # protect the FFT loop so unexpected exceptions are logged
                # and bring attention to the cause instead of silently
                # letting the task die.
                pass
            except Exception:
                # placeholder to keep structure; actual loop below
                pass
            try:
            # FFT main loop
                if frames_sent == 0:
                    print(f"[fft_stream_task] running for session {s.get('session_id')} (n_max={n_max})", flush=True)
                pos = int(s.get("position", 0))  # where playback has advanced to
                i1 = min(pos, n_max)

                # If not enough to fill first FFT window, sleep a bit
                if i1 < nfft:
                    await asyncio.sleep((hop / fs) / speed * 0.25)
                    continue

                # Wait for hop advancement
                if (i1 - last_i1) < hop:
                    await asyncio.sleep(0)  # yield
                    continue

                i0 = i1 - nfft
                # Build multi-channel segment (assumes np.ndarray for zero-copy slice views)
                segs = {}
                ok = True
                for ch in channels:
                    arr = np.asarray(data[ch])
                    if i1 > len(arr):
                        ok = False
                        break
                    segs[ch] = arr[i0:i1]
                if not ok:
                    # Data not ready or mismatch; wait briefly
                    await asyncio.sleep(0.005)
                    continue

            # Compute FFT
                freqs, spectra = compute_rfft_multichannel(
                    segs_by_channel=segs, fs=fs,
                    window_type=window_type, detrend=detrend,
                    output=output, db=db, bin_stride=bin_stride, max_freq_hz=max_freq_hz
                )

                # Emit payload
                i_center = i0 + (nfft // 2)
                t_center = i_center / fs
                payload = {
                    "fs": fs,
                    "nfft": nfft,
                    "overlap": overlap,
                    "hop": hop,
                    "speed": speed,
                    "t_center": t_center,
                    "i_center": i_center,
                    "freqs": freqs.tolist(),
                    "output": output,
                    "db": db,
                    "channels": {ch: spectra[ch].tolist() for ch in channels}
                }

                # Broadcast if any listeners (but run regardless)
                subscribers = list(s.get("fft_subscribers", []))
                print(f"[fft_stream_task] subscribers={len(subscribers)}", flush=True)
                # don't print full payload by default (can be noisy)
                try:
                    print(f"[fft_stream_task] emitting i_center={i_center} t_center={t_center} freqs_len={len(freqs)}", flush=True)
                except Exception:
                    pass
                if subscribers:
                    await asyncio.gather(*(_safe_put(q, payload, 1.0) for q in subscribers), return_exceptions=True)

                # Publish feature event for downstream online agents
                try:
                    from events import publish_feature
                    evt = {"type": "fft", "session_id": s.get("session_id"), "i_center": i_center, "t_center": t_center, "payload": {"freqs": freqs.tolist()}}
                    await publish_feature(s.get("session_id"), evt)
                except Exception:
                    pass

                last_i1 = i1
                frames_sent += 1

                # Drift-resistant scheduling by hop/speed
                target_elapsed = (frames_sent * hop / fs) / speed
                delay = (start_wall + target_elapsed) - time.perf_counter()
                await asyncio.sleep(delay if delay > 0 else 0)
            except Exception as e:
                    import traceback
                    print(f"[fft_stream_task] exception in loop: {e}", flush=True)
                    traceback.print_exc()
                    # stop the FFT loop on unexpected error so operator can inspect
                    s["running_fft"] = False
                    break
    finally:
        print(f"[fft_stream_task] finishing for session {s.get('session_id')}", flush=True)
        s["fft_task"] = None
        s["running_fft"] = False
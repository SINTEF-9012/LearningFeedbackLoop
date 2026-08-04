# fft_streamer.py
import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from .computation import compute_rfft_multichannel  # your helper
from .metadata_utils import get_sample_frequency

logger = logging.getLogger(__name__)

from .events import publish_feature

async def _safe_put(q: asyncio.Queue, item, timeout: float = 1.0):
    try:
        await asyncio.wait_for(q.put(item), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass

async def fft_stream_task(session: Any):
    s = session
    logger.info("[fft_stream_task] starting")
    data: Dict[str, np.ndarray] = s["data"]
    metadata: Dict[str, Any] = s["metadata"]
    fs = get_sample_frequency(metadata)

    # Channels from selection -> config -> all
    sel = s.get("selection", {})
    selected_channels = sel.get("channels") or s["config"].get("channels")
    channels = selected_channels or list(data.keys())

    # FFT config
    cfg_fft = s.get("fft_config", {})
    nfft = int(cfg_fft.get("nfft", 1024))
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

    # Absolute frame scheduling
    start_wall = time.perf_counter()
    frames_sent = 0
    last_i1 = 0  # last right-edge sample used
    logger.debug("IN FFT STREAMER TASK")
    # Diagnostic: report key session values so we can understand early exits
    try:
        logger.debug(
            "[fft_stream_task] diag session_id=%s running_fft=%s position=%s nfft=%s channels=%s",
            s.get("session_id"),
            s.get("running_fft"),
            s.get("position"),
            nfft,
            channels,
        )
        # also print per-channel lengths to detect mismatches
        try:
            lengths = {ch: len(s['data'].get(ch, [])) for ch in channels}
            logger.debug("[fft_stream_task] channel_lengths=%s", lengths)
        except Exception:
            pass
    except Exception:
        pass
    try:
        # Main loop - will only run while `s['running_fft']` is True. If this
        # is False at task start the loop will not execute and the task will
        # exit immediately; the diagnostics above help pinpoint that state.
        if not s.get("running_fft", False):
            logger.debug(
                "[fft_stream_task] not starting main loop because running_fft=%s",
                s.get("running_fft"),
            )
        while s.get("running_fft", False):
            try:
            # FFT main loop
                channels = selected_channels or list(data.keys())
                if not channels:
                    if not s.get("running", False):
                        break
                    await asyncio.sleep(0.05)
                    continue

                try:
                    n_max = min(len(data.get(ch, [])) for ch in channels)
                except Exception:
                    if not s.get("running", False):
                        break
                    await asyncio.sleep(0.05)
                    continue

                if frames_sent == 0:
                    logger.info(
                        "[fft_stream_task] running for session %s (n_max=%s)",
                        s.get("session_id"),
                        n_max,
                    )
                pos = int(s.get("position", 0))  # where playback has advanced to
                i1 = min(pos, n_max)

                # If not enough to fill first FFT window, sleep a bit
                if i1 < nfft:
                    if not s.get("running", False) and i1 >= n_max:
                        break
                    await asyncio.sleep((hop / fs) / speed * 0.25)
                    continue

                # Wait for hop advancement
                if (i1 - last_i1) < hop:
                    if not s.get("running", False) and i1 >= n_max:
                        break
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
                logger.debug("[fft_stream_task] subscribers=%s", len(subscribers))
                # don't print full payload by default (can be noisy)
                try:
                    logger.debug(
                        "[fft_stream_task] emitting i_center=%s t_center=%s freqs_len=%s",
                        i_center,
                        t_center,
                        len(freqs),
                    )
                except Exception:
                    pass
                if subscribers:
                    await asyncio.gather(*(_safe_put(q, payload, 1.0) for q in subscribers), return_exceptions=True)

                # Publish feature event for downstream online agents
                try:
                    evt = {"type": "fft", "session_id": s.get("session_id"), "i_center": i_center, "t_center": t_center, "payload": {"freqs": freqs.tolist()}}
                    await publish_feature(s.get("session_id"), evt)
                except Exception:
                    logger.warning("Feature publish failed (fft); continuing stream", exc_info=True)

                last_i1 = i1
                frames_sent += 1

                # Drift-resistant scheduling by hop/speed
                target_elapsed = (frames_sent * hop / fs) / speed
                delay = (start_wall + target_elapsed) - time.perf_counter()
                await asyncio.sleep(delay if delay > 0 else 0)
            except Exception as e:
                    logger.exception("[fft_stream_task] exception in loop: %s", e)
                    # stop the FFT loop on unexpected error so operator can inspect
                    s["running_fft"] = False
                    break
    finally:
        logger.info("[fft_stream_task] finishing for session %s", s.get("session_id"))
        s["fft_task"] = None
        s["running_fft"] = False
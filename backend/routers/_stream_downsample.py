"""WebSocket frame downsampling helpers (Agent Q Round 20, 2026-04-24).

Server-side LTTB downsampling for WS stream frames so that clients
never need to plot more than a few thousand points per update.

Two frame shapes are supported:

1. **Time-domain chunk frame** emitted by ``playback_task`` when
   ``samples_per_tick > 1``::

       {"t0": float, "t1": float, "i0": int, "i1": int, "fs": float,
        "<channel_name>": list[float] | np.ndarray, ...}

   For every channel value that is an array of length >= threshold,
   we build a time x-axis (``np.linspace(t0, t1, len(values))``) and
   apply LTTB. The result replaces the channel value in-place (on a
   shallow copy).

2. **FFT frame** emitted by ``fft_stream_task``::

       {"freqs": list[float], "channels": {ch: list[float]}, ...}

   We downsample each spectrum against the ``freqs`` axis and emit
   a single downsampled ``freqs`` array shared by every channel —
   so all channels use the same x bins, matching how the UI plots.

Per-sample frames (``samples_per_tick == 1``) have scalar channel
values and are returned unchanged.

EOS frames and anything without the expected keys are also returned
unchanged — failure must be fail-open (stream must not break).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

import numpy as np

from backend.agents.processing.downsample import lttb

logger = logging.getLogger(__name__)


__all__ = ["maybe_downsample_frame", "DEFAULT_STREAM_DOWNSAMPLE_THRESHOLD"]


# Only frames whose per-channel array length meets or exceeds this many
# points are eligible for downsampling. Below this, LTTB is a no-op
# anyway, so we skip the overhead entirely.
DEFAULT_STREAM_DOWNSAMPLE_THRESHOLD: int = 2000


def _is_array_like(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        return v.ndim == 1 and v.dtype.kind in "fi"
    if isinstance(v, (list, tuple)) and len(v) > 0:
        # Peek first element only — mixed-type lists should not be downsampled.
        first = v[0]
        return isinstance(first, (int, float))
    return False


def _downsample_time_chunk(frame: Dict[str, Any], threshold: int) -> Dict[str, Any]:
    """Downsample a time-domain chunk frame per channel."""
    t0 = frame.get("t0")
    t1 = frame.get("t1")
    if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
        return frame

    # Shallow-copy so callers that retain the original frame aren't affected.
    out: Dict[str, Any] = dict(frame)
    reserved = {"t0", "t1", "i0", "i1", "fs", "eos", "final_i"}
    downsampled_any = False

    for key, value in frame.items():
        if key in reserved:
            continue
        if not _is_array_like(value):
            continue
        arr = np.asarray(value, dtype=np.float64).ravel()
        n = arr.shape[0]
        if n < threshold:
            continue
        xs = np.linspace(float(t0), float(t1), n)
        _, ys = lttb(xs, arr, threshold)
        out[key] = ys.tolist()
        downsampled_any = True

    if downsampled_any:
        # Publish the downsampled x-axis once (all channels share it).
        sampled_x = np.linspace(float(t0), float(t1), threshold)
        out["t_downsampled"] = sampled_x.tolist()
        out["downsampled"] = True
        out["downsample_threshold"] = int(threshold)
    return out


def _downsample_fft(frame: Dict[str, Any], threshold: int) -> Dict[str, Any]:
    """Downsample an FFT frame's spectra against the ``freqs`` axis."""
    freqs = frame.get("freqs")
    channels = frame.get("channels")
    if not _is_array_like(freqs) or not isinstance(channels, Mapping):
        return frame
    freqs_arr = np.asarray(freqs, dtype=np.float64).ravel()
    n = freqs_arr.shape[0]
    if n < threshold:
        return frame

    out_channels: Dict[str, Any] = {}
    sampled_x: np.ndarray | None = None
    for ch, spectrum in channels.items():
        if not _is_array_like(spectrum):
            out_channels[ch] = spectrum
            continue
        ys_arr = np.asarray(spectrum, dtype=np.float64).ravel()
        if ys_arr.shape[0] != n:
            # Mismatched lengths — don't touch.
            out_channels[ch] = spectrum
            continue
        xs_out, ys_out = lttb(freqs_arr, ys_arr, threshold)
        if sampled_x is None:
            sampled_x = xs_out
        out_channels[ch] = ys_out.tolist()

    out: Dict[str, Any] = dict(frame)
    out["channels"] = out_channels
    if sampled_x is not None:
        out["freqs"] = sampled_x.tolist()
        out["downsampled"] = True
        out["downsample_threshold"] = int(threshold)
    return out


def maybe_downsample_frame(frame: Any, threshold: int) -> Any:
    """Return a possibly downsampled copy of ``frame``.

    Args:
        frame: The outgoing WS payload (typically a dict). Non-dict
            payloads are returned unchanged.
        threshold: LTTB target point count. ``<= 0`` disables
            downsampling entirely; values between 1 and
            ``DEFAULT_STREAM_DOWNSAMPLE_THRESHOLD`` are accepted but
            the caller should not expect a meaningful visual
            reduction below a few hundred points.

    The function is fail-open: any internal error returns the
    original frame and logs a warning.
    """
    if not isinstance(frame, dict) or threshold <= 2:
        return frame
    try:
        if "freqs" in frame and "channels" in frame:
            return _downsample_fft(frame, threshold)
        if "t0" in frame and "t1" in frame:
            return _downsample_time_chunk(frame, threshold)
    except Exception:
        logger.warning("maybe_downsample_frame: error; passing frame through", exc_info=True)
    return frame

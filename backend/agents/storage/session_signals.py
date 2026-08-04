"""Durable per-session signal binding — Agent N (2026-04-24).

On session upload we persist the raw per-channel sample arrays plus the
metadata needed to slice windows (sampling frequency, channel order) to
``data/sessions/{session_id}.npz``. Memories reference the session by
``Memory.session_id`` and store a ``TimeRange(i0, i1, fs)``; combined
with the durable signal file, we can serve
``GET /memory/{id}/signal?channels=Fx,Fy&margin_s=0.5`` after a restart
without needing the original session to still be loaded in memory.

Module is pure storage: it does not import FastAPI, the orchestrator, or
the sessions dict. All side-effecting calls (save/load) go through the
helpers here so tests can point them at a tmpdir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_SESSIONS_DIR = Path(os.environ.get("LFL_SESSIONS_DIR", "data/sessions"))
_META_ARRAY_KEY = "__lfl_meta__"
_FS_KEY = "__lfl_fs__"
_CHANNELS_KEY = "__lfl_channels__"


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_dir(sessions_dir: Optional[os.PathLike]) -> Path:
    return Path(sessions_dir) if sessions_dir is not None else DEFAULT_SESSIONS_DIR


def session_signal_path(session_id: str, sessions_dir: Optional[os.PathLike] = None) -> Path:
    """Return the on-disk `.npz` path for a session id."""
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in ("-", "_"))
    if not safe:
        safe = "session"
    return _resolve_dir(sessions_dir) / f"{safe}.npz"


def save_session_signal(
    session_id: str,
    data: Dict[str, Sequence[float]],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    fs: Optional[float] = None,
    sessions_dir: Optional[os.PathLike] = None,
) -> Optional[Path]:
    """Persist per-channel sample arrays + metadata to ``.npz``.

    Returns the written path, or ``None`` if the input had no numeric
    channels to save (we never create an empty file).
    """
    if not data:
        return None

    arrays: Dict[str, np.ndarray] = {}
    for name, samples in data.items():
        if samples is None:
            continue
        try:
            arr = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 1 or arr.size == 0:
            continue
        arrays[str(name)] = arr

    if not arrays:
        return None

    target = session_signal_path(session_id, sessions_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    meta_to_store: Dict[str, Any] = dict(metadata or {})
    if fs is not None:
        meta_to_store.setdefault("fs", float(fs))
    meta_to_store["_session_id"] = str(session_id)
    meta_to_store["_channels"] = list(arrays.keys())

    # NumPy's `np.savez` won't accept arbitrary Python objects, so the
    # metadata goes in as a 0-d bytes array holding a JSON blob.
    try:
        meta_blob = json.dumps(meta_to_store, default=_json_default).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.warning("save_session_signal: metadata not JSON-serialisable: %s", exc)
        meta_blob = json.dumps({"_session_id": str(session_id)}).encode("utf-8")

    payload = dict(arrays)
    payload[_META_ARRAY_KEY] = np.frombuffer(meta_blob, dtype=np.uint8)

    tmp = target.with_name(target.stem + ".tmp")
    tmp_written = tmp.with_suffix(tmp.suffix + ".npz")  # np.savez_compressed appends .npz
    try:
        np.savez_compressed(str(tmp), **payload)
        os.replace(tmp_written, target)
    except Exception:
        for p in (tmp_written, tmp):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_session_signal(
    session_id: str,
    sessions_dir: Optional[os.PathLike] = None,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """Load an ``.npz`` previously written by :func:`save_session_signal`.

    Returns ``(data, metadata)`` or ``None`` if the file is missing /
    malformed. Does NOT raise on read errors — the signal binding is a
    best-effort channel.
    """
    path = session_signal_path(session_id, sessions_dir)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta: Dict[str, Any] = {}
            if _META_ARRAY_KEY in npz.files:
                try:
                    meta = json.loads(bytes(npz[_META_ARRAY_KEY].tobytes()).decode("utf-8"))
                except Exception:
                    meta = {}
            data: Dict[str, np.ndarray] = {}
            for name in npz.files:
                if name == _META_ARRAY_KEY:
                    continue
                data[name] = np.asarray(npz[name])
            return data, meta
    except Exception as exc:
        logger.warning("load_session_signal(%s) failed: %s", session_id, exc)
        return None


def list_persisted_sessions(sessions_dir: Optional[os.PathLike] = None) -> List[str]:
    """Return session ids that have a persisted `.npz` file on disk."""
    target_dir = _resolve_dir(sessions_dir)
    if not target_dir.exists():
        return []
    out: List[str] = []
    for p in target_dir.glob("*.npz"):
        out.append(p.stem)
    return sorted(out)


# ─────────────────────────────────────────────────────────────────────────────
# Window extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_signal_window(
    data: Dict[str, Sequence[float]],
    channels: Optional[Iterable[str]],
    i0: int,
    i1: int,
    *,
    fs: float,
    margin_s: float = 0.0,
) -> Dict[str, Any]:
    """Return a JSON-safe windowed slice.

    Clamps indices to the channel bounds, applies symmetric margin in
    samples derived from ``margin_s * fs``, and returns one flat list per
    requested channel plus the final effective window.
    """
    try:
        i0 = int(i0)
        i1 = int(i1)
    except (TypeError, ValueError):
        i0, i1 = 0, 0
    if i1 < i0:
        i0, i1 = i1, i0

    try:
        fs_f = float(fs)
    except (TypeError, ValueError):
        fs_f = 0.0
    margin_samples = max(0, int(round(max(0.0, float(margin_s)) * max(0.0, fs_f))))

    wi0 = max(0, i0 - margin_samples)
    wi1 = i1 + margin_samples

    channel_order: List[str]
    if channels is None:
        channel_order = [c for c in data.keys() if not c.startswith("_")]
    else:
        channel_order = [str(c) for c in channels]

    out: Dict[str, List[float]] = {}
    effective_len = 0
    for name in channel_order:
        arr = data.get(name)
        if arr is None:
            continue
        try:
            np_arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            continue
        if np_arr.ndim != 1 or np_arr.size == 0:
            continue
        lo = max(0, wi0)
        hi = min(np_arr.size, max(lo, wi1))
        slice_arr = np_arr[lo:hi]
        effective_len = max(effective_len, int(slice_arr.size))
        out[name] = slice_arr.astype(float, copy=False).tolist()

    t0 = (wi0 / fs_f) if fs_f > 0 else 0.0
    t1 = (wi1 / fs_f) if fs_f > 0 else 0.0

    return {
        "i0": int(wi0),
        "i1": int(wi1),
        "t0": float(t0),
        "t1": float(t1),
        "fs": float(fs_f),
        "margin_s": float(max(0.0, margin_s)),
        "margin_samples": int(margin_samples),
        "requested_range": {"i0": int(i0), "i1": int(i1)},
        "channels": out,
        "channel_length": int(effective_len),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Digest (tamper / rebinding detection)
# ─────────────────────────────────────────────────────────────────────────────


def compute_signal_digest(
    data: Dict[str, Sequence[float]],
    i0: int,
    i1: int,
    *,
    channels: Optional[Iterable[str]] = None,
) -> str:
    """SHA-1 over the sample bytes in ``[i0, i1)`` for stable rebinding checks.

    Returns a hex string (40 chars) or ``""`` if no channel contributed.
    """
    try:
        i0 = int(i0)
        i1 = int(i1)
    except (TypeError, ValueError):
        return ""
    if i1 <= i0:
        return ""

    order: List[str]
    if channels is None:
        order = sorted(c for c in data.keys() if not c.startswith("_"))
    else:
        order = sorted(str(c) for c in channels)

    h = hashlib.sha1()
    any_bytes = False
    for name in order:
        arr = data.get(name)
        if arr is None:
            continue
        try:
            np_arr = np.asarray(arr, dtype=np.float32)
        except Exception:
            continue
        if np_arr.ndim != 1 or np_arr.size == 0:
            continue
        lo = max(0, i0)
        hi = min(np_arr.size, i1)
        if hi <= lo:
            continue
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(np_arr[lo:hi].tobytes())
        any_bytes = True
    return h.hexdigest() if any_bytes else ""

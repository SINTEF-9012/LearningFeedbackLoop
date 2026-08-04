"""SINDIT live property bridge (Phase 0).

Subscribes to the LFL feature bus and pushes current machine state / cutting
parameters onto the resolved SINDIT machine asset as property *values*, so the
digital twin reflects live state instead of only static specs.

Design (mirrors the guardrails from ISS-34):
  - **Throttled** to ~1 write-batch per second *per machine* — never per tick.
  - **Best-effort**: every SINDIT call is guarded; failures are logged, never raised.
  - **Gated** by ``SINDIT_LIVE_BRIDGE`` (default off), like the experiment push.
  - Writes only to the existing physical machine assets (``urn:lfl:asset:*``),
    using the same ``{machine_iri}:{name}`` property-URI scheme the seeded
    ``machine_state`` / ``spindle_speed`` properties already use, so writes
    **upsert** rather than duplicate.

Values pushed (when available in the stream):
  - ``machine_state``  — "running" while a session streams (liveness heartbeat).
  - ``spindle_speed`` / ``feed_rate`` — from the resolved cutting context.
  - ``vibration_rms`` — a genuinely live-changing value from the feature metrics.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SAMM_PROP = "urn:samm:sindit.sintef.no:1.0.0#AbstractAssetProperty"
_THROTTLE_S = 1.0

_task: Optional["asyncio.Task[Any]"] = None
_stop: Optional[asyncio.Event] = None


def _enabled() -> bool:
    return os.environ.get("SINDIT_LIVE_BRIDGE", "false").lower() in ("1", "true", "yes")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def start_live_bridge() -> None:
    """Start the live bridge background task (no-op unless enabled)."""
    global _task, _stop
    if not _enabled():
        logger.info("SINDIT live bridge disabled (set SINDIT_LIVE_BRIDGE=true to enable)")
        return
    if _task is not None and not _task.done():
        return
    _stop = asyncio.Event()
    _task = asyncio.create_task(_run(), name="sindit-live-bridge")
    logger.info("SINDIT live bridge started (throttle=%.1fs/machine)", _THROTTLE_S)


async def stop_live_bridge() -> None:
    global _task, _stop
    if _stop is not None:
        _stop.set()
    if _task is not None:
        try:
            await asyncio.wait_for(_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _task.cancel()
        except Exception:  # pragma: no cover - defensive
            pass
        _task = None
    _stop = None


async def _run() -> None:
    from backend.events import subscribe_features
    from backend.agents.sindit.client import SinditClient

    queue = subscribe_features()  # global feature channel
    url = os.environ.get("SINDIT_API_URL", "http://localhost:9017")
    user = os.environ.get("SINDIT_USERNAME", "sindit")
    pw = os.environ.get("SINDIT_PASSWORD", "sindit")

    session_meta_cache: Dict[str, Dict[str, Any]] = {}
    last_write: Dict[str, float] = {}

    async with SinditClient(base_url=url) as client:
        try:
            if not await client.authenticate(user, pw):
                logger.warning("SINDIT live bridge: auth failed — bridge inactive")
                return
        except Exception as exc:
            logger.warning("SINDIT live bridge: auth error %s — bridge inactive", exc)
            return

        while _stop is None or not _stop.is_set():
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
            try:
                await _handle(client, payload, session_meta_cache, last_write)
            except Exception as exc:  # never let one bad tick kill the bridge
                logger.debug("SINDIT live bridge tick failed: %s", exc)


def _update_session_cache(
    cache: Dict[str, Dict[str, Any]], session_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    meta = cache.setdefault(session_id, {})
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if v is not None:
                meta[k] = v
    for k in ("machine_id", "machine", "case_dir", "source", "machine_family"):
        if payload.get(k) is not None:
            meta[k] = payload[k]
    return meta


def _resolve_machine_iri(session_meta: Dict[str, Any], payload: Dict[str, Any]) -> Optional[str]:
    from backend.agents.sindit.runtime_context import resolve_runtime_metadata

    try:
        merged = resolve_runtime_metadata(dict(session_meta), payload)
    except Exception:
        merged = session_meta
    iri = merged.get("sindit_asset_iri") or merged.get("machine_iri")
    return str(iri) if iri else None


def _cutting_values(session_meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from backend.agents.core.context import extract_context_from_metadata

        out = extract_context_from_metadata(session_meta).model_dump()
    except Exception:
        out = {}
    # Direct fallback: cutting params are often plain keys in the session metadata.
    for k in ("spindle_speed", "feed_rate"):
        if out.get(k) is None:
            v = session_meta.get(k)
            if isinstance(v, (int, float)):
                out[k] = v
    return out


def _live_rms(payload: Dict[str, Any]) -> Optional[float]:
    # 1) a pre-computed metric if a downstream stage attached one
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for k in ("rms", "vibration_rms", "amplitude", "max_amplitude", "energy", "peak"):
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    # 2) otherwise compute RMS from the raw frame samples (genuinely live-changing).
    #    Use the channel with the largest RMS — the active vibration/signal
    #    channel — so a flat index/time channel doesn't mask the real signal.
    frame = payload.get("frame")
    if isinstance(frame, dict):
        best: Optional[float] = None
        for arr in frame.values():
            if isinstance(arr, (list, tuple)) and arr:
                vals = [float(x) for x in arr[:4096] if isinstance(x, (int, float))]
                if vals:
                    rms = (sum(v * v for v in vals) / len(vals)) ** 0.5
                    if best is None or rms > best:
                        best = rms
        return best
    return None


async def _handle(
    client: Any,
    payload: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    last_write: Dict[str, float],
) -> None:
    if not isinstance(payload, dict):
        return
    session_id = payload.get("session_id")
    if not session_id:
        return

    session_meta = _update_session_cache(cache, session_id, payload)
    machine_iri = _resolve_machine_iri(session_meta, payload)
    if not machine_iri:
        return

    now = time.monotonic()
    if now - last_write.get(machine_iri, 0.0) < _THROTTLE_S:
        return
    last_write[machine_iri] = now

    cc = _cutting_values(session_meta)
    updates = [("machine_state", "running", "", "string")]
    ss = cc.get("spindle_speed")
    if isinstance(ss, (int, float)):
        updates.append(("spindle_speed", round(float(ss), 1), "rpm", "float"))
    fr = cc.get("feed_rate")
    if isinstance(fr, (int, float)):
        updates.append(("feed_rate", round(float(fr), 1), "mm/min", "float"))
    rms = _live_rms(payload)
    if rms is not None:
        updates.append(("vibration_rms", round(rms, 4), "", "float"))

    ts = _now_iso()
    for name, value, unit, dtype in updates:
        prop = {
            "class_uri": _SAMM_PROP,
            "uri": f"{machine_iri}:{name}",
            "label": name.replace("_", " ").title(),
            "propertyName": name,
            "propertyValue": str(value),
            "propertyUnit": unit,
            "propertyDataType": dtype,
            "propertyValueTimestamp": ts,
            "assetUri": machine_iri,
        }
        try:
            await client.post_property(prop)
        except Exception as exc:
            logger.debug("live bridge post_property %s failed: %s", name, exc)

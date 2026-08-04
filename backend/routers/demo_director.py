"""Demo Director endpoints — fire curated scripted events for the video /
presentation demo, so the presenter can drive the sequence deterministically
instead of hand-running scripts. Not part of the operator product surface.

Events are the fixtures under ``scripts/demo_data/`` and are processed through
the exact same event pipeline as live events (scoring / alerting / learning).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo-director", tags=["demo-director"])

_DEMO_DIR = Path(__file__).resolve().parents[2] / "scripts" / "demo_data"

# Curated, ordered steps for the demo flow. event A (chatter) → operator
# feedback → event B (similar chatter, links back to A so the feedback effect
# shows) → escalations → a normal (no-alert) event.
_DEMO_EVENTS: List[Dict[str, str]] = [
    {"key": "chatter", "file": "event_2_chatter.json", "label": "Chatter alert — event A"},
    {"key": "similar_chatter", "file": "event_4_similar_to_chatter.json", "label": "Similar chatter — event B (shows feedback effect)"},
    {"key": "tool_wear", "file": "event_5_tool_wear_drill.json", "label": "Tool-wear (drill)"},
    {"key": "breakage_risk", "file": "event_6_breakage_risk_titanium.json", "label": "Breakage risk (titanium)"},
    {"key": "spindle_overload", "file": "event_9_spindle_overload_steel.json", "label": "Spindle overload (steel)"},
    {"key": "normal", "file": "event_1_normal.json", "label": "Normal — no alert"},
]
_BY_KEY = {e["key"]: e for e in _DEMO_EVENTS}

# Fallback machine for scripted events so their alerts resolve to a real
# SINDIT machine/tool context (curated fixtures carry material/tool but no
# machine). Used only when the session has no derivable machine and the
# presenter did not override. Presenter can override per fire-event.
_DEFAULT_DEMO_MACHINE = "Site_b - MACHINE_B1 - CASE_B1"


def _machine_for_session(session_id: str) -> str:
    """Derive the machine label from the running session so a scripted event's
    alert context matches the session selector on screen (e.g. a SITE_C session
    should not show the Site_b default). Falls back to the default when the
    session carries no case_dir.
    """
    try:
        from backend.app import get_session_or_404
        session = get_session_or_404(session_id)
        metadata = session.get("metadata") or {}
        source_config = session.get("source_config") or metadata.get("source_config") or {}
        casedata = metadata.get("casedata") or {}
        case_dir = (source_config.get("case_dir") or casedata.get("case_dir") or "").strip()
        if case_dir:
            return case_dir
    except Exception:
        logger.debug("could not derive machine for session %s", session_id, exc_info=True)
    return _DEFAULT_DEMO_MACHINE


class FireEventRequest(BaseModel):
    session_id: str
    event: str
    machine_id: str | None = None


@router.get("/events")
async def list_demo_events() -> Dict[str, Any]:
    """List the demo events whose fixtures are present."""
    available = [e for e in _DEMO_EVENTS if (_DEMO_DIR / e["file"]).exists()]
    return {"events": available}


@router.post("/fire-event")
async def fire_demo_event(req: FireEventRequest) -> Dict[str, Any]:
    """Inject one curated demo event into the given session through the real
    event-processing pipeline."""
    entry = _BY_KEY.get(req.event)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown demo event: {req.event}")
    path = _DEMO_DIR / entry["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"fixture missing: {entry['file']}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read fixture: {exc}")

    payload.pop("_description", None)
    # Keep the fixture's curated explanation as a demo-only FALLBACK: threaded via
    # event metadata, it is used only if the live LLM call fails, so a take still
    # shows polished prose instead of the terse deterministic fallback. The live
    # LLM remains primary and overrides it whenever it succeeds.
    curated = payload.pop("_explanation", None)
    payload["session_id"] = req.session_id
    if curated:
        md = payload.get("metadata") or {}
        md.setdefault("curated_explanation", curated)
        payload["metadata"] = md

    # Attach a machine so the scripted event's alert resolves to real SINDIT
    # machine/tool context (fixtures carry material/tool but no machine).
    machine = req.machine_id or _machine_for_session(req.session_id)
    cc = payload.get("cutting_context")
    if isinstance(cc, dict) and not cc.get("machine_id"):
        cc["machine_id"] = machine
        payload["cutting_context"] = cc

    # Reuse the real /agent/memory/events path so the demo event is scored,
    # alerted and learned from exactly like a live event.
    from backend.agents.memory.router import ProcessEventRequest, process_event

    # Clear rate-limit / dismiss-cooldown so this scripted event reliably alerts
    # (a recent dismiss would otherwise suppress it for 30 s — nothing to open).
    try:
        from backend.agents.memory.dispatcher import get_dispatcher
        get_dispatcher().reset_session_gating(req.session_id)
    except Exception:
        logger.debug("could not reset alert gating for demo event", exc_info=True)

    # Pause live playback so the scripted event stays the presented focus instead
    # of being replaced/buried a second later by the session's own live-stream
    # alerts. The presenter drives the sequence deterministically and can Resume.
    try:
        from backend.app import get_session_or_404
        get_session_or_404(req.session_id)["paused"] = True
    except Exception:
        logger.debug("could not pause session for demo event", exc_info=True)

    try:
        request = ProcessEventRequest(**payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"fixture is not a valid event: {exc}")

    result = await process_event(request)
    return {
        "event": req.event,
        "label": entry["label"],
        "memory_id": getattr(result, "memory_id", None),
        "significant": getattr(result, "significant", None),
        "action": getattr(getattr(result, "action", None), "value", None),
    }


class SeedRequest(BaseModel):
    session_id: str
    count: int = 12
    confirm_rate: float = 0.75
    machine_id: str | None = None


@router.post("/seed")
async def seed_demo_graph(req: SeedRequest) -> Dict[str, Any]:
    """Fire a batch of curated events and apply synthetic operator feedback, so
    the memory graph, learned priors, co-occurrence edges and batch-review data
    populate quickly instead of requiring a long manual run.

    Cycles through the significant scripted events (so multiple contexts/patterns
    are exercised, not just A/B) and confirms/dismisses each one with a
    deterministic, seeded mix (``confirm_rate``). LLM explanations are skipped
    for speed. Presenter/demo tool only — same learning path as real feedback.
    """
    import random as _random

    from backend.agents.memory.router import ProcessEventRequest, process_event
    from backend.agents.memory.feedback import MemoryFeedbackRequest, FeedbackAction
    from backend.agents.memory.orchestrator import get_orchestrator

    sig_events = [e for e in _DEMO_EVENTS if e["key"] != "normal"]
    if not sig_events:
        raise HTTPException(status_code=500, detail="no scripted events available")

    count = max(1, min(int(req.count), 100))  # cap to keep a single call bounded
    confirm_rate = max(0.0, min(float(req.confirm_rate), 1.0))
    machine = req.machine_id or _machine_for_session(req.session_id)
    orch = get_orchestrator()
    rng = _random.Random(20260715)  # deterministic → reproducible demos

    fired = 0
    confirmed = 0
    dismissed = 0
    skipped = 0
    memory_ids: List[str] = []

    for i in range(count):
        entry = sig_events[i % len(sig_events)]
        path = _DEMO_DIR / entry["file"]
        if not path.exists():
            skipped += 1
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            skipped += 1
            continue
        payload.pop("_description", None)
        payload.pop("_explanation", None)
        payload["session_id"] = req.session_id
        # Skip the (slow) LLM explanation during bulk seeding.
        meta = payload.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta["generate_explanations_override"] = False
        payload["metadata"] = meta
        cc = payload.get("cutting_context")
        if isinstance(cc, dict) and not cc.get("machine_id"):
            cc["machine_id"] = machine
            payload["cutting_context"] = cc

        try:
            request = ProcessEventRequest(**payload)
            result = await process_event(request)
        except Exception:
            logger.debug("seed: fire failed for %s", entry["key"], exc_info=True)
            skipped += 1
            continue

        mid = getattr(result, "memory_id", None)
        if not mid:
            skipped += 1
            continue
        fired += 1
        memory_ids.append(mid)

        action = FeedbackAction.CONFIRM if rng.random() < confirm_rate else FeedbackAction.DISMISS
        try:
            if orch and getattr(orch, "feedback_handler", None):
                await orch.feedback_handler.process_feedback(
                    mid,
                    MemoryFeedbackRequest(
                        action=action,
                        user_id="demo-seed",
                        reason="synthetic demo feedback",
                    ),
                )
                if action == FeedbackAction.CONFIRM:
                    confirmed += 1
                else:
                    dismissed += 1
        except Exception:
            logger.debug("seed: feedback failed for %s", mid, exc_info=True)

    return {
        "requested": count,
        "fired": fired,
        "confirmed": confirmed,
        "dismissed": dismissed,
        "skipped": skipped,
        "memory_ids": memory_ids[:50],
    }


class SeedFleetRequest(BaseModel):
    # Full context is required by the fleet aggregator (k-anonymity is per-context).
    machine_type: str = "gantry_mill"
    tool_type: str = "face_mill"
    material: str = "casting_steel"
    regime: str = "roughing"
    sites: int = 3


@router.post("/seed-fleet")
async def seed_fleet(req: SeedFleetRequest) -> Dict[str, Any]:
    """Append a few synthetic per-site knowledge packs for one context so the
    'transfers across the fleet' panel has something to aggregate (k-anonymity
    met). Demo-only: clearly synthetic sites/priors, not real learned state.
    """
    import os
    from datetime import datetime, timezone

    from backend.agents.knowledge import KnowledgePack
    from backend.agents.knowledge.fleet import append_pack_to_store, fleet_store_path

    context = {
        "machine_type": req.machine_type,
        "tool_type": req.tool_type,
        "material": req.material,
        "regime": req.regime,
    }
    # A small pattern-prior profile per site, jittered so the aggregate is not
    # trivially identical across sites.
    base_priors = {
        "TOOL_WEAR_RISK": 0.78,
        "ANOMALY_HIGH_VIBRATION": 0.71,
        "RATIO_Fx_Fy:>5": 0.64,
        "POWER_SPIKE_SUSTAINED": 0.58,
    }
    n = max(1, min(int(req.sites), 12))
    store = fleet_store_path((os.environ.get("FLEET_KNOWLEDGE_STORE_PATH") or "").strip() or "data/fleet_packs.jsonl")
    now = datetime.now(timezone.utc).isoformat()
    stored = 0
    site_names = []
    for i in range(n):
        jitter = (i - n / 2) * 0.03
        priors = {k: round(min(0.98, max(0.4, v + jitter)), 3) for k, v in base_priors.items()}
        site = f"demo-site-{i + 1}"
        site_names.append(site)
        pack = KnowledgePack(
            site=site,
            built_at=now,
            license="internal-only",
            context=dict(context),
            pattern_priors=priors,
            notes=["synthetic demo pack"],
        )
        try:
            stored = append_pack_to_store(pack, store)
        except Exception:
            logger.debug("seed-fleet: append failed for %s", site, exc_info=True)
    return {"context": context, "sites": site_names, "stored_count": stored}


class FillRequest(BaseModel):
    session_id: str
    count: int = 5
    confirm_rate: float = 0.7
    machine_id: str | None = None


# Background events used to fill the gap between scripted A and B — everything
# EXCEPT the chatter signatures, so the A→B feedback-drift story stays clean.
_FILL_KEYS = ("tool_wear", "breakage_risk", "spindle_overload", "normal")


@router.post("/fill")
async def fill_between_events(req: FillRequest) -> Dict[str, Any]:
    """Fire several synthetic BACKGROUND events to create a realistic run of
    activity between the scripted A and B events. Deliberately excludes the
    chatter events so the operator's confirm-on-A → drift-on-B demo stays clean.
    Significant background events get light synthetic feedback; 'normal' does not.
    """
    import random as _random

    from backend.agents.memory.router import ProcessEventRequest, process_event
    from backend.agents.memory.feedback import MemoryFeedbackRequest, FeedbackAction
    from backend.agents.memory.orchestrator import get_orchestrator

    bg = [e for e in _DEMO_EVENTS if e["key"] in _FILL_KEYS and (_DEMO_DIR / e["file"]).exists()]
    if not bg:
        raise HTTPException(status_code=500, detail="no background events available")

    count = max(1, min(int(req.count), 50))
    confirm_rate = max(0.0, min(float(req.confirm_rate), 1.0))
    machine = req.machine_id or _machine_for_session(req.session_id)
    orch = get_orchestrator()
    rng = _random.Random()

    fired = 0
    confirmed = 0
    dismissed = 0
    skipped = 0
    memory_ids: List[str] = []

    for i in range(count):
        entry = bg[i % len(bg)]
        try:
            payload = json.loads((_DEMO_DIR / entry["file"]).read_text())
        except Exception:
            skipped += 1
            continue
        payload.pop("_description", None)
        payload.pop("_explanation", None)
        payload["session_id"] = req.session_id
        meta = payload.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta["generate_explanations_override"] = False  # skip slow LLM for filler
        payload["metadata"] = meta
        cc = payload.get("cutting_context")
        if isinstance(cc, dict) and not cc.get("machine_id"):
            cc["machine_id"] = machine
            payload["cutting_context"] = cc

        try:
            result = await process_event(ProcessEventRequest(**payload))
        except Exception:
            logger.debug("fill: fire failed for %s", entry["key"], exc_info=True)
            skipped += 1
            continue

        mid = getattr(result, "memory_id", None)
        if not mid:
            skipped += 1
            continue
        fired += 1
        memory_ids.append(mid)

        # Light synthetic feedback on significant background events only.
        if entry["key"] != "normal" and orch and getattr(orch, "feedback_handler", None):
            action = FeedbackAction.CONFIRM if rng.random() < confirm_rate else FeedbackAction.DISMISS
            try:
                await orch.feedback_handler.process_feedback(
                    mid,
                    MemoryFeedbackRequest(
                        action=action,
                        user_id="demo-fill",
                        reason="synthetic background event",
                    ),
                )
                if action == FeedbackAction.CONFIRM:
                    confirmed += 1
                else:
                    dismissed += 1
            except Exception:
                logger.debug("fill: feedback failed for %s", mid, exc_info=True)

    return {
        "requested": count,
        "fired": fired,
        "confirmed": confirmed,
        "dismissed": dismissed,
        "skipped": skipped,
        "memory_ids": memory_ids[:50],
    }

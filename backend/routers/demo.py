"""Demo session helpers for naturally scored playback demos.

The UI and CLI demo paths use this module to resolve a dataset-specific
session source plus a deterministic seek anchor so playback can start inside
an interesting region without injecting synthetic memory events.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from backend.session_logs import append_session_log as _append_demo_log
from backend.session_logs import session_log_path as demo_log_path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_TEST_DATA_DIR = _PROJECT_ROOT / "test_data"
_CASEDATA_SESSION_FILE = os.environ.get("DEMO_CASEDATA_SESSION_FILE", "casedata_session.json")
_SITE_A_LINE2_SESSION_FILES = (
    "site_a_line2_part0001_of00015_session.json",
    "site_a_line2_session.json",
)
_CASEDATA_ROOT = _PROJECT_ROOT / "data" / "casedata"
_SITE_A_ROOT = _PROJECT_ROOT / "data" / "site_a"
_DEFAULT_OPERATION_ID = "OF00001"
_SITE_C_CASE_DIR = "SITE_C - MACHINE_C1 - CASE_C1"
_SITE_A_CASE_DIR = "Site_a - MACHINE_A1 - CASE_A1"
_SITE_B_CASE_DIR = "Site_b - MACHINE_B1 - CASE_B1"

# ── Demo mode definitions ────────────────────────────────────────────────────

DemoMode = str  # "default" | "labeled" | "casedata" | "site_c" | "site_a" | "site_b" | "site_a_line2"


def _resolve_demo_session_path(session_file_override: Optional[str]) -> Optional[Path]:
    raw = str(session_file_override or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (_TEST_DATA_DIR / candidate).resolve()
    return candidate


def _site_a_line2_session_path(session_override: Optional[Path]) -> Path:
    if session_override is not None:
        if not session_override.exists():
            raise FileNotFoundError(f"Demo session file not found: {session_override}")
        return session_override

    session_path = next(
        (
            _TEST_DATA_DIR / candidate
            for candidate in _SITE_A_LINE2_SESSION_FILES
            if (_TEST_DATA_DIR / candidate).exists()
        ),
        _TEST_DATA_DIR / _SITE_A_LINE2_SESSION_FILES[0],
    )
    if not session_path.exists():
        raise FileNotFoundError(
            "Site_a_line2 session not found. Run: "
            "python scripts/load_site_a_line2_session.py "
            "or, for the legacy fixed OF00013 export, python scripts/build_site_a_line2_demo.py"
        )
    return session_path


def _fallback_demo_session(*candidates: str) -> Path:
    for candidate in candidates:
        path = _TEST_DATA_DIR / candidate
        if path.exists():
            return path
    return _TEST_DATA_DIR / candidates[-1]


def _get_demo_config(
    mode: DemoMode,
    session_file_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the dataset + seek configuration for a demo mode."""

    session_override = _resolve_demo_session_path(session_file_override)

    if mode == "site_c":
        return {
            "mode": mode,
            "source": "simulated_casedata",
            "casedata_root": str(_CASEDATA_ROOT),
            "case_dir": _SITE_C_CASE_DIR,
            "operation_id": _DEFAULT_OPERATION_ID,
            "start_at_first_cutting_row": True,
            "requested_start_position": 0,
            "rationale": "Starts at the first cutting row to skip the long idle lead-in on the 1 Hz SITE_C operation.",
        }

    if mode == "site_a":
        return {
            "mode": mode,
            "source": "simulated_casedata",
            "casedata_root": str(_SITE_A_ROOT),
            "case_dir": _SITE_A_CASE_DIR,
            "operation_id": _DEFAULT_OPERATION_ID,
            "start_at_first_cutting_row": True,
            "requested_start_position": 0,
            "rationale": "Starts at the first cutting row so the shortest 1 Hz casedata operation reaches an active span immediately.",
        }

    if mode == "site_b":
        return {
            "mode": mode,
            "source": "simulated_casedata",
            "casedata_root": str(_CASEDATA_ROOT),
            "case_dir": _SITE_B_CASE_DIR,
            "operation_id": _DEFAULT_OPERATION_ID,
            "start_at_first_cutting_row": True,
            "requested_start_position": 0,
            "rationale": "Starts at the first cutting row on the Site_b case under data/casedata to avoid replaying hours of steady idle data.",
        }

    if mode == "casedata":
        config = _get_demo_config("site_c", session_file_override=session_file_override)
        return {
            **config,
            "mode": mode,
            "rationale": "Legacy casedata demo mode now points at the default SITE_C cutting region instead of injecting events.",
        }

    if mode == "site_a_line2":
        return {
            "mode": mode,
            "source": "simulated_file",
            "session_path": _site_a_line2_session_path(session_override),
            "requested_start_position": 0,
            "start_at_label": "pre_break",
            "start_label_lead_in_samples": 30,
            "rationale": "Starts 30 samples before the first pre_break label so the operator sees the tool_wear to pre_break transition on naturally scored alerts.",
        }

    if mode == "labeled":
        session_path = session_override or _fallback_demo_session("cnc_session.json", "sample_session.json")
        if not session_path.exists():
            raise FileNotFoundError(f"Demo session file not found: {session_path}")
        return {
            "mode": mode,
            "source": "simulated_file",
            "session_path": session_path,
            "requested_start_position": 0,
            "rationale": "Legacy labeled demo uses a prerecorded session file and natural inference only.",
        }

    session_path = session_override or _fallback_demo_session("sample_session.json")
    if not session_path.exists():
        raise FileNotFoundError(f"Demo session file not found: {session_path}")
    return {
        "mode": mode,
        "source": "simulated_file",
        "session_path": session_path,
        "requested_start_position": 0,
        "rationale": "Fallback demo mode replays the bundled sample session without injected events.",
    }


async def _wait_for_demo_delay(session: Optional[Dict[str, Any]], delay_s: float) -> bool:
    """Count down demo delay only while playback is actively running."""
    remaining = max(0.0, float(delay_s))
    while remaining > 0:
        if session is not None:
            if not session.get("running", False):
                return False
            if session.get("paused", False):
                await asyncio.sleep(0.1)
                continue
        step = min(0.1, remaining)
        await asyncio.sleep(step)
        remaining -= step
    return session is None or bool(session.get("running", False))


async def run_demo_sequence(
    session_id: str,
    mode: DemoMode = "default",
    sleep_s: float = 3.0,
    reset_priors: bool = True,
    session: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record demo lifecycle for a naturally scored playback session.

    Kept as a lightweight helper for callers that still want demo bus/log
    notifications, but no synthetic events or automated feedback are injected.
    """
    from backend.agents.memory.orchestrator import get_orchestrator
    from backend.events import bus

    config = _get_demo_config(mode)

    if reset_priors:
        try:
            get_orchestrator().scorer.reset_priors()
            logger.info("Demo: priors reset")
        except Exception:
            logger.debug("Demo: prior reset failed (continuing)", exc_info=True)

    seek_summary = {
        "requested_start_position": config.get("requested_start_position"),
        "start_at_first_cutting_row": config.get("start_at_first_cutting_row", False),
        "start_at_label": config.get("start_at_label"),
        "start_label_lead_in_samples": config.get("start_label_lead_in_samples"),
        "rationale": config.get("rationale"),
    }

    await bus.publish(f"demo.{session_id}", {
        "phase": "started",
        "session_id": session_id,
        "mode": mode,
        "source": config.get("source"),
        "seek": seek_summary,
    })
    _append_demo_log(session_id, {
        "phase": "started",
        "session_id": session_id,
        "mode": mode,
        "source": config.get("source"),
        "seek": seek_summary,
    })
    await bus.publish(f"demo.{session_id}", {
        "phase": "done",
        "session_id": session_id,
        "source": config.get("source"),
        "sleep_s": float(sleep_s),
    })
    _append_demo_log(session_id, {
        "phase": "done",
        "session_id": session_id,
        "mode": mode,
        "source": config.get("source"),
        "sleep_s": float(sleep_s),
    })

    return {
        "session_id": session_id,
        "mode": mode,
        "source": config.get("source"),
        "seek": seek_summary,
        "events_injected": 0,
        "feedback_applied": 0,
        "details": [],
    }

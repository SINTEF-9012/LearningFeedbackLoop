"""
LLM-backed operator-comment classifier.

On every ``FeedbackAction.COMMENT`` the handler calls ``classify_comment``
to extract a small structured record:

    {
        "root_cause":  Optional[str],   # short free-text category
        "action_taken": Optional[str],  # short free-text verb phrase
        "tool_change": bool,            # did the operator change the tool?
        "source": "llm" | "heuristic",  # which path produced the result
    }

When an LLM is available, we prompt for JSON; when it isn't (or parsing
fails) we fall back to a fast regex heuristic. The classifier is
best-effort — feedback processing must never block or fail on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_TOOL_CHANGE_PATTERNS = (
    re.compile(r"\btool[-\s]*chang\w*", re.I),
    re.compile(r"\b(replac|swap|switch|new)\w*\s+(the\s+)?tool\b", re.I),
    re.compile(r"\bchang\w+\s+(the\s+)?tool\b", re.I),
    re.compile(r"\binsert\w*\s+(replac|chang|swap)\w*", re.I),
)

_ACTION_PATTERNS = (
    (re.compile(r"\breplac(ed|ing)?\s+\w+", re.I), "replacement"),
    (re.compile(r"\bchang(ed|ing)?\s+\w+", re.I), "change"),
    (re.compile(r"\bclean(ed|ing)?", re.I), "cleaned"),
    (re.compile(r"\bresum(ed|ing)?|restart(ed|ing)?|rerun", re.I), "restarted"),
    (re.compile(r"\bstop(ped|ping)?|halt(ed|ing)?", re.I), "stopped"),
    (re.compile(r"\badjust(ed|ing)?\s+\w+", re.I), "adjusted"),
)

_ROOT_CAUSE_KEYWORDS = {
    "tool_wear": ("wear", "worn", "blunt", "dull"),
    "tool_break": ("broke", "broken", "snap", "chipped", "fracture"),
    "chatter": ("chatter", "vibration", "resonance"),
    "feed_rate": ("feed rate", "overfeed", "too fast", "too slow"),
    "coolant": ("coolant", "overheat", "temperature", "cooling"),
    "material": ("hard spot", "inclusion", "workpiece", "material"),
    "program": ("program", "g-code", "gcode", "nc code"),
    "operator_error": ("misload", "wrong setup", "operator error"),
}


def _classify_heuristic(text: str) -> Dict[str, Any]:
    norm = text.strip()
    if not norm:
        return {
            "root_cause": None,
            "action_taken": None,
            "tool_change": False,
            "source": "heuristic",
        }

    tool_change = any(p.search(norm) for p in _TOOL_CHANGE_PATTERNS)

    action_taken: Optional[str] = None
    for pat, label in _ACTION_PATTERNS:
        m = pat.search(norm)
        if m:
            action_taken = label
            break

    root_cause: Optional[str] = None
    lower = norm.lower()
    for cause, kws in _ROOT_CAUSE_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            root_cause = cause
            break

    return {
        "root_cause": root_cause,
        "action_taken": action_taken,
        "tool_change": tool_change,
        "source": "heuristic",
    }


def _coerce_llm_result(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from raw LLM output; return None on failure."""
    if not raw:
        return None
    # Try straight JSON first.
    candidates = []
    try:
        candidates.append(json.loads(raw))
    except Exception:
        pass
    # Then fenced-block / embedded JSON.
    if not candidates:
        m = re.search(r"\{[^{}]*\}", raw, re.S)
        if m:
            try:
                candidates.append(json.loads(m.group(0)))
            except Exception:
                return None
    if not candidates:
        return None
    obj = candidates[0]
    if not isinstance(obj, dict):
        return None

    def _opt_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        s = str(value).strip()
        return s or None

    tool_change_val = obj.get("tool_change")
    if isinstance(tool_change_val, str):
        tool_change = tool_change_val.strip().lower() in ("true", "yes", "1")
    else:
        tool_change = bool(tool_change_val)

    return {
        "root_cause": _opt_str(obj.get("root_cause")),
        "action_taken": _opt_str(obj.get("action_taken")),
        "tool_change": tool_change,
        "source": "llm",
    }


def _build_prompt(text: str) -> str:
    return (
        "You are classifying an operator's comment on a CNC monitoring alert. "
        "Extract three fields and return ONLY a JSON object with keys "
        "`root_cause`, `action_taken`, `tool_change` (boolean). "
        "`root_cause` is a short category (tool_wear, tool_break, chatter, "
        "feed_rate, coolant, material, program, operator_error, or other). "
        "`action_taken` is a short verb phrase (e.g. 'replaced tool', "
        "'cleaned nozzle', 'no action'). Use null when unknown. "
        "Do NOT include any text outside the JSON object.\n\n"
        f"Comment: {text!r}\n\nJSON:"
    )


async def classify_comment(
    text: str,
    *,
    llm_agent: Optional[Any] = None,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """Classify an operator comment; heuristic fallback on any failure."""
    fallback = _classify_heuristic(text)
    if not text or not text.strip():
        return fallback
    if llm_agent is None:
        return fallback

    is_avail = getattr(llm_agent, "is_available", None)
    try:
        if callable(is_avail) and not is_avail():
            return fallback
    except Exception:
        return fallback

    prompt = _build_prompt(text)
    try:
        res = await asyncio.wait_for(
            llm_agent.handle_request("comment_classifier", "query", {"question": prompt}, {}),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.info("classify_comment: LLM timeout after %.1fs — using heuristic", timeout_s)
        return fallback
    except Exception as exc:
        logger.info("classify_comment: LLM call failed (%s) — using heuristic", exc)
        return fallback

    raw = ""
    if isinstance(res, dict):
        raw = str(res.get("answer") or res.get("result") or "")
    parsed = _coerce_llm_result(raw)
    if parsed is None:
        return fallback
    return parsed

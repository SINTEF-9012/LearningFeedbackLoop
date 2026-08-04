from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional

from backend.agents.core.batch_context import BatchContext
from backend.agents.llm.explainer import LLMExplainer
from backend.agents.memory.reconfig import (
    ParameterDelta,
    ProcessReconfiguration,
    RecipeEdit,
    ToolAction,
    normalize_context,
)


logger = logging.getLogger(__name__)


_RISK_RANK = {"low": 0, "medium": 1, "high": 2}

_PATTERN_RULES: Dict[str, Dict[str, Any]] = {
    "SPINDLE_LOAD_RAMP": {
        "risk": "medium",
        "parameter_deltas": [
            {
                "parameter": "feed_rate",
                "direction": "decrease",
                "magnitude_pct": -5.0,
                "confidence": 0.72,
                "rationale": "Observed spindle-load ramp suggests reducing feed to slow load growth during the cut.",
            }
        ],
        "tool_actions": [
            {
                "action": "inspect",
                "reason_code": "wear_indicated_by_power_ramp",
                "confidence": 0.7,
            }
        ],
    },
    "SPINDLE_POWER_SURGE": {
        "risk": "medium",
        "parameter_deltas": [
            {
                "parameter": "feed_rate",
                "direction": "decrease",
                "magnitude_pct": -10.0,
                "confidence": 0.82,
                "rationale": "Sustained spindle-power surge supports a conservative feed reduction before continuing the cut.",
            }
        ],
        "tool_actions": [
            {
                "action": "inspect",
                "reason_code": "load_surge_requires_tool_check",
                "confidence": 0.78,
            }
        ],
    },
    "VIBRATION_REGIME_SHIFT": {
        "risk": "medium",
        "parameter_deltas": [
            {
                "parameter": "spindle_speed",
                "direction": "decrease",
                "magnitude_pct": -5.0,
                "confidence": 0.79,
                "rationale": "A vibration regime shift is consistent with chatter onset, so a small spindle-speed shift is the safest first intervention.",
            }
        ],
        "tool_actions": [
            {
                "action": "inspect",
                "reason_code": "vibration_shift_requires_balance_check",
                "confidence": 0.7,
            }
        ],
    },
    "FEED_OVERRIDE_DROP": {
        "risk": "low",
        "parameter_deltas": [
            {
                "parameter": "feed_rate",
                "direction": "decrease",
                "magnitude_pct": -8.0,
                "confidence": 0.68,
                "rationale": "Repeated feed-override drops imply the programmed feed is too aggressive for this context.",
            }
        ],
        "tool_actions": [],
    },
    "ENERGY_ACCUMULATION": {
        "risk": "medium",
        "parameter_deltas": [
            {
                "parameter": "depth_of_cut",
                "direction": "decrease",
                "magnitude_pct": -10.0,
                "confidence": 0.71,
                "rationale": "Rising energy accumulation supports reducing engagement before heat and wear compound further.",
            }
        ],
        "tool_actions": [
            {
                "action": "inspect",
                "reason_code": "energy_rise_requires_wear_check",
                "confidence": 0.66,
            }
        ],
    },
}


def _merged_confidence(base: float, observed: Optional[float]) -> float:
    if observed is None:
        return round(base, 2)
    return round(min(0.95, (float(base) + float(observed)) / 2.0), 2)


def compose_reconfiguration_prompt(
    *,
    triggered_by: Iterable[str],
    context: Mapping[str, Any],
    batch: Optional[BatchContext] = None,
    pattern_scores: Optional[Mapping[str, float]] = None,
) -> ProcessReconfiguration:
    normalized_scores = {
        str(key).strip().upper(): float(value)
        for key, value in (pattern_scores or {}).items()
        if isinstance(value, (int, float))
    }
    parameter_deltas: list[ParameterDelta] = []
    tool_actions: list[ToolAction] = []
    risk = "low"
    notes: list[str] = []
    seen_parameters: set[str] = set()
    seen_actions: set[tuple[str, Optional[int], Optional[str], str]] = set()

    for raw_key in triggered_by:
        canonical_key = str(raw_key).strip().upper()
        rule = _PATTERN_RULES.get(canonical_key)
        if rule is None:
            continue

        if _RISK_RANK[rule["risk"]] > _RISK_RANK[risk]:
            risk = rule["risk"]
        observed_score = normalized_scores.get(canonical_key)

        for delta_spec in rule.get("parameter_deltas", []):
            parameter = str(delta_spec["parameter"])
            if parameter in seen_parameters:
                continue
            seen_parameters.add(parameter)
            parameter_deltas.append(
                ParameterDelta(
                    parameter=delta_spec["parameter"],
                    direction=delta_spec["direction"],
                    magnitude_pct=delta_spec["magnitude_pct"],
                    confidence=_merged_confidence(delta_spec["confidence"], observed_score),
                    evidence=[canonical_key],
                    rationale=delta_spec["rationale"],
                )
            )

        for action_spec in rule.get("tool_actions", []):
            action_key = (
                str(action_spec["action"]),
                action_spec.get("tool_number"),
                action_spec.get("tool_id"),
                str(action_spec["reason_code"]),
            )
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            tool_actions.append(
                ToolAction(
                    action=action_spec["action"],
                    tool_number=action_spec.get("tool_number"),
                    tool_id=action_spec.get("tool_id"),
                    reason_code=action_spec["reason_code"],
                    confidence=_merged_confidence(action_spec["confidence"], observed_score),
                    evidence=[canonical_key],
                )
            )

    if not parameter_deltas and not tool_actions:
        notes.append("no deterministic reconfiguration rule matched the triggered patterns")

    recipe_edits: list[RecipeEdit] = []
    if batch is not None and parameter_deltas:
        has_next_unit = batch.unit_count is None or batch.unit_index is None or (batch.unit_index + 1 < batch.unit_count)
        if has_next_unit:
            recipe_edits.append(
                RecipeEdit(
                    target="next_unit",
                    recipe_id=batch.recipe_id,
                    edits=[ParameterDelta.model_validate(delta.model_dump(mode="python")) for delta in parameter_deltas],
                    notes="bounded candidate update for the next unit in this batch; operator confirmation required",
                )
            )
            notes.append("added batch-level recipe edit target=next_unit from deterministic deltas")

    notes.append("composed via deterministic reconfig prompt")
    return ProcessReconfiguration(
        triggered_by=list(triggered_by),
        context=normalize_context(context),
        batch=batch,
        parameter_deltas=parameter_deltas,
        tool_actions=tool_actions,
        recipe_edits=recipe_edits,
        risk=risk,
        notes=notes,
    )


def _bool_env(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


async def maybe_enrich_reconfiguration_with_llm(
    proposal: ProcessReconfiguration,
    *,
    enable: bool = False,
    memory_snippets: Optional[Iterable[str]] = None,
    doc_snippets: Optional[Iterable[str]] = None,
) -> ProcessReconfiguration:
    if not enable and not _bool_env("RECONFIG_ENABLE_LLM_NARRATION", default=False):
        return proposal

    explainer = LLMExplainer()
    if not explainer.is_available():
        updated = ProcessReconfiguration.model_validate(proposal.model_dump(mode="python"))
        updated.notes.append("llm narration skipped: provider unavailable")
        return updated

    param_specs = [
        {
            "parameter": delta.parameter,
            "direction": delta.direction,
            "magnitude_pct": delta.magnitude_pct,
            "confidence": delta.confidence,
            "rationale": delta.rationale,
        }
        for delta in proposal.parameter_deltas
    ]
    tool_specs = [
        {
            "action": action.action,
            "tool_number": action.tool_number,
            "tool_id": action.tool_id,
            "reason_code": action.reason_code,
            "confidence": action.confidence,
        }
        for action in proposal.tool_actions
    ]
    evidence = {
        "triggered_by": list(proposal.triggered_by),
        "context": proposal.context,
        "batch": proposal.batch.model_dump(mode="json") if proposal.batch is not None else None,
        "risk": proposal.risk,
        "parameter_deltas": param_specs,
        "tool_actions": tool_specs,
        "memory_snippets": [str(item).strip() for item in (memory_snippets or []) if str(item).strip()][:5],
        "doc_snippets": [str(item).strip() for item in (doc_snippets or []) if str(item).strip()][:5],
    }
    prompt = (
        "You are writing an operator-facing reconfiguration narration for a CNC event. "
        "Do not invent parameter changes. Use only the provided deterministic proposal evidence. "
        "Return strict JSON with keys: summary (string), parameter_rationales (object mapping parameter->string)."
        "\n\nEvidence JSON:\n"
        f"{json.dumps(evidence, ensure_ascii=True, sort_keys=True)}"
    )

    updated = ProcessReconfiguration.model_validate(proposal.model_dump(mode="python"))
    try:
        payload = await explainer._call_llm_json_async(prompt, use_system_role=True)
        summary = str(payload.get("summary") or "").strip()
        rationale_map_raw = payload.get("parameter_rationales")
        rationale_map = rationale_map_raw if isinstance(rationale_map_raw, dict) else {}

        if summary:
            updated.notes.append(f"llm narrative: {summary}")

        for delta in updated.parameter_deltas:
            extra = str(rationale_map.get(delta.parameter) or "").strip()
            if extra:
                delta.rationale = extra

        updated.notes.append("llm narration source: llm")
        return updated
    except Exception:
        logger.debug("LLM narration failed for proposal %s", proposal.proposal_id, exc_info=True)
        updated.notes.append("llm narration failed; kept deterministic rationale")
        return updated

# ── LLM-generated end-of-batch reconfiguration ───────────────────────────────
# The LLM *authors* the proposal (which bounded parameter to change, tool action,
# and a plain-language reasoning) from the batch evidence — anomalies + operator
# feedback. Extension point: add parameters to _RECONFIG_BOUNDS and more fields
# to the `evidence` dict (e.g. MaaS / DPP data) as they become available.

_RECONFIG_BOUNDS: Dict[str, Dict[str, Any]] = {
    "feed_rate":     {"max_pct": 15.0, "directions": ["decrease", "hold"]},
    "spindle_speed": {"max_pct": 10.0, "directions": ["decrease", "increase", "hold"]},
    "feed_override": {"max_pct": 15.0, "directions": ["decrease", "hold"]},
    "depth_of_cut":  {"max_pct": 10.0, "directions": ["decrease", "hold"]},
    "stepover":      {"max_pct": 10.0, "directions": ["decrease", "hold"]},
    "coolant_flow":  {"max_pct": 20.0, "directions": ["increase", "hold"]},
}
_TOOL_ACTIONS = ["inspect", "replace", "rotate", "regrind", "no_action"]


async def generate_batch_reconfiguration(
    *,
    context: Mapping[str, Any],
    evidence: Dict[str, Any],
    fallback: ProcessReconfiguration,
    explainer: Optional[LLMExplainer] = None,
) -> ProcessReconfiguration:
    """Generate an end-of-batch reconfiguration proposal from the batch evidence.

    The LLM chooses bounded parameter changes + tool actions and writes a plain
    reasoning, grounded ONLY in the observed anomalies + operator feedback. Falls
    back to the deterministic ``fallback`` proposal when the LLM is unavailable or
    returns nothing usable. Always clamped to safety bounds; always requires
    operator confirmation.
    """
    explainer = explainer or LLMExplainer()

    def _with_provenance(rec: ProcessReconfiguration, generator: str, reasoning: Optional[str] = None) -> ProcessReconfiguration:
        rec.generator = generator
        if reasoning:
            rec.reasoning = reasoning
        rec.source_evidence = {
            "confirmed": evidence.get("confirmed"),
            "dismissed": evidence.get("dismissed"),
            "anomalies": evidence.get("anomalies"),
            "feedback": evidence.get("feedback"),
            "events": evidence.get("events"),
            "generator": generator,
        }
        return rec

    if not explainer.is_available():
        fallback.notes.append("llm reconfiguration unavailable — deterministic proposal")
        return _with_provenance(fallback, "deterministic")

    bounds_desc = {
        p: {"max_change_pct": b["max_pct"], "allowed_directions": b["directions"]}
        for p, b in _RECONFIG_BOUNDS.items()
    }
    prompt_obj = {
        "context": dict(context),
        "confirmed_catches": evidence.get("confirmed"),
        "dismissed": evidence.get("dismissed"),
        "anomalies": evidence.get("anomalies"),          # [{signature, count}]
        "operator_feedback": evidence.get("feedback"),   # [{action, reason, note}]
        "allowed_parameters": bounds_desc,
        "allowed_tool_actions": _TOOL_ACTIONS,
    }
    prompt = (
        "You are a manufacturing process engineer. Based ONLY on the observed anomalies and operator "
        "feedback below, propose a conservative reconfiguration of the CNC process for the NEXT unit in "
        "this batch. Typical responses: reduce feed_rate or spindle_speed when high vibration / chatter or "
        "confirmed tool breakage is observed; recommend a tool inspection or replacement when breakage is "
        "confirmed. Use ONLY the allowed parameters and stay within the bounds. If evidence is weak or "
        "mixed, prefer a tool inspection and a small change (or hold). A full optimisation would need "
        "additional process / MaaS data that is NOT available here — do not invent it. Ground every "
        "rationale in the given evidence.\n\n"
        "Return STRICT JSON with keys: reasoning (string, 1-3 sentences citing the anomalies/feedback), "
        "risk (low|medium|high), parameter_deltas (array of {parameter, direction, magnitude_pct, "
        "rationale}), tool_actions (array of {action, reason}). Do not wrap in markdown.\n\nINPUT:\n"
        f"{json.dumps(prompt_obj, ensure_ascii=True, sort_keys=True)}"
    )

    anomaly_keys = list(evidence.get("anomaly_keys") or [])
    n = int(evidence.get("confirmed", 0) or 0) + int(evidence.get("dismissed", 0) or 0)
    conf = round((int(evidence.get("confirmed", 0) or 0)) / (n + 4), 3) if n else 0.0
    try:
        payload = await explainer._call_llm_json_async(prompt, use_system_role=True)
        reasoning = str(payload.get("reasoning") or "").strip()
        risk = str(payload.get("risk") or "").strip().lower()
        if risk not in ("low", "medium", "high"):
            risk = fallback.risk

        deltas = []
        for d in (payload.get("parameter_deltas") or []):
            if not isinstance(d, dict):
                continue
            param = str(d.get("parameter", "")).strip()
            b = _RECONFIG_BOUNDS.get(param)
            if b is None:
                continue
            direction = str(d.get("direction", "")).strip().lower()
            if direction not in b["directions"]:
                direction = "decrease" if "decrease" in b["directions"] else b["directions"][0]
            mag = d.get("magnitude_pct")
            try:
                mag = float(mag) if mag is not None else None
            except (TypeError, ValueError):
                mag = None
            if mag is not None:
                mag = round(max(0.0, min(mag, b["max_pct"])), 1)
            rationale = str(d.get("rationale") or "").strip() or f"Precautionary {direction} of {param} from the batch evidence."
            deltas.append(ParameterDelta(parameter=param, direction=direction, magnitude_pct=mag,
                                         confidence=conf, evidence=anomaly_keys, rationale=rationale))

        tools = []
        for t in (payload.get("tool_actions") or []):
            if not isinstance(t, dict):
                continue
            action = str(t.get("action", "")).strip().lower()
            if action not in _TOOL_ACTIONS or action == "no_action":
                continue
            tools.append(ToolAction(action=action, tool_id=evidence.get("tool_id"),
                                    tool_number=evidence.get("tool_number"),
                                    reason_code="llm_batch_reconfig",
                                    confidence=round(min(0.9, 0.6 + 0.1 * int(evidence.get("confirmed", 0) or 0)), 2),
                                    evidence=anomaly_keys))

        if not deltas and not tools:
            fallback.notes.append("llm reconfiguration returned nothing usable — deterministic proposal")
            return _with_provenance(fallback, "deterministic")

        rec = ProcessReconfiguration(
            triggered_by=list(fallback.triggered_by),
            context=normalize_context(context),
            parameter_deltas=deltas,
            tool_actions=tools or list(fallback.tool_actions),
            risk=risk,
            requires_operator_confirmation=True,
            notes=[f"end-of-batch reconfiguration generated by LLM from "
                   f"{evidence.get('confirmed', 0)} confirmed / {evidence.get('dismissed', 0)} dismissed"],
        )
        return _with_provenance(rec, "llm", reasoning=reasoning or None)
    except Exception:
        logger.debug("LLM batch reconfiguration failed", exc_info=True)
        fallback.notes.append("llm reconfiguration failed — deterministic proposal")
        return _with_provenance(fallback, "deterministic")

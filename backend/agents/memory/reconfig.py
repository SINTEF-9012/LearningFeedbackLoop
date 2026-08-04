from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from backend.agents.core.batch_context import BatchContext
from backend.agents.knowledge.pack import ContextKeys

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_context(context: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    return ContextKeys(
        machine_type=(str(context.get("machine_type") or "").strip() or None),
        tool_type=(str(context.get("tool_type") or "").strip() or None),
        material=(str(context.get("material") or "").strip() or None),
        regime=(str(context.get("regime") or "").strip() or None),
    ).to_dict()


def context_matches(record_context: Mapping[str, Any], target_context: Mapping[str, Any]) -> bool:
    normalized_record = normalize_context(record_context)
    normalized_target = normalize_context(target_context)
    return all(normalized_record.get(key) == value for key, value in normalized_target.items())


class ParameterDelta(BaseModel):
    parameter: Literal[
        "feed_rate",
        "spindle_speed",
        "feed_override",
        "depth_of_cut",
        "stepover",
        "coolant_flow",
    ]
    direction: Literal["increase", "decrease", "hold"]
    magnitude_pct: Optional[float] = None
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    rationale: str


class ToolAction(BaseModel):
    action: Literal["inspect", "replace", "rotate", "regrind", "no_action"]
    tool_number: Optional[int] = None
    tool_id: Optional[str] = None
    reason_code: str
    confidence: float
    evidence: List[str] = Field(default_factory=list)


class RecipeEdit(BaseModel):
    target: Literal["next_unit", "this_batch", "recipe_template"]
    recipe_id: Optional[str] = None
    edits: List[ParameterDelta] = Field(default_factory=list)
    notes: Optional[str] = None


class ProcessReconfiguration(BaseModel):
    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=_now_utc)
    triggered_by: List[str] = Field(default_factory=list)
    context: Dict[str, Optional[str]]
    batch: Optional[BatchContext] = None
    parameter_deltas: List[ParameterDelta] = Field(default_factory=list)
    tool_actions: List[ToolAction] = Field(default_factory=list)
    recipe_edits: List[RecipeEdit] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]
    requires_operator_confirmation: bool = True
    operator_decision: Optional[Literal["accept", "reject", "modify", "manual"]] = None
    operator_decision_at: Optional[datetime] = None
    operator_decision_by: Optional[str] = None
    applied: bool = False
    applied_via: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    # Plain-language reasoning for the proposal (LLM-generated when available).
    reasoning: Optional[str] = None
    # Traceable provenance so the operator can see *why*: the feedback items and
    # the event trace the proposal was built from, plus how it was generated.
    # Kept as an open dict so it can be extended with more data (e.g. MaaS/DPP).
    source_evidence: Optional[Dict[str, Any]] = None
    # "llm" | "deterministic" — how the proposal itself was produced.
    generator: Optional[str] = None


def reconfig_store_path(path: str | Path | None = None) -> Path:
    return Path(path or "data/reconfig_outbox.jsonl")


def append_reconfig_record(record: ProcessReconfiguration, path: str | Path) -> int:
    target = reconfig_store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.model_dump(mode="json"), sort_keys=True, default=str) + "\n")
    return len(load_reconfig_records(target))


def load_reconfig_records(path: str | Path) -> List[ProcessReconfiguration]:
    target = reconfig_store_path(path)
    if not target.is_file():
        return []
    records: List[ProcessReconfiguration] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                records.append(ProcessReconfiguration.model_validate(payload))
            except (json.JSONDecodeError, ValidationError):
                logger.warning("reconfig store: skipping corrupt line in %s", target)
                continue
    return records


def latest_reconfig_records(
    path: str | Path,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> List[ProcessReconfiguration]:
    latest: Dict[str, ProcessReconfiguration] = {}
    for record in load_reconfig_records(path):
        if context is not None and not context_matches(record.context, context):
            continue
        if record.proposal_id in latest:
            del latest[record.proposal_id]
        latest[record.proposal_id] = record
    return list(reversed(list(latest.values())))


def get_latest_reconfig(path: str | Path, proposal_id: str) -> Optional[ProcessReconfiguration]:
    for record in latest_reconfig_records(path):
        if record.proposal_id == proposal_id:
            return record
    return None


def apply_reconfig_decision(
    record: ProcessReconfiguration,
    *,
    decision: Literal["accept", "reject", "modify"],
    operator_id: str,
    reason: Optional[str] = None,
    applied_via: Optional[str] = None,
    parameter_deltas: Optional[Iterable[ParameterDelta]] = None,
    tool_actions: Optional[Iterable[ToolAction]] = None,
    recipe_edits: Optional[Iterable[RecipeEdit]] = None,
) -> ProcessReconfiguration:
    updated = ProcessReconfiguration.model_validate(record.model_dump(mode="python"))
    updated.operator_decision = decision
    updated.operator_decision_at = _now_utc()
    updated.operator_decision_by = operator_id.strip()

    if decision == "accept":
        updated.applied = True
        updated.applied_via = (applied_via or updated.applied_via or "manual").strip()
    elif decision == "reject":
        updated.applied = False
        updated.applied_via = None
        if reason:
            updated.notes.append(f"operator reject reason: {reason}")
    elif decision == "modify":
        if parameter_deltas is not None:
            updated.parameter_deltas = list(parameter_deltas)
        if tool_actions is not None:
            updated.tool_actions = list(tool_actions)
        if recipe_edits is not None:
            updated.recipe_edits = list(recipe_edits)
        updated.applied = True
        updated.applied_via = (applied_via or updated.applied_via or "manual").strip()
        if reason:
            updated.notes.append(f"operator modify reason: {reason}")

    return updated


# volume-shrinkage constant shared with the MaaS evidence exporter: a recipe-level
# generalisation is only as confident as the feedback volume behind it.
_RECONFIG_PRIOR_N = 20.0


def compose_tool_condition_reconfiguration(
    *,
    context: Mapping[str, Any],
    severity: Literal["chipped", "broken"] = "broken",
    confirmed: int = 0,
    dismissed: int = 0,
    tool_id: Optional[str] = None,
    tool_number: Optional[int] = None,
    evidence: Optional[Iterable[str]] = None,
    triggered_by: Optional[Iterable[str]] = None,
    impact_co2_kg_per_catch: Optional[float] = None,
    suggest_parameter_delta: bool = True,
    feed_reduction_pct: float = 10.0,
) -> ProcessReconfiguration:
    """Compose a bounded reconfiguration proposal from confirmed tool-condition feedback.

    Honest by construction:
      - A confirmed break/chip directly warrants a TOOL ACTION (replace / inspect) on the
        immediate tool — high confidence, since one confirmation justifies acting on that
        tool. This is the directly evidence-backed part.
      - A recipe-level PARAMETER CHANGE (a small feed reduction for future units) is only
        a PRECAUTIONARY hypothesis: its confidence is shrunk by feedback volume and its
        magnitude is bounded and small, with a rationale that says so. It is never claimed
        to be the computed-optimal parameter.
    The proposal always `requires_operator_confirmation=True` — nothing here is autonomous.
    """
    evidence = list(evidence or [])
    n = confirmed + dismissed
    confirm_rate = (confirmed / n) if n else 0.0
    volume_conf = n / (n + _RECONFIG_PRIOR_N) if n else 0.0
    co2_note = (
        f"each confirmed catch avoids ~{impact_co2_kg_per_catch:.0f} kg CO2eq (DPP processing stage)"
        if impact_co2_kg_per_catch is not None else None
    )

    tool_action = ToolAction(
        action="replace" if severity == "broken" else "inspect",
        tool_number=tool_number,
        tool_id=tool_id,
        reason_code=f"confirmed_tool_{severity}",
        confidence=round(min(0.95, 0.6 + 0.1 * confirmed) * max(confirm_rate, 0.5), 2),
        evidence=evidence,
    )

    parameter_deltas: List[ParameterDelta] = []
    risk: Literal["low", "medium", "high"] = "low"
    if suggest_parameter_delta and confirmed >= 1:
        rationale = (
            f"precautionary: {confirmed} confirmed tool {severity} event(s) in this context; "
            f"a small feed reduction may lower breakage risk pending operator judgement "
            f"(not a computed-optimal value)."
        )
        if co2_note:
            rationale += f" {co2_note}."
        parameter_deltas.append(ParameterDelta(
            parameter="feed_rate",
            direction="decrease",
            magnitude_pct=round(min(feed_reduction_pct, 15.0), 1),
            confidence=round(confirm_rate * volume_conf, 3),
            evidence=evidence,
            rationale=rationale,
        ))
        risk = "medium"

    notes: List[str] = [
        f"composed from operator feedback: {confirmed} confirmed / {dismissed} dismissed",
    ]
    if co2_note:
        notes.append(co2_note)

    return ProcessReconfiguration(
        triggered_by=list(triggered_by or evidence),
        context=normalize_context(context),
        parameter_deltas=parameter_deltas,
        tool_actions=[tool_action],
        recipe_edits=[],
        risk=risk,
        requires_operator_confirmation=True,
        notes=notes,
    )


def build_manual_operator_action(
    *,
    triggered_by: Iterable[str],
    context: Mapping[str, Any],
    batch: Optional[BatchContext],
    parameter_deltas: Iterable[ParameterDelta],
    tool_actions: Iterable[ToolAction],
    recipe_edits: Iterable[RecipeEdit],
    risk: Literal["low", "medium", "high"],
    operator_id: str,
    reason: str,
    notes: Optional[Iterable[str]] = None,
) -> ProcessReconfiguration:
    record = ProcessReconfiguration(
        triggered_by=list(triggered_by),
        context=normalize_context(context),
        batch=batch,
        parameter_deltas=list(parameter_deltas),
        tool_actions=list(tool_actions),
        recipe_edits=list(recipe_edits),
        risk=risk,
        requires_operator_confirmation=False,
        operator_decision="manual",
        operator_decision_at=_now_utc(),
        operator_decision_by=operator_id.strip(),
        applied=True,
        applied_via="manual",
        notes=list(notes or []),
    )
    record.notes.append(f"manual operator reason: {reason}")
    return record
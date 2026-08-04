from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.core.batch_context import BatchContext
from backend.agents.llm.reconfig_prompt import (
    compose_reconfiguration_prompt,
    generate_batch_reconfiguration,
    maybe_enrich_reconfiguration_with_llm,
)
from backend.agents.knowledge.pack import ContextKeys
from backend.agents.memory.reconfig import (
    ParameterDelta,
    ProcessReconfiguration,
    RecipeEdit,
    ToolAction,
    append_reconfig_record,
    apply_reconfig_decision,
    build_manual_operator_action,
    compose_tool_condition_reconfiguration,
    get_latest_reconfig,
    latest_reconfig_records,
    load_reconfig_records,
    normalize_context,
    reconfig_store_path,
)

router = APIRouter(prefix="/reconfig", tags=["reconfig"])


def _store_path() -> Path:
    raw = (os.environ.get("RECONFIG_OUTBOX_PATH") or "").strip() or "data/reconfig_outbox.jsonl"
    return reconfig_store_path(raw)


def _validated_context(raw_context: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    ctx = ContextKeys(
        machine_type=raw_context.get("machine_type"),
        tool_type=raw_context.get("tool_type"),
        material=raw_context.get("material"),
        regime=raw_context.get("regime"),
    )
    missing = ctx.missing_required_keys()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "reconfiguration records require a complete context",
                "missing_context_keys": missing,
            },
        )
    return normalize_context(ctx.to_dict())


class ProposalCreateRequest(BaseModel):
    triggered_by: List[str] = Field(default_factory=list)
    context: Dict[str, Optional[str]]
    batch: Optional[BatchContext] = None
    parameter_deltas: List[ParameterDelta] = Field(default_factory=list)
    tool_actions: List[ToolAction] = Field(default_factory=list)
    recipe_edits: List[RecipeEdit] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]
    notes: List[str] = Field(default_factory=list)


class ReconfigListResponse(BaseModel):
    items: List[ProcessReconfiguration] = Field(default_factory=list)


class DecisionRequest(BaseModel):
    operator_id: str
    reason: Optional[str] = None
    applied_via: Optional[str] = None


class ModifyDecisionRequest(DecisionRequest):
    parameter_deltas: Optional[List[ParameterDelta]] = None
    tool_actions: Optional[List[ToolAction]] = None
    recipe_edits: Optional[List[RecipeEdit]] = None


class ManualOperatorActionRequest(ProposalCreateRequest):
    operator_id: str
    reason: str


class ComposeProposalRequest(BaseModel):
    triggered_by: List[str] = Field(default_factory=list)
    pattern_scores: Dict[str, float] = Field(default_factory=dict)
    context: Dict[str, Optional[str]]
    batch: Optional[BatchContext] = None
    llm_narrate: bool = False
    memory_snippets: List[str] = Field(default_factory=list)
    doc_snippets: List[str] = Field(default_factory=list)


@router.post("/compose", response_model=ProcessReconfiguration)
async def compose_reconfig_proposal(request: ComposeProposalRequest) -> ProcessReconfiguration:
    record = compose_reconfiguration_prompt(
        triggered_by=request.triggered_by,
        pattern_scores=request.pattern_scores,
        context=_validated_context(request.context),
        batch=request.batch,
    )
    record = await maybe_enrich_reconfiguration_with_llm(
        record,
        enable=request.llm_narrate,
        memory_snippets=request.memory_snippets,
        doc_snippets=request.doc_snippets,
    )
    if request.llm_narrate and (record.applied_via or "").strip() == "":
        record.applied_via = "llm_assisted_narration"
    append_reconfig_record(record, _store_path())
    return record


class ComposeBatchRequest(BaseModel):
    """End-of-batch reconfiguration request. A batch = one completed process
    (for now, one OF / workpiece, keyed by session_id)."""
    session_id: str
    operator_id: str = "operator"
    # Only propose for a context group with at least this many confirmed catches.
    min_confirmed: int = 1


class ComposeBatchResponse(BaseModel):
    session_id: str
    proposals: List[ProcessReconfiguration] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


def _session_tag(session_id: str) -> str:
    return f"session:{session_id}"


def _feedback_items(orch: Any, mem_id: str) -> List[Dict[str, Any]]:
    """Operator feedback for one memory, with reasons/notes, for proposal
    provenance (so the operator can see *why*)."""
    events: List[Any] = []
    store = getattr(orch, "store", None)
    if store is not None and hasattr(store, "list_feedback_events"):
        try:
            events = list(store.list_feedback_events(mem_id, limit=50))
        except Exception:
            events = []
    if not events:
        try:
            events = list(orch.feedback_handler.get_feedback_history(mem_id))
        except Exception:
            events = []
    out: List[Dict[str, Any]] = []
    for e in events:
        action = getattr(e, "action", None)
        if action is None and isinstance(e, dict):
            action = e.get("action")
        action = getattr(action, "value", action)
        data = getattr(e, "data", None)
        if data is None and isinstance(e, dict):
            data = e.get("data")
        data = data if isinstance(data, dict) else {}
        reason = data.get("reason") or data.get("comment")
        if str(action) in ("confirm", "dismiss", "comment"):
            out.append({"action": str(action), "reason": (str(reason).strip() if reason else None), "memory_id": mem_id})
    return out


def _memory_context(mem: Any) -> Dict[str, Optional[str]]:
    meta = getattr(mem, "metadata", None) or {}
    cc = meta.get("cutting_context") if isinstance(meta.get("cutting_context"), dict) else {}
    return {
        "machine_type": cc.get("machine_type") or cc.get("machine_id"),
        "tool_type": cc.get("tool_type"),
        "material": cc.get("workpiece_material") or cc.get("material"),
        "regime": cc.get("operating_regime") or cc.get("regime"),
    }


@router.post("/compose-batch", response_model=ComposeBatchResponse)
async def compose_batch_reconfiguration(request: ComposeBatchRequest) -> ComposeBatchResponse:
    """Aggregate a completed batch's anomalies + operator feedback into
    reconfiguration suggestions — the end-of-batch, strategic tier of the
    two-tier recommendation model. Groups the session's memories by cutting
    context, sums confirmed/dismissed feedback, and composes one bounded
    proposal per context that accumulated confirmed catches. Every proposal
    stays ``requires_operator_confirmation=True`` — nothing is applied here.
    """
    try:
        from backend.agents.memory.orchestrator import get_orchestrator
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"orchestrator unavailable: {exc}")

    orch = get_orchestrator()
    memories = orch.list_memories(session_id=request.session_id)

    # Aggregate feedback by cutting context.
    groups: Dict[tuple, Dict[str, Any]] = {}
    total_confirmed = 0
    total_dismissed = 0
    for mem in memories:
        try:
            stats = orch.feedback_handler.get_feedback_stats(getattr(mem, "id", "")) or {}
        except Exception:
            stats = {}
        confirmed = int(stats.get("confirms", 0) or 0)
        dismissed = int(stats.get("dismisses", 0) or 0)
        total_confirmed += confirmed
        total_dismissed += dismissed
        context = _memory_context(mem)
        key = tuple(context.get(k) for k in ("machine_type", "tool_type", "material", "regime"))
        group = groups.setdefault(key, {
            "context": context,
            "confirmed": 0,
            "dismissed": 0,
            "evidence": set(),
            "anomaly_counts": {},
            "events": [],
            "feedback": [],
            "tool_id": None,
            "tool_number": None,
        })
        group["confirmed"] += confirmed
        group["dismissed"] += dismissed
        mem_id = getattr(mem, "id", "")
        if mem_id:
            group["events"].append(mem_id)
            group["feedback"].extend(_feedback_items(orch, mem_id))
        for p in (getattr(mem, "pattern_keys", None) or []):
            key_str = getattr(p, "key", None) or (p if isinstance(p, str) else None)
            if key_str:
                group["evidence"].add(str(key_str))
                group["anomaly_counts"][str(key_str)] = group["anomaly_counts"].get(str(key_str), 0) + 1
        meta = getattr(mem, "metadata", None) or {}
        cc = meta.get("cutting_context") if isinstance(meta.get("cutting_context"), dict) else {}
        if group["tool_id"] is None and cc.get("tool_id"):
            group["tool_id"] = cc.get("tool_id")
        if group["tool_number"] is None and cc.get("tool_number") is not None:
            try:
                group["tool_number"] = int(cc.get("tool_number"))
            except (TypeError, ValueError):
                pass

    # Idempotency: use a deterministic proposal_id per (session, context) so
    # re-running "Close batch" refreshes the same proposal instead of piling up
    # duplicates. Preserve any proposal the operator already decided on.
    tag = _session_tag(request.session_id)
    decided_ids: set = set()
    latest_for_session: Dict[str, ProcessReconfiguration] = {}
    for rec in load_reconfig_records(_store_path()):
        if tag in (rec.triggered_by or []):
            latest_for_session[rec.proposal_id] = rec
    for pid, rec in latest_for_session.items():
        if rec.operator_decision is not None:
            decided_ids.add(pid)

    proposals: List[ProcessReconfiguration] = []
    for group in groups.values():
        if group["confirmed"] < request.min_confirmed:
            continue
        ctx_key = "|".join(str(group["context"].get(k) or "") for k in ("machine_type", "tool_type", "material", "regime"))
        det_id = "batch-" + hashlib.sha1(f"{request.session_id}|{ctx_key}".encode()).hexdigest()[:16]
        if det_id in decided_ids:
            # Operator already accepted/rejected/modified this one — don't reopen it.
            proposals.append(latest_for_session[det_id])
            continue
        evidence_keys = sorted(group["evidence"])
        # Deterministic proposal — used as the fallback when the LLM is unavailable.
        deterministic = compose_tool_condition_reconfiguration(
            context=group["context"],
            severity="chipped",  # batch aggregate → conservative inspect + precautionary delta
            confirmed=group["confirmed"],
            dismissed=group["dismissed"],
            tool_id=group["tool_id"],
            tool_number=group["tool_number"],
            evidence=evidence_keys,
            triggered_by=[_session_tag(request.session_id), *evidence_keys],
        )
        # Structured batch evidence for the LLM generator (extendable — add more
        # data sources here, e.g. MaaS / DPP, without changing the generator).
        anomalies = sorted(group["anomaly_counts"].items(), key=lambda kv: -kv[1])[:8]
        batch_evidence: Dict[str, Any] = {
            "confirmed": group["confirmed"],
            "dismissed": group["dismissed"],
            "anomalies": [{"signature": k.replace("_", " ").replace(":", " ").strip().lower(), "count": v} for k, v in anomalies],
            "anomaly_keys": evidence_keys,
            "feedback": [f for f in group["feedback"] if f.get("action") in ("confirm", "dismiss") or f.get("reason")][:12],
            "events": group["events"][:20],
            "tool_id": group["tool_id"],
            "tool_number": group["tool_number"],
        }
        # The LLM authors the reconfiguration from the evidence (bounded, with a
        # traceable reasoning); falls back to the deterministic proposal.
        record = await generate_batch_reconfiguration(
            context=group["context"],
            evidence=batch_evidence,
            fallback=deterministic,
        )
        record.proposal_id = det_id
        record.notes.append(f"end-of-batch aggregate for {_session_tag(request.session_id)}")
        proposals.append(record)
        append_reconfig_record(record, _store_path())

    summary = {
        "n_memories": len(memories),
        "n_contexts": len(groups),
        "n_proposals": len(proposals),
        "total_confirmed": total_confirmed,
        "total_dismissed": total_dismissed,
    }
    return ComposeBatchResponse(session_id=request.session_id, proposals=proposals, summary=summary)


@router.get("/batch/{session_id}", response_model=ReconfigListResponse)
async def list_batch_proposals(session_id: str) -> ReconfigListResponse:
    """List reconfiguration proposals composed for a given batch (session).

    Returns the latest record per proposal_id tagged with this session, newest
    first, so the review UI can show the operator what to accept / modify."""
    tag = _session_tag(session_id)
    records = load_reconfig_records(_store_path())
    # Keep the latest state per proposal_id (records are append-only; decisions
    # are re-appended), filtered to this session.
    latest: Dict[str, ProcessReconfiguration] = {}
    for rec in records:
        if tag in (rec.triggered_by or []):
            latest[rec.proposal_id] = rec
    items = sorted(latest.values(), key=lambda r: r.created_at, reverse=True)
    return ReconfigListResponse(items=items)


@router.get("/batch/{session_id}/summary")
async def batch_summary(session_id: str) -> Dict[str, Any]:
    """Fast, read-only feedback summary for a batch — no LLM, no persistence.

    Aggregates this session's memories + operator feedback exactly like
    ``compose-batch`` does, but skips the (slow) LLM proposal authoring. Lets the
    review UI always show how many confirmed catches the batch holds and how they
    split across cutting contexts, immediately and independent of composing.
    """
    try:
        from backend.agents.memory.orchestrator import get_orchestrator
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"orchestrator unavailable: {exc}")

    orch = get_orchestrator()
    memories = orch.list_memories(session_id=session_id)

    total_confirmed = 0
    total_dismissed = 0
    groups: Dict[tuple, Dict[str, Any]] = {}
    for mem in memories:
        try:
            stats = orch.feedback_handler.get_feedback_stats(getattr(mem, "id", "")) or {}
        except Exception:
            stats = {}
        confirmed = int(stats.get("confirms", 0) or 0)
        dismissed = int(stats.get("dismisses", 0) or 0)
        total_confirmed += confirmed
        total_dismissed += dismissed
        context = _memory_context(mem)
        key = tuple(context.get(k) for k in ("machine_type", "tool_type", "material", "regime"))
        group = groups.setdefault(key, {"context": context, "confirmed": 0, "dismissed": 0, "events": 0})
        group["confirmed"] += confirmed
        group["dismissed"] += dismissed
        group["events"] += 1

    contexts = sorted(groups.values(), key=lambda g: (-g["confirmed"], -g["events"]))
    return {
        "session_id": session_id,
        "n_memories": len(memories),
        "total_confirmed": total_confirmed,
        "total_dismissed": total_dismissed,
        "n_contexts": len(groups),
        "contexts": contexts,
    }


def _compute_total_summary() -> Dict[str, Any]:
    """Heavy, synchronous all-time aggregation. Run OFF the event loop (via
    asyncio.to_thread) — it scans every memory's feedback and must never block
    request handling."""
    from backend.agents.memory.orchestrator import get_orchestrator

    orch = get_orchestrator()
    memories = orch.list_memories(session_id=None)

    total_confirmed = 0
    total_dismissed = 0
    sessions: set = set()
    groups: Dict[tuple, Dict[str, Any]] = {}
    for mem in memories:
        sid = getattr(mem, "session_id", None)
        if sid:
            sessions.add(sid)
        try:
            stats = orch.feedback_handler.get_feedback_stats(getattr(mem, "id", "")) or {}
        except Exception:
            stats = {}
        confirmed = int(stats.get("confirms", 0) or 0)
        dismissed = int(stats.get("dismisses", 0) or 0)
        total_confirmed += confirmed
        total_dismissed += dismissed
        context = _memory_context(mem)
        key = tuple(context.get(k) for k in ("machine_type", "tool_type", "material", "regime"))
        group = groups.setdefault(key, {"context": context, "confirmed": 0, "dismissed": 0, "events": 0})
        group["confirmed"] += confirmed
        group["dismissed"] += dismissed
        group["events"] += 1

    contexts = sorted(groups.values(), key=lambda g: (-g["confirmed"], -g["events"]))

    # Prefer uncapped graph aggregates for the headline totals — the memory scan
    # above is capped (list_all LIMIT) and a live stream can flood it with
    # unfeedbacked memories, which would understate confirms/dismisses.
    store = getattr(orch, "store", None)
    gft = getattr(store, "global_feedback_totals", None)
    if callable(gft):
        try:
            total_confirmed, total_dismissed = gft()
        except Exception:
            pass
    cm = getattr(store, "count_memories", None)
    n_memories = len(memories)
    if callable(cm):
        try:
            n_memories = cm()
        except Exception:
            pass

    return {
        "scope": "total",
        "n_sessions": len(sessions),
        "n_memories": n_memories,
        "total_confirmed": total_confirmed,
        "total_dismissed": total_dismissed,
        "n_contexts": len(groups),
        "contexts": contexts,
    }


# Short TTL cache so repeated UI polls don't re-scan the whole store.
_TOTAL_SUMMARY_CACHE: Dict[str, Any] = {"value": None, "at": 0.0}
_TOTAL_SUMMARY_TTL = 60.0


@router.get("/summary/total")
async def total_summary() -> Dict[str, Any]:
    """All-time feedback summary across every operation on this system — the
    'Total' contrast to a single batch. Cached (60s) and computed off the event
    loop so a full-store scan never wedges the server.
    """
    import asyncio
    import time

    now = time.time()
    cached = _TOTAL_SUMMARY_CACHE.get("value")
    if cached is not None and (now - float(_TOTAL_SUMMARY_CACHE.get("at", 0.0))) < _TOTAL_SUMMARY_TTL:
        return cached
    try:
        result = await asyncio.to_thread(_compute_total_summary)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"total summary failed: {exc}")
    _TOTAL_SUMMARY_CACHE["value"] = result
    _TOTAL_SUMMARY_CACHE["at"] = time.time()
    return result


@router.get("/batch-sessions")
async def list_batch_sessions() -> Dict[str, List[str]]:
    """Session ids that have a stored batch reconfiguration proposal.

    The live session dict is cleaned up when playback ends, but the composed
    reconfiguration persists in the reconfig store. This lets the review UI offer
    *completed* batches (whose live session already ended) — a batch review is,
    by definition, done at the end of the batch. Newest first.
    """
    records = load_reconfig_records(_store_path())
    sessions: List[str] = []
    seen: set = set()
    for rec in reversed(records):  # append-only; reverse for newest-first
        for tag in (rec.triggered_by or []):
            if isinstance(tag, str) and tag.startswith("session:"):
                sid = tag.split("session:", 1)[1].strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    sessions.append(sid)
    return {"sessions": sessions}


@router.post("/proposal", response_model=ProcessReconfiguration)
async def create_reconfig_proposal(request: ProposalCreateRequest) -> ProcessReconfiguration:
    record = ProcessReconfiguration(
        triggered_by=list(request.triggered_by),
        context=_validated_context(request.context),
        batch=request.batch,
        parameter_deltas=list(request.parameter_deltas),
        tool_actions=list(request.tool_actions),
        recipe_edits=list(request.recipe_edits),
        risk=request.risk,
        notes=list(request.notes),
    )
    append_reconfig_record(record, _store_path())
    return record


@router.get("/proposal", response_model=ReconfigListResponse)
async def list_reconfig_proposals(
    machine_type: str = Query(...),
    tool_type: str = Query(...),
    material: str = Query(...),
    regime: str = Query(...),
) -> ReconfigListResponse:
    context = _validated_context(
        {
            "machine_type": machine_type,
            "tool_type": tool_type,
            "material": material,
            "regime": regime,
        }
    )
    return ReconfigListResponse(items=latest_reconfig_records(_store_path(), context=context))


def _require_existing_record(proposal_id: str) -> ProcessReconfiguration:
    record = get_latest_reconfig(_store_path(), proposal_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"message": f"proposal not found: {proposal_id}"})
    return record


@router.post("/{proposal_id}/accept", response_model=ProcessReconfiguration)
async def accept_reconfig_proposal(proposal_id: str, request: DecisionRequest) -> ProcessReconfiguration:
    updated = apply_reconfig_decision(
        _require_existing_record(proposal_id),
        decision="accept",
        operator_id=request.operator_id,
        applied_via=request.applied_via,
    )
    append_reconfig_record(updated, _store_path())
    return updated


@router.post("/{proposal_id}/reject", response_model=ProcessReconfiguration)
async def reject_reconfig_proposal(proposal_id: str, request: DecisionRequest) -> ProcessReconfiguration:
    reason = (request.reason or "").strip()
    if not reason:
        raise HTTPException(status_code=422, detail={"message": "reject requires a reason"})
    updated = apply_reconfig_decision(
        _require_existing_record(proposal_id),
        decision="reject",
        operator_id=request.operator_id,
        reason=reason,
    )
    append_reconfig_record(updated, _store_path())
    return updated


@router.post("/{proposal_id}/modify", response_model=ProcessReconfiguration)
async def modify_reconfig_proposal(proposal_id: str, request: ModifyDecisionRequest) -> ProcessReconfiguration:
    reason = (request.reason or "").strip()
    if not reason:
        raise HTTPException(status_code=422, detail={"message": "modify requires a reason"})
    updated = apply_reconfig_decision(
        _require_existing_record(proposal_id),
        decision="modify",
        operator_id=request.operator_id,
        reason=reason,
        applied_via=request.applied_via,
        parameter_deltas=request.parameter_deltas,
        tool_actions=request.tool_actions,
        recipe_edits=request.recipe_edits,
    )
    append_reconfig_record(updated, _store_path())
    return updated


@router.post("/manual", response_model=ProcessReconfiguration)
async def create_manual_operator_action(request: ManualOperatorActionRequest) -> ProcessReconfiguration:
    reason = request.reason.strip()
    if not reason:
        raise HTTPException(status_code=422, detail={"message": "manual action requires a reason"})
    record = build_manual_operator_action(
        triggered_by=request.triggered_by,
        context=_validated_context(request.context),
        batch=request.batch,
        parameter_deltas=request.parameter_deltas,
        tool_actions=request.tool_actions,
        recipe_edits=request.recipe_edits,
        risk=request.risk,
        operator_id=request.operator_id,
        reason=reason,
        notes=request.notes,
    )
    append_reconfig_record(record, _store_path())
    return record
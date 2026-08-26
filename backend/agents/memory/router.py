"""
Memory API Router - FastAPI endpoints for the memory system.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module provides HTTP and WebSocket endpoints for the memory system.
# Simple implementation for prototyping.
# ===========================================================================

Endpoints:
- POST /memory/events - Process a memory event
- GET /memory/{id} - Get a specific memory
- GET /memory/session/{session_id} - List memories for a session
- PATCH /memory/{id}/feedback - Add feedback to a memory
- WS /memory/alerts/{session_id} - Subscribe to alerts
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query, Body, Request
from pydantic import BaseModel, Field

from backend.agents.llm.docs_backend import get_docs_backend
from backend.agents.llm.alert_doc_linker import (
    DEFAULT_DOC_LINK_LIMIT,
    DEFAULT_DOC_LINK_SCORE_FLOOR,
    DEFAULT_DOC_LINK_TOP_K,
    propose_alert_doc_links,
)
from backend.agents.usecase import resolve_usecase

from ..core.schemas import PatternKey, PatternType, TimeRange, Memory, CaptureMemoryRequest, CaptureMemoryResponse
from ..core.batch_context import extract_batch_context
from ..core.context import CuttingContext
from ..core.metrics import MetricsComputer, compute_feature_vector
from ..patterns.generator import PatternGenerator
from .orchestrator import (
    MemoryEvent,
    get_orchestrator,
    get_scorer,
    get_store,
    get_explainer,
    get_orchestrator_config,
)
from .dispatcher import get_dispatcher
from .experiment_routes import router as experiment_router
from .experiment_summary_routes import router as experiment_summary_router
from .graph_routes import router as graph_router
from .init import persist_runtime_overrides
from .memory_feedback_routes import router as memory_feedback_router

logger = logging.getLogger(__name__)

# [PROTOTYPE_LLM_MEMORY_V1] - Create router
router = APIRouter(prefix="/memory", tags=["memory"])
router.include_router(experiment_summary_router)
router.include_router(memory_feedback_router)
# Must be included before the catch-all ``/{memory_id}`` route below, or
# ``/memory/graph/...`` and ``/memory/experiment/...`` would be matched as a
# memory id.
router.include_router(graph_router)
router.include_router(experiment_router)


def _convert_pattern_strings(
    pattern_strings: List[Any],
    pattern_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PatternKey]:
    patterns: List[PatternKey] = []
    for p in pattern_strings or []:
        if isinstance(p, PatternKey):
            patterns.append(p)
            continue
        if isinstance(p, dict):
            try:
                patterns.append(PatternKey(**p))
                continue
            except Exception:
                pass

        parts = str(p).split(":")
        if len(parts) < 2:
            continue
        category = parts[0].upper()
        key = str(p)
        if category in ("FREQ", "SPECTRAL", "PSD"):
            ptype = PatternType.SPECTRAL_PEAK
        elif category in ("RATIO", "AMP"):
            ptype = PatternType.RATIO
        elif category in ("ANOMALY", "OUTLIER"):
            ptype = PatternType.ANOMALY
        elif category in ("CLUSTER",):
            ptype = PatternType.CLUSTER
        else:
            ptype = PatternType.CUSTOM
        metadata = (pattern_metadata or {}).get(key, {})
        patterns.append(PatternKey(
            pattern_type=ptype,
            key=key,
            confidence=float(metadata.get("confidence", 1.0)) if metadata else 1.0,
            source_metric=metadata.get("source_metric") if metadata else None,
            additional=metadata or None,
        ))
    return patterns


# ============================================================================
# Helper Functions
# ============================================================================

def _memory_to_summary(mem: Memory) -> "MemorySummary":
    """Convert Memory to MemorySummary (reduces code duplication)."""
    return MemorySummary(
        id=mem.id,
        session_id=mem.session_id,
        created_at=mem.created_at.isoformat(),
        patterns=[p.key for p in mem.pattern_keys],
        label=mem.label,
        tags=mem.tags,
        significance_score=mem.metadata.get("significance_score") if mem.metadata else None,
        annotation_preview=mem.annotation_text[:100] if mem.annotation_text else None,
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class ProcessEventRequest(BaseModel):
    """Request to process a memory event."""
    session_id: str
    
    # Time range
    time_range: Optional[Dict[str, Any]] = None
    
    # Patterns (simplified input)
    pattern_keys: List[str] = Field(default_factory=list)

    # When True, ignore any supplied pattern_keys and derive them from raw metrics
    derive_patterns: bool = False
    
    # Cutting context (optional)
    cutting_context: Optional[Dict[str, Any]] = None
    
    # External signals
    external_signals: Optional[Dict[str, Any]] = None
    
    # Flat feature dict (e.g. 17 CNC features from casedata)
    # Passed through to MemoryEvent.raw_metrics for classical model scoring.
    metrics: Optional[Dict[str, Any]] = None
    
    # Channels
    channels: List[str] = Field(default_factory=list)

    # Arbitrary metadata (experiment flags, labels, etc.)
    metadata: Optional[Dict[str, Any]] = None

    # Optional annotation text
    annotation_text: Optional[str] = None

    # Tags
    tags: List[str] = Field(default_factory=list)


class ProcessEventResponse(BaseModel):
    """Response from processing a memory event."""
    processed: bool
    significant: bool
    memory_id: Optional[str] = None
    significance_score: float
    action: str
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None
    alert_line: Optional[str] = None
    alert_line_source: Optional[str] = None
    similar_memory_count: int = 0
    alert_dispatched: bool = False
    error: Optional[str] = None
    pattern_keys_used: List[str] = Field(default_factory=list)
    # Agent C (2026-04-24): per-model attribution snapshot.
    model_breakdown: Dict[str, Any] = Field(default_factory=dict)
    prior_boost: float = 0.0
    pattern_rule_score: float = 0.0
    triggered_rules: List[str] = Field(default_factory=list)


def _derive_request_pattern_keys(
    pattern_keys: List[str],
    raw_metrics: Optional[Dict[str, float]],
    derive_patterns: bool,
) -> List[str]:
    if not derive_patterns:
        return [str(key).strip() for key in (pattern_keys or []) if str(key).strip()]
    if not raw_metrics:
        return []
    try:
        from backend.agents.patterns.registry import detect_patterns

        derived = detect_patterns(raw_metrics).get("fired", [])
        return [str(key).strip() for key in derived if str(key).strip()]
    except Exception:
        logger.debug("Server-side pattern derivation failed", exc_info=True)
        return []


def _attach_feedback_scope(
    cutting_context: Optional[CuttingContext],
    metadata: Optional[Dict[str, Any]],
) -> Optional[CuttingContext]:
    scope_user_id = str((metadata or {}).get("feedback_scope_user_id") or "").strip()
    if not scope_user_id:
        return cutting_context
    if cutting_context is None:
        return CuttingContext(extra={"feedback_scope_user_id": scope_user_id})
    extra = dict(cutting_context.extra or {})
    extra["feedback_scope_user_id"] = scope_user_id
    return cutting_context.model_copy(update={"extra": extra})


class MemorySummary(BaseModel):
    """Brief memory summary for listing."""
    id: str
    session_id: str
    created_at: str
    patterns: List[str]
    label: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    significance_score: Optional[float] = None
    annotation_preview: Optional[str] = None


class ListMemoriesResponse(BaseModel):
    """Response for listing memories."""
    memories: List[MemorySummary]
    total_count: int


class MemoryDetailResponse(BaseModel):
    """Full memory detail response."""
    memory: Dict[str, Any]
    feedback_stats: Dict[str, Any]
    doc_links: List[ProposedDocLink] = Field(default_factory=list)


class TraceListResponse(BaseModel):
    memory_id: str
    traces: List[Dict[str, Any]]


class ProposedDocQuery(BaseModel):
    pattern_key: str
    query: str


class ProposedDocLink(BaseModel):
    id: Optional[str] = None
    citation: Optional[str] = None
    score: Optional[float] = None
    page: Optional[Any] = None
    file_name: Optional[str] = None
    source: Optional[str] = None
    usecase: Optional[str] = None
    machine: Optional[str] = None
    text: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    query_used: str
    pattern_key: str
    doc_feedback: Optional[str] = None
    helpful_count: int = 0
    not_helpful_count: int = 0
    feedback_score: float = 0.0
    evidence_entities: List[Dict[str, Any]] = Field(default_factory=list)


class ProposedDocLinksResponse(BaseModel):
    memory_id: str
    usecase: Optional[str] = None
    machine: Optional[str] = None
    top_k: int
    score_floor: float
    query_candidates: List[ProposedDocQuery] = Field(default_factory=list)
    doc_links: List[ProposedDocLink] = Field(default_factory=list)


def _alert_doc_request_matches_persisted_defaults(top_k: int, score_floor: float) -> bool:
    return int(top_k) == int(DEFAULT_DOC_LINK_TOP_K) and math.isclose(
        float(score_floor),
        float(DEFAULT_DOC_LINK_SCORE_FLOOR),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _query_candidates_from_doc_links(doc_links: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for link in doc_links:
        pattern_key = str(link.get("pattern_key") or "").strip()
        query_used = str(link.get("query_used") or "").strip()
        if not pattern_key or not query_used:
            continue
        key = (pattern_key, query_used)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"pattern_key": pattern_key, "query": query_used})
    return candidates


class ExternalSignalRequest(BaseModel):
    """Request to process an external signal."""
    session_id: str
    signal_type: str
    signal_value: Any
    time_range: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/capture", response_model=CaptureMemoryResponse)
async def capture_memory(request: CaptureMemoryRequest):
    """Create a memory from a UI capture/brush selection.

    The UI may include the selected samples; if present, the backend computes
    metrics/patterns and stores the memory immediately.
    """

    orchestrator = get_orchestrator()

    # Build TimeRange from sample indices.
    i0 = int(request.window.i0)
    i1 = int(request.window.i1)
    fs = float(request.window.fs)
    time_range = TimeRange(i0=i0, i1=i1, t0=i0 / fs, t1=i1 / fs, fs=fs)

    channels = list(request.channels or [])

    metrics_obj = None
    patterns: List[PatternKey] = []
    feature_vector: Optional[List[float]] = None

    if request.samples:
        # Determine channel order.
        if not channels:
            channels = [str(k) for k in request.samples.keys()]

        # Validate and shape data (n_channels, n_samples)
        arrays: List[List[float]] = []
        n_samples: Optional[int] = None
        for ch in channels:
            if ch not in request.samples:
                raise HTTPException(status_code=400, detail=f"samples missing channel '{ch}'")
            vals = request.samples.get(ch) or []
            if n_samples is None:
                n_samples = len(vals)
            elif len(vals) != n_samples:
                raise HTTPException(status_code=400, detail="all channel sample arrays must be same length")
            arrays.append([float(x) for x in vals])

        # Compute metrics/patterns from raw samples.
        import numpy as np

        data = np.asarray(arrays, dtype=float)
        if request.compute_metrics:
            metrics_obj = MetricsComputer(sample_rate=fs).compute(data, sample_rate=fs)

        if request.compute_patterns and metrics_obj is not None:
            generator = PatternGenerator()
            pattern_strings = generator.generate(metrics_obj)
            pattern_metadata = generator.get_pattern_metadata(metrics_obj)
            patterns = _convert_pattern_strings(pattern_strings, pattern_metadata=pattern_metadata)

        if request.include_feature_vector:
            try:
                vec = compute_feature_vector(data, sample_rate=fs, max_channels=4)
                feature_vector = [float(x) for x in vec.tolist()]
            except Exception:
                feature_vector = None

    # Store operator memory (always stores, bypassing significance gating).
    memory = await orchestrator.create_operator_memory(
        session_id=request.session_id,
        time_range=time_range,
        channels=channels,
        annotation_text=request.annotation_text,
        tags=request.tags,
        label=request.label,
        created_by=request.created_by,
        metrics=metrics_obj,
        patterns=patterns,
        feature_vector=feature_vector,
        metadata=request.metadata,
    )

    return CaptureMemoryResponse(ok=True, memory_id=str(memory.id))


# ============================================================================
# HTTP Endpoints
# ============================================================================

@router.post("/events", response_model=ProcessEventResponse)
async def process_event(request: ProcessEventRequest):
    """
    Process a memory event.
    
    [PROTOTYPE_LLM_MEMORY_V1] - Main event processing endpoint.
    
    Call this when patterns are detected by the feature extractor.
    """
    try:
        orchestrator = get_orchestrator()
        
        # Build time range
        time_range = TimeRange(
            i0=request.time_range.get("i0", 0) if request.time_range else 0,
            i1=request.time_range.get("i1", 100) if request.time_range else 100,
            t0=request.time_range.get("t0", 0.0) if request.time_range else 0.0,
            t1=request.time_range.get("t1", 1.0) if request.time_range else 1.0,
            fs=request.time_range.get("fs", 1000.0) if request.time_range else 1000.0,
        )
        
        # Build raw_metrics for classical model scoring (flat feature dict)
        raw_metrics = None
        if request.metrics:
            raw_metrics = {
                k: float(v) for k, v in request.metrics.items()
                if isinstance(v, (int, float))
            }

        pattern_keys_used = _derive_request_pattern_keys(
            request.pattern_keys,
            raw_metrics,
            request.derive_patterns,
        )
        patterns = [
            PatternKey(pattern_type=PatternType.CUSTOM, key=key)
            for key in pattern_keys_used
        ]
        
        # Build cutting context
        cutting_context = None
        if request.cutting_context:
            cutting_context = CuttingContext(**request.cutting_context)
        cutting_context = _attach_feedback_scope(cutting_context, request.metadata)

        # Create event
        event = MemoryEvent(
            session_id=request.session_id,
            time_range=time_range,
            patterns=patterns,
            cutting_context=cutting_context,
            external_signals=request.external_signals or {},
            channels=request.channels,
            raw_metrics=raw_metrics,
            batch=extract_batch_context(request.metadata),
            metadata=request.metadata,
        )
        
        # Process
        result = await orchestrator.process_event(event)

        # Bridge experiment feature data to the PubSub "features" channel
        # so the SinditBridge (if running) picks it up and pushes to SINDIT KG.
        if raw_metrics:
            try:
                from backend.events import publish_feature
                await publish_feature(
                    session_id=request.session_id,
                    payload={"features": raw_metrics, "session_id": request.session_id},
                )
            except Exception:
                pass  # non-fatal — don't break event processing

        return ProcessEventResponse(
            processed=result.processed,
            significant=result.significant,
            memory_id=result.memory_id,
            significance_score=result.significance_score,
            action=result.action.value,
            explanation=result.explanation,
            explanation_source=getattr(result, 'explanation_source', None),
            alert_line=getattr(result, 'alert_line', None),
            alert_line_source=getattr(result, 'alert_line_source', None),
            similar_memory_count=len(result.similar_memories),
            alert_dispatched=result.alert_dispatched,
            error=result.error,
            pattern_keys_used=pattern_keys_used,
            model_breakdown=getattr(result, 'model_breakdown', {}) or {},
            prior_boost=float(getattr(result, 'prior_boost', 0.0) or 0.0),
            pattern_rule_score=float(getattr(result, 'pattern_rule_score', 0.0) or 0.0),
            triggered_rules=list(getattr(result, 'triggered_rules', []) or []),
        )
        
    except Exception as e:
        logger.error(f"Event processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Batch event processing — submit many events in a single HTTP round-trip
# ------------------------------------------------------------------
class BatchProcessEventsRequest(BaseModel):
    """Batch of events to process."""
    events: List[ProcessEventRequest]


class BatchProcessEventsResponse(BaseModel):
    """Results for a batch of events."""
    results: List[ProcessEventResponse]


@router.post("/events/batch", response_model=BatchProcessEventsResponse)
async def process_events_batch(request: BatchProcessEventsRequest):
    """Process a batch of memory events in a single request.

    This is semantically equivalent to calling ``POST /events`` for each
    element, but eliminates per-event HTTP round-trip overhead when running
    experiments.  Events are processed sequentially (maintaining arrival
    order) but without the network cost of individual requests.
    """
    results: List[ProcessEventResponse] = []
    orchestrator = get_orchestrator()

    for req in request.events:
        try:
            time_range = TimeRange(
                i0=req.time_range.get("i0", 0) if req.time_range else 0,
                i1=req.time_range.get("i1", 100) if req.time_range else 100,
                t0=req.time_range.get("t0", 0.0) if req.time_range else 0.0,
                t1=req.time_range.get("t1", 1.0) if req.time_range else 1.0,
                fs=req.time_range.get("fs", 1000.0) if req.time_range else 1000.0,
            )

            raw_metrics = None
            if req.metrics:
                raw_metrics = {
                    k: float(v) for k, v in req.metrics.items()
                    if isinstance(v, (int, float))
                }

            pattern_keys_used = _derive_request_pattern_keys(
                req.pattern_keys,
                raw_metrics,
                req.derive_patterns,
            )
            patterns = [
                PatternKey(pattern_type=PatternType.CUSTOM, key=key)
                for key in pattern_keys_used
            ]

            cutting_context = None
            if req.cutting_context:
                cutting_context = CuttingContext(**req.cutting_context)
            cutting_context = _attach_feedback_scope(cutting_context, req.metadata)

            event = MemoryEvent(
                session_id=req.session_id,
                time_range=time_range,
                patterns=patterns,
                cutting_context=cutting_context,
                external_signals=req.external_signals or {},
                channels=req.channels,
                raw_metrics=raw_metrics,
                batch=extract_batch_context(req.metadata),
                metadata=req.metadata,
            )

            result = await orchestrator.process_event(event)

            results.append(ProcessEventResponse(
                processed=result.processed,
                significant=result.significant,
                memory_id=result.memory_id,
                significance_score=result.significance_score,
                action=result.action.value,
                explanation=result.explanation,
                explanation_source=getattr(result, 'explanation_source', None),
                alert_line=getattr(result, 'alert_line', None),
                alert_line_source=getattr(result, 'alert_line_source', None),
                similar_memory_count=len(result.similar_memories),
                alert_dispatched=result.alert_dispatched,
                error=result.error,
                pattern_keys_used=pattern_keys_used,
                model_breakdown=getattr(result, 'model_breakdown', {}) or {},
                prior_boost=float(getattr(result, 'prior_boost', 0.0) or 0.0),
                pattern_rule_score=float(getattr(result, 'pattern_rule_score', 0.0) or 0.0),
                triggered_rules=list(getattr(result, 'triggered_rules', []) or []),
            ))
        except Exception as e:
            logger.warning("Batch item failed: %s", e)
            results.append(ProcessEventResponse(
                processed=False,
                significant=False,
                significance_score=0.0,
                action="IGNORE",
                error=str(e),
            ))

    return BatchProcessEventsResponse(results=results)


@router.post("/signals", response_model=ProcessEventResponse)
async def process_external_signal(request: ExternalSignalRequest):
    """
    Process an external signal (e.g., from classical models).
    
    [PROTOTYPE_LLM_MEMORY_V1] - For external model integration.
    """
    try:
        orchestrator = get_orchestrator()
        
        time_range = None
        if request.time_range:
            time_range = TimeRange(**request.time_range)
        
        result = await orchestrator.process_external_signal(
            session_id=request.session_id,
            signal_type=request.signal_type,
            signal_value=request.signal_value,
            time_range=time_range,
            metadata=request.metadata,
        )
        
        return ProcessEventResponse(
            processed=result.processed,
            significant=result.significant,
            memory_id=result.memory_id,
            significance_score=result.significance_score,
            action=result.action.value,
            explanation=result.explanation,
            explanation_source=getattr(result, 'explanation_source', None),
            alert_line=getattr(result, 'alert_line', None),
            alert_line_source=getattr(result, 'alert_line_source', None),
            similar_memory_count=len(result.similar_memories),
            alert_dispatched=result.alert_dispatched,
            error=result.error,
        )
        
    except Exception as e:
        logger.error(f"Signal processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Pattern Registry Endpoints
# ============================================================================

@router.get("/patterns/discovered")
async def list_discovered_patterns(
    promoted_only: bool = Query(default=False, description="Only return promoted patterns"),
):
    """Return all automatically discovered patterns with provenance.

    Each pattern includes:
    - ``key`` — unique pattern key (``discovered:...``)
    - ``features`` — feature → direction map
    - ``confirmation_count`` — how many confirmed events contributed
    - ``promoted`` — whether the pattern is actively scoring
    - ``prior`` — current Bayesian prior
    - ``first_seen`` / ``last_seen`` — timestamps
    - ``source_events`` — list of contributing events with memory_id,
      session_id, timestamp, deviations, z_scores

    If Neo4j is the backend, also includes ``source_memory_ids`` from
    the graph (`:DISCOVERED_FROM` edges).
    """
    orch = get_orchestrator()
    patterns = orch.pattern_discovery.get_patterns()

    # Merge with Neo4j data if available
    neo4j_lookup: Dict[str, Dict[str, Any]] = {}
    if hasattr(orch.store, 'list_discovered_patterns'):
        try:
            for dp in orch.store.list_discovered_patterns(promoted_only=promoted_only):
                neo4j_lookup[dp["key"]] = dp
        except Exception:
            pass

    result = []
    for key, pat in sorted(patterns.items(), key=lambda kv: kv[1].last_seen, reverse=True):
        if promoted_only and not pat.promoted:
            continue
        d = pat.to_dict()
        d["source_events"] = [se.to_dict() for se in pat.source_events]
        # Enrich with Neo4j graph data if available
        neo = neo4j_lookup.get(key, {})
        if neo.get("source_memory_ids"):
            d["source_memory_ids"] = neo["source_memory_ids"]
        result.append(d)

    return {"discovered_patterns": result, "count": len(result)}


@router.get("/patterns")
async def list_patterns(
    category: Optional[str] = Query(default=None, description="Filter by category (fault, domain, ts_derived)"),
    polarity: Optional[str] = Query(default=None, description="Filter by polarity (fault_supporting, protective, uninformative)"),
    enabled_only: bool = Query(default=False),
):
    """List all registered pattern definitions.

    Returns the shared pattern catalogue used by both the live pipeline
    and experiment evaluator.
    """
    from ..patterns.registry import list_patterns_dict
    return {
        "patterns": list_patterns_dict(
            enabled_only=enabled_only,
            category=category,
            polarity=polarity,
        )
    }


@router.post("/patterns/detect")
async def detect_patterns_endpoint(
    body: Dict[str, Any] = Body(...),
):
    """Run all enabled pattern detectors on a feature vector.

    Body::

        {
          "features": {"power_spindle_delta_max": 18.5, ...},
          "thresholds": {},          // optional calibrated thresholds
          "include_details": true    // optional, default false
        }

    Returns ``{fired, count, details?}``.
    """
    from ..patterns.registry import detect_patterns as _detect
    features = body.get("features", {})
    thresholds = body.get("thresholds")
    include_details = body.get("include_details", False)
    return _detect(features, thresholds, include_details=include_details)


@router.get("/priors")
async def list_pattern_priors(
    pattern_key: Optional[str] = Query(
        default=None, description="Return a single pattern's prior record instead of all"
    ),
):
    """Learned pattern priors as structured, auditable records.

    Each record pairs the derived prior with the confirm/dismiss feedback counts
    behind it and a volume-based confidence (see ``PatternPrior``). Read-only —
    does not change scoring or persistence; surfaces §5.8 "auditable, traceable
    priors" for inspection.
    """
    orchestrator = get_orchestrator()
    scorer = getattr(orchestrator, "scorer", None)
    if scorer is None:
        raise HTTPException(status_code=503, detail="scorer unavailable")
    if pattern_key:
        return {"prior": scorer.get_pattern_prior_record(pattern_key).to_dict()}
    records = scorer.list_pattern_prior_records()
    return {"count": len(records), "priors": [r.to_dict() for r in records]}


# ============================================================================
# Knowledge Graph Endpoints
# ============================================================================

@router.get("/session/{session_id}", response_model=ListMemoriesResponse)
async def list_session_memories(
    session_id: str,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    List memories for a session.
    """
    orchestrator = get_orchestrator()
    
    memories = orchestrator.list_memories(session_id=session_id)
    total_count = len(memories)
    
    # Paginate
    memories = memories[offset:offset + limit]
    
    # Convert to summaries using helper
    summaries = [_memory_to_summary(mem) for mem in memories]
    
    return ListMemoriesResponse(
        memories=summaries,
        total_count=total_count,
    )


@router.get("/", response_model=ListMemoriesResponse)
async def list_all_memories(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    List all memories.
    """
    orchestrator = get_orchestrator()
    
    memories = orchestrator.list_memories()
    total_count = len(memories)
    
    # Paginate
    memories = memories[offset:offset + limit]
    
    # Convert to summaries using helper
    summaries = [_memory_to_summary(mem) for mem in memories]
    
    return ListMemoriesResponse(
        memories=summaries,
        total_count=total_count,
    )


@router.websocket("/alerts/{session_id}")
async def websocket_alerts(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time alerts.
    
    [PROTOTYPE_LLM_MEMORY_V1] - Simple alert streaming.
    
    Connect to receive alerts for a specific session.
    Use session_id="all" for all sessions.
    """
    # Validate session exists (unless subscribing to all)
    if session_id != "all":
        sessions = getattr(websocket.app.state, "sessions", {})
        if session_id not in sessions:
            await websocket.close(code=4404)
            return

    await websocket.accept()
    
    dispatcher = get_dispatcher()
    
    # Subscribe to alerts
    if session_id == "all":
        queue = dispatcher.subscribe(session_id=None)
    else:
        queue = dispatcher.subscribe(session_id=session_id)
    
    logger.info(f"Client subscribed to alerts: session={session_id}")
    
    try:
        while True:
            # Wait for alert
            alert = await queue.get()
            
            # Send to client
            await websocket.send_text(alert.to_json())
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from alerts: session={session_id}")
    except Exception as e:
        logger.error(f"Alert WebSocket error: {e}")
    finally:
        dispatcher.unsubscribe(queue, session_id if session_id != "all" else None)


@router.websocket("/alerts")
async def websocket_alerts_global(websocket: WebSocket):
    """WebSocket endpoint for all alerts (global subscription)."""
    await websocket_alerts(websocket, "all")


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/config")
async def get_memory_config():
    """
    Return the active memory-system configuration (safe subset).

    Useful for the UI to discover which storage backend, LLM model,
    SINDIT settings, and scoring thresholds are active without hard-coding
    them on the client side.
    """
    from ..config import get_config
    cfg = get_config()
    orch = get_orchestrator()
    return {
        "storage_backend": cfg.storage_backend,
        "neo4j_uri": cfg.neo4j_uri,
        "neo4j_database": cfg.neo4j_database,
        "sindit_enabled": cfg.sindit_enabled,
        "sindit_api_url": cfg.sindit_api_url,
        "llm_provider": cfg.llm_provider,
        "ollama_url": cfg.ollama_url,
        "ollama_model": cfg.ollama_model,
        "groq_model": cfg.groq_model,
        "thresholds": {
            "store": cfg.thresholds.store_threshold,
            "alert": cfg.thresholds.alert_threshold,
            "critical": cfg.thresholds.critical_threshold,
        },
        "enable_ann": cfg.enable_ann,
        "enable_embeddings": cfg.enable_embeddings,
        "generate_explanations": orch.config.generate_explanations,
        "dispatch_alerts": orch.config.dispatch_alerts,
        "llm_available": orch.explainer.is_available() if hasattr(orch, "explainer") else False,
    }


@router.patch("/config")
async def patch_memory_config(body: Dict[str, Any]):
    """Toggle runtime configuration flags.

    Supported keys:
        generate_explanations (bool) — Enable/disable LLM-powered explanations.
            When disabled, the system uses fast heuristic fallbacks and never
            contacts Ollama, eliminating the timeout errors you see when no
            LLM server is running.
        dispatch_alerts (bool) — Enable/disable WebSocket alert dispatch.

    Returns the updated config snapshot.
    """
    config = get_orchestrator_config()
    changed = {}
    pending = {}

    if "generate_explanations" in body:
        val = bool(body["generate_explanations"])
        pending["generate_explanations"] = val

    if "dispatch_alerts" in body:
        val = bool(body["dispatch_alerts"])
        pending["dispatch_alerts"] = val

    persisted = {}
    if pending:
        try:
            persisted = persist_runtime_overrides(pending)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not persist runtime config: {exc}") from exc

    if "dispatch_alerts" in pending:
        val = pending["dispatch_alerts"]
        config.dispatch_alerts = val
        changed["dispatch_alerts"] = val
        logger.info("Runtime config: dispatch_alerts → %s", val)

    if "generate_explanations" in pending:
        val = pending["generate_explanations"]
        config.generate_explanations = val
        changed["generate_explanations"] = val
        logger.info("Runtime config: generate_explanations → %s", val)

    return {"ok": True, "changed": changed, "persisted": persisted}


@router.get("/stats/overview")
async def get_memory_stats():
    """
    Get overview statistics of the memory system.
    """
    orchestrator = get_orchestrator()
    
    all_memories = orchestrator.list_memories()
    
    # Count by action
    action_counts: Dict[str, int] = {}
    session_counts: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    
    for mem in all_memories:
        # By action
        action = mem.metadata.get("significance_action", "unknown") if mem.metadata else "unknown"
        action_counts[action] = action_counts.get(action, 0) + 1
        
        # By session
        session_counts[mem.session_id] = session_counts.get(mem.session_id, 0) + 1
        
        # By pattern
        for p in mem.pattern_keys:
            pattern_type = p.key.split(":")[0] if ":" in p.key else p.key.split("_")[0]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    return {
        "total_memories": len(all_memories),
        "by_action": action_counts,
        "by_session": session_counts,
        "by_pattern_type": pattern_counts,
        "scorer_priors": dict(list((orchestrator.scorer._pattern_priors or {}).items())[:10]),
        "dispatch_alerts": bool(getattr(orchestrator, "config", None) and getattr(orchestrator.config, "dispatch_alerts", False)),
    }


@router.post("/scorer/reset-priors")
async def reset_scorer_priors():
    """
    Reset all pattern priors to default.
    
    [PROTOTYPE_LLM_MEMORY_V1] - For testing/debugging.
    """
    scorer = get_scorer()
    if hasattr(scorer, "reset_feedback_state"):
        scorer.reset_feedback_state()
    else:
        scorer._pattern_priors.clear()
        if hasattr(scorer, "_local_feedback_counts"):
            scorer._local_feedback_counts.clear()
    return {
        "message": "Pattern priors cache reset (durable feedback not deleted)",
        "note": "Priors are derived from feedback events; delete feedback events separately if desired.",
    }


@router.get("/scorer/priors")
async def get_scorer_priors(limit: int = Query(default=50, ge=1, le=500)):
    """Return current pattern priors (highest first).

    This is primarily for demos/visualizations to show how operator feedback
    shifts significance scoring over time.
    """
    scorer = get_scorer()
    if hasattr(scorer, "refresh_priors"):
        try:
            scorer.refresh_priors()
        except Exception:
            pass
    priors = dict(scorer._pattern_priors or {})
    diagnostics = (
        scorer.get_feedback_diagnostics()
        if hasattr(scorer, "get_feedback_diagnostics")
        else {}
    )
    items = sorted(priors.items(), key=lambda kv: float(kv[1]), reverse=True)
    if limit:
        items = items[:limit]
    return {
        "count": len(priors),
        "returned": len(items),
        "priors": [
            {
                "pattern": k,
                "prior": float(v),
                "effective_weight_total": float((diagnostics.get(k) or {}).get("effective_weight_total", 0.0) or 0.0),
                "passive_outcome_count": int((diagnostics.get(k) or {}).get("passive_outcome_count", 0) or 0),
                "severity_calibration": (diagnostics.get(k) or {}).get("severity_calibration") or {
                    "average_delta": 0.0,
                    "weight_total": 0.0,
                    "targets": {"info": 0.0, "warning": 0.0, "critical": 0.0},
                },
            }
            for k, v in items
        ],
    }


@router.get("/scorer/prior")
async def get_scorer_prior(
    pattern_key: str = Query(..., description="Pattern key string, e.g. RATIO_Fx_Fy:>5"),
    session_id: Optional[str] = Query(default=None, description="Optional session context (currently informational)"),
):
    """Return the derived prior for a single pattern.

    Note: context-specific priors are computed inside scoring using CuttingContext.
    """
    _ = session_id
    scorer = get_scorer()
    prior = scorer.get_pattern_prior(pattern_key, context=None)
    diagnostics = (
        scorer.get_feedback_diagnostics(pattern_key)
        if hasattr(scorer, "get_feedback_diagnostics")
        else {}
    )
    return {
        "pattern": pattern_key,
        "prior": float(prior),
        "effective_weight_total": float(diagnostics.get("effective_weight_total", 0.0) or 0.0),
        "passive_outcome_count": int(diagnostics.get("passive_outcome_count", 0) or 0),
        "severity_calibration": diagnostics.get("severity_calibration") or {
            "average_delta": 0.0,
            "weight_total": 0.0,
            "targets": {"info": 0.0, "warning": 0.0, "critical": 0.0},
        },
    }


@router.get("/counterfactual")
async def get_counterfactual(
    snapshot: Optional[str] = Query(
        default=None,
        description="Override snapshot path; defaults to the full-stack Site_a_line2 case study.",
    ),
):
    """Return the measured 'what did operator feedback change' summary (plan 2.4).

    Serves a **measured** counterfactual over the validated Site_a_line2 breakage
    case study (a stored replay snapshot), not a live re-run of the current
    session. Returns ``{available: false}`` when no snapshot is present so the
    UI can hide the panel gracefully.
    """
    from .counterfactual import load_counterfactual_summary

    summary = load_counterfactual_summary(snapshot)
    if summary is None:
        return {"available": False}
    return {"available": True, **summary}


@router.get("/maas-evidence")
async def get_maas_evidence(
    path: Optional[str] = Query(
        default=None,
        description="Override the capability-evidence artifact path.",
    ),
):
    """Return the MaaS capability-evidence artifact (read-only UI surface).

    The aggregate that 'propagates up' to a MaaS matchmaking platform:
    per-(plant, context, capability) declared→measured capability with
    confirm-rate, volume-shrunk confidence, and CO₂ avoided per confirmed catch —
    never raw signals or memory contents. **Illustrative**: no live matchmaking
    service consumes it (see the handoff doc's claim boundary). Returns
    ``{available: false}`` when the artifact is absent.
    """
    from ..maas.evidence_summary import load_evidence_summary

    summary = load_evidence_summary(path)
    if summary is None:
        return {"available": False}
    return {"available": True, **summary}


@router.get("/maas-evidence/facets")
async def get_maas_evidence_facets():
    """Return all MaaS evidence facets (read-only UI surface).

    Loads the four evidence objects the loop exposes to a matchmaking platform —
    capability, fault & lead-time, availability-adjustment, and realised
    sustainability — each an aggregate over confirmed operator feedback (and, for
    sustainability, plant-catalogue + DPP figures). **Illustrative**: no live
    matchmaking service consumes these. Facets whose artifact is absent come back
    ``None`` so the UI hides that panel.
    """
    from ..maas.evidence_summary import load_evidence_facets

    return load_evidence_facets()


@router.get("/loop_metrics")
async def get_loop_metrics(
    session_id: Optional[str] = Query(default=None, description="Scope to a single session"),
    persist: bool = Query(default=False, description="Append a daily snapshot to data/loop_metrics/"),
):
    """Return feedback-loop quality metrics.

    Agent A (2026-04-24). Aggregates:
      - ``rule_performance``: per-rule precision / recall / F1 / sample count,
        sourced from ``SignificanceScorer._rule_performance`` (sliding window).
      - ``adaptive_thresholds``: current alert / store / critical thresholds
        and moving precision from ``AdaptiveThresholds``.
      - ``session``: optional per-session confirm/dismiss counts and rate.
      - ``totals``: overall confirm/dismiss counts across the memory store.
      - ``retrainer``: last retrain state if available.

    Caveat: current precision is computed on the same feedback window the
    rule weights were tuned on (self-report). A hold-out evaluation is on
    the roadmap; this endpoint is a monitoring surface, not a model
    evaluation.
    """
    orchestrator = get_orchestrator()
    scorer = orchestrator.scorer

    # Per-rule performance.
    rule_perf: Dict[str, Any] = {}
    try:
        rule_perf = scorer.get_rule_performance() if hasattr(scorer, "get_rule_performance") else {}
    except Exception:
        rule_perf = {}

    # Adaptive thresholds snapshot.
    adaptive: Dict[str, Any] = {}
    try:
        at = getattr(scorer, "_adaptive_thresholds", None)
        if at is not None and hasattr(at, "to_dict"):
            adaptive = at.to_dict()
    except Exception:
        adaptive = {}

    model_trust: Dict[str, Any] = {}
    try:
        if hasattr(scorer, "get_model_trust_diagnostics"):
            model_trust = scorer.get_model_trust_diagnostics()
    except Exception:
        model_trust = {}

    # Session-scoped feedback roll-up.
    session_block: Optional[Dict[str, Any]] = None
    if session_id:
        confirm = 0
        dismiss = 0
        try:
            if hasattr(orchestrator.store, "list_by_session"):
                mems = orchestrator.store.list_by_session(session_id, limit=500)
            else:
                mems = orchestrator.cached_memories_for_session(session_id)
            for mem in mems:
                stats = orchestrator.feedback_handler.get_feedback_stats(mem.id)
                confirm += int(stats.get("confirm_count", 0) or 0)
                dismiss += int(stats.get("dismiss_count", 0) or 0)
        except Exception:
            pass
        total = confirm + dismiss
        session_block = {
            "session_id": session_id,
            "confirm_count": confirm,
            "dismiss_count": dismiss,
            "precision": (confirm / total) if total else None,
            "sample_count": total,
        }

    # Retrainer state if exposed.
    retrainer_block: Optional[Dict[str, Any]] = None
    try:
        rt = getattr(orchestrator, "model_retrainer", None) or getattr(orchestrator, "retrainer", None)
        if rt is not None and hasattr(rt, "status"):
            retrainer_block = rt.status()
        elif rt is not None:
            retrainer_block = {
                "labeled_samples": getattr(rt, "labeled_sample_count", None),
                "last_retrain_at": getattr(rt, "last_retrain_at", None),
                "last_metrics": getattr(rt, "last_metrics", None),
            }
    except Exception:
        retrainer_block = None

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule_performance": rule_perf,
        "adaptive_thresholds": adaptive,
        "model_trust": model_trust,
        "session": session_block,
        "retrainer": retrainer_block,
    }

    if persist:
        try:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "..", "data", "loop_metrics")
            out_dir = os.path.abspath(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = os.path.join(out_dir, f"{day}.jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            payload["persisted_to"] = path
        except Exception as exc:
            logger.warning("loop_metrics: persist failed: %s", exc)

    return payload


@router.delete("/")
async def delete_all_memories():
    """
    Delete ALL memories. Use with caution.
    
    Clears both in-memory storage and database (if using SQLite backend).
    Also clears pattern priors.
    """
    orchestrator = get_orchestrator()
    
    # Drop the in-process cache, keeping the count it held
    count = orchestrator.clear_memory_cache()
    
    # Clear database if using SQLite backend
    if hasattr(orchestrator.store, 'clear') and callable(getattr(orchestrator.store, 'clear')):
        orchestrator.store.clear()
    
    # Also reset priors
    if hasattr(orchestrator.scorer, "reset_feedback_state"):
        orchestrator.scorer.reset_feedback_state()
    else:
        orchestrator.scorer._pattern_priors.clear()
    
    return {
        "message": "All memories deleted",
        "deleted_count": count,
        "priors_reset": True,
    }


# ============================================================================
# LLM Explanation Endpoints
# ============================================================================

@router.get("/llm/status")
async def get_llm_status():
    """
    Check if LLM service is available.
    """
    explainer = get_explainer()
    available = explainer.is_available()
    ecfg = explainer.config
    return {
        "available": available,
        "provider": ecfg.provider,
        "model": ecfg.model,
        "ollama_url": ecfg.ollama_url if ecfg.provider == "ollama" else None,
        "groq_api_url": ecfg.groq_api_url if ecfg.provider == "groq" else None,
    }


@router.get("/llm/diagnostics")
async def get_llm_diagnostics():
    """
    Detailed LLM diagnostics for debugging connectivity issues.

    Returns internal availability state, call/fallback counts, and timing
    information that is useful when the demo script is not producing LLM
    descriptions.
    """
    explainer = get_explainer()
    return explainer.get_diagnostics()


@router.post("/llm/warmup")
async def warmup_llm():
    """
    Force the server-side LLMExplainer to re-check Ollama availability.

    The demo script should call this *after* confirming that Ollama is
    reachable (via its own preflight check).  This sets the server-side
    explainer's ``_available`` flag so that the first batch of events
    gets real LLM descriptions instead of falling through to fallback text.

    For cloud models that do not appear in ``/api/tags``, the endpoint
    automatically force-enables the explainer.  To force-enable manually
    regardless of model name, use ``POST /llm/force-available``.
    """
    explainer = get_explainer()
    explainer = explainer

    # First attempt a normal availability check to refresh the cached flag
    was_available = explainer.is_available()

    # If still unavailable and the model is a cloud model, force-enable
    if not was_available and explainer._is_cloud_model_name(explainer.config.model or ""):
        logger.info("Cloud model detected during warmup — force-enabling LLM")
        explainer.force_available(True)

    now_available = explainer.is_available()
    diag = explainer.get_diagnostics()
    diag["warmup_result"] = {
        "was_available": was_available,
        "now_available": now_available,
        "forced": (not was_available and now_available),
    }
    return diag


@router.post("/llm/force-available")
async def force_llm_available():
    """
    Unconditionally mark the server-side LLMExplainer as available.

    Use this when you have independently verified that the Ollama endpoint
    serves the configured model (e.g., after a manual `ollama pull`).
    """
    explainer = get_explainer()
    explainer.force_available(True)
    return explainer.get_diagnostics()




# ── Experiment management endpoints ────────────────────────────────────────
# These serve the React ExperimentDashboard — list, inspect, trigger, and
# evaluate experiment runs that live on disk under
#   data/breakage_patterns/stoppage_experiment/<run_dir>/experiment_results.json

import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


# =========================================================================
# Model Registry — list / inspect available models
# =========================================================================

_MODELS_DIR = _PROJECT_ROOT / "data" / "models"


@router.get("/models")
async def list_models():
    """List all model files in data/models/.

    Returns ``{models: [{filename, size_bytes, modified, type}]}``.
    """
    if not _MODELS_DIR.is_dir():
        return {"models": []}

    models: List[Dict[str, Any]] = []
    for f in sorted(_MODELS_DIR.iterdir()):
        if f.is_file() and f.suffix in (".pkl", ".joblib", ".json", ".npz"):
            stat = f.stat()
            models.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "priors" if "priors" in f.name else (
                    "rl_agent" if "rl_agent" in f.name else "seed_model"
                ),
            })
    return {"models": models}


@router.get("/models/{filename}/info")
async def get_model_info(filename: str):
    """Return metadata about a specific model file.

    For joblib/pkl models, returns file stats.
    For JSON models (priors, rl_agent) returns the JSON content.
    """
    safe = pathlib.Path(filename).name  # prevent path traversal
    path = _MODELS_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model not found: {safe}")

    stat = path.stat()
    info: Dict[str, Any] = {
        "filename": safe,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

    if path.suffix == ".json":
        try:
            info["content"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            info["content"] = None
    else:
        info["content"] = None
        info["note"] = "Binary model — use Python to inspect"

    return info


# =========================================================================
# Experiment Config Schema — expose ExperimentConfig fields to UI
# =========================================================================


# =====================================================================
# Model Retraining  
# =====================================================================


class RetrainResponse(BaseModel):
    success: bool
    message: str
    n_samples_used: int = 0
    n_confirmed: int = 0
    n_dismissed: int = 0
    previous_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    duration_s: float = 0.0


class RetrainerStatusResponse(BaseModel):
    total_feedback: int = 0
    since_last_retrain: int = 0
    retrain_threshold: int = 20
    buffer_size: int = 0
    confirmed_in_buffer: int = 0
    dismissed_in_buffer: int = 0
    should_retrain: bool = False
    retrain_count: int = 0
    last_retrain: Optional[str] = None
    # Aliases expected by the React UI
    threshold: int = 20
    last_retrain_at: Optional[str] = None


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_model():
    """Manually trigger model retraining from accumulated feedback.

    The retrainer collects feature vectors from confirmed (true positive)
    and dismissed (false positive) memories. When triggered, it retrains
    the classical anomaly detection model so future scoring quality improves.
    """
    from .retrainer import get_retrainer

    config = get_orchestrator_config()
    scorer = get_scorer()
    retrainer = get_retrainer(
        model_path=(
            pathlib.Path(config.seed_model_path)
            if getattr(config, "seed_model_path", None) else None
        ),
        model_confidence_path=getattr(scorer, "_model_confidence_path", None),
    )
    result = retrainer.retrain()

    return RetrainResponse(
        success=result.success,
        message=result.message,
        n_samples_used=result.n_samples_used,
        n_confirmed=result.n_confirmed,
        n_dismissed=result.n_dismissed,
        previous_accuracy=result.previous_accuracy,
        new_accuracy=result.new_accuracy,
        duration_s=result.duration_s,
    )


@router.get("/retrain/status", response_model=RetrainerStatusResponse)
async def retrain_status():
    """Check model retraining status — feedback buffer, readiness."""
    try:
        from .retrainer import get_retrainer

        config = get_orchestrator_config()
        scorer = get_scorer()
        retrainer = get_retrainer(
            model_path=(
                pathlib.Path(config.seed_model_path)
                if getattr(config, "seed_model_path", None) else None
            ),
            model_confidence_path=getattr(scorer, "_model_confidence_path", None),
        )
        status = retrainer.get_status()
        # Add UI-expected alias fields
        status["threshold"] = status.get("retrain_threshold", 20)
        status["last_retrain_at"] = status.get("last_retrain")
        return RetrainerStatusResponse(**status)
    except Exception as exc:
        logger.warning("retrain_status failed, returning defaults: %s", exc)
        return RetrainerStatusResponse()


# =====================================================================
# LLM Chat — conversational explanation for memory events
# =====================================================================


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    mode: Literal["focused", "clarification"] = "focused"


class ChatResponse(BaseModel):
    reply: str
    source: str = "llm"  # "llm" | "fallback"
    memory_context: Optional[Dict[str, Any]] = None


_CLARIFICATION_HINTS = (
    "clarify",
    "clarification",
    "before",
    "previous",
    "related",
    "similar",
    "history",
    "happened before",
    "past event",
    "last time",
)


def _wants_clarification(request: ChatRequest) -> bool:
    if request.mode == "clarification":
        return True
    message = str(request.message or "").strip().lower()
    return any(hint in message for hint in _CLARIFICATION_HINTS)


def _strict_usecase_grounding_enabled() -> bool:
    return os.environ.get("STRICT_USECASE_GROUNDING", "true").strip().lower() not in {"0", "false", "no"}


def _memory_usecase(memory: Memory) -> Optional[str]:
    metadata = memory.metadata or {}
    return resolve_usecase(
        metadata=metadata,
        machine_uri=getattr(memory, "machine_uri", None),
        fallback_generic=False,
    )


def _doc_lines(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No documentation context available."
    lines: List[str] = []
    for match in matches[:3]:
        citation = match.get("citation") or match.get("file_name") or "documentation"
        excerpt = str(match.get("text") or "").replace("\n", " ").strip()
        evidence_entities = match.get("evidence_entities") or []
        evidence_suffix = ""
        if evidence_entities:
            entity_labels = []
            for entity in evidence_entities[:3]:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name") or "").strip()
                entity_type = str(entity.get("type") or "").strip()
                if name and entity_type:
                    entity_labels.append(f"{name} ({entity_type})")
                elif name:
                    entity_labels.append(name)
            if entity_labels:
                evidence_suffix = f" [entities: {', '.join(entity_labels)}]"
        if excerpt:
            lines.append(f"- {citation}{evidence_suffix}: {excerpt}")
        else:
            lines.append(f"- {citation}{evidence_suffix}")
    return "\n".join(lines)


def _pattern_feedback_lines(orch: Any, patterns: List[str], cutting_context: Any = None) -> str:
    """Learned confirm/dismiss history for each pattern, for the chat prompt.

    Read-only view over the scorer's pattern priors (the same data the Learnings
    page shows): how often operators confirmed vs dismissed this pattern and the
    resulting learned prior. Lets the assistant answer "has this been confirmed
    before / how reliable is this signal?" from real feedback, not guesswork.
    """
    scorer = getattr(orch, "scorer", None)
    if scorer is None or not patterns:
        return ""
    lines: List[str] = []
    for pk in patterns[:6]:
        try:
            rec = scorer.get_pattern_prior_record(pk, context=cutting_context)
            d = rec.to_dict()
        except Exception:
            continue
        confirmed = int(d.get("confirmed") or 0)
        dismissed = int(d.get("dismissed") or 0)
        total = confirmed + dismissed
        prior = d.get("prior_strength")
        if total <= 0:
            lines.append(f"- {pk}: no operator feedback yet (learned prior {float(prior or 0.5):.2f}).")
        else:
            lines.append(
                f"- {pk}: confirmed {confirmed}/{total} times, dismissed {dismissed}/{total} "
                f"(learned prior {float(prior or 0.5):.2f}, confidence {float(d.get('confidence') or 0.0):.2f})."
            )
    return "\n".join(lines)


_DPP_REGISTRY_CACHE: Any = None
_DPP_REGISTRY_LOADED = False


def _carbon_context_lines(usecase: Optional[str]) -> str:
    """Embodied-carbon-at-stake context for the event's usecase (backend-only).

    Resolves a per-part Product Carbon Footprint strictly within the event's
    usecase (never across usecases) and frames it as *modeled / at-stake* — the
    embodied CO2e a confirmed catch would protect from scrap/rework, not carbon
    "saved". Returns "" when no DPP exists for the usecase so the prompt simply
    omits sustainability context. Not surfaced in the UI; available so the
    assistant can answer a sustainability question if the operator asks.
    """
    global _DPP_REGISTRY_CACHE, _DPP_REGISTRY_LOADED
    if not usecase:
        return ""
    if not _DPP_REGISTRY_LOADED:
        _DPP_REGISTRY_LOADED = True
        try:
            from ..maas import DPPRegistry
            _DPP_REGISTRY_CACHE = DPPRegistry.from_dir(
                os.getenv("DPP_DIR", "data/supplementary_data")
            )
        except Exception:
            logger.debug("carbon context: DPP registry load failed", exc_info=True)
            _DPP_REGISTRY_CACHE = None
    registry = _DPP_REGISTRY_CACHE
    if registry is None:
        return ""
    try:
        impact = registry.resolve_for_usecase(usecase)
    except Exception:
        return ""
    if impact is None:
        return ""
    return (
        f"Embodied-carbon context (modeled, from Digital Product Passport {impact.source}; "
        f"illustrative, not measured plant data): a representative part in this usecase carries "
        f"~{impact.pcf_total_kg:.0f} kg CO2e cradle-to-gate (A1-A3), of which "
        f"~{impact.pcf_processing_kg:.0f} kg CO2e is the manufacturing/processing stage (A3). "
        f"Acting on a confirmed fault that would otherwise scrap the part protects roughly the A3 "
        f"embodied carbon that is at stake — this is a modeled figure, not carbon already saved."
    )


async def _related_history_lines(
    orch: Any,
    memory_id: str,
    usecase: Optional[str],
) -> tuple[str, List[Dict[str, Any]]]:
    store = getattr(orch, "store", None)
    if store is None or not hasattr(store, "get_similar_with_resolution"):
        return "", []

    try:
        related = await asyncio.to_thread(
            store.get_similar_with_resolution,
            memory_id,
            k=4,
            include_dismissed=True,
        )
    except Exception:
        return "", []

    filtered: List[Dict[str, Any]] = []
    for item in related:
        item_usecase = resolve_usecase(
            machine_uri=item.get("machine_uri"),
            metadata=item,
            fallback_generic=False,
        )
        if usecase and item_usecase != usecase:
            continue
        filtered.append(item)

    if not filtered:
        return "", []

    lines: List[str] = []
    for item in filtered[:3]:
        feedback = item.get("feedback") or {}
        lines.append(
            "- Memory {id}: shared_patterns={patterns}, last_action={last_action}, "
            "confirm={confirm}, dismiss={dismiss}".format(
                id=item.get("id"),
                patterns=", ".join(item.get("shared_pattern_keys") or []) or "none",
                last_action=feedback.get("last_action") or "unknown",
                confirm=feedback.get("confirm_count", 0),
                dismiss=feedback.get("dismiss_count", 0),
            )
        )
    return "\n".join(lines), filtered[:3]

# =====================================================================
# Experiment → Feedback Bridge
# =====================================================================


@router.get("/alerts/{memory_id}/doc_links", response_model=ProposedDocLinksResponse)
async def get_memory_alert_doc_links(
    memory_id: str,
    top_k: int = Query(default=3, ge=1, le=10),
    score_floor: float = Query(default=0.6, ge=0.0, le=1.0),
):
    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    metadata = memory.metadata or {}
    cutting_context = metadata.get("cutting_context") if isinstance(metadata.get("cutting_context"), dict) else {}
    usecase = _memory_usecase(memory)
    machine_hint = (
        cutting_context.get("machine_id")
        or metadata.get("machine_id")
        or getattr(memory, "machine_uri", None)
        or cutting_context.get("machine_type")
    )
    pattern_keys = [
        str(pattern.key).strip()
        for pattern in (memory.pattern_keys or [])
        if str(getattr(pattern, "key", "")).strip()
    ]

    store = getattr(orchestrator, "store", None)
    if (
        store is not None
        and hasattr(store, "get_doc_links")
        and _alert_doc_request_matches_persisted_defaults(top_k, score_floor)
    ):
        try:
            persisted_links = list(
                store.get_doc_links(
                    memory_id,
                    score_floor=score_floor,
                    limit=DEFAULT_DOC_LINK_LIMIT,
                )
                or []
            )
        except Exception:
            logger.debug("Persisted alert doc-link lookup failed for %s", memory_id, exc_info=True)
        else:
            if persisted_links:
                return ProposedDocLinksResponse(
                    memory_id=memory_id,
                    usecase=usecase,
                    machine=machine_hint,
                    top_k=top_k,
                    score_floor=score_floor,
                    query_candidates=[ProposedDocQuery(**entry) for entry in _query_candidates_from_doc_links(persisted_links)],
                    doc_links=[ProposedDocLink(**entry) for entry in persisted_links],
                )

    try:
        link_payload = await propose_alert_doc_links(
            get_docs_backend(),
            pattern_keys=pattern_keys,
            usecase=usecase,
            machine=machine_hint,
            cutting_context=cutting_context,
            channel_names=list(memory.channels or []),
            top_k=top_k,
            score_floor=score_floor,
        )
    except Exception as exc:
        logger.error("Alert doc-link lookup failed for %s: %s", memory_id, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Documentation lookup failed") from exc

    return ProposedDocLinksResponse(
        memory_id=memory_id,
        usecase=usecase,
        machine=machine_hint,
        top_k=top_k,
        score_floor=score_floor,
        query_candidates=[ProposedDocQuery(**entry) for entry in link_payload.get("query_candidates") or []],
        doc_links=[ProposedDocLink(**entry) for entry in link_payload.get("doc_links") or []],
    )

@router.get("/{memory_id}", response_model=MemoryDetailResponse)
async def get_memory(memory_id: str):
    """
    Get a specific memory by ID.
    """
    orchestrator = get_orchestrator()
    
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Get feedback stats
    feedback_stats = orchestrator.feedback_handler.get_feedback_stats(memory_id)
    doc_links: List[Dict[str, Any]] = []
    store = getattr(orchestrator, "store", None)
    if store is not None and hasattr(store, "get_doc_links"):
        try:
            doc_links = list(
                store.get_doc_links(
                    memory_id,
                    score_floor=0.0,
                    limit=DEFAULT_DOC_LINK_LIMIT,
                )
                or []
            )
        except Exception:
            logger.debug("Memory detail doc-link lookup failed for %s", memory_id, exc_info=True)
            doc_links = []
    elif isinstance(getattr(memory, "metadata", None), dict):
        doc_links = [
            dict(link)
            for link in (memory.metadata.get("doc_links") or [])
            if isinstance(link, dict)
        ]
    
    return MemoryDetailResponse(
        memory=memory.model_dump() if hasattr(memory, 'model_dump') else memory.__dict__,
        feedback_stats=feedback_stats,
        doc_links=[ProposedDocLink(**entry) for entry in doc_links],
    )


@router.get("/{memory_id}/traces", response_model=TraceListResponse)
async def get_memory_traces(
    memory_id: str,
    trace_type: Optional[str] = Query(default=None, description="Filter by trace type, e.g. score|retrieve"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Return scoring/retrieval traces for a memory (auditability)."""
    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    traces: List[Dict[str, Any]] = []
    if hasattr(orchestrator.store, "list_traces"):
        try:
            traces = list(
                orchestrator.store.list_traces(
                    memory_id=memory_id,
                    trace_type=trace_type,
                    limit=int(limit),
                )
            )
        except Exception:
            traces = []

    return TraceListResponse(memory_id=memory_id, traces=traces)


@router.get("/{memory_id}/resolution_chain")
async def get_memory_resolution_chain(
    memory_id: str,
    k: int = Query(default=5, ge=1, le=50, description="Max similar memories"),
    include_dismissed: bool = Query(default=True),
):
    """Return "has this happened before, how was it resolved?" context.

    Agent E (2026-04-24). Surfaces:
      - ``memory``: the target memory (summary fields).
      - ``own_feedback``: ordered feedback events on the target memory.
      - ``similar``: up to ``k`` prior memories sharing patterns / :SIMILAR_TO,
        each with its own feedback roll-up (confirm / dismiss counts, last
        action, last comment).

    Feeds the UI timeline component and ``ExplanationContext.similar_memories``
    so the LLM can say "last time this happened operator confirmed and changed
    the tool." Graceful degradation when the store lacks
    ``get_similar_with_resolution`` (e.g. in-memory test store).
    """
    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    # Own feedback events.
    own_feedback: List[Dict[str, Any]] = []
    if hasattr(orchestrator.store, "list_feedback_events"):
        try:
            own_feedback = list(orchestrator.store.list_feedback_events(memory_id, limit=200))
        except Exception:
            own_feedback = []
    if not own_feedback:
        try:
            own_feedback = [
                r.model_dump() for r in orchestrator.feedback_handler.get_feedback_history(memory_id)
            ]
        except Exception:
            own_feedback = []

    # Similar memories with resolutions.
    similar: List[Dict[str, Any]] = []
    if hasattr(orchestrator.store, "get_similar_with_resolution"):
        try:
            similar = list(
                await asyncio.to_thread(
                    orchestrator.store.get_similar_with_resolution,
                    memory_id,
                    k=int(k),
                    include_dismissed=bool(include_dismissed),
                )
            )
        except Exception as exc:
            logger.warning("resolution_chain: get_similar_with_resolution failed: %s", exc)
            similar = []

    return {
        "memory": {
            "id": memory.id,
            "session_id": memory.session_id,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "pattern_keys": [p.key for p in memory.pattern_keys],
            "label": memory.label,
            "annotation_text": memory.annotation_text,
        },
        "own_feedback": own_feedback,
        "similar": similar,
    }


@router.get("/{memory_id}/signal")
async def get_memory_signal(
    memory_id: str,
    request: Request,
    channels: Optional[str] = Query(default=None, description="Comma-separated channel names"),
    margin_s: float = Query(default=0.0, ge=0.0, le=60.0, description="Symmetric margin in seconds"),
):
    """Return the raw windowed samples for a memory's time range.

    Agent N (2026-04-24). Lookup order:
      1. Live session dict on ``app.state.sessions`` (fastest, zero disk I/O).
      2. Persisted ``data/sessions/{session_id}.npz`` (written at upload).
    Returns 404 if the memory is unknown, 200 with ``available=False`` if
    the signal binding cannot be resolved.
    """
    from ..storage.session_signals import (
        extract_signal_window,
        load_session_signal,
        compute_signal_digest,
    )

    orchestrator = get_orchestrator()
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    tr = getattr(memory, "time_range", None)
    if tr is None:
        return {"available": False, "reason": "memory has no time_range", "memory_id": memory_id}
    i0 = int(getattr(tr, "i0", 0))
    i1 = int(getattr(tr, "i1", 0))
    tr_fs = float(getattr(tr, "fs", 0.0) or 0.0)

    # Parse requested channels.
    wanted: Optional[List[str]]
    if channels:
        wanted = [c.strip() for c in channels.split(",") if c.strip()]
        if not wanted:
            wanted = None
    else:
        wanted = None

    # Try the live session first.
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}
    source = "unavailable"
    session_id = getattr(memory, "session_id", None)
    if session_id:
        live_sessions = getattr(request.app.state, "sessions", None) or {}
        live = live_sessions.get(session_id) if isinstance(live_sessions, dict) else None
        if isinstance(live, dict) and isinstance(live.get("data"), dict) and live["data"]:
            data = live["data"]
            metadata = live.get("metadata") or {}
            source = "live"
        else:
            loaded = load_session_signal(session_id)
            if loaded is not None:
                data, metadata = loaded
                source = "disk"

    if data is None:
        return {
            "available": False,
            "reason": "no signal source for session (not loaded and no persisted file)",
            "memory_id": memory_id,
            "session_id": session_id,
        }

    # Resolve fs: time_range wins, then metadata, then default.
    fs = tr_fs if tr_fs > 0 else float(metadata.get("fs") or metadata.get("sample_rate_hz") or 0.0)
    if fs <= 0:
        return {
            "available": False,
            "reason": "unable to determine sample rate",
            "memory_id": memory_id,
            "session_id": session_id,
        }

    window = extract_signal_window(data, wanted, i0, i1, fs=fs, margin_s=margin_s)

    # Mutation-detection digest over the original [i0, i1).
    stored_digest = None
    try:
        stored_digest = (memory.metadata or {}).get("signal_digest") if hasattr(memory, "metadata") else None
    except Exception:
        stored_digest = None
    current_digest = compute_signal_digest(data, i0, i1, channels=wanted)

    return {
        "available": True,
        "memory_id": memory_id,
        "session_id": session_id,
        "source": source,
        "signal_digest": {
            "current": current_digest,
            "stored": stored_digest,
            "match": (stored_digest is None) or (stored_digest == current_digest),
        },
        **window,
    }


# ---------------------------------------------------------------------------
# False-negative feedback (Issue #14 fix, 2026-04-14)
# ---------------------------------------------------------------------------

class MissedEventRequest(BaseModel):
    """Report that the system missed an event that should have been flagged."""
    session_id: str = Field(..., description="Session where the event occurred")
    pattern_keys: List[str] = Field(default_factory=list, description="Patterns that should have fired")
    derive_patterns: bool = False
    user_id: str = "operator"
    reason: Optional[str] = Field(None, description="Why this should have been flagged")
    timestamp: Optional[str] = Field(None, description="Approximate time of the missed event (ISO)")
    raw_metrics: Optional[Dict[str, Any]] = Field(None, description="Sensor readings at the time")


@router.post("/feedback/missed-event")
async def report_missed_event(request: MissedEventRequest):
    """Report a false-negative: the system failed to flag a significant event.

    This closes the feedback loop's blind spot \u2014 without this endpoint the
    system can only learn from events it already detected.  Missed-event
    reports:

    1. Lower the scorer's action thresholds (adaptive thresholds nudge down)
    2. Boost the historical prior for the reported pattern keys
    3. Record a feedback event for audit
    """
    scorer = get_scorer()
    store = get_store()
    scorer = scorer

    raw_metrics = None
    if request.raw_metrics:
        raw_metrics = {
            k: float(v) for k, v in request.raw_metrics.items()
            if isinstance(v, (int, float))
        }
    pattern_keys_used = _derive_request_pattern_keys(
        request.pattern_keys,
        raw_metrics,
        request.derive_patterns,
    )

    # 1) Nudge adaptive thresholds downward so future similar events cross
    #    the alert boundary.  Use the current alert threshold as the
    #    "score" and mark it as a missed confirmation.
    if hasattr(scorer, "record_feedback_for_adaptive_thresholds"):
        try:
            # Score just below the alert threshold to signal a miss
            eff_alert = getattr(scorer, "_adaptive_thresholds", None)
            alert_threshold = (
                eff_alert.alert_threshold if eff_alert else scorer.config.alert_threshold
            )
            scorer.record_feedback_for_adaptive_thresholds(
                score=alert_threshold - 0.01,
                action="ignore",  # it was below threshold
                was_confirmed=True,  # operator says it was real
            )
        except Exception:
            pass

    persisted = False

    # 2) Record in the durable feedback store
    if store and hasattr(store, "add_feedback_event"):
        try:
            store.add_feedback_event(
                memory_id=None,
                action="missed",
                user_id=request.user_id,
                pattern_keys=pattern_keys_used,
                context_key=None,
                context=None,
                data={
                    "reason": request.reason,
                    "timestamp": request.timestamp,
                    "raw_metrics_keys": list((request.raw_metrics or {}).keys())[:20],
                    "false_negative": True,
                },
            )
            persisted = True
        except Exception:
            pass

    if not persisted:
        for pk in pattern_keys_used:
            try:
                scorer.update_pattern_prior(pk, was_significant=True)
            except Exception:
                pass

    return {
        "success": True,
        "message": f"Missed-event report recorded for patterns {pattern_keys_used}",
        "patterns_boosted": pattern_keys_used,
        "thresholds_nudged": True,
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """
    Delete a specific memory by ID.
    """
    orchestrator = get_orchestrator()
    
    if not orchestrator.delete(memory_id):
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    return {"message": f"Memory {memory_id} deleted", "deleted": True}


@router.post("/{memory_id}/explain")
async def explain_memory(memory_id: str):
    """
    Generate an LLM explanation for a specific memory.
    
    This endpoint calls the LLM to generate a human-readable explanation
    of the memory's significance, patterns, and context.
    """
    orchestrator = get_orchestrator()
    
    memory = orchestrator.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    
    # Check if LLM is available
    if not orchestrator.explainer.is_available():
        # Return fallback explanation
        fallback = orchestrator.explainer._fallback_memory_summary(memory)
        return {
            "memory_id": memory_id,
            "explanation": fallback,
            "llm_used": False,
            "message": "LLM not available, using fallback",
        }
    
    # Generate LLM explanation
    try:
        explanation = await orchestrator.explainer.generate_memory_summary_async(
            memory=memory,
            context=None,  # Could extract from memory metadata if available
        )
        return {
            "memory_id": memory_id,
            "explanation": explanation,
            "llm_used": True,
        }
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}", exc_info=True)
        fallback = orchestrator.explainer._fallback_memory_summary(memory)
        return {
            "memory_id": memory_id,
            "explanation": fallback,
            "llm_used": False,
            "error": str(e),
        }


# ── Breakage experiment results ────────────────────────────────────────────

@router.post("/{memory_id}/chat", response_model=ChatResponse)
async def chat_about_memory(memory_id: str, request: ChatRequest):
    """Conversational LLM chat about a specific memory event.

    Loads memory context (patterns, scores, sensor data, similar memories)
    and uses the RAG agent + LLM to answer the operator's question.
    This lets operators ask 'Why did you flag this?' or 'What should I do?'
    directly from the memory detail modal.
    """
    orch = get_orchestrator()

    # Load memory
    memory = None
    if orch.store:
        try:
            memory = orch.store.get(memory_id)
        except Exception:
            pass

    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    # Build context from memory data
    meta = memory.metadata or {}
    action = meta.get("action") or meta.get("significance_action") or "N/A"
    patterns = [p.key for p in (memory.pattern_keys or [])]
    resolved_usecase = _memory_usecase(memory)
    clarification_mode = _wants_clarification(request)
    cutting_context = meta.get("cutting_context") if isinstance(meta.get("cutting_context"), dict) else {}
    machine_hint = (
        cutting_context.get("machine_id")
        or getattr(memory, "machine_uri", None)
        or cutting_context.get("machine_type")
    )

    context_parts = [
        f"Memory ID: {memory_id}",
        f"Session: {memory.session_id or 'unknown'}",
        f"Patterns detected: {', '.join(patterns) if patterns else 'none'}",
        f"Significance score: {meta.get('significance_score', 'N/A')}",
        f"Action: {action}",
    ]
    if getattr(memory, "machine_uri", None):
        context_parts.append(f"Machine URI: {memory.machine_uri}")
    if resolved_usecase:
        context_parts.append(f"Usecase: {resolved_usecase}")

    if cutting_context:
        context_parts.append(f"Machine: {cutting_context.get('machine_type', 'unknown')}")
        context_parts.append(f"Tool: {cutting_context.get('tool_type', 'unknown')}")
        context_parts.append(f"Material: {cutting_context.get('workpiece_material', 'unknown')}")

    if meta.get("explanation"):
        context_parts.append(f"LLM explanation: {meta['explanation']}")

    doc_matches: List[Dict[str, Any]] = []
    docs_scope_reason = "scoped_search"
    store = getattr(orch, "store", None)
    if store is not None and hasattr(store, "get_doc_links"):
        try:
            doc_matches = list(
                store.get_doc_links(
                    memory_id,
                    score_floor=0.0,
                    limit=DEFAULT_DOC_LINK_LIMIT,
                )
                or []
            )
        except Exception:
            logger.debug("Persisted memory-chat doc-link lookup failed for %s", memory_id, exc_info=True)
            doc_matches = []
    if doc_matches:
        docs_scope_reason = "persisted_links"
    elif _strict_usecase_grounding_enabled() and resolved_usecase is None and machine_hint is None:
        docs_scope_reason = "no_usecase_scope"
    else:
        docs_backend = get_docs_backend()
        docs_result = await docs_backend.search(
            request.message,
            top_k=3,
            usecase=resolved_usecase,
            machine=machine_hint,
        )
        doc_matches = list(docs_result.get("matches") or [])
    docs_context = _doc_lines(doc_matches)

    related_context = ""
    related_items: List[Dict[str, Any]] = []
    if clarification_mode:
        related_context, related_items = await _related_history_lines(
            orch,
            memory_id,
            resolved_usecase,
        )

    # Learned pattern feedback (confirm/dismiss history) — graph/scorer-backed,
    # always available so the assistant can speak to signal reliability.
    feedback_lines = _pattern_feedback_lines(orch, patterns)

    # Embodied-carbon-at-stake context, strictly scoped to this event's usecase
    # (backend-only; omitted when no DPP exists for the usecase).
    carbon_lines = _carbon_context_lines(resolved_usecase)

    memory_context_str = "\n".join(context_parts)

    # Build conversation history for the LLM
    history_str = ""
    for msg in request.history[-5:]:  # Last 5 messages
        history_str += f"\n{msg.role.upper()}: {msg.content}"

    prompt = (
        f"You are an expert manufacturing process analyst assisting an operator.\n"
        f"Context about the flagged event:\n{memory_context_str}\n\n"
        f"Documentation graph context (same usecase only):\n{docs_context}\n\n"
        f"Learned pattern feedback history:\n{feedback_lines or 'No pattern feedback history available.'}\n\n"
        f"{(carbon_lines + chr(10) + chr(10)) if carbon_lines else ''}"
        f"Related memory/feedback context ({'enabled' if clarification_mode else 'disabled'}):\n"
        f"{related_context or 'Not requested for this turn.'}\n"
        f"{history_str}\n"
        f"OPERATOR: {request.message}\n"
        f"ANALYST:"
    )

    reply = ""
    source = "fallback"

    try:
        if orch.explainer.is_available():
            reply = await orch.explainer._call_llm_async(prompt, use_system_role=True)
            if reply:
                source = "llm"
    except Exception as e:
        logger.debug(f"Memory chat LLM call failed: {e}")

    # Fallback: deterministic explanation
    if not reply:
        reply = (
            f"This event was flagged with patterns: {', '.join(patterns) if patterns else 'none'}. "
            f"The significance score was {meta.get('significance_score', 'N/A')} "
            f"(action: {action}). "
        )
        if meta.get("explanation"):
            reply += f"\nPrevious analysis: {meta['explanation']}"
        if feedback_lines:
            reply += f"\nPattern feedback history:\n{feedback_lines}"
        if doc_matches:
            reply += "\nRelevant documentation: " + "; ".join(
                str(match.get("citation") or match.get("file_name") or "documentation")
                for match in doc_matches[:3]
            )
        if related_items:
            reply += f"\nRelated prior events in the same usecase: {len(related_items)}"
        reply += "\n\n(LLM unavailable — showing deterministic summary. Start the configured LLM provider for AI-powered analysis.)"
        source = "fallback"

    return ChatResponse(
        reply=reply,
        source=source,
        memory_context={
            "patterns": patterns,
            "score": meta.get("significance_score"),
            "action": action,
            "mode": "clarification" if clarification_mode else "focused",
            "usecase": resolved_usecase,
            "docs_scope_reason": docs_scope_reason,
            "documents": [match.get("citation") for match in doc_matches[:3]],
            "related_event_count": len(related_items),
            "has_pattern_feedback": bool(feedback_lines),
            "has_carbon_context": bool(carbon_lines),
        },
    )

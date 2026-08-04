"""
Memory Event Orchestrator - Coordinates the memory system event flow.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module is the main entry point for the memory system.
# Orchestrates: detection -> significance -> retrieval -> explanation -> alert
# ===========================================================================

Main flow:
1. Receive event trigger (pattern detected, external signal)
2. Score significance
3. If significant: store memory, retrieve similar, generate explanation
4. Dispatch alert to clients

[INTEGRATION_POINT] This should be called from the main processing pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Mapping, TYPE_CHECKING

from backend.json_utils import json_safe

from ..core.schemas import (
    Memory,
    PatternKey,
    NumericMetrics,
    TimeRange,
    MemoryProvenance,
)
from ..core.batch_context import BatchContext, extract_batch_context
from ..core.context import CuttingContext, extract_context_from_metadata
from .scorer import (
    SignificanceScorer,
    SignificanceResult,
    SignificanceAction,
    SignificanceConfig,
)
from .retriever import MemoryRetriever, MemoryMatch, RetrievalConfig
from .dispatcher import AlertDispatcher, dispatch_significant_event, get_dispatcher
from .feedback import MemoryFeedbackHandler
from .cycle_tracker import CycleEnded
from .outcome_correlator import attach_passive_outcome
from ..core.metrics import WindowMetrics
from ..patterns.discovery import PatternDiscovery
from ..storage.in_memory_store import InMemoryStoreAdapter
from .enrichment import enrich_with_classical_scores, enrich_with_harmonic_score
from .explanation_context import build_explanation_context
from backend.session_logs import append_session_log
from backend.learning_emitter import publish_scored_learning, publish_insight_learning

# Lazy import to avoid circular dependency with llm.explainer
if TYPE_CHECKING:
    from ..llm.explainer import ExplanationContext, LLMExplainer, ExplainerConfig

logger = logging.getLogger(__name__)


def _curated_event_metadata_snapshot(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    snapshot: Dict[str, Any] = {}

    source = metadata.get("source")
    if isinstance(source, str) and source:
        snapshot["source"] = source

    for key in (
        "machine_family",
        "dataset_id",
        "source_dataset_id",
        "machine_uri",
        "machine_iri",
        "sindit_asset_iri",
        "sample_frequency",
        "ground_truth_label",
        "ground_truth_index",
    ):
        value = json_safe(metadata.get(key))
        if isinstance(value, (str, int, float, bool)) and value not in ("", None):
            snapshot[key] = value

    casedata = metadata.get("casedata")
    if isinstance(casedata, dict):
        safe_casedata: Dict[str, Any] = {}
        for key in ("operation_id", "tool_id", "root", "case_dir", "dataset_id"):
            value = casedata.get(key)
            if value is None:
                continue
            if isinstance(value, Path):
                safe_casedata[key] = str(value)
                continue
            safe_value = json_safe(value)
            if isinstance(safe_value, (str, int, float, bool)) and safe_value not in ("", None):
                safe_casedata[key] = safe_value
            elif safe_value not in (None, ""):
                safe_casedata[key] = str(safe_value)
        if safe_casedata:
            snapshot["casedata"] = safe_casedata

    batch = extract_batch_context(metadata)
    if batch is not None:
        snapshot["batch"] = batch.model_dump(mode="json")

    return snapshot


def _dispatch_context_snapshot(
    cutting_context: Optional[CuttingContext],
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    context = cutting_context.model_dump() if cutting_context else None
    if not isinstance(metadata, dict):
        return context

    ground_truth_label = metadata.get("ground_truth_label")
    if isinstance(ground_truth_label, str) and ground_truth_label.strip():
        if context is None:
            context = {}
        context["ground_truth_label"] = ground_truth_label.strip()

    ground_truth_index = metadata.get("ground_truth_index")
    if isinstance(ground_truth_index, int) and not isinstance(ground_truth_index, bool) and ground_truth_index >= 0:
        if context is None:
            context = {}
        context["ground_truth_index"] = ground_truth_index

    return context


# [PROTOTYPE_LLM_MEMORY_V1] - Event input
@dataclass
class MemoryEvent:
    """
    Input event for memory processing.
    
    [INTEGRATION_POINT] Feature extractor should produce this.
    """
    session_id: str
    time_range: TimeRange
    
    # From pattern generator
    patterns: List[PatternKey] = field(default_factory=list)
    
    # From metrics computer
    metrics: Optional[WindowMetrics] = None
    
    # From session metadata
    cutting_context: Optional[CuttingContext] = None
    
    # External signals (classical models, etc.)
    external_signals: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: channels involved
    channels: List[str] = field(default_factory=list)
    
    # Optional: pre-computed feature vector
    feature_vector: Optional[List[float]] = None

    # Optional: flat feature dict (e.g. 17 CNC features from casedata)
    # Used by classical model enrichment when WindowMetrics is not available.
    raw_metrics: Optional[Dict[str, float]] = None

    # Optional: batch identity for batch-level learning/reconfiguration.
    batch: Optional[BatchContext] = None

    # Optional: arbitrary metadata dict (experiment flags, labels, etc.)
    metadata: Optional[Dict[str, Any]] = None


# [PROTOTYPE_LLM_MEMORY_V1] - Processing result
@dataclass
class MemoryEventResult:
    """Result of processing a memory event."""
    processed: bool
    significant: bool
    memory_id: Optional[str] = None
    significance_score: float = 0.0
    action: SignificanceAction = SignificanceAction.IGNORE
    explanation: Optional[str] = None
    explanation_source: Optional[str] = None
    alert_line: Optional[str] = None
    alert_line_source: Optional[str] = None
    # [LLM_GUARDRAILS_V1] Audit trail of the Tier-1 output-rail outcome applied
    # to the grounded explanation: {"action", "reasons", "checks"} or None.
    guardrail_outcome: Optional[Dict[str, Any]] = None
    similar_memories: List[MemoryMatch] = field(default_factory=list)
    alert_dispatched: bool = False
    reconfig_proposal_id: Optional[str] = None
    error: Optional[str] = None
    prior_boost: float = 0.0
    pattern_rule_score: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    # Agent C (2026-04-24): normalized per-model attribution snapshot built
    # from ``event.external_signals`` at processing time. See
    # ``backend/agents/memory/model_breakdown.py`` for the schema.
    model_breakdown: Dict[str, Any] = field(default_factory=dict)


# [PROTOTYPE_LLM_MEMORY_V1] - Orchestrator configuration
@dataclass
class OrchestratorConfig:
    """Configuration for the memory orchestrator."""
    # Component configs
    significance_config: Optional[SignificanceConfig] = None
    retrieval_config: Optional[RetrievalConfig] = None
    explainer_config: Optional[Any] = None  # ExplainerConfig, lazy imported
    
    # Paths for persistence
    priors_path: Optional[str] = None  # Path to persist pattern priors
    model_confidence_path: Optional[str] = None  # Path to persist model-confidence feedback state
    
    # [PROTOTYPE_CLASSICAL_RL_V1] - Optional classical anomaly detector
    use_classical_models: bool = True  # Enable seed model + RL anomaly detector
    seed_model_path: Optional[str] = None  # Path to trained seed model pickle
    rl_agent_path: Optional[str] = None  # Path to RL agent state JSON
    casedata_path: Optional[str] = None  # Path to case data for on-the-fly training
    # When True and no cached seed model exists, training runs on a
    # background thread so orchestrator init does not block on the
    # ~30 s casedata read. The detector returns no seed-model score
    # until training completes.
    lazy_seed_training: bool = False

    # [HARMONIC_CONTEXT_V1] - Harmonic context-weighted CNN scorer
    harmonic_config: Optional[Any] = None  # HarmonicContextConfig instance
    enable_harmonic_scorer: bool = True  # Auto-load if trained model exists

    # Behavior
    always_store: bool = False  # Store even non-significant events
    min_score_for_retrieval: float = 0.3  # Min significance to retrieve similar
    top_k_similar: int = 5  # Number of similar memories to retrieve
    generate_explanations: bool = False  # Use LLM for explanations (off by default — requires Ollama)
    # Tier-1 deterministic output rail over LLM explanations [LLM_GUARDRAILS_V1].
    # Only effective when generate_explanations is on. Default on.
    llm_guardrails_enabled: bool = True
    dispatch_alerts: bool = True  # Push alerts to clients
    compose_reconfig_proposals: bool = False  # Auto-compose proposal records for significant alerts
    reconfig_outbox_path: Optional[str] = None  # JSONL path for proposal persistence
    reconfig_min_score: float = 0.6  # Minimum score required for auto-composition


# [PROTOTYPE_LLM_MEMORY_V1] - Main orchestrator class
class MemoryEventOrchestrator:
    """
    Orchestrates the full memory event processing pipeline.
    
    [INTEGRATION_POINT] This is the main entry point.
    Call process_event() when patterns are detected.
    """
    
    def __init__(
        self,
        memory_store: Any = None,  # MemoryStore instance
        config: Optional[OrchestratorConfig] = None,
    ):
        self.config = config or OrchestratorConfig()
        
        # [PROTOTYPE_LLM_MEMORY_V1] - In-memory storage fallback
        # This adapter allows retriever and feedback to access in-memory storage
        self._memories: Dict[str, Memory] = {}
        
        # Use provided store or create in-memory adapter
        self.store = memory_store or self._create_in_memory_adapter()
        
        # Initialize components with persistence paths
        self.scorer = SignificanceScorer(
            config=self.config.significance_config,
            priors_path=self.config.priors_path,
            feedback_store=self.store,
            model_confidence_path=self.config.model_confidence_path,
        )
        
        # Always create retriever with proper store reference
        self.retriever = MemoryRetriever(self.store, self.config.retrieval_config)
        
        # Lazy import LLMExplainer to avoid circular import
        from ..llm.explainer import LLMExplainer
        self.explainer = LLMExplainer(self.config.explainer_config)

        # Dedicated executor for Neo4j I/O so store/trace/explanation-persist run
        # off the event loop AND don't queue behind the shared default thread pool
        # (which serves LLM-related to_thread work). Fixes the ISS-44 store-await
        # inflation: the persist itself is ~0.07s; the delay was loop contention.
        import concurrent.futures as _cf
        self._neo4j_executor = _cf.ThreadPoolExecutor(
            max_workers=6, thread_name_prefix="neo4j-io"
        )

        # [LLM_GUARDRAILS_V1] Tier-1 deterministic output rail. Applied to the
        # operator-facing LLM explanation before it is persisted/broadcast.
        # Tier 2 (semantic_checker) is intentionally left as None — not built.
        self.output_guardrail = None
        if getattr(self.config, "llm_guardrails_enabled", True):
            try:
                from ..llm.guardrails import OutputGuardrail
                import os as _os
                soft = _os.environ.get("LLM_GUARDRAIL_SOFT_BLOCK", "false").lower() in ("1", "true", "yes")
                self.output_guardrail = OutputGuardrail(soft_block=soft)
                logger.info("LLM output guardrail (Tier 1) enabled (soft_block=%s)", soft)
            except Exception as e:
                logger.warning("LLM output guardrail unavailable: %s", e)

        # Session score calibration (plan 1.2), opt-in. When enabled, the raw
        # one-class model score is replaced per session by a session-relative
        # rolling percentile before scoring — this fixes cross-session score
        # shift (offline: alerts on ~29% of windows → far fewer). OFF by default
        # because it needs ANOMALY_SCORE_THRESHOLD retuned for the percentile
        # scale (~0.95) and a live-session validation of the warm-up window.
        import os as _os
        self._calibrate_model_score = _os.environ.get(
            "MEMORY_CALIBRATE_MODEL_SCORE", "0"
        ).strip().lower() in ("1", "true", "yes")
        self._score_calibrators: Dict[str, Any] = {}
        self._calibrate_warmup = int(_os.environ.get("MEMORY_CALIBRATE_WARMUP", "30"))
        self._calibrate_window = int(_os.environ.get("MEMORY_CALIBRATE_WINDOW", "480"))
        if self._calibrate_model_score:
            logger.info(
                "Session score calibration ENABLED (warmup=%d window=%d) — ensure "
                "ANOMALY_SCORE_THRESHOLD is set for the percentile scale",
                self._calibrate_warmup, self._calibrate_window,
            )

        self.feedback_handler = MemoryFeedbackHandler(self.store, self.scorer)
        self.feedback_handler.retriever = self.retriever  # Wire retriever for feedback boost
        self.alert_dispatcher = get_dispatcher()

        # Pattern discovery engine — learns new patterns from confirmed events
        discovery_dir = Path(self.config.priors_path).parent if self.config.priors_path else Path("data")
        self.pattern_discovery = PatternDiscovery(data_dir=discovery_dir)
        self.feedback_handler.pattern_discovery = self.pattern_discovery

        # Wire Neo4j persistence callback for promoted patterns
        self.pattern_discovery.on_pattern_event = self._on_pattern_discovered

        # [PROTOTYPE_CLASSICAL_RL_V1] - Optional classical anomaly detector
        self.anomaly_detector = None
        if self.config.use_classical_models:
            try:
                from ..processing.classical_models import (
                    create_seed_model,
                    create_online_detector,
                )
                seed = create_seed_model(
                    casedata_path=self.config.casedata_path or "data/casedata",
                    model_path=self.config.seed_model_path,
                    lazy_training=self.config.lazy_seed_training,
                )
                self.anomaly_detector = create_online_detector(
                    seed_model=seed,
                    rl_path=self.config.rl_agent_path,
                )
                logger.info("Classical anomaly detector loaded")

                # Wire RL agent into the scorer for adaptive weight tuning
                if self.anomaly_detector.rl_agent is not None:
                    self.scorer.set_rl_agent(self.anomaly_detector.rl_agent)
                    logger.info("RL agent wired into scorer for adaptive weights")

                # Record model training time for age-decay (Improvement 7)
                if seed.is_trained:
                    train_ts = seed._training_stats.get("trained_at")
                    if train_ts:
                        # trained_at may be an ISO string – normalise to epoch float
                        if isinstance(train_ts, str):
                            from datetime import datetime, timezone
                            train_ts = datetime.fromisoformat(train_ts).timestamp()
                        self.scorer.set_model_retrained_at(float(train_ts))
                    else:
                        # Approximate: use model file mtime if available
                        model_path = self.config.seed_model_path
                        if model_path and Path(model_path).exists():
                            self.scorer.set_model_retrained_at(
                                Path(model_path).stat().st_mtime
                            )
            except Exception as e:
                logger.warning("Classical models not available: %s", e)

        # [HARMONIC_CONTEXT_V1] - Optional harmonic context scorer
        self.harmonic_scorer = None
        if self.config.enable_harmonic_scorer:
            try:
                from ..processing.harmonic_runtime import (
                    ensure_harmonic_scorer,
                    harmonic_torch_available,
                    resolve_runtime_harmonic_config,
                )

                # Fall back to an env-selected dataset preset (HARMONIC_RUNTIME_DATASET)
                # so the live stream loads a real trained checkpoint instead of the
                # non-existent default path.
                harmonic_config = self.config.harmonic_config or resolve_runtime_harmonic_config()

                if harmonic_torch_available(harmonic_config):
                    scorer = ensure_harmonic_scorer(harmonic_config)
                    if scorer is not None:
                        self.harmonic_scorer = scorer
                        logger.info(
                            "Harmonic scorer loaded (kind=%s dataset=%s)",
                            getattr(scorer.config, "scorer_kind", "context"),
                            scorer.config.dataset_name,
                        )
                    else:
                        logger.debug("Harmonic scorer: no trained model found")
                else:
                    logger.debug("Harmonic scorer unavailable: torch not installed")
            except Exception as e:
                logger.debug("Harmonic scorer init failed: %s", e)

        self._harmonic_row_history: Dict[str, Any] = {}

        # Background LLM tasks — keep strong references so they aren't GC'd
        self._background_tasks: set = set()

        logger.info("MemoryEventOrchestrator initialized")

        # Automated co-occurrence decay: prune stale edges on startup
        # so the graph stays current (Issue #7 fix, 2026-04-14).
        self._run_co_occurrence_decay()

    def _run_co_occurrence_decay(self) -> None:
        """Best-effort decay of stale co-occurrence edges on startup.

        Only activates when the store supports ``decay_old_co_occurrence``
        (Neo4j).  For SQLite, a similar pruning is performed in-place.
        """
        if not self.store:
            return
        try:
            if hasattr(self.store, "decay_old_co_occurrence"):
                pruned = self.store.decay_old_co_occurrence(
                    max_age_days=30, decay_factor=0.5, prune_below=1,
                )
                if pruned:
                    logger.info("Startup co-occurrence decay: pruned %d stale edges", pruned)
            elif hasattr(self.store, "_get_connection"):
                # SQLite fallback: delete edges with weight <= 1 and older than 30 days
                from datetime import timedelta
                cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                with self.store._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM co_occurrence WHERE weight <= 1 AND last_session < ?",
                        (cutoff,),
                    )
                    pruned = cursor.rowcount
                    conn.commit()
                if pruned:
                    logger.info("Startup SQLite co-occurrence decay: pruned %d stale edges", pruned)
        except Exception as e:
            logger.debug("Co-occurrence decay on startup failed: %s", e)
    
    def _create_in_memory_adapter(self) -> InMemoryStoreAdapter:
        """Create an in-memory store adapter that wraps self._memories."""
        return InMemoryStoreAdapter(self._memories)


    def _extract_feature_dict(self, event: MemoryEvent) -> Dict[str, float]:
        """Extract a flat numeric feature dict from event for pattern discovery."""
        feature_dict: Dict[str, float] = {}

        def _ingest(mapping: Any) -> None:
            # Accept dicts directly; convert dataclasses / objects via vars()
            if not isinstance(mapping, dict):
                try:
                    mapping = vars(mapping)
                except TypeError:
                    return
            for key, value in mapping.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_dict[str(key)] = float(value)

        if event.raw_metrics:
            _ingest(event.raw_metrics)
        elif event.metrics:
            _ingest(event.metrics)
        return feature_dict


    # ------------------------------------------------------------------
    # Evidence gathering for grounded LLM explanations
    # ------------------------------------------------------------------


    def _on_pattern_discovered(self, pat: Any) -> None:
        """Callback fired when PatternDiscovery promotes or updates a pattern.

        Persists the :DiscoveredPattern node and [:DISCOVERED_FROM] edges
        into the Neo4j graph (if the backend is Neo4j).  Also records an
        audit trace so the discovery event is visible in the timeline.
        """
        from datetime import datetime, timezone

        store = self.store
        if hasattr(store, 'store_discovered_pattern'):
            try:
                source_mids = [
                    se.memory_id for se in getattr(pat, 'source_events', [])
                    if se.memory_id
                ]
                store.store_discovered_pattern(
                    key=pat.key,
                    features=pat.features,
                    confirmation_count=pat.confirmation_count,
                    promoted=pat.promoted,
                    prior=pat.prior,
                    first_seen=datetime.fromtimestamp(pat.first_seen, tz=timezone.utc).isoformat(),
                    last_seen=datetime.fromtimestamp(pat.last_seen, tz=timezone.utc).isoformat(),
                    source_memory_ids=source_mids,
                )
                logger.info("Persisted discovered pattern '%s' to Neo4j (%d source memories)", pat.key, len(source_mids))
            except Exception as e:
                logger.debug("Neo4j persistence for discovered pattern failed: %s", e)

        # Audit trace
        if hasattr(store, 'add_trace'):
            try:
                store.add_trace(
                    session_id=None,
                    memory_id=None,
                    trace_type="pattern_discovered",
                    payload={
                        "pattern_key": pat.key,
                        "features": pat.features,
                        "confirmation_count": pat.confirmation_count,
                        "promoted": pat.promoted,
                        "source_memory_ids": [se.memory_id for se in getattr(pat, 'source_events', []) if se.memory_id],
                    },
                )
            except Exception:
                pass

    async def process_event(self, event: MemoryEvent) -> MemoryEventResult:
        """
        Process a memory event through the full pipeline.
        
        Flow:
        1. Score significance
        2. If significant enough, create and store memory
        3. Retrieve similar memories
        4. Generate explanation
        5. Dispatch alert
        
        Args:
            event: The memory event to process
        
        Returns:
            MemoryEventResult with processing outcome
        """
        try:
            # Experiment fast-path: when event metadata contains
            # experiment_fast_path=True, skip storage, retrieval, and broadcast
            # to reduce per-sample overhead.  Scoring + pattern discovery still
            # run so the API returns the authoritative score.
            _fast_path = bool(
                (event.metadata or {}).get("experiment_fast_path", False)
            )

            # Step 0: Enrich with classical model scores
            # [PROTOTYPE_CLASSICAL_RL_V1]
            if self.anomaly_detector is not None:
                detector_signals = enrich_with_classical_scores(
                    event, anomaly_detector=self.anomaly_detector
                )
                merged = dict(detector_signals)
                merged.update(event.external_signals or {})
                event.external_signals = merged

            # Step 0a: Enrich with harmonic context score [HARMONIC_CONTEXT_V1]
            if self.harmonic_scorer is not None and self.harmonic_scorer.is_available():
                try:
                    hc_signals = enrich_with_harmonic_score(
                        event,
                        harmonic_scorer=self.harmonic_scorer,
                        row_history=self._harmonic_row_history,
                    )
                    if hc_signals:
                        event.external_signals = {
                            **(event.external_signals or {}),
                            **hc_signals,
                        }
                except Exception as hc_err:
                    logger.debug("Harmonic context enrichment failed: %s", hc_err)

            # Step 0b: Update pattern-discovery baseline with every event
            feature_dict = self._extract_feature_dict(event)
            if feature_dict:
                self.pattern_discovery.update_baseline(
                    feature_dict,
                    cutting_context=event.cutting_context,
                )

            # Step 0c: Match discovered patterns and append to event patterns
            if feature_dict:
                discovered_keys = self.pattern_discovery.match_event(
                    feature_dict,
                    cutting_context=event.cutting_context,
                )
                if discovered_keys:
                    from ..core.schemas import PatternType
                    for dk in discovered_keys:
                        event.patterns = list(event.patterns or []) + [
                            PatternKey(
                                key=dk,
                                pattern_type=PatternType.CLUSTER,
                                confidence=0.7,
                            )
                        ]

            # Step 0d: Session score calibration (plan 1.2, opt-in)
            score_signals = self._maybe_calibrate_model_score(event)

            # Step 1: Score significance
            significance = self.scorer.score(
                patterns=event.patterns,
                metrics=event.metrics,
                context=event.cutting_context,
                session_id=event.session_id,
                external_signals=score_signals,
            )

            logger.debug(
                f"Event scored: {significance.score:.2f} "
                f"({significance.action.value}), "
                f"rules: {significance.triggered_rules}"
            )
            
            # Determine if we should process further
            should_store = (
                significance.is_significant or 
                self.config.always_store
            )
            event.batch = self._normalized_event_batch(event)
            
            # Agent C (2026-04-24): build model_breakdown from the enriched
            # external_signals. Cheap (dict-walk), safe on empty input.
            from .model_breakdown import build_model_breakdown
            _model_breakdown = build_model_breakdown(event.external_signals)

            if not should_store:
                # Persist a scoring trace even when not storing a memory.
                if not _fast_path:
                    self._add_trace(
                        session_id=event.session_id,
                        memory_id=None,
                        trace_type="score",
                        payload={
                            "patterns": [p.key for p in (event.patterns or [])],
                            "external_signals": event.external_signals or {},
                            "cutting_context": event.cutting_context.model_dump() if event.cutting_context else None,
                            "significance": significance.to_dict(),
                        },
                    )
                # Still broadcast the scored event (no memory) so the inference panel
                # shows sub-threshold / ignored scores live.
                if not _fast_path and self.config.dispatch_alerts:
                    try:
                        try:
                            _ignore_metrics = self._build_metrics_for_alert(event, significance)
                        except Exception:
                            _ignore_metrics = None
                        await self.alert_dispatcher.broadcast_scored_event(
                            event=event,
                            memory=None,
                            significance=significance,
                            cutting_context=_dispatch_context_snapshot(event.cutting_context, event.metadata),
                            metrics_summary=_ignore_metrics,
                        )
                    except Exception as e:
                        logger.debug(f"Ignore-action scored broadcast failed (non-critical): {e}")
                return MemoryEventResult(
                    processed=True,
                    significant=False,
                    significance_score=significance.score,
                    action=significance.action,
                    model_breakdown=_model_breakdown,
                    prior_boost=float(getattr(significance, "prior_boost", 0.0) or 0.0),
                    pattern_rule_score=float(getattr(significance, "pattern_rule_score", 0.0) or 0.0),
                    triggered_rules=list(getattr(significance, "triggered_rules", []) or []),
                )
            
            # Step 2: Create memory (skip storage in fast-path mode)
            memory = self._create_memory(event, significance)
            if _fast_path:
                memory_id = memory.id  # use generated ID but don't persist
            else:
                memory_id = await self._store_memory(memory)

            # Persist scoring trace linked to the stored memory.
            if not _fast_path:
                self._add_trace(
                    session_id=event.session_id,
                    memory_id=memory_id,
                    trace_type="score",
                    payload={
                        "patterns": [p.key for p in (event.patterns or [])],
                        "external_signals": event.external_signals or {},
                        "cutting_context": event.cutting_context.model_dump() if event.cutting_context else None,
                        "significance": significance.to_dict(),
                    },
                )
            
            # Step 3: Retrieve similar memories (skip in fast-path mode)
            similar_memories: List[MemoryMatch] = []
            reconfig_proposal_id: Optional[str] = None
            if not _fast_path:
                try:
                    reconfig_proposal_id = self._maybe_compose_reconfig_proposal(event, significance)
                except Exception:
                    logger.debug("Reconfig proposal composition failed for %s", memory_id, exc_info=True)
            if (
                not _fast_path
                and self.retriever 
                and significance.score >= self.config.min_score_for_retrieval
            ):
                import functools as _functools
                _loop = asyncio.get_running_loop()
                similar_memories = await _loop.run_in_executor(
                    self._neo4j_executor,
                    _functools.partial(
                    self.retriever.retrieve,
                    query_patterns=event.patterns,
                    query_metrics=event.metrics,
                    query_context=event.cutting_context,
                    session_id=event.session_id,
                    top_k=self.config.top_k_similar,
                    exclude_ids={memory_id},
                ))

                self._add_trace(
                    session_id=event.session_id,
                    memory_id=memory_id,
                    trace_type="retrieve",
                    payload={
                        "query_patterns": [p.key for p in (event.patterns or [])],
                        "returned": [
                            {
                                "memory_id": m.memory.id,
                                "score": float(m.relevance_score),
                                "reasons": list(m.match_reasons or []),
                            }
                            for m in (similar_memories or [])
                        ],
                        "exclude_ids": [memory_id],
                        "top_k": int(self.config.top_k_similar),
                    },
                )

            # Step 4: Check eligibility for alert dispatch BEFORE calling the
            # LLM, so we don't waste expensive inference on events that will
            # be rate-limited or suppressed.  (Issue #11)
            wants_alert = (
                not _fast_path
                and self.config.dispatch_alerts
                and significance.action in (SignificanceAction.ALERT, SignificanceAction.CRITICAL)
            )
            will_dispatch = False
            if wants_alert:
                will_dispatch = self.alert_dispatcher._check_rate_limit(
                    event.session_id
                ) and not self.alert_dispatcher._is_in_cooldown(event.session_id)
                if not will_dispatch:
                    logger.debug(
                        "Event rate-limited/in-cooldown — skipping LLM explanation "
                        "(session=%s, score=%.2f)",
                        event.session_id, significance.score,
                    )

            # Step 4b: Schedule LLM explanation in the background.
            # The LLM calls (explain_grounded_async + alert-line generation)
            # take 4-25 s each.  Rather than blocking the scoring pipeline we
            # fire them off as a background asyncio task.  The alert is
            # dispatched *immediately* with a template/fallback summary; the
            # full LLM explanation is broadcast as an ``explanation_update``
            # message once ready and also persisted on the memory record.
            explanation: Optional[str] = None
            explanation_source: Optional[str] = None
            alert_line: Optional[str] = None
            alert_line_source: Optional[str] = None
            _llm_pending = False
            # Event-level override of the explanation flag (metadata wins over config).
            _explanations_enabled = self.config.generate_explanations
            _expl_override = (event.metadata or {}).get("generate_explanations_override")
            if _expl_override is not None:
                _explanations_enabled = bool(_expl_override)
            if _explanations_enabled and will_dispatch:
                # Build the evidence context *now* (fast, <1 ms) so the
                # background task has everything it needs.
                expl_ctx = build_explanation_context(
                    event,
                    significance,
                    similar_memories,
                    scorer=self.scorer,
                    store=self.store,
                )
                _llm_pending = True
                task = asyncio.create_task(
                    self._generate_explanation_background(
                        memory=memory,
                        memory_id=memory_id,
                        event=event,
                        significance=significance,
                        similar_memories=similar_memories,
                        expl_ctx=expl_ctx,
                    )
                )
                # Prevent GC from collecting the task before it completes
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            
            # Step 5: Dispatch alert immediately (no LLM wait).
            # The dispatcher generates a template/fallback summary when
            # summary=None, so notifications appear instantly.
            alert_dispatched = False
            alert_doc_links: List[Dict[str, Any]] = []
            alert_context = _dispatch_context_snapshot(event.cutting_context, event.metadata)
            if wants_alert and will_dispatch:
                try:
                    alert_doc_links = await self.alert_dispatcher.propose_doc_links_for_memory(
                        memory=memory,
                        cutting_context=event.cutting_context.model_dump() if event.cutting_context else None,
                    )
                except Exception:
                    logger.debug("Pre-dispatch doc-link lookup failed for %s", memory_id, exc_info=True)
                    alert_doc_links = []
                similar_history: List[Dict[str, Any]] = []
                if hasattr(self.store, "get_similar_with_resolution"):
                    try:
                        similar_history = list(self.store.get_similar_with_resolution(
                            memory_id, k=self.config.top_k_similar,
                        ) or [])
                    except Exception:
                        logger.debug("similar-history resolution failed for %s", memory_id, exc_info=True)
                alert_dispatched = await self.alert_dispatcher.dispatch(
                    memory=memory,
                    significance=significance,
                    similar_memories=[m.to_query_result() for m in similar_memories],
                    similar_history=similar_history,
                    summary=None,          # LLM summary arrives later
                    summary_source=None,
                    explanation=None,       # LLM explanation arrives later
                    explanation_source=None,
                    cutting_context=alert_context,
                    metrics_summary=self._build_metrics_for_alert(event, significance),
                    doc_links=alert_doc_links,
                )
                if alert_doc_links and hasattr(self.store, "persist_doc_links"):
                    try:
                        await asyncio.to_thread(
                            self.store.persist_doc_links,
                            memory_id=memory_id,
                            pattern_keys=[p.key for p in (memory.pattern_keys or [])],
                            doc_links=alert_doc_links,
                        )
                    except Exception:
                        logger.debug("Doc-link persistence failed for %s", memory_id, exc_info=True)

            # Step 5b: Always broadcast scored event for inference panel
            # (includes sub-threshold events the alert dispatcher would skip)
            if not _fast_path and self.config.dispatch_alerts and not alert_dispatched:
                try:
                    await self.alert_dispatcher.broadcast_scored_event(
                        memory=memory,
                        significance=significance,
                        cutting_context=alert_context,
                        metrics_summary=self._build_metrics_for_alert(event, significance),
                    )
                except Exception as e:
                    logger.debug(f"Scored event broadcast failed (non-critical): {e}")

            # Step 5c: publish a scored-learning envelope onto the learning bus for
            # any stored event (the upstream feed for fleet/MaaS learning aggregation).
            if memory_id is not None:
                try:
                    await publish_scored_learning(
                        session_id=event.session_id,
                        memory_id=memory_id,
                        significance=significance,
                        patterns=event.patterns,
                        external_signals=event.external_signals,
                        model_breakdown=_model_breakdown,
                        alert_dispatched=alert_dispatched,
                        similar_memory_count=len(similar_memories),
                        time_range=event.time_range,
                        batch=event.batch.to_dict() if getattr(getattr(event, "batch", None), "to_dict", None) else None,
                    )
                except Exception:
                    logger.debug("publish_scored_learning failed (non-critical)", exc_info=True)

            logger.info(
                f"Event processed: memory_id={memory_id}, "
                f"significant={significance.is_significant}, "
                f"similar={len(similar_memories)}, "
                f"alerted={alert_dispatched}, llm_pending={_llm_pending}"
            )
            
            return MemoryEventResult(
                processed=True,
                significant=significance.is_significant,
                memory_id=memory_id,
                significance_score=significance.score,
                action=significance.action,
                explanation=None if _llm_pending else explanation,
                explanation_source=None if _llm_pending else explanation_source,
                alert_line=None if _llm_pending else alert_line,
                alert_line_source=None if _llm_pending else alert_line_source,
                similar_memories=similar_memories,
                alert_dispatched=alert_dispatched,
                reconfig_proposal_id=reconfig_proposal_id,
                model_breakdown=_model_breakdown,
                prior_boost=float(getattr(significance, "prior_boost", 0.0) or 0.0),
                pattern_rule_score=float(getattr(significance, "pattern_rule_score", 0.0) or 0.0),
                triggered_rules=list(getattr(significance, "triggered_rules", []) or []),
            )
            
        except Exception as e:
            logger.error(f"Event processing failed: {e}", exc_info=True)
            return MemoryEventResult(
                processed=False,
                significant=False,
                error=str(e),
            )

    def _add_trace(
        self,
        *,
        session_id: Optional[str],
        memory_id: Optional[str],
        trace_type: str,
        payload: Dict[str, Any],
    ) -> None:
        if not self.store or not hasattr(self.store, "add_trace"):
            return

        def _do() -> None:
            try:
                self.store.add_trace(
                    session_id=session_id,
                    memory_id=memory_id,
                    trace_type=trace_type,
                    payload=payload,
                )
            except Exception:
                logger.debug(
                    "Failed to persist memory trace %s for session %s memory %s",
                    trace_type, session_id, memory_id, exc_info=True,
                )

        # Traces are best-effort audit data — never block the event loop on them.
        # Schedule the Neo4j write on the dedicated executor and return immediately.
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self._neo4j_executor, _do)
        except RuntimeError:
            _do()  # no running loop (sync context) — run inline

    async def process_external_signal(
        self,
        session_id: str,
        signal_type: str,
        signal_value: Any,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEventResult:
        """
        Process an external signal (e.g., classical model alert).
        
        [INTEGRATION_POINT] Call this when external models detect events.
        
        Args:
            session_id: Session identifier
            signal_type: Type of signal (breakage_prediction, tool_wear, etc.)
            signal_value: Signal value
            time_range: Time range of the signal
            metadata: Additional metadata (including cutting conditions)
        """
        metadata = metadata or {}
        
        # Extract cutting context
        cutting_context = None
        if metadata:
            cutting_context = extract_context_from_metadata(metadata)
        
        # Build time range if not provided
        if time_range is None:
            # [PROTOTYPE_LLM_MEMORY_V1] - Dummy time range
            time_range = TimeRange(i0=0, i1=100, t0=0.0, t1=1.0, fs=1000.0)
        
        # Create a pattern for the external signal
        from ..core.schemas import PatternType
        signal_pattern = PatternKey(
            pattern_type=PatternType.CUSTOM,
            key=f"EXTERNAL_{signal_type.upper()}:{signal_value}",
            source_metric=signal_type,
        )
        
        # Create event
        event = MemoryEvent(
            session_id=session_id,
            time_range=time_range,
            patterns=[signal_pattern],
            cutting_context=cutting_context,
            external_signals={signal_type: signal_value},
        )
        
        return await self.process_event(event)
    
    def _create_memory(self, event: MemoryEvent, significance: SignificanceResult) -> Memory:
        """Create Memory object from event."""
        # Cross-reference to SINDIT digital twin (plan point 2).
        # Prefer an explicit machine_uri from event.metadata, else derive
        # from cutting_context.machine_id, else fall back to the default
        # single-machine URN. Stored verbatim on the Memory so the
        # /graph/unified endpoint can join against SINDIT by IRI without
        # an extra lookup per node.
        machine_uri: Optional[str] = None
        if event.metadata and isinstance(event.metadata.get("machine_uri"), str):
            machine_uri = event.metadata["machine_uri"]
        elif event.cutting_context and event.cutting_context.machine_id:
            mid = str(event.cutting_context.machine_id).strip()
            if mid:
                # Slugify roughly (lowercase; alnum/dash only).
                slug = "".join(
                    ch.lower() if (ch.isalnum() or ch in ("-", "_")) else "-"
                    for ch in mid
                ).strip("-") or "unknown"
                machine_uri = f"urn:lfl:asset:{slug}"
        if not machine_uri:
            machine_uri = "urn:lfl:asset:cnc-machine-1"

        harmonic_context = None
        if event.metadata and isinstance(event.metadata.get("harmonic_context"), dict):
            raw_harmonic = event.metadata["harmonic_context"]
            safe_harmonic: Dict[str, Any] = {}

            source = raw_harmonic.get("source")
            if isinstance(source, str) and source:
                safe_harmonic["source"] = source

            labels = raw_harmonic.get("feature_labels")
            if isinstance(labels, list):
                safe_labels = [str(label) for label in labels if label is not None]
                if safe_labels:
                    safe_harmonic["feature_labels"] = safe_labels

            feature_values = raw_harmonic.get("feature_values")
            if isinstance(feature_values, list):
                safe_values = []
                for value in feature_values:
                    if isinstance(value, (int, float)):
                        numeric = float(value)
                        if numeric == numeric and numeric not in (float("inf"), float("-inf")):
                            safe_values.append(numeric)
                if safe_values:
                    safe_harmonic["feature_values"] = safe_values

            context_weights = raw_harmonic.get("context_weights")
            if isinstance(context_weights, list):
                safe_weights = []
                for value in context_weights:
                    if isinstance(value, (int, float)):
                        numeric = float(value)
                        if numeric == numeric and numeric not in (float("inf"), float("-inf")):
                            safe_weights.append(numeric)
                if safe_weights:
                    safe_harmonic["context_weights"] = safe_weights

            if safe_harmonic:
                harmonic_context = safe_harmonic

        harmonic_runtime = None
        if event.metadata and isinstance(event.metadata.get("harmonic_runtime"), dict):
            raw_runtime = event.metadata["harmonic_runtime"]
            safe_runtime: Dict[str, Any] = {}

            scorer_kind = raw_runtime.get("scorer_kind")
            if isinstance(scorer_kind, str) and scorer_kind:
                safe_runtime["scorer_kind"] = scorer_kind

            dataset_name = raw_runtime.get("dataset")
            if isinstance(dataset_name, str) and dataset_name:
                safe_runtime["dataset"] = dataset_name

            if safe_runtime:
                harmonic_runtime = safe_runtime

        preserved_metadata = _curated_event_metadata_snapshot(event.metadata)
        memory_metadata = {
            **preserved_metadata,
            "significance_score": significance.score,
            "significance_action": significance.action.value,
            "triggered_rules": significance.triggered_rules,
            "significance": significance.to_dict(),
            "cutting_context": event.cutting_context.model_dump() if event.cutting_context else None,
            "batch": event.batch.model_dump(mode="json") if event.batch else None,
            "external_signals": {
                k: v for k, v in (event.external_signals or {}).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            # Store raw CNC features so feedback handler can extract
            # them for pattern discovery and model retraining.
            "raw_metrics": {
                k: v for k, v in (event.raw_metrics or {}).items()
                if isinstance(v, (int, float))
            } if event.raw_metrics else None,
            "harmonic_context": harmonic_context,
            "harmonic_runtime": harmonic_runtime,
        }

        return Memory(
            id=str(uuid.uuid4()),
            session_id=event.session_id,
            created_at=datetime.now(timezone.utc),
            created_by="system",
            time_range=event.time_range,
            channels=event.channels,
            annotation_text="; ".join(significance.reasons),
            tags=[significance.action.value],
            label=None,
            metrics=self._convert_metrics(event.metrics) or NumericMetrics(),
            pattern_keys=event.patterns,
            numeric_vector=event.feature_vector,
            machine_uri=machine_uri,
            provenance=MemoryProvenance(
                compute_version="prototype_v1",
                data_source=event.session_id,
            ),
            metadata=memory_metadata,
        )
        return memory

    def _normalized_event_batch(self, event: MemoryEvent) -> Optional[BatchContext]:
        if isinstance(event.batch, BatchContext):
            return event.batch
        if isinstance(event.batch, Mapping):
            return extract_batch_context({"batch": event.batch}, event.metadata or {})
        return extract_batch_context(event.metadata or {})

    def _reconfig_context_from_event(self, event: MemoryEvent) -> Optional[Dict[str, Optional[str]]]:
        if event.cutting_context is None:
            return None
        regime = event.cutting_context.operating_regime
        regime_value = regime.value if hasattr(regime, "value") else regime
        context = {
            "machine_type": event.cutting_context.machine_type,
            "tool_type": event.cutting_context.tool_type,
            "material": event.cutting_context.workpiece_material,
            "regime": regime_value,
        }
        if not all(str(context.get(key) or "").strip() for key in ("machine_type", "tool_type", "material", "regime")):
            return None
        return context

    def _maybe_calibrate_model_score(self, event: "MemoryEvent") -> Optional[Dict[str, Any]]:
        """Return external_signals for scoring, with the model score replaced by a
        session-relative rolling percentile when calibration is enabled (plan 1.2).

        Returns the event's signals unchanged when the flag is off, no signals
        exist, or no anomaly score is present — so default behaviour is identical.
        The calibrator is per-session and updated in event order.
        """
        signals = event.external_signals
        if not self._calibrate_model_score or not signals:
            return signals
        raw = signals.get("anomaly_detector_score")
        if raw is None or not isinstance(raw, (int, float)):
            return signals
        try:
            from ..processing.score_calibration import SessionScoreCalibrator
            sid = event.session_id or "_global"
            cal = self._score_calibrators.get(sid)
            if cal is None:
                cal = SessionScoreCalibrator(
                    warmup=self._calibrate_warmup, window=self._calibrate_window
                )
                self._score_calibrators[sid] = cal
            calibrated = cal.update(float(raw))
            out = dict(signals)
            out["anomaly_detector_score"] = round(calibrated.percentile, 4)
            out["anomaly_detector_score_raw"] = round(float(raw), 4)
            out["anomaly_detector_calibrated"] = True
            return out
        except Exception:
            logger.debug("session score calibration failed; using raw score", exc_info=True)
            return signals

    def _maybe_compose_reconfig_proposal(
        self,
        event: MemoryEvent,
        significance: SignificanceResult,
    ) -> Optional[str]:
        if not self.config.compose_reconfig_proposals:
            return None
        if significance.score < float(self.config.reconfig_min_score):
            return None
        if significance.action not in (SignificanceAction.ALERT, SignificanceAction.CRITICAL):
            return None

        pattern_keys = [str(pattern.key).strip() for pattern in (event.patterns or []) if str(pattern.key).strip()]
        if not pattern_keys:
            return None

        context = self._reconfig_context_from_event(event)
        if context is None:
            return None

        from ..llm.reconfig_prompt import compose_reconfiguration_prompt
        from .reconfig import append_reconfig_record

        pattern_scores: Dict[str, float] = {}
        for pattern in (event.patterns or []):
            key = str(pattern.key).strip()
            if not key:
                continue
            prior_score = float(self.scorer.get_pattern_prior(key, context=event.cutting_context))
            if isinstance(pattern.confidence, (int, float)):
                pattern_scores[key] = round((prior_score + float(pattern.confidence)) / 2.0, 4)
            else:
                pattern_scores[key] = prior_score
        proposal = compose_reconfiguration_prompt(
            triggered_by=pattern_keys,
            pattern_scores=pattern_scores,
            context=context,
            batch=self._normalized_event_batch(event),
        )
        append_reconfig_record(
            proposal,
            self.config.reconfig_outbox_path or "data/reconfig_outbox.jsonl",
        )
        return proposal.proposal_id

    def _apply_output_guardrail(
        self,
        explanation: Optional[str],
        explanation_source: Optional[str],
        expl_ctx: Any,
    ) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """[LLM_GUARDRAILS_V1] Run the Tier-1 output rail over the explanation.

        Returns ``(explanation, explanation_source, guardrail_outcome)``:
          - ``block``    → swap in the deterministic fallback explanation and
                           mark the source as ``"guardrail_fallback"``.
          - ``annotate`` → use the guardrail's annotated text.
          - ``pass``     → unchanged.

        Always degrades to the original text on any error so the loop is never
        broken (matches the codebase's degrade-to-fallback convention).
        """
        if self.output_guardrail is None or not explanation:
            return explanation, explanation_source, None
        try:
            result = self.output_guardrail.check(explanation, expl_ctx)
            outcome = result.to_dict()
            if result.action == "block":
                fallback = None
                try:
                    fallback = self.explainer._fallback_grounded(expl_ctx)
                except Exception:
                    fallback = explanation
                logger.warning(
                    "LLM explanation BLOCKED by guardrail (reasons=%s) — using fallback",
                    result.reasons,
                )
                return fallback, "guardrail_fallback", outcome
            if result.action == "annotate":
                logger.info(
                    "LLM explanation ANNOTATED by guardrail (reasons=%s)",
                    result.reasons,
                )
                return result.text, explanation_source, outcome
            return explanation, explanation_source, outcome
        except Exception as exc:
            logger.debug("Output guardrail errored, passing explanation through: %s", exc)
            return explanation, explanation_source, None

    def _persist_explanation_on_memory(
        self,
        memory_id: str,
        explanation: Optional[str],
        explanation_source: Optional[str],
        alert_line: Optional[str],
        alert_line_source: Optional[str],
        guardrail_outcome: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None,
    ) -> None:
        """Persist LLM explanation fields in memory metadata.

        This makes explanations retrievable via GET /agent/memory/{id}
        even after the transient MemoryEventResult is gone.
        """
        if not explanation and not alert_line and not guardrail_outcome and not recommendation:
            return
        if not self.store:
            return
        try:
            # The store protocol exposes get()/update()/store() — NOT get_memory()
            # or update_memory(). Resolve whichever read/write the backend offers so
            # the explanation is actually persisted (this was silently no-op'ing on
            # the Neo4j backend, so GET /agent/memory/{id} only ever saw the fallback).
            getter = getattr(self.store, "get_memory", None) or getattr(self.store, "get", None)
            mem = getter(memory_id) if getter else None
            if not (mem and isinstance(getattr(mem, "metadata", None), dict)):
                return
            if explanation:
                mem.metadata["explanation"] = explanation
                mem.metadata["explanation_source"] = explanation_source
            if alert_line:
                mem.metadata["alert_line"] = alert_line
                mem.metadata["alert_line_source"] = alert_line_source
            if recommendation:
                mem.metadata["recommendation"] = recommendation
            if guardrail_outcome is not None:
                mem.metadata["guardrail_outcome"] = guardrail_outcome

            if hasattr(self.store, "update_memory"):
                self.store.update_memory(mem)
            elif hasattr(self.store, "update"):
                # update() sets attributes on a fresh copy then re-stores; pass the
                # full merged metadata so existing fields are preserved.
                self.store.update(memory_id, {"metadata": mem.metadata})
            elif hasattr(self.store, "store"):
                self.store.store(mem)
        except Exception as exc:
            logger.debug("Failed to persist explanation on memory %s: %s", memory_id, exc)

    # ------------------------------------------------------------------
    # Background LLM explanation generation
    # ------------------------------------------------------------------

    async def _generate_explanation_background(
        self,
        *,
        memory: Memory,
        memory_id: str,
        event: "MemoryEvent",
        significance: "SignificanceResult",
        similar_memories: list,
        expl_ctx: Any,
    ) -> None:
        """Run LLM explanation generation in the background.

        Called via ``asyncio.create_task`` so the main event-processing
        pipeline is never blocked by slow LLM inference (4-25 s per call).

        On completion the method:
        1. Persists the explanation on the memory record.
        2. Broadcasts an ``explanation_update`` message so the frontend
           can patch the alert in place.
        """
        explanation: Optional[str] = None
        explanation_source: Optional[str] = None
        alert_line: Optional[str] = None
        alert_line_source: Optional[str] = None
        recommendation: Optional[str] = None
        guardrail_outcome: Optional[Dict[str, Any]] = None
        try:
            # Grounded explanation (detailed, 3-5 sentences)
            explanation, explanation_source = await self.explainer.explain_grounded_async(
                expl_ctx,
            )

            # [LLM_GUARDRAILS_V1] Tier-1 output rail. On 'block' fall back to the
            # deterministic explanation; on 'annotate' use the annotated text.
            # Wrapped so it can never break the loop.
            explanation, explanation_source, guardrail_outcome = self._apply_output_guardrail(
                explanation, explanation_source, expl_ctx,
            )

            # Short alert line (~20 words, for notifications)
            numeric_metrics = (
                self._convert_metrics(event.metrics)
                or self._numeric_metrics_from_raw(event.raw_metrics or {})
            )
            # Pull the just-emitted recurrence snapshot so the LLM can phrase
            # the alert as recurring / first-time appropriately.
            recurrence_snapshot = None
            try:
                recurrence_snapshot = self.alert_dispatcher.get_signature_lifecycle(
                    memory.session_id,
                    [p.key for p in (event.patterns or [])],
                )
            except Exception:
                recurrence_snapshot = None
            if similar_memories:
                alert_line, alert_line_source, recommendation = await self.explainer.summarize_with_history_for_alert_async(
                    current_memory=memory,
                    similar_memories=similar_memories,
                    significance=significance,
                    recurrence=recurrence_snapshot,
                )
            else:
                alert_line, alert_line_source, recommendation = await self.explainer.explain_significance_for_alert_async(
                    patterns=event.patterns,
                    significance=significance,
                    context=event.cutting_context,
                    metrics=numeric_metrics,
                    recurrence=recurrence_snapshot,
                )

            # 1. Persist on memory record (incl. guardrail audit trail) — off the
            #    event loop so this background task doesn't block concurrent stores.
            import functools as _functools
            _loop = asyncio.get_running_loop()
            await _loop.run_in_executor(
                self._neo4j_executor,
                _functools.partial(
                    self._persist_explanation_on_memory,
                    memory_id, explanation, explanation_source,
                    alert_line, alert_line_source,
                    guardrail_outcome=guardrail_outcome,
                    recommendation=recommendation,
                ),
            )

            # 2. Broadcast explanation_update so the frontend can patch
            #    the alert / scored-event card in real time.
            await self.alert_dispatcher.broadcast_explanation_update(
                memory_id=memory_id,
                session_id=memory.session_id,
                explanation=explanation,
                explanation_source=explanation_source,
                alert_line=alert_line,
                alert_line_source=alert_line_source,
                guardrail_outcome=guardrail_outcome,
                recommendation=recommendation,
            )

            # 3. Publish an insight-learning envelope onto the learning bus.
            try:
                await publish_insight_learning(
                    session_id=memory.session_id,
                    memory_id=memory_id,
                    explanation=explanation,
                    explanation_source=explanation_source,
                    alert_line=alert_line,
                    alert_line_source=alert_line_source,
                )
            except Exception:
                logger.debug("publish_insight_learning failed (non-critical)", exc_info=True)

            logger.debug(
                "Background LLM explanation ready for %s (source=%s)",
                memory_id, explanation_source,
            )
        except Exception as exc:
            logger.warning(
                "Background LLM explanation failed for %s: %s",
                memory_id, exc,
            )

    async def create_operator_memory(
        self,
        *,
        session_id: str,
        time_range: TimeRange,
        channels: Optional[List[str]] = None,
        annotation_text: str = "",
        tags: Optional[List[str]] = None,
        label: Optional[str] = None,
        created_by: str = "operator",
        metrics: Optional[WindowMetrics] = None,
        patterns: Optional[List[PatternKey]] = None,
        feature_vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Create and persist a memory directly from an operator/UI capture.

        This bypasses significance gating (it always stores), but still computes
        and stores patterns/metrics if provided.
        """

        memory = Memory(
            id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            time_range=time_range,
            channels=list(channels or []),
            annotation_text=str(annotation_text or ""),
            tags=list(dict.fromkeys([*(tags or []), "operator_capture"])),
            label=label,
            metrics=self._convert_metrics(metrics) or NumericMetrics(),
            pattern_keys=list(patterns or []),
            numeric_vector=list(feature_vector) if feature_vector is not None else None,
            provenance=MemoryProvenance(
                compute_version="ui_capture_v1",
                data_source=session_id,
            ),
            metadata={
                **(metadata or {}),
                "capture": {
                    "i0": int(time_range.i0),
                    "i1": int(time_range.i1),
                    "fs": float(time_range.fs),
                },
            },
        )

        memory_id = await self._store_memory(memory)
        self._add_trace(
            session_id=session_id,
            memory_id=memory_id,
            trace_type="capture",
            payload={
                "created_by": created_by,
                "label": label,
                "tags": list(memory.tags or []),
                "channels": list(memory.channels or []),
                "time_range": time_range.model_dump() if hasattr(time_range, "model_dump") else None,
                "patterns": [p.key for p in (patterns or [])],
                "has_metrics": bool(metrics is not None),
                "has_feature_vector": bool(feature_vector is not None),
            },
        )
        return memory
    
    async def _store_memory(self, memory: Memory) -> str:
        """Store memory and return ID.

        After persisting, records pattern co-occurrence edges for every
        pair of pattern keys in the memory (Gap #2 — co-occurrence in
        the live API pipeline).
        """
        # Always store in _memories for local access
        self._memories[memory.id] = memory

        if self.store:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._neo4j_executor, self._persist_memory_to_store, memory)

        return memory.id

    def _persist_memory_to_store(self, memory: Memory) -> None:
        """Run synchronous store operations off the event loop."""
        try:
            if hasattr(self.store, 'store'):
                self.store.store(memory)
            elif hasattr(self.store, 'save'):
                self.store.save(memory)
        except Exception as e:
            logger.warning(f"Failed to store in MemoryStore: {e}")

        if (
            hasattr(self.store, "upsert_co_occurrence")
            and len(memory.pattern_keys) >= 2
        ):
            keys = [pk.key for pk in memory.pattern_keys]
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    try:
                        self.store.upsert_co_occurrence(a, b, memory.session_id)
                    except Exception as e:
                        logger.debug("Co-occurrence upsert failed (%s, %s): %s", a, b, e)
    
    def _convert_metrics(self, metrics: Optional[WindowMetrics]) -> Optional[NumericMetrics]:
        """Convert WindowMetrics to NumericMetrics schema."""
        if metrics is None:
            return None
        
        return NumericMetrics(
            means={f"ch_{i}": v for i, v in enumerate(metrics.channel_means)},
            stds={f"ch_{i}": v for i, v in enumerate(metrics.channel_stds)},
            rms={f"ch_{i}": v for i, v in enumerate(metrics.channel_rms)},
            peaks={f"ch_{i}": v for i, v in enumerate(metrics.channel_peaks)},
            dominant_freqs={f"ch_{i}": v for i, v in enumerate(metrics.dominant_frequencies)},
            spectral_centroids={f"ch_{i}": v for i, v in enumerate(metrics.spectral_centroids)},
        )

    @staticmethod
    def _numeric_metrics_from_raw(raw: Dict[str, float]) -> Optional[NumericMetrics]:
        """Build a :class:`NumericMetrics` from a flat feature dict.

        Used as a fallback when a real :class:`WindowMetrics` object is
        unavailable (common during experiments where features arrive as
        flat dicts from the evaluator).  Groups features into the
        NumericMetrics buckets by name heuristics.
        """
        if not raw:
            return None
        rms: Dict[str, float] = {}
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        peaks: Dict[str, float] = {}
        dom_freqs: Dict[str, float] = {}
        for key, val in raw.items():
            if not isinstance(val, (int, float)):
                continue
            kl = key.lower()
            if "rms" in kl:
                rms[key] = float(val)
            elif kl.endswith("_std"):
                stds[key] = float(val)
            elif kl.endswith("_max") or "peak" in kl:
                peaks[key] = float(val)
            elif "freq" in kl or "chatter" in kl:
                dom_freqs[key] = float(val)
            elif kl.endswith("_mean"):
                means[key] = float(val)
        if not (rms or means or peaks):
            return None
        return NumericMetrics(
            means=means,
            stds=stds,
            rms=rms,
            peaks=peaks,
            dominant_freqs=dom_freqs,
        )
    
    def _get_metrics_summary(self, metrics: Optional[WindowMetrics]) -> Optional[Dict[str, Any]]:
        """Get brief metrics summary for alerts."""
        if metrics is None:
            return None
        
        return {
            "rms": metrics.channel_rms[:4] if metrics.channel_rms else [],
            "dominant_freq": metrics.dominant_frequencies[:4] if metrics.dominant_frequencies else [],
            "total_energy": metrics.total_energy,
        }

    def _build_metrics_for_alert(
        self, event: MemoryEvent, significance: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build metrics dict for alert payload.

        Prefers raw_metrics (17 CNC features) from the classical model
        pipeline, merged with external_signals (which contains model
        inference results like anomaly_detector_score).  Falls back to
        the legacy WindowMetrics summary.

        When *significance* is given, the scoring breakdown is merged so
        the live UI can render per-event score charts.
        """
        result: Dict[str, Any] = {}

        # Start with raw CNC features if available
        if event.raw_metrics:
            result.update(event.raw_metrics)

        # Merge inference results + model/harmonic provenance from external_signals
        if event.external_signals:
            for key in (
                "anomaly_detector_score",
                "model_confidence",
                "model_source",
                "breakage_prediction",
                "tool_wear_estimate",
                "harmonic_context_score",
                "harmonic_context_source",
                "harmonic_pair_score",
                "harmonic_pair_source",
                "stoppage_probability",
                "stoppage_label",
                "stoppage_eta_s",
            ):
                if key in event.external_signals:
                    result[key] = event.external_signals[key]

        # Sample-rate provenance so the UI can align time axes.
        fs = getattr(getattr(event, "time_range", None), "fs", None)
        if fs:
            result["sample_rate_hz"] = float(fs)
            result["fs"] = float(fs)

        # Merge significance scoring breakdown for live charts
        if significance is not None:
            result["significance_score"] = getattr(significance, "score", 0.0)
            result["prior_boost"] = getattr(significance, "prior_boost", 0.0)
            result["prior_damping_factor"] = getattr(significance, "prior_damping_factor", 1.0)
            result["prior_evidence_count"] = getattr(significance, "prior_evidence_count", 0)
            result["n_rules_triggered"] = len(getattr(significance, "triggered_rules", []) or [])

        if result:
            return result

        # Legacy fallback
        return self._get_metrics_summary(event.metrics)
    
    # [PROTOTYPE_LLM_MEMORY_V1] - Helper methods
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID."""
        if self.store and hasattr(self.store, 'get'):
            return self.store.get(memory_id)
        return self._memories.get(memory_id)
    
    def list_memories(self, session_id: Optional[str] = None) -> List[Memory]:
        """List memories, optionally filtered by session."""
        # Prefer the backing store if it can list persisted memories.
        if self.store and hasattr(self.store, "list"):
            try:
                return list(self.store.list(session_id=session_id))
            except Exception:
                pass
        # Session-scoped query: use the store's own scoped method when present
        # (e.g. Neo4jMemoryStore.list_by_session runs a MATCH ...-[:IN_SESSION]->
        # Cypher). The list_all(limit=1000)+python-filter fallback below is
        # fragile on a large graph — if the session's memories aren't within the
        # newest 1000 rows the filter silently returns nothing (ISS-24).
        if session_id and self.store and hasattr(self.store, "list_by_session"):
            try:
                return list(self.store.list_by_session(session_id, limit=500))
            except Exception:
                pass
        if self.store and hasattr(self.store, "list_all"):
            try:
                all_mems = list(self.store.list_all(limit=1000))
                if session_id:
                    all_mems = [m for m in all_mems if m.session_id == session_id]
                return all_mems
            except Exception:
                pass

        memories = list(self._memories.values())
        if session_id:
            memories = [m for m in memories if m.session_id == session_id]
        return sorted(memories, key=lambda m: m.created_at, reverse=True)

    async def attach_passive_cycle_outcome(self, cycle: CycleEnded) -> int:
        """Correlate a finished cycle back to memories in the same time window."""
        memories = self.list_memories(session_id=cycle.session_id)
        affected = attach_passive_outcome(
            cycle=cycle,
            memories=memories,
            store=self.store,
            scorer=self.scorer,
        )

        try:
            append_session_log(
                cycle.session_id,
                {
                    "phase": "cycle_end",
                    "part_id": cycle.part_id,
                    "operation_id": cycle.operation_id,
                    "started_at": cycle.started_at,
                    "ended_at": cycle.ended_at,
                    "passive_feedback_count": affected,
                },
            )
        except Exception:
            logger.debug("Failed to append cycle_end session log", exc_info=True)

        return affected

    def delete(self, memory_id: str) -> bool:
        """Delete a memory (best-effort across in-memory and backing store)."""
        deleted = False
        if memory_id in self._memories:
            self._memories.pop(memory_id, None)
            deleted = True

        if self.store and hasattr(self.store, "delete"):
            try:
                deleted = bool(self.store.delete(memory_id)) or deleted
            except Exception:
                pass
        return deleted


# [PROTOTYPE_LLM_MEMORY_V1] - Global orchestrator instance
_orchestrator: Optional[MemoryEventOrchestrator] = None


def set_orchestrator(orchestrator: MemoryEventOrchestrator):
    """Set the global orchestrator instance (called by init module)."""
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator(
    memory_store: Any = None,
    config: Optional[OrchestratorConfig] = None,
) -> MemoryEventOrchestrator:
    """Get or create the global MemoryEventOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MemoryEventOrchestrator(memory_store, config)
    return _orchestrator


async def process_memory_event(event: MemoryEvent) -> MemoryEventResult:
    """Convenience function to process via global orchestrator."""
    orchestrator = get_orchestrator()
    return await orchestrator.process_event(event)


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function
def create_orchestrator(
    memory_store: Any = None,
    config: Optional[OrchestratorConfig] = None,
) -> MemoryEventOrchestrator:
    """Create a new MemoryEventOrchestrator instance."""
    return MemoryEventOrchestrator(memory_store, config)

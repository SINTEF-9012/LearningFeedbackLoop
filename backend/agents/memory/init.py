"""
Memory System Initialization - Wire up all components.

# ===========================================================================
# [PROTOTYPE_LLM_MEMORY_V1] - Central initialization for memory system
# This module initializes MemoryStore, PatternIndex, and Orchestrator
# with proper configuration.
# ===========================================================================

Usage:
    from backend.agents.memory.init import initialize_memory_system, get_memory_components

    # Initialize once at startup
    initialize_memory_system()
    
    # Get components anywhere
    store, orchestrator = get_memory_components()
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import asdict, replace
from typing import Optional, Tuple, TYPE_CHECKING
from pathlib import Path

from ..config import EMBEDDING_MODEL, get_config, MemorySystemConfig
from ..storage.store import MemoryStore
from ..storage.protocol import MemoryStoreProtocol
from ..storage.pattern_index import PatternIndex
from .orchestrator import MemoryEventOrchestrator, OrchestratorConfig, set_orchestrator
from .scorer import SignificanceConfig

if TYPE_CHECKING:
    from .orchestrator import MemoryEventOrchestrator

logger = logging.getLogger(__name__)

_RUNTIME_OVERRIDES_FILENAME = "runtime_overrides.json"
_RUNTIME_OVERRIDE_KEYS = frozenset({"generate_explanations", "dispatch_alerts"})


def _sqlite_fallback_allowed() -> bool:
    return os.environ.get("ALLOW_SQLITE_FALLBACK", "").strip().lower() in {"1", "true", "yes"}


def _runtime_overrides_path(config: Optional[MemorySystemConfig] = None) -> Path:
    cfg = config or get_config()
    return Path(cfg.db_path).parent / _RUNTIME_OVERRIDES_FILENAME


def load_runtime_overrides(config: Optional[MemorySystemConfig] = None) -> dict[str, Any]:
    path = _runtime_overrides_path(config)
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read runtime override file at %s", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: bool(value)
        for key, value in data.items()
        if key in _RUNTIME_OVERRIDE_KEYS
    }


def persist_runtime_overrides(
    overrides: dict[str, Any],
    config: Optional[MemorySystemConfig] = None,
) -> dict[str, Any]:
    path = _runtime_overrides_path(config)
    merged = load_runtime_overrides(config)
    for key, value in dict(overrides or {}).items():
        if key in _RUNTIME_OVERRIDE_KEYS:
            merged[key] = bool(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return merged


def apply_runtime_overrides(config: MemorySystemConfig, overrides: Optional[dict[str, Any]] = None) -> MemorySystemConfig:
    effective = dict(load_runtime_overrides(config) if overrides is None else (overrides or {}))
    filtered = {
        key: bool(value)
        for key, value in effective.items()
        if key in _RUNTIME_OVERRIDE_KEYS
    }
    if not filtered:
        return config
    return replace(config, **filtered)


# ============================================================================
# Global instances
# ============================================================================

_memory_store: Optional[MemoryStoreProtocol] = None
_pattern_index: Optional[PatternIndex] = None
_orchestrator: Optional[MemoryEventOrchestrator] = None
_initialized: bool = False


def initialize_memory_system(
    config: Optional[MemorySystemConfig] = None,
    force: bool = False,
) -> Tuple[Optional[MemoryStore], Optional["MemoryEventOrchestrator"]]:
    """
    Initialize the complete memory system with all components wired together.
    
    Args:
        config: Configuration (uses env-based config if None)
        force: Force re-initialization even if already initialized
    
    Returns:
        Tuple of (MemoryStore, MemoryEventOrchestrator)
    """
    global _memory_store, _pattern_index, _orchestrator, _initialized
    
    if _initialized and not force:
        logger.debug("Memory system already initialized, returning existing instances")
        return _memory_store, _orchestrator
    
    config = config or get_config()
    effective_config = apply_runtime_overrides(config)
    if effective_config != config:
        logger.info(
            "Applied runtime memory config overrides: %s",
            {
                key: getattr(effective_config, key)
                for key in _RUNTIME_OVERRIDE_KEYS
                if getattr(effective_config, key) != getattr(config, key)
            },
        )
    logger.info("Initializing memory system with config: %s", config)
    
    # Ensure data directory exists
    data_dir = Path(config.db_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize store backend based on configuration
    if config.storage_backend == "neo4j":
        logger.info("Initializing Neo4jMemoryStore at %s", config.neo4j_uri)
        try:
            from ..storage.neo4j_store import Neo4jMemoryStore

            _memory_store = Neo4jMemoryStore(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database,
                connect_timeout_s=config.neo4j_connect_timeout_s,
                max_pool_size=config.neo4j_max_pool_size,
                max_transaction_retry_s=config.neo4j_tx_retry_s,
                graph_outbox_path=config.neo4j_graph_outbox_path,
            )
        except Exception as exc:
            if not _sqlite_fallback_allowed():
                logger.exception(
                    "Neo4j memory store unavailable and SQLite fallback is disabled"
                )
                raise RuntimeError(
                    "Neo4j memory store unavailable and ALLOW_SQLITE_FALLBACK is not enabled"
                ) from exc
            effective_config = replace(effective_config, storage_backend="sqlite")
            logger.warning(
                "Neo4j memory store unavailable; falling back to SQLite at %s: %s",
                effective_config.db_path,
                exc,
            )
            _memory_store = MemoryStore(
                db_path=effective_config.db_path,
                enable_ann=effective_config.enable_ann,
                enable_embeddings=effective_config.enable_embeddings,
                embedding_model=EMBEDDING_MODEL if effective_config.enable_embeddings else None,
            )
    else:
        logger.info("Initializing SQLite MemoryStore at %s", config.db_path)
        _memory_store = MemoryStore(
            db_path=config.db_path,
            enable_ann=config.enable_ann,
            enable_embeddings=config.enable_embeddings,
            embedding_model=EMBEDDING_MODEL if config.enable_embeddings else None,
        )
    logger.info("Resolved memory store: %s", type(_memory_store).__name__)
    
    # 2. Initialize PatternIndex (in-memory, optional persistence via store)
    # SQLite store creates its own PatternIndex; for Neo4j we create a standalone one.
    logger.info("PatternIndex initialized (in-memory)")
    _pattern_index = (
        _memory_store.pattern_index
        if hasattr(_memory_store, "pattern_index")
        else PatternIndex()
    )
    
    # 3. Build OrchestratorConfig from system config
    threshold_values = asdict(config.thresholds)
    sig_field_names = set(SignificanceConfig.__dataclass_fields__)
    sig_kwargs = {
        key: value for key, value in threshold_values.items() if key in sig_field_names
    }
    ignored_threshold_keys = sorted(
        key for key in threshold_values.keys() if key not in sig_field_names
    )
    if ignored_threshold_keys:
        logger.debug(
            "Ignoring SignificanceThresholds fields not supported by SignificanceConfig: %s",
            ", ".join(ignored_threshold_keys),
        )
    sig_config = SignificanceConfig(**sig_kwargs)
    
    # Build ExplainerConfig from the system config so that programmatic
    # overrides (e.g. in tests) propagate to the LLM explainer rather
    # than being silently ignored.  (Issue #4 — config wiring gap)
    from ..llm.explainer import ExplainerConfig  # deferred to break circular import
    explainer_cfg = ExplainerConfig(
        ollama_url=config.ollama_url,
        model=config.ollama_model,
        timeout=config.ollama_timeout,
    )
    logger.info(
        "ExplainerConfig built from MemorySystemConfig: model=%s, url=%s, timeout=%.1f",
        explainer_cfg.model, explainer_cfg.ollama_url, explainer_cfg.timeout,
    )

    orch_kwargs = {
        "significance_config": sig_config,
        "explainer_config": explainer_cfg,
        "priors_path": effective_config.pattern_priors_path,
        "model_confidence_path": effective_config.model_confidence_path,
        "generate_explanations": effective_config.generate_explanations,
        "llm_guardrails_enabled": getattr(
            effective_config, "llm_guardrails_enabled", True
        ),
        "dispatch_alerts": effective_config.dispatch_alerts,
        "top_k_similar": effective_config.top_k_similar,
        "use_classical_models": effective_config.use_classical_models,
        "lazy_seed_training": effective_config.lazy_seed_training,
    }
    orch_field_names = set(OrchestratorConfig.__dataclass_fields__)
    ignored_orch_keys = sorted(
        key for key in orch_kwargs.keys() if key not in orch_field_names
    )
    if ignored_orch_keys:
        logger.debug(
            "Ignoring MemorySystemConfig fields not supported by OrchestratorConfig: %s",
            ", ".join(ignored_orch_keys),
        )
    orch_config = OrchestratorConfig(
        **{key: value for key, value in orch_kwargs.items() if key in orch_field_names}
    )
    
    # 4. Initialize Orchestrator with MemoryStore
    logger.info("Initializing MemoryEventOrchestrator")
    _orchestrator = MemoryEventOrchestrator(
        memory_store=_memory_store,
        config=orch_config,
    )

    # Demo-friendly behavior:
    # - Never silently disable explanations (that makes demos look "non-human-readable").
    # - Optionally fail-fast if REQUIRE_LLM=true so a demo can't run without the LLM.
    try:
        if _orchestrator.config.generate_explanations and not _orchestrator.explainer.is_available():
            msg = (
                "LLM not available (or model missing); alerts will fall back to non-LLM summaries. "
                "Set REQUIRE_LLM=true to fail-fast for demos."
            )
            if bool(getattr(effective_config, "require_llm", False)):
                raise RuntimeError(msg)
            logger.warning(msg)
    except Exception:
        # Best effort; don't break startup unless explicitly strict.
        if bool(getattr(effective_config, "require_llm", False)):
            raise
        pass
    
    # 5. Wire SINDIT context provider into the memory bridge (Gap #1)
    if effective_config.sindit_enabled:
        try:
            from ..sindit.client import SinditClient
            from ..sindit.context_provider import SinditContextProvider
            from .feature_stream_bridge import set_sindit_provider

            sindit_client = SinditClient(
                base_url=effective_config.sindit_api_url,
                timeout=effective_config.sindit_timeout_s,
            )
            sindit_provider = SinditContextProvider(
                client=sindit_client,
                machine_asset_iri=getattr(effective_config, "sindit_machine_iri", None),
            )
            set_sindit_provider(sindit_provider)
            logger.info(
                "SINDIT context provider registered (url=%s)",
                effective_config.sindit_api_url,
            )

            # Also wire into the scorer for context-conditioned profiles
            if _orchestrator and hasattr(_orchestrator, 'scorer'):
                _orchestrator.scorer.set_sindit_provider(sindit_provider)
        except Exception as exc:
            logger.warning("Failed to initialise SINDIT provider: %s", exc)
    else:
        logger.debug("SINDIT disabled — bridge will not enrich context")

    # Register as global orchestrator
    set_orchestrator(_orchestrator)
    
    _initialized = True
    logger.info("Memory system initialization complete")
    
    return _memory_store, _orchestrator


def get_memory_components() -> Tuple[Optional[MemoryStoreProtocol], Optional["MemoryEventOrchestrator"]]:
    """
    Get the initialized memory system components.
    
    Returns:
        Tuple of (store, orchestrator) or (None, None) if not initialized.
        Store conforms to MemoryStoreProtocol (SQLite or Neo4j).
    """
    return _memory_store, _orchestrator


def get_store() -> Optional[MemoryStoreProtocol]:
    """Get the MemoryStore instance."""
    return _memory_store


def get_pattern_index() -> Optional[PatternIndex]:
    """Get the PatternIndex instance."""
    return _pattern_index


def is_initialized() -> bool:
    """Check if memory system is initialized."""
    return _initialized


def shutdown_memory_system():
    """Clean shutdown of memory system (persist indices, close connections)."""
    global _memory_store, _pattern_index, _orchestrator, _initialized
    
    logger.info("Shutting down memory system")
    
    if _pattern_index:
        try:
            config = get_config()
            _pattern_index.save(config.pattern_index_path)
            logger.info("PatternIndex saved")
        except Exception as e:
            logger.error("Failed to save PatternIndex: %s", e)
    
    if _memory_store:
        try:
            _memory_store.close()
            logger.info("MemoryStore closed")
        except Exception as e:
            logger.error("Failed to close MemoryStore: %s", e)
    
    _memory_store = None
    _pattern_index = None
    _orchestrator = None
    _initialized = False
    
    logger.info("Memory system shutdown complete")

"""Agents package: registry and helpers.

This package contains small agent implementations and a router used to
dispatch incoming requests to agent handlers.

# ===========================================================================
# [PROTOTYPE_LLM_MEMORY_V1] - Restructured into logical subpackages:
# - core: Shared schemas, context, metrics
# - memory: LLM memory orchestration (scorer, retriever, feedback, dispatcher)
# - storage: Persistence layer (store, ann_index, pattern_index)
# - patterns: Pattern detection (generator)
# - llm: LLM integration (explainer, rag, ingest)
# - processing: Real-time agents (compute, online)
# ===========================================================================
#
# Imports are lazy to avoid loading all sub-packages at import time.
# This reduces startup latency and makes the dependency graph explicit.
"""

__all__ = [
    "router",
    # Memory system
    "MemoryEventOrchestrator",
    "MemoryEvent",
    "MemoryEventResult",
    "get_orchestrator",
    "SignificanceScorer",
    "SignificanceResult",
    "SignificanceAction",
    "CuttingContext",
    "extract_context_from_metadata",
    "get_dispatcher",
    "dispatch_significant_event",
    "MemoryFeedbackHandler",
    "MemoryFeedbackRequest",
    # Core schemas
    "Memory",
    "PatternKey",
    "PatternType",
    "NumericMetrics",
    "TimeRange",
    # Storage
    "MemoryStore",
    "ANNIndex",
    "PatternIndex",
    # Patterns
    "PatternGenerator",
    # LLM
    "LLMExplainer",
    "LLMAgent",
    "Ingestor",
    # Processing
    "ComputeAgent",
    "OnlineAgent",
]


# ── Lazy import mapping ─────────────────────────────────────────────────────
# Maps symbol name → (subpackage, attribute) for deferred loading.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "router":                       (".router", "router"),
    # memory
    "MemoryEventOrchestrator":      (".memory", "MemoryEventOrchestrator"),
    "MemoryEvent":                  (".memory", "MemoryEvent"),
    "MemoryEventResult":            (".memory", "MemoryEventResult"),
    "get_orchestrator":             (".memory", "get_orchestrator"),
    "SignificanceScorer":           (".memory", "SignificanceScorer"),
    "SignificanceResult":           (".memory", "SignificanceResult"),
    "SignificanceAction":           (".memory", "SignificanceAction"),
    "get_dispatcher":               (".memory", "get_dispatcher"),
    "dispatch_significant_event":   (".memory", "dispatch_significant_event"),
    "MemoryFeedbackHandler":        (".memory", "MemoryFeedbackHandler"),
    "MemoryFeedbackRequest":        (".memory", "MemoryFeedbackRequest"),
    # core
    "CuttingContext":               (".core", "CuttingContext"),
    "extract_context_from_metadata":(".core", "extract_context_from_metadata"),
    "Memory":                       (".core", "Memory"),
    "PatternKey":                   (".core", "PatternKey"),
    "PatternType":                  (".core", "PatternType"),
    "NumericMetrics":               (".core", "NumericMetrics"),
    "TimeRange":                    (".core", "TimeRange"),
    # storage
    "MemoryStore":                  (".storage", "MemoryStore"),
    "ANNIndex":                     (".storage", "ANNIndex"),
    "PatternIndex":                 (".storage", "PatternIndex"),
    # patterns
    "PatternGenerator":             (".patterns", "PatternGenerator"),
    # llm
    "LLMExplainer":                 (".llm", "LLMExplainer"),
    "LLMAgent":                     (".llm", "LLMAgent"),
    "Ingestor":                     (".llm", "Ingestor"),
    # processing
    "ComputeAgent":                 (".processing", "ComputeAgent"),
    "OnlineAgent":                  (".processing", "OnlineAgent"),
}


def __getattr__(name: str):
    """Lazy-load symbols on first access."""
    if name in _LAZY_IMPORTS:
        subpkg, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(subpkg, __name__)
        value = getattr(mod, attr)
        # Cache on the module so subsequent access is instant
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

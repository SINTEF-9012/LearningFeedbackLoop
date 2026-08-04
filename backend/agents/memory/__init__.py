"""
Memory module - LLM-augmented memory system for pattern detection and explanation.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This is the main memory system module.
# ===========================================================================

This module contains:
- orchestrator: MemoryEventOrchestrator - main entry point
- scorer: SignificanceScorer - evaluates if patterns are significant
- retriever: MemoryRetriever - context-aware memory retrieval
- feedback: MemoryFeedbackHandler - user feedback processing
- dispatcher: AlertDispatcher - WebSocket alert broadcasting
- router: FastAPI router for memory endpoints
- init: System initialization and wiring
- bridge: Feature bus to memory system connector
"""

# Import scorer FIRST to avoid circular imports with llm.explainer
from .scorer import (
    SignificanceScorer,
    SignificanceResult,
    SignificanceAction,
    SignificanceConfig,
)
from .retriever import (
    MemoryRetriever,
    MemoryMatch,
    RetrievalConfig,
)
from .feedback import (
    MemoryFeedbackHandler,
    MemoryFeedbackRequest,
    MemoryFeedbackResponse,
    FeedbackAction,
)
from .dispatcher import (
    AlertDispatcher,
    dispatch_significant_event,
    get_dispatcher,
)

# Import orchestrator AFTER scorer to prevent circular import
from .orchestrator import (
    MemoryEventOrchestrator,
    MemoryEvent,
    MemoryEventResult,
    get_orchestrator,
)

from .router import router as memory_router
from .init import (
    initialize_memory_system,
    get_memory_components,
    get_store,
    get_pattern_index,
    is_initialized,
    shutdown_memory_system,
)
from .feature_stream_bridge import (
    start_memory_processor,
    stop_memory_processor,
    create_memory_event_from_feature,
)

__all__ = [
    # Orchestrator
    "MemoryEventOrchestrator",
    "MemoryEvent", 
    "MemoryEventResult",
    "get_orchestrator",
    # Scorer
    "SignificanceScorer",
    "SignificanceResult",
    "SignificanceAction",
    "SignificanceConfig",
    # Retriever
    "MemoryRetriever",
    "MemoryMatch",
    "RetrievalConfig",
    # Feedback
    "MemoryFeedbackHandler",
    "MemoryFeedbackRequest",
    "MemoryFeedbackResponse",
    "FeedbackAction",
    # Dispatcher
    "AlertDispatcher",
    "dispatch_significant_event",
    "get_dispatcher",
    # Router
    "memory_router",
    # Init
    "initialize_memory_system",
    "get_memory_components",
    "get_store",
    "get_pattern_index",
    "is_initialized",
    "shutdown_memory_system",
    # Bridge
    "start_memory_processor",
    "stop_memory_processor",
    "create_memory_event_from_feature",
]

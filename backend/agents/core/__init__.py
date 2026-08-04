"""
Core module - Shared data models and utilities.

This module contains foundational types used across the agents package:
- schemas: Memory, PatternKey, TimeRange, etc.
- context: CuttingContext, OperatingRegime
- metrics: WindowMetrics
"""

from .schemas import (
    PatternType,
    PatternKey,
    NumericMetrics,
    TimeRange,
    MemoryProvenance,
    Memory,
    MemoryQueryResult,
)
from .batch_context import BatchContext, extract_batch_context

from .context import (
    OperatingRegime,
    CuttingContext,
    ContextTolerance,
    get_default_tolerance,
    extract_context_from_metadata,
)

from .metrics import (
    WindowMetrics,
    MetricsComputer,
)

__all__ = [
    # schemas
    "PatternType",
    "PatternKey",
    "NumericMetrics",
    "TimeRange",
    "MemoryProvenance",
    "Memory",
    "MemoryQueryResult",
    "BatchContext",
    "extract_batch_context",
    # context
    "OperatingRegime",
    "CuttingContext",
    "ContextTolerance",
    "get_default_tolerance",
    "extract_context_from_metadata",
    # metrics
    "WindowMetrics",
    "MetricsComputer",
]

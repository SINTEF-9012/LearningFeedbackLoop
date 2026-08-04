"""
Patterns module - Pattern detection and generation.

This module contains:
- generator: PatternGenerator for creating symbolic pattern keys from metrics
"""

from .generator import (
    PatternThresholds,
    PatternGenerator,
)

__all__ = [
    "PatternThresholds",
    "PatternGenerator",
]

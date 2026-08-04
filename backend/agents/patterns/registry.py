"""
Shared Pattern Registry — unified pattern definitions for both live pipeline
and experiment evaluation.

Wraps the canonical ``backend.agents.experiment.pattern_registry`` and
exposes it as a backend module so that:

  1. The API can serve available patterns (``GET /memory/patterns``).
  2. The experiment evaluator and real-time pipeline share the *same*
     pattern definitions and detectors.
  3. New patterns registered at runtime (domain-expert or ML-derived)
     are visible to both code paths.

Usage (backend)::

    from backend.agents.patterns.registry import get_registry, list_patterns_dict
    reg = get_registry()
    fired = reg.detect_all(features_dict, thresholds)

Usage (API)::

    GET /memory/patterns          → all pattern definitions
    GET /memory/patterns/detect   → run detectors on a feature vector
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Re-export the canonical types and singleton from the experiment module.
from ..experiment.pattern_registry import (
    PatternDefinition,
    PatternRegistry,
    get_registry,
    reset_registry,
)

logger = logging.getLogger(__name__)


def list_patterns_dict(
    enabled_only: bool = False,
    category: Optional[str] = None,
    polarity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all pattern definitions as JSON-serialisable dicts.

    Convenience wrapper used by the API endpoint.
    """
    reg = get_registry()
    patterns = reg.list_patterns(
        category=category,
        enabled_only=enabled_only,
        polarity=polarity,
    )
    return [
        {
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "severity": p.severity,
            "enabled": p.enabled,
            "source": p.source,
            "default_prior": p.default_prior,
            "discrimination_ratio": p.discrimination_ratio,
            "discrimination_score": p.discrimination_score,
            "fire_rate_normal": p.fire_rate_normal,
            "fire_rate_event": p.fire_rate_event,
            "polarity": p.polarity,
            "columns": p.columns,
        }
        for p in patterns
    ]


def detect_patterns(
    features: Dict[str, float],
    thresholds: Optional[Dict[str, Any]] = None,
    include_details: bool = False,
) -> Dict[str, Any]:
    """Run all enabled pattern detectors on a feature vector.

    Parameters
    ----------
    features : dict
        Flat dict of column→value from one sample (e.g., CNC sensor features).
    thresholds : dict, optional
        Calibrated threshold overrides (from training phase).
    include_details : bool
        If True, return full pattern metadata for each fired pattern.

    Returns
    -------
    dict with keys:
        - ``fired``: list of pattern names that triggered
        - ``count``: number of patterns triggered
        - ``details``: (optional) list of dicts with full pattern info
    """
    reg = get_registry()
    fired = reg.detect_all(features, thresholds)

    result: Dict[str, Any] = {
        "fired": fired,
        "count": len(fired),
    }

    if include_details:
        result["details"] = reg.detect_with_details(features, thresholds)

    return result

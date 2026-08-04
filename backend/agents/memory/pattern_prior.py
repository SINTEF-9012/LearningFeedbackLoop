"""Structured, auditable view of a learned pattern prior (deliverable §5.8).

A ``PatternPrior`` is a **read-only aggregation** produced by
``SignificanceScorer.get_pattern_prior_record``: it pairs the derived prior with
the confirm/dismiss feedback counts it was computed from and a volume-based
confidence, so priors can be inspected and exported *with the evidence behind
them*. It does not change how priors are computed, weighted, or persisted — the
scorer's float ``get_pattern_prior`` remains the source of truth used in scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PatternPrior:
    """One pattern's learned prior plus the feedback evidence behind it.

    Fields
    ------
    pattern_key:       normalized pattern key.
    prior_strength:    the derived prior in [0, 1] (== ``get_pattern_prior``).
    confidence:        volume-based support, ``n / (n + k)`` over the evidence
                       count ``n`` — 0.0 with no feedback, rising with volume.
    confirmed/dismissed: feedback counts at the resolved context level.
    evidence_count:    confirmed + dismissed.
    context_key:       the most-specific context level that had feedback
                       (``None`` = global / no context).
    evidence_memory_ids: optional provenance (memory ids); empty unless populated
                       from the feedback store.
    """

    pattern_key: str
    prior_strength: float
    confidence: float
    confirmed: float
    dismissed: float
    evidence_count: float
    context_key: Optional[str] = None
    evidence_memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "pattern_key": self.pattern_key,
            "context_key": self.context_key,
            "prior_strength": round(self.prior_strength, 4),
            "confidence": round(self.confidence, 4),
            "confirmed": self.confirmed,
            "dismissed": self.dismissed,
            "evidence_count": self.evidence_count,
            "evidence_memory_ids": list(self.evidence_memory_ids),
        }

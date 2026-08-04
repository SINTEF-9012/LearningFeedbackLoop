"""
Automatic Pattern Discovery
============================

Discovers new patterns from confirmed events whose feature signatures
don't match any pre-defined pattern well.  Discovered patterns are
``CLUSTER``-type and are characterised by **which features deviated**
from the running baseline and in **which direction**.

Design rationale
----------------
When an operator confirms an anomalous event, we know something real
happened — but the existing pattern library may not describe it well.
By analysing the feature z-scores of confirmed events we can identify
recurring *new* feature-deviation combinations and promote them to
first-class patterns.

Self-correction is built-in: every discovered pattern starts with a
neutral Bayesian prior (0.5).  If it turns out to be diagnostic noise,
operators will dismiss future events that match it, the prior drops,
and the pattern stops contributing to scoring.  If it's genuine, the
prior rises naturally through the existing feedback loop.

Lifecycle::

    Confirmed event
        → compute feature z-scores against running baseline
        → ≥ N_MIN features exceed DISCOVERY_Z (default 2.5)?
        → build a feature "signature" (set of deviating feature names + direction)
        → does this signature overlap ≥ MERGE_OVERLAP (80 %) with an existing
          discovered pattern?
            YES → merge (strengthen existing pattern)
            NO  → create new CLUSTER pattern
        → register with scorer (prior = 0.5)
        → persist to ``data/discovered_patterns.json``

Usage::

    from backend.agents.patterns.discovery import PatternDiscovery

    discovery = PatternDiscovery(priors_dir="data")
    # On confirmed event:
    new_patterns = discovery.analyse_confirmed_event(
        features={"power_spindle_mean": 85.3, ...},
        existing_pattern_keys=["BREAKAGE_POWER_SPIKE"],
        scorer=scorer,          # optional — auto-registers if given
    )
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.context import CuttingContext
from .registry import get_registry as get_pattern_registry

logger = logging.getLogger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────

#: Minimum z-score for a feature to count as "deviating"
DISCOVERY_Z: float = 2.5

#: Minimum number of co-deviating features to form a candidate pattern
N_MIN_FEATURES: int = 2

#: Maximum number of features in a single discovered pattern
N_MAX_FEATURES: int = 6

#: Minimum Jaccard overlap to merge into an existing discovered pattern
MERGE_OVERLAP: float = 0.80

#: How many confirmed events with the same signature before promoting
#: the candidate to a real pattern.  Raised from 2 to 4 (2026-04-14,
#: Issue #15) to reduce false promotions in noisy CNC environments.
MIN_CONFIRMATIONS: int = 4

#: Maximum number of discovered patterns to keep (prevent unbounded growth)
MAX_DISCOVERED: int = 50

#: Initial Bayesian prior for a newly discovered pattern
INITIAL_PRIOR: float = 0.50

#: Running-statistics EMA decay factor (same as scorer's recency_decay)
STATS_DECAY: float = 0.95


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class FeatureDeviation:
    """A single feature that deviated from baseline."""
    feature: str
    z_score: float
    direction: str  # "high" or "low"
    raw_value: float


@dataclass
class SourceEvent:
    """Provenance record: which confirmed event contributed to a discovery."""
    memory_id: Optional[str] = None
    session_id: Optional[str] = None
    context_key: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    deviations: Dict[str, str] = field(default_factory=dict)  # feature → direction
    z_scores: Dict[str, float] = field(default_factory=dict)  # feature → z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "context_key": self.context_key,
            "timestamp": self.timestamp,
            "deviations": self.deviations,
            "z_scores": {k: round(v, 3) for k, v in self.z_scores.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceEvent":
        return cls(
            memory_id=d.get("memory_id"),
            session_id=d.get("session_id"),
            context_key=d.get("context_key"),
            timestamp=d.get("timestamp", 0.0),
            deviations=d.get("deviations", {}),
            z_scores=d.get("z_scores", {}),
        )


@dataclass
class DiscoveredPattern:
    """A pattern discovered from confirmed-event feature signatures."""
    key: str                                    # e.g. "discovered:power_spindle_mean_H+vib_severity_x_mean_H"
    features: Dict[str, str]                    # feature_name → direction ("high" / "low")
    context_key: Optional[str] = None           # machine/tool/material/regime scope
    confirmation_count: int = 0                 # how many times seen
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    promoted: bool = False                      # registered with scorer?
    prior: float = INITIAL_PRIOR
    source_events: List[SourceEvent] = field(default_factory=list)  # provenance chain

    @property
    def signature_set(self) -> Set[str]:
        """Canonical set for Jaccard comparison: ``{feat:dir, ...}``."""
        return {f"{f}:{d}" for f, d in self.features.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "features": self.features,
            "context_key": self.context_key,
            "confirmation_count": self.confirmation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "promoted": self.promoted,
            "prior": self.prior,
            "source_events": [se.to_dict() for se in self.source_events],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveredPattern":
        return cls(
            key=d["key"],
            features=d["features"],
            context_key=d.get("context_key"),
            confirmation_count=d.get("confirmation_count", 0),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            promoted=d.get("promoted", False),
            prior=d.get("prior", INITIAL_PRIOR),
            source_events=[SourceEvent.from_dict(se) for se in d.get("source_events", [])],
        )


# ── Running feature statistics ─────────────────────────────────────────────

class _RunningStats:
    """Per-feature EMA mean and variance tracker."""

    def __init__(self, decay: float = STATS_DECAY):
        self.decay = decay
        self.means: Dict[str, float] = {}
        self.variances: Dict[str, float] = {}
        self.n: int = 0

    def update(self, features: Dict[str, float]) -> None:
        self.n += 1
        for feat, val in features.items():
            if not math.isfinite(val):
                continue
            if feat not in self.means:
                self.means[feat] = val
                self.variances[feat] = 0.0
            else:
                old_mean = self.means[feat]
                self.means[feat] = self.decay * old_mean + (1 - self.decay) * val
                diff = val - old_mean
                old_var = self.variances[feat]
                self.variances[feat] = self.decay * old_var + (1 - self.decay) * diff * diff

    def z_score(self, feat: str, val: float) -> float:
        """Return z-score of ``val`` relative to running stats for ``feat``."""
        if feat not in self.means or self.n < 10:
            return 0.0
        std = math.sqrt(max(self.variances.get(feat, 0.0), 1e-12))
        return (val - self.means[feat]) / std

    def to_dict(self) -> Dict[str, Any]:
        return {"means": dict(self.means), "variances": dict(self.variances), "n": self.n}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.means = d.get("means", {})
        self.variances = d.get("variances", {})
        self.n = d.get("n", 0)


# ── Main discovery engine ──────────────────────────────────────────────────

class PatternDiscovery:
    """Discovers new patterns from confirmed events.

    Parameters
    ----------
    data_dir : str or Path
        Directory where ``discovered_patterns.json`` is persisted.
    discovery_z : float
        Z-score threshold for a feature to count as deviating.
    n_min_features : int
        Minimum number of co-deviating features to form a candidate.
    min_confirmations : int
        Number of times a candidate must be seen before being promoted.
    max_discovered : int
        Hard cap on total discovered patterns.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        discovery_z: float = DISCOVERY_Z,
        n_min_features: int = N_MIN_FEATURES,
        min_confirmations: int = MIN_CONFIRMATIONS,
        max_discovered: int = MAX_DISCOVERED,
    ):
        self.data_dir = Path(data_dir)
        self.discovery_z = discovery_z
        self.n_min_features = n_min_features
        self.min_confirmations = min_confirmations
        self.max_discovered = max_discovered

        self._stats = _RunningStats()
        self._context_stats: Dict[str, _RunningStats] = {}
        self._patterns: Dict[str, DiscoveredPattern] = {}

        # Optional callback invoked when a pattern is promoted or updated.
        # Signature: callback(pattern: DiscoveredPattern) -> None
        self.on_pattern_event: Optional[Any] = None

        self._load()

    # ── Public API ─────────────────────────────────────────────────────

    def update_baseline(
        self,
        features: Dict[str, float],
        cutting_context: Optional[CuttingContext] = None,
    ) -> None:
        """Feed a feature vector into the running baseline.

        Call this for **every** event (not just confirmed ones) so the
        baseline tracks normal operating conditions.
        """
        self._stats.update(features)
        stats = self._baseline_stats(cutting_context, create=True)
        if stats is not None and stats is not self._stats:
            stats.update(features)

    def analyse_confirmed_event(
        self,
        features: Dict[str, float],
        existing_pattern_keys: Optional[List[str]] = None,
        scorer: Any = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        cutting_context: Optional[CuttingContext] = None,
    ) -> List[DiscoveredPattern]:
        """Analyse a confirmed event for potential new patterns.

        Parameters
        ----------
        features
            Feature dict (the 28 FEATURE_NAMES or a subset).
        existing_pattern_keys
            Pattern keys already matched on this event.  If a strong
            existing match is present, discovery may be skipped.
        scorer
            Optional ``SignificanceScorer`` — if provided, promoted
            patterns are registered automatically.
        memory_id
            ID of the confirmed memory (provenance tracking).
        session_id
            Session that produced the confirmed event (provenance).

        Returns
        -------
        list[DiscoveredPattern]
            Newly created or strengthened patterns.
        """
        stats = self._baseline_stats(cutting_context)
        if stats is None or stats.n < 20:
            # Not enough baseline data yet
            return []

        # Curated patterns should remain the primary explanation path.
        # Discovery is reserved for confirmed events whose evidence is not
        # already well-covered by strong built-in or domain-expert patterns.
        if self._has_strong_curated_match(existing_pattern_keys):
            return []

        # 1. Compute z-scores
        deviations = self._compute_deviations(features, stats=stats)
        if len(deviations) < self.n_min_features:
            return []

        # 2. Build signature from top deviations (capped at N_MAX_FEATURES)
        top = sorted(deviations, key=lambda d: abs(d.z_score), reverse=True)[:N_MAX_FEATURES]
        signature: Dict[str, str] = {d.feature: d.direction for d in top}
        sig_set = {f"{f}:{d}" for f, d in signature.items()}
        context_key = self._context_key(cutting_context)

        # 2b. Build provenance record for this confirmed event
        source_event = SourceEvent(
            memory_id=memory_id,
            session_id=session_id,
            context_key=context_key,
            timestamp=time.time(),
            deviations=dict(signature),
            z_scores={d.feature: d.z_score for d in top},
        )

        # 3. Check overlap with existing discovered patterns → merge or new
        result: List[DiscoveredPattern] = []
        merged = False
        for pat in self._patterns.values():
            if pat.context_key != context_key:
                continue
            jaccard = self._jaccard(sig_set, pat.signature_set)
            if jaccard >= MERGE_OVERLAP:
                # Merge: strengthen existing candidate
                pat.confirmation_count += 1
                pat.last_seen = time.time()
                pat.source_events.append(source_event)
                # Merge any new features into the signature
                for f, d in signature.items():
                    if f not in pat.features:
                        pat.features[f] = d
                merged = True

                # Promote if enough confirmations and not yet promoted
                if pat.confirmation_count >= self.min_confirmations and not pat.promoted:
                    self._promote(pat, scorer)

                result.append(pat)
                break

        if not merged:
            # Create new candidate
            if len(self._patterns) >= self.max_discovered:
                self._evict_weakest()

            key = self._build_key(signature, context_key=context_key, prefix="discovered")
            pat = DiscoveredPattern(
                key=key,
                features=dict(signature),
                context_key=context_key,
                confirmation_count=1,
                first_seen=time.time(),
                last_seen=time.time(),
                source_events=[source_event],
            )

            # Immediate promotion if min_confirmations <= 1
            if self.min_confirmations <= 1:
                self._promote(pat, scorer)

            self._patterns[key] = pat
            result.append(pat)
            logger.info(
                "Pattern discovery: new candidate '%s' from %d deviating features",
                key, len(signature),
            )

        self._save()
        return result

    def analyse_dismissed_event(
        self,
        features: Dict[str, float],
        existing_pattern_keys: Optional[List[str]] = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        cutting_context: Optional[CuttingContext] = None,
    ) -> List[DiscoveredPattern]:
        """Learn suppression patterns from dismissed events.

        When an operator **dismisses** an event, features that deviated
        but weren't actually significant form a "suppression signature".
        If the same false-alarm signature recurs, the system can proactively
        lower its score (negative prior).

        The suppression patterns use the same clustering as positive
        discovery, but keys are prefixed ``suppressed:`` and their priors
        start at 0.3 (below neutral 0.5) so the scorer naturally penalises
        them.
        """
        stats = self._baseline_stats(cutting_context)
        if stats is None or stats.n < 20:
            return []

        deviations = self._compute_deviations(features, stats=stats)
        if len(deviations) < self.n_min_features:
            return []

        top = sorted(deviations, key=lambda d: abs(d.z_score), reverse=True)[:N_MAX_FEATURES]
        signature: Dict[str, str] = {d.feature: d.direction for d in top}
        sig_set = {f"{f}:{d}" for f, d in signature.items()}
        context_key = self._context_key(cutting_context)

        source_event = SourceEvent(
            memory_id=memory_id,
            session_id=session_id,
            context_key=context_key,
            timestamp=time.time(),
            deviations=dict(signature),
            z_scores={d.feature: d.z_score for d in top},
        )

        result: List[DiscoveredPattern] = []
        merged = False
        for pat in self._patterns.values():
            if not pat.key.startswith("suppressed:"):
                continue
            if pat.context_key != context_key:
                continue
            jaccard = self._jaccard(sig_set, pat.signature_set)
            if jaccard >= MERGE_OVERLAP:
                pat.confirmation_count += 1
                pat.last_seen = time.time()
                pat.source_events.append(source_event)
                for f, d in signature.items():
                    if f not in pat.features:
                        pat.features[f] = d
                merged = True
                if pat.confirmation_count >= self.min_confirmations and not pat.promoted:
                    pat.promoted = True
                    pat.prior = 0.30  # below neutral → suppressive
                    logger.info(
                        "Suppression pattern PROMOTED: '%s' after %d dismissals",
                        pat.key, pat.confirmation_count,
                    )
                    if self.on_pattern_event:
                        try:
                            self.on_pattern_event(pat)
                        except Exception:
                            pass
                result.append(pat)
                break

        if not merged:
            if len(self._patterns) >= self.max_discovered:
                self._evict_weakest()
            key = self._build_key(signature, context_key=context_key, prefix="suppressed")
            pat = DiscoveredPattern(
                key=key,
                features=dict(signature),
                context_key=context_key,
                confirmation_count=1,
                first_seen=time.time(),
                last_seen=time.time(),
                prior=0.30,
                source_events=[source_event],
            )
            if self.min_confirmations <= 1:
                pat.promoted = True
            self._patterns[key] = pat
            result.append(pat)
            logger.info(
                "Suppression candidate: '%s' from %d deviating features",
                key, len(signature),
            )

        self._save()
        return result

    def match_event(
        self,
        features: Dict[str, float],
        cutting_context: Optional[CuttingContext] = None,
    ) -> List[str]:
        """Return keys of discovered patterns that match the given features.

        A discovered pattern matches if **all** its constituent features
        deviate in the expected direction (z-score ≥ half the discovery
        threshold).
        """
        stats = self._baseline_stats(cutting_context)
        if stats is None or stats.n < 20:
            return []

        matched: List[str] = []
        match_z = self.discovery_z * 0.5  # relaxed threshold for matching
        context_key = self._context_key(cutting_context)

        for pat in self._patterns.values():
            if not pat.promoted:
                continue
            if pat.context_key != context_key:
                continue
            all_match = True
            for feat, expected_dir in pat.features.items():
                val = features.get(feat)
                if val is None:
                    all_match = False
                    break
                z = stats.z_score(feat, val)
                if expected_dir == "high" and z < match_z:
                    all_match = False
                    break
                if expected_dir == "low" and z > -match_z:
                    all_match = False
                    break
            if all_match:
                matched.append(pat.key)

        return matched

    def get_patterns(self) -> Dict[str, DiscoveredPattern]:
        """Return all discovered patterns (candidates + promoted)."""
        return dict(self._patterns)

    def get_promoted_patterns(self) -> Dict[str, DiscoveredPattern]:
        """Return only promoted (active) patterns."""
        return {k: v for k, v in self._patterns.items() if v.promoted}

    # ── Internal helpers ───────────────────────────────────────────────

    def _compute_deviations(
        self,
        features: Dict[str, float],
        *,
        stats: Optional[_RunningStats] = None,
    ) -> List[FeatureDeviation]:
        """Identify features that deviate significantly from baseline."""
        active_stats = stats or self._stats
        devs: List[FeatureDeviation] = []
        for feat, val in features.items():
            if not math.isfinite(val):
                continue
            z = active_stats.z_score(feat, val)
            if abs(z) >= self.discovery_z:
                devs.append(FeatureDeviation(
                    feature=feat,
                    z_score=z,
                    direction="high" if z > 0 else "low",
                    raw_value=val,
                ))
        return devs

    def _baseline_stats(
        self,
        cutting_context: Optional[CuttingContext],
        *,
        create: bool = False,
    ) -> Optional[_RunningStats]:
        context_key = self._context_key(cutting_context)
        if context_key is None:
            return self._stats
        stats = self._context_stats.get(context_key)
        if stats is None and create:
            stats = _RunningStats(decay=self._stats.decay)
            self._context_stats[context_key] = stats
        return stats

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _build_key(
        signature: Dict[str, str],
        *,
        context_key: Optional[str] = None,
        prefix: str = "discovered",
    ) -> str:
        """Build a deterministic key from the feature signature."""
        parts = sorted(
            f"{feat}_{'H' if direction == 'high' else 'L'}"
            for feat, direction in signature.items()
        )
        signature_key = "+".join(parts)
        if context_key:
            return f"{prefix}:{context_key}::{signature_key}"
        return f"{prefix}:{signature_key}"

    @staticmethod
    def _context_key(cutting_context: Optional[CuttingContext]) -> Optional[str]:
        if cutting_context is None:
            return None
        dims: List[tuple[str, Optional[str]]] = [
            ("machine_type", getattr(cutting_context, "machine_type", None)),
            ("tool_type", getattr(cutting_context, "tool_type", None)),
            ("workpiece_material", getattr(cutting_context, "workpiece_material", None)),
            (
                "operating_regime",
                getattr(cutting_context, "operating_regime", None).value
                if getattr(cutting_context, "operating_regime", None)
                else None,
            ),
        ]
        parts = [f"{key}={str(value).strip()}" for key, value in dims if value is not None and str(value).strip()]
        if not parts:
            return None
        return "|".join(parts)

    @staticmethod
    def _has_strong_curated_match(existing_pattern_keys: Optional[List[str]]) -> bool:
        """Return True when the event already has a strong curated explanation.

        This keeps discovery focused on confirmed events that are genuinely novel,
        instead of creating a new discovered signature for every event that already
        matches built-in or domain-expert patterns.
        """
        if not existing_pattern_keys:
            return False

        registry = get_pattern_registry()
        for key in existing_pattern_keys:
            if not key:
                continue
            if key.startswith(("discovered:", "suppressed:")):
                continue
            if key.startswith(("fault:", "hypothesis:")):
                return True

            pdef = registry.get(str(key).strip())
            if pdef is None:
                continue
            if pdef.source in {"builtin", "domain_expert"}:
                return True
            if pdef.category in {"fault", "domain"} and float(pdef.severity or 0.0) >= 0.7:
                return True

        return False

    def _promote(self, pat: DiscoveredPattern, scorer: Any = None) -> None:
        """Register a candidate as an active pattern."""
        pat.promoted = True
        pat.prior = INITIAL_PRIOR
        logger.info(
            "Pattern discovery: PROMOTED '%s' after %d confirmations "
            "(features: %s)",
            pat.key, pat.confirmation_count,
            ", ".join(f"{f}={d}" for f, d in pat.features.items()),
        )

        # Register prior with the scorer so it participates in scoring
        if scorer and hasattr(scorer, "_pattern_priors"):
            scorer._pattern_priors[pat.key] = INITIAL_PRIOR
            if hasattr(scorer, "_local_feedback_counts"):
                scorer._local_feedback_counts.setdefault(
                    pat.key, {"confirm": 0, "dismiss": 0}
                )
            if hasattr(scorer, "_save_priors"):
                scorer._save_priors()

        # Fire external callback (e.g. Neo4j persistence)
        if self.on_pattern_event:
            try:
                self.on_pattern_event(pat)
            except Exception as e:
                logger.debug("on_pattern_event callback failed: %s", e)

    def _evict_weakest(self) -> None:
        """Remove the discovered pattern with lowest confirmation count + highest age."""
        if not self._patterns:
            return
        weakest = min(
            self._patterns.values(),
            key=lambda p: (p.confirmation_count, -p.first_seen),
        )
        logger.info("Pattern discovery: evicting weakest pattern '%s'", weakest.key)
        del self._patterns[weakest.key]

    # ── Persistence ────────────────────────────────────────────────────

    def _path(self) -> Path:
        return self.data_dir / "discovered_patterns.json"

    def _save(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "patterns": {k: v.to_dict() for k, v in self._patterns.items()},
                "baseline_stats": self._stats.to_dict(),
                "baseline_stats_by_context": {
                    key: stats.to_dict()
                    for key, stats in self._context_stats.items()
                },
                "version": 2,
            }
            with open(self._path(), "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save discovered patterns: %s", e)

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            with open(p) as f:
                data = json.load(f)
            for k, v in data.get("patterns", {}).items():
                self._patterns[k] = DiscoveredPattern.from_dict(v)
            stats = data.get("baseline_stats")
            if stats:
                self._stats.load_dict(stats)
            raw_context_stats = data.get("baseline_stats_by_context") or {}
            if isinstance(raw_context_stats, dict):
                for context_key, payload in raw_context_stats.items():
                    if not isinstance(payload, dict):
                        continue
                    stats_for_context = _RunningStats(decay=self._stats.decay)
                    stats_for_context.load_dict(payload)
                    self._context_stats[str(context_key)] = stats_for_context
            logger.info(
                "Loaded %d discovered patterns (%d promoted)",
                len(self._patterns),
                sum(1 for p in self._patterns.values() if p.promoted),
            )
        except Exception as e:
            logger.warning("Failed to load discovered patterns: %s", e)

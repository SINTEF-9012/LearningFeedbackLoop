"""
Significance Scorer - Determine if detected patterns warrant storage and alerting.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module is a draft implementation for significance detection.
# Scoring rules are placeholder heuristics - expected to evolve based on
# operator feedback and domain expertise.
# ===========================================================================

Evaluates incoming pattern/metric combinations against multiple criteria:
1. Classical model alerts (external signals)
2. Pattern-based rules (symbolic matching)
3. Statistical anomaly detection (deviation from baseline)
4. Historical knowledge (learned priors from feedback)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from ..core.schemas import PatternKey, PatternType, NumericMetrics
from ..core.context import CuttingContext
from ..core.metrics import WindowMetrics
from ..patterns.registry import get_registry as get_pattern_registry
from ..patterns.signatures import normalize_signature_key
from .prior_math import DEFAULT_RECENCY_DECAY, effective_feedback_count, prior_from_counts

logger = logging.getLogger(__name__)

FEEDBACK_WEIGHTS: Dict[str, float] = {
    "passive_cycle_completed_without_intervention": 0.25,
    "dismiss_oneclick": 0.50,
    "dismiss_with_reason": 0.75,
    "severity_correction": 1.00,
    "confirm_explicit": 1.00,
    "outcome_explicit": 1.25,
}

_SEVERITY_TARGET_SCORES: Dict[str, float] = {
    "info": 0.45,
    "warning": 0.72,
    "critical": 0.92,
}

# ---------------------------------------------------------------------------
# Pattern key normalization (P11 fix)
# Maps legacy UPPER_CASE keys to generator-format keys so that historical
# priors actually match incoming patterns.
#
# Agent D (2026-04-24): aliases moved to data/pattern_aliases.json. Built-in
# defaults below remain as a safe fallback when the data file is missing or
# malformed; the data file wins when both define the same key.
# ---------------------------------------------------------------------------
_PATTERN_KEY_ALIASES_BUILTIN: Dict[str, str] = {
    # Old priors file keys → generator-format keys
    "FAULT_CHATTER": "fault:chatter",
    "CHATTER_DETECTED": "fault:chatter",
    "TOOL_WEAR_RISK": "fault:tool_breakage",
    "ANOMALY_HIGH_VIBRATION": "spectral:hf_burst",
    "VIB_SEVERITY_HIGH": "spectral:hf_burst",
    "ANOMALY_HIGH_POWER": "SPINDLE_POWER_SURGE",
    "POWER_SPIKE_SUSTAINED": "SPINDLE_POWER_SURGE",
    # Legacy experiment pattern names → renamed observables
    "BREAKAGE_POWER_SPIKE": "SPINDLE_POWER_SURGE",
    "BREAKAGE_VIB_SHIFT": "VIBRATION_REGIME_SHIFT",
    "BREAKAGE_FEED_OVERRIDE_DROP": "FEED_OVERRIDE_DROP",
    "BREAKAGE_DECORRELATION": "SENSOR_DECORRELATION",
    # Stat-burst keys emitted by the stoppage-experiment registry → canonical
    # (prior-learnable, pattern-rule-whitelisted) keys. Mirrored in
    # data/pattern_aliases.json; kept here too because data/ is gitignored and
    # a fresh checkout must score identically without the file.
    "HF_ENERGY_BURST": "spectral:hf_burst",
    "IMPULSE_BURST": "temporal:impulsive_burst",
}


def _load_pattern_aliases() -> Dict[str, str]:
    """Merge built-in aliases with data/pattern_aliases.json (file wins)."""
    merged = dict(_PATTERN_KEY_ALIASES_BUILTIN)
    # data/ lives at repo root — two parents up from this file's package dir.
    candidate = Path(__file__).resolve().parents[3] / "data" / "pattern_aliases.json"
    try:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            file_aliases = raw.get("aliases") if isinstance(raw, dict) else None
            if isinstance(file_aliases, dict):
                for k, v in file_aliases.items():
                    if isinstance(k, str) and isinstance(v, str):
                        merged[k] = v
    except Exception:
        logger.warning("Failed to load data/pattern_aliases.json; using built-in defaults", exc_info=True)
    return merged


_PATTERN_KEY_ALIASES: Dict[str, str] = _load_pattern_aliases()


def normalize_pattern_key(key: str) -> str:
    """Normalize a pattern key to the canonical generator-format name.

    Handles both old UPPER_CASE convention and new colon-delimited convention.
    Returns the original key unchanged if no alias is found.
    """
    canonical = _PATTERN_KEY_ALIASES.get(key, key)
    return normalize_signature_key(canonical)


def is_signature_pattern_key(key: str) -> bool:
    canonical = normalize_pattern_key(str(key).strip())
    return canonical.startswith("signature:")


def _parse_ratio_bucket_lower_bound(key: str) -> Optional[float]:
    """Lower bound of a bucketed ratio pattern key, e.g. ``RATIO_Fx_Fy:>5`` -> 5.0.

    Supports the four bucket forms emitted by the casedata loader:
      ``:>5`` -> 5.0, ``:2-5`` -> 2.0, ``:0.5-2`` -> 0.5, ``:<0.5`` -> 0.0.
    Returns None when the key carries no parseable bucket spec.
    """
    if ":" not in str(key):
        return None
    spec = str(key).rsplit(":", 1)[1].strip()
    try:
        if spec.startswith(">"):
            return float(spec[1:])
        if spec.startswith("<"):
            return 0.0  # an upper-bounded bucket has a lower bound of zero
        if "-" in spec:
            return float(spec.split("-", 1)[0])
        return float(spec)
    except (ValueError, IndexError):
        return None


def is_hypothesis_pattern_key(key: str) -> bool:
    """Return True for high-level fault / hypothesis pattern keys.

    These keys represent semantic fault hypotheses that later normalize into
    signature-oriented keys for explanation and retrieval.
    """
    raw = str(key or "").strip().lower()
    return (
        raw.startswith("fault:")
        or raw.startswith("hypothesis:")
        or is_signature_pattern_key(key)
    )


# [PROTOTYPE_LLM_MEMORY_V1] - Action recommendations
class SignificanceAction(str, Enum):
    """Recommended action based on significance score."""
    IGNORE = "ignore"  # Not significant, don't store
    STORE = "store"  # Store memory but don't alert
    ALERT = "alert"  # Store and push alert to clients
    CRITICAL = "critical"  # Store with high priority, immediate alert


# [PROTOTYPE_LLM_MEMORY_V1] - Result container
@dataclass
class SignificanceResult:
    """Result of significance evaluation."""
    is_significant: bool
    score: float  # 0.0 - 1.0 composite score
    action: SignificanceAction
    reasons: List[str]  # Human-readable explanations
    triggered_rules: List[str]  # Which rules fired
    pattern_priors: Dict[str, float] = field(default_factory=dict)  # Pattern -> prior significance
    prior_boost: float = 0.0  # Additive boost in additive mode; raw prior in multiplicative mode
    pattern_rule_score: float = 0.0  # Weighted contribution from the pattern_match rule (post-weight)
    # In multiplicative prior mode the final score is base * prior_factor;
    # left as None in additive mode so consumers can detect which math applied.
    prior_factor: Optional[float] = None
    prior_mode: str = "multiplicative"
    historical_prior: float = 0.5
    prior_evidence_count: int = 0
    prior_damping_factor: float = 1.0
    # Agent D (2026-04-24): ordered per-component contribution trace for
    # explainability — each entry is a dict with keys:
    #   component (str), value (float), source (str)
    # Consumers (UI waterfall, diagnostics) must tolerate missing/new keys.
    score_trace: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_significant": self.is_significant,
            "score": self.score,
            "action": self.action.value,
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules,
            "pattern_priors": self.pattern_priors,
            "prior_boost": self.prior_boost,
            "pattern_rule_score": self.pattern_rule_score,
            "prior_factor": self.prior_factor,
            "prior_mode": self.prior_mode,
            "historical_prior": self.historical_prior,
            "prior_evidence_count": self.prior_evidence_count,
            "prior_damping_factor": self.prior_damping_factor,
            "score_trace": self.score_trace,
            "timestamp": self.timestamp.isoformat(),
        }


# [PROTOTYPE_LLM_MEMORY_V1] - Configuration
@dataclass
class SignificanceConfig:
    """
    Configuration for significance scoring.
    
    [INTEGRATION_POINT] These thresholds should be tunable via API
    or configuration file in production.
    """
    # Score thresholds for actions
    store_threshold: float = 0.3  # Score above this -> STORE
    alert_threshold: float = 0.6  # Score above this -> ALERT
    critical_threshold: float = 0.85  # Score above this -> CRITICAL
    severity_correction_max_delta: float = 0.20  # Max score shift from severity calibration
    
    # Rule weights (how much each rule contributes to final score)
    weight_classical_alert: float = 0.4  # External model says significant
    weight_harmonic_alert: float = 0.1  # Harmonic scorer says significant
    weight_pattern_rule: float = 0.25  # Pattern-based rules match
    weight_protective_pattern: float = 0.2  # Normality-supporting patterns suppress score
    weight_anomaly_deviation: float = 0.2  # Statistical anomaly
    weight_historical_prior: float = 0.30  # Learned from feedback (increased for faster convergence)
    weight_aad: float = 0.5  # AAD combiner rule (feedback-trained model over detectors)
    
    # Anomaly detection
    baseline_window_size: int = 100  # Samples for rolling baseline
    anomaly_z_threshold: float = 4.0  # Standard deviations for anomaly (was 3.0)
    # Require ≥2 detector rules for an ALERT; a lone rule below the critical band
    # caps at STORE. Opt-in (default off) — also suppresses legitimate single-rule
    # alerts. Enable via SIG_REQUIRE_MULTI_RULE_ALERT=1.
    require_multi_rule_alert: bool = False

    # Pattern-specific thresholds
    chatter_ratio_threshold: float = 5.0  # Fx/Fy ratio indicating chatter
    anomaly_score_threshold: float = 0.7  # OnlineAgent anomaly score
    
    # Classical model blending
    # Controls how much the classical anomaly model contributes relative
    # to pattern/prior rules.  final = (1 - w) * pattern_prior_score + w * model_score.
    # When model_confidence is low the model contribution is further scaled down.
    anomaly_model_blend_weight: float = 0.3

    # --- Adaptive scoring (Improvements 1-7) ---

    # RL agent integration (Improvement 1)
    enable_rl_weight_tuning: bool = True  # Apply RL agent weight adjustments
    rl_safe_mode_threshold: float = 0.60  # Only explore on events below ALERT level

    # Multiplicative prior mode (Improvement 2)
    prior_mode: str = "multiplicative"  # "additive" (legacy) or "multiplicative"
    prior_multiplier_range: float = 0.8  # prior=0→0.6x, prior=0.5→1.0x, prior=1.0→1.4x
    prior_evidence_damping_k: float = 20.0  # pull low-support priors toward neutral

    # Sub-neutral prior damping (ISS-13). Historically only above-neutral priors
    # reached the multiplicative factor, so operator dismissals could never damp
    # a score. When enabled, a sub-neutral prior applies IF (a) no pattern on
    # the event carries an above-neutral (boosting) prior — confirmations always
    # win over dismissals — and (b) the dismissal evidence volume meets the
    # minimum. The factor is floored, and events whose base evidence already
    # reaches the critical band are immune to damping entirely.
    prior_allow_subneutral: bool = False
    prior_factor_floor: float = 0.85  # lowest multiplicative factor damping may apply
    prior_subneutral_min_evidence: float = 3.0  # feedback volume required to damp

    # Co-occurrence gating (plan 1.7). A single-feature "supporting" pattern
    # (requires_corroboration=True in the registry) that fires alone contributes
    # only this much to the pattern rule — STORE-band, below the alert threshold —
    # so a lone weak spike does not raise an operator alert by itself. It escalates
    # to full severity when corroborated: a 2nd distinct pattern co-fires, or the
    # classical/anomaly rule agrees (that agreement adds its own weighted score).
    supporting_uncorroborated_cap: float = 0.5
    enable_cooccurrence_gating: bool = True

    # D1: partial-pooling of context-scoped priors (empirical-Bayes shrinkage).
    # OFF by default -> first-match behaviour is preserved exactly. When ON, a
    # context prior is shrunk toward its parent level:
    #   p_ctx = (n_ctx * p_local + kappa * p_parent) / (n_ctx + kappa)
    # so a sparsely-observed (tool/material) context borrows strength from the
    # broader (machine) context instead of fragmenting evidence. As kappa -> 0
    # the estimate is fully local; as evidence n_ctx grows it becomes local
    # regardless of kappa. See EXPERIMENT_IMPROVEMENT_PLAN_2026-06-15 (D1).
    context_prior_pooling: bool = False
    context_prior_pooling_kappa: float = 8.0  # parent-pull strength (pseudo-counts)

    # Pluggable model/rule selection. None -> the default rule set (unchanged).
    # Set to a list of registered rule names to run a chosen SET of models, or
    # to add a newly-registered model rule (e.g. ["classical_alert",
    # "pattern_match", "historical_prior", "aad_combiner"]).
    enabled_rules: Optional[List[str]] = None

    # Rule agreement scoring (Improvement 3)
    enable_rule_agreement: bool = True
    agreement_bonus_model_pattern: float = 0.08
    agreement_bonus_model_zscore: float = 0.06
    agreement_bonus_pattern_zscore: float = 0.05
    agreement_bonus_cap: float = 0.25

    # Online precision/recall tracking (Improvement 4)
    enable_rule_performance_tracking: bool = True
    rule_performance_window: int = 100  # sliding window of feedback events
    weight_adaptation_rate: float = 0.15  # how fast weights adapt to performance

    # Context-conditioned weight profiles (Improvement 5)
    enable_context_profiles: bool = True
    profile_learning_rate: float = 0.05  # nudge rate per feedback event
    profiles_path: Optional[str] = None  # persistence path (auto-set from priors_path)

    # When True, load priors from a file marked ``bootstrap_seeded`` (e.g. a fleet/repo
    # seed). Default False so a fresh site starts cold rather than silently inheriting
    # seeded priors; real local feedback always loads regardless of this flag.
    bootstrap_pattern_priors: bool = False

    # Adaptive action thresholds (Improvement 6)
    enable_adaptive_thresholds: bool = True
    target_alert_precision: float = 0.70  # desired precision for ALERT actions
    threshold_history_window: int = 100  # sliding window for tracking
    threshold_step: float = 0.02  # how much to nudge thresholds

    # Model confidence age decay (Improvement 7)
    enable_model_age_decay: bool = True
    model_age_decay_rate: float = 0.01  # weight loss per hour since last retrain
    model_age_decay_floor: float = 0.50  # minimum fraction of classical weight


# ---------------------------------------------------------------------------
# Adaptive scoring data structures (Improvements 1-7)
# ---------------------------------------------------------------------------

@dataclass
class WeightProfile:
    """Per-context weight profile learned from feedback.

    Persisted alongside pattern priors so that different operating regimes,
    tool types, and materials can have independently tuned weights.
    """
    classical: float = 0.40
    harmonic: float = 0.10
    pattern: float = 0.25
    anomaly: float = 0.20
    historical: float = 0.30
    n_events: int = 0
    n_feedbacks: int = 0
    # Optional SINDIT metadata (filled when context comes from digital twin)
    sindit_asset_iri: Optional[str] = None
    sindit_machine_state: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classical": round(self.classical, 4),
            "harmonic": round(self.harmonic, 4),
            "pattern": round(self.pattern, 4),
            "anomaly": round(self.anomaly, 4),
            "historical": round(self.historical, 4),
            "n_events": self.n_events,
            "n_feedbacks": self.n_feedbacks,
            "sindit_asset_iri": self.sindit_asset_iri,
            "sindit_machine_state": self.sindit_machine_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeightProfile":
        return cls(
            classical=float(d.get("classical", 0.40)),
            harmonic=float(d.get("harmonic", 0.10)),
            pattern=float(d.get("pattern", 0.25)),
            anomaly=float(d.get("anomaly", 0.20)),
            historical=float(d.get("historical", 0.30)),
            n_events=int(d.get("n_events", 0)),
            n_feedbacks=int(d.get("n_feedbacks", 0)),
            sindit_asset_iri=d.get("sindit_asset_iri"),
            sindit_machine_state=d.get("sindit_machine_state"),
        )


@dataclass
class RulePerformance:
    """Sliding-window precision/recall tracker for a single rule.

    Each entry in ``_history`` is ``(rule_fired: bool, was_confirmed: bool)``.
    """
    _history: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, rule_fired: bool, was_confirmed: bool) -> None:
        self._history.append((rule_fired, was_confirmed))

    @property
    def true_positives(self) -> int:
        return sum(1 for fired, confirmed in self._history if fired and confirmed)

    @property
    def false_positives(self) -> int:
        return sum(1 for fired, confirmed in self._history if fired and not confirmed)

    @property
    def false_negatives(self) -> int:
        return sum(1 for fired, confirmed in self._history if not fired and confirmed)

    @property
    def true_negatives(self) -> int:
        return sum(1 for fired, confirmed in self._history if not fired and not confirmed)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.5

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.5

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def n_samples(self) -> int:
        return len(self._history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "n_samples": self.n_samples,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


class AdaptiveThresholds:
    """Self-tuning action thresholds based on operator feedback.

    Tracks the precision of ALERT-level events and nudges the threshold
    up (if too many false alarms) or down (if precision is much higher
    than target, indicating we can catch more).
    """

    def __init__(
        self,
        base_alert: float = 0.60,
        base_store: float = 0.30,
        base_critical: float = 0.85,
        target_precision: float = 0.70,
        window_size: int = 100,
        step: float = 0.02,
    ):
        self.base_alert = base_alert
        self.base_store = base_store
        self.base_critical = base_critical
        self.target_precision = target_precision
        self.step = step
        self._alert_history: deque = deque(maxlen=window_size)
        self._current_alert: float = base_alert
        self._current_store: float = base_store
        self._current_critical: float = base_critical

    def record_feedback(
        self,
        score: float,
        action: str,
        confirmed: bool,
        weight: float = 1.0,
    ) -> None:
        """Record feedback outcome for an alert/store event."""
        try:
            effective_weight = max(0.0, float(weight))
        except (TypeError, ValueError):
            effective_weight = 1.0
        if effective_weight <= 0.0:
            return
        if action in ("alert", "critical"):
            self._alert_history.append((score, confirmed, effective_weight))
            self._recalculate()

    def _recalculate(self) -> None:
        if len(self._alert_history) < 10:
            return
        total_weight = sum(weight for _, _, weight in self._alert_history)
        if total_weight <= 0.0:
            return
        precision = sum(weight for _, confirmed, weight in self._alert_history if confirmed) / total_weight
        if precision < self.target_precision:
            # Too many false alarms — raise threshold
            self._current_alert = min(0.90, self._current_alert + self.step)
        elif precision > self.target_precision + 0.10:
            # Over-target — can afford to catch more
            self._current_alert = max(0.40, self._current_alert - self.step)
        # Keep critical relative to alert
        self._current_critical = max(
            self._current_alert + 0.15,
            self.base_critical,
        )

    @property
    def alert_threshold(self) -> float:
        return self._current_alert

    @property
    def store_threshold(self) -> float:
        return self._current_store

    @property
    def critical_threshold(self) -> float:
        return self._current_critical

    @property
    def current_precision(self) -> Optional[float]:
        if len(self._alert_history) < 5:
            return None
        total_weight = sum(weight for _, _, weight in self._alert_history)
        if total_weight <= 0.0:
            return None
        return sum(weight for _, confirmed, weight in self._alert_history if confirmed) / total_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_threshold": round(self._current_alert, 3),
            "store_threshold": round(self._current_store, 3),
            "critical_threshold": round(self._current_critical, 3),
            "current_precision": round(self.current_precision, 3) if self.current_precision is not None else None,
            "n_samples": len(self._alert_history),
            "target_precision": self.target_precision,
        }


# Rule agreement pairs (Improvement 3) — which rule combinations get a bonus.
# Agent P (2026-04-24): externalized to `data/rule_agreement_pairs.json` so
# new rule pairs (e.g. adding a 5th rule) don't require a code change.
# Each entry: (rule_a, rule_b, config_attr_for_bonus_amount).
_RULE_AGREEMENT_PAIRS_BUILTIN: List[Tuple[str, str, str]] = [
    ("classical_alert", "pattern_match", "agreement_bonus_model_pattern"),
    ("classical_alert", "anomaly_deviation", "agreement_bonus_model_zscore"),
    ("harmonic_alert", "pattern_match", "agreement_bonus_model_pattern"),
    ("harmonic_alert", "anomaly_deviation", "agreement_bonus_model_zscore"),
    ("pattern_match", "anomaly_deviation", "agreement_bonus_pattern_zscore"),
]


def _load_rule_agreement_pairs() -> List[Tuple[str, str, str]]:
    """Load rule-agreement pairs, merging built-ins with data/rule_agreement_pairs.json.

    File schema (v1):
        {
          "schema_version": 1,
          "pairs": [
            {"a": "classical_alert", "b": "pattern_match",
             "bonus_attr": "agreement_bonus_model_pattern"},
            ...
          ]
        }

    Malformed files log a warning and fall back to built-ins.
    """
    candidate = Path(__file__).resolve().parents[3] / "data" / "rule_agreement_pairs.json"
    pairs: List[Tuple[str, str, str]] = list(_RULE_AGREEMENT_PAIRS_BUILTIN)
    try:
        if not candidate.is_file():
            return pairs
        with candidate.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.warning("rule_agreement_pairs: load failed (%s); using built-ins", exc)
        return pairs
    file_pairs = raw.get("pairs") if isinstance(raw, dict) else None
    if not isinstance(file_pairs, list):
        return pairs
    # File fully replaces built-ins when provided (explicit override semantics).
    merged: List[Tuple[str, str, str]] = []
    for entry in file_pairs:
        if not isinstance(entry, dict):
            continue
        a = entry.get("a"); b = entry.get("b"); attr = entry.get("bonus_attr")
        if isinstance(a, str) and isinstance(b, str) and isinstance(attr, str):
            merged.append((a, b, attr))
    return merged if merged else pairs


_RULE_AGREEMENT_PAIRS: List[Tuple[str, str, str]] = _load_rule_agreement_pairs()


# ======================================================================
# Pluggable scoring-rule registry
# ----------------------------------------------------------------------
# Each scoring rule consumes a model's output (classical anomaly score,
# harmonic score, pattern matches, statistical deviation, learned prior, ...)
# and contributes a weighted signal to the fused significance score. Models are
# therefore plugged in AS RULES. Register a new rule factory by name and select
# the active set per run (constructor `enabled_rules=` / `rules=`, or
# `SignificanceConfig.enabled_rules`). The default set is unchanged, so existing
# behaviour is preserved.
# ======================================================================

SCORING_RULE_REGISTRY: Dict[str, Callable[[], "_SignificanceRule"]] = {}

# The default active set (and order). Models added later (e.g. an AAD combiner
# or a temporal-model rule) register themselves and are opted in by name.
DEFAULT_RULE_ORDER: List[str] = [
    "classical_alert",
    "harmonic_alert",
    "pattern_match",
    "anomaly_deviation",
    "historical_prior",
]


def register_scoring_rule(name: str, factory: Callable[[], "_SignificanceRule"]) -> None:
    """Register a scoring-rule factory under ``name`` (plug-in point for models)."""
    SCORING_RULE_REGISTRY[name] = factory


def build_rules(enabled_rules: Optional[List[str]] = None) -> List["_SignificanceRule"]:
    """Instantiate the selected rule set from the registry.

    ``None`` -> the default set in default order. Unknown names raise so a
    typo in a run config fails loudly rather than silently dropping a model.
    """
    names = enabled_rules if enabled_rules is not None else DEFAULT_RULE_ORDER
    rules: List["_SignificanceRule"] = []
    for n in names:
        if n not in SCORING_RULE_REGISTRY:
            raise KeyError(
                f"Unknown scoring rule '{n}'. Registered: {sorted(SCORING_RULE_REGISTRY)}"
            )
        rules.append(SCORING_RULE_REGISTRY[n]())
    return rules


# [PROTOTYPE_LLM_MEMORY_V1] - Main scorer class
class SignificanceScorer:
    """
    Evaluates pattern/metric combinations for significance.
    
    Architecture:
    - Multiple rule evaluators run independently
    - Each rule returns (triggered: bool, score: float, reason: str)
    - Final score is weighted combination
    - Thresholds determine action
    
    [INTEGRATION_POINT] Pattern priors are now persisted to disk.
    """
    
    def __init__(
        self,
        config: Optional[SignificanceConfig] = None,
        priors_path: Optional[str] = None,
        feedback_store: Optional[Any] = None,
        model_confidence_path: Optional[str | Path] = None,
        rules: Optional[List["_SignificanceRule"]] = None,
    ):
        self.config = config or SignificanceConfig()
        # Stash a caller-injected rule set (takes precedence over the registry
        # selection); resolved into self._rules below.
        self._injected_rules = rules
        self._pattern_priors: Dict[str, float] = {}
        self._priors_lock = threading.Lock()

        # Optional store for durable feedback events and traces.
        self.feedback_store = feedback_store
        
        # Path for persisting pattern priors
        self._priors_path = Path(priors_path) if priors_path else None
        from ..model_confidence import resolve_model_confidence_path

        if model_confidence_path is not None:
            self._model_confidence_path = resolve_model_confidence_path(model_confidence_path)
        elif self._priors_path:
            self._model_confidence_path = resolve_model_confidence_path(
                self._priors_path.parent / "model_confidence.json"
            )
        else:
            self._model_confidence_path = resolve_model_confidence_path()
        self._model_confidence_paths: Dict[str, Path] = {
            "anomaly_detector_score": self._model_confidence_path,
            "harmonic_context_score": self._model_confidence_path.with_name(
                f"{self._model_confidence_path.stem}_harmonic_context"
                f"{self._model_confidence_path.suffix or '.json'}"
            ),
            "breakage_prediction": self._model_confidence_path.with_name(
                f"{self._model_confidence_path.stem}_breakage_prediction"
                f"{self._model_confidence_path.suffix or '.json'}"
            ),
        }
        
        # Derived priors cache (primarily for diagnostics/endpoints).
        self._pattern_priors: Dict[str, float] = {}
        self._context_pattern_priors: Dict[str, Dict[str, float]] = {}
        # Lock for thread-safe access to _pattern_priors when background
        # LLM tasks and feedback endpoints run concurrently.
        self._priors_lock = threading.Lock()

        # Local feedback counts when no store is provided.
        # pattern_key -> {"confirm": float, "dismiss": float}
        self._local_feedback_counts: Dict[str, Dict[str, float]] = {}
        self._context_feedback_counts: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._severity_calibration: Dict[str, Dict[str, float]] = {}
        self._feedback_observability: Dict[str, Dict[str, Any]] = {}
        
        # Try to load existing priors from disk
        if self._priors_path:
            self._load_priors()
        
        # Rolling baseline per session (session_id -> feature buffer)
        self._session_baselines: Dict[str, _RollingBaseline] = {}
        
        # Active scoring rules (== the selected set of models). Precedence:
        #   1. rules injected at the constructor,
        #   2. config.enabled_rules (registry selection),
        #   3. the default set (unchanged behaviour).
        if self._injected_rules is not None:
            self._rules: List[_SignificanceRule] = list(self._injected_rules)
        else:
            self._rules = build_rules(getattr(self.config, "enabled_rules", None))

        # --- Adaptive scoring state (Improvements 1-7) ---

        # Improvement 1: RL agent reference (wired from orchestrator)
        self._rl_agent: Optional[Any] = None  # RLAgent from classical_models

        # Improvement 4: Per-rule precision/recall tracking
        self._rule_performance: Dict[str, RulePerformance] = {
            rule.name: RulePerformance(
                _history=deque(maxlen=self.config.rule_performance_window)
            )
            for rule in self._rules
        }

        # Improvement 5: Context-conditioned weight profiles
        self._weight_profiles: Dict[str, WeightProfile] = {}
        self._profiles_path: Optional[Path] = None
        if self._priors_path:
            self._profiles_path = self._priors_path.parent / "weight_profiles.json"
            if self.config.profiles_path:
                self._profiles_path = Path(self.config.profiles_path)
            self._load_weight_profiles()

        # Improvement 6: Adaptive action thresholds
        self._adaptive_thresholds = AdaptiveThresholds(
            base_alert=self.config.alert_threshold,
            base_store=self.config.store_threshold,
            base_critical=self.config.critical_threshold,
            target_precision=self.config.target_alert_precision,
            window_size=self.config.threshold_history_window,
            step=self.config.threshold_step,
        )

        # Improvement 7: Model last-retrain timestamp (set externally)
        self._model_last_retrained_at: Optional[float] = None  # epoch seconds

        # SINDIT context provider reference (set by orchestrator/init)
        self._sindit_provider: Optional[Any] = None

        # Reconcile the on-disk prior cache with live store feedback: the priors file
        # is only a fast-load cache, so a stale cached value must not win over the
        # store's actual confirm/dismiss history for the same pattern key.
        self._hydrate_priors_from_store()

    def _hydrate_priors_from_store(self) -> None:
        """Override file-cached priors with store-derived ones (the store is
        authoritative over the on-disk cache). Two override paths, in precedence order:

        1. A direct priors snapshot (``get_pattern_priors_snapshot``) — e.g. an
           in-memory store that already holds the live priors.
        2. Recomputed priors from feedback history (``list_pattern_keys_with_feedback``).

        Keys the store knows nothing about keep their cached value."""
        store = self.feedback_store
        if store is None:
            return

        # Path 1: direct snapshot wins outright for the keys it carries.
        if hasattr(store, "get_pattern_priors_snapshot"):
            try:
                snapshot = store.get_pattern_priors_snapshot() or {}
            except Exception:
                snapshot = {}
            normalized: Dict[str, float] = {}
            for k, prior in snapshot.items():
                canonical = normalize_pattern_key(str(k).strip())
                if not canonical or is_signature_pattern_key(canonical):
                    continue
                try:
                    normalized[canonical] = float(prior)
                except (TypeError, ValueError):
                    continue
            if normalized:
                with self._priors_lock:
                    self._pattern_priors.update(normalized)
            return

        # Path 2: recompute from feedback history.
        if not hasattr(store, "list_pattern_keys_with_feedback"):
            return
        try:
            keys = list(store.list_pattern_keys_with_feedback() or [])
        except Exception:
            return
        computed: Dict[str, float] = {}
        for k in keys:
            canonical = normalize_pattern_key(str(k).strip())
            if not canonical or is_signature_pattern_key(canonical):
                continue
            try:
                computed[canonical] = float(self.get_pattern_prior(canonical, context=None))
            except Exception:
                continue
        if computed:
            with self._priors_lock:
                self._pattern_priors.update(computed)

    def score(
        self,
        patterns: List[PatternKey],
        metrics: Optional[WindowMetrics] = None,
        context: Optional[CuttingContext] = None,
        session_id: Optional[str] = None,
        external_signals: Optional[Dict[str, Any]] = None,
    ) -> SignificanceResult:
        """
        Evaluate significance of a pattern/metric combination.

        Improvements over the fixed-weight version:
        1. RL agent adjustments applied when available (safe-mode for high scores)
        2. Multiplicative prior instead of additive boost
        3. Rule agreement bonus for corroborating rules
        4. Performance-adjusted weights (online precision/recall)
        5. Context-conditioned weight profiles (per regime/tool/material)
        6. Adaptive action thresholds (self-tuning)
        7. Model confidence age decay for stale models
        8. Context-scoped model trust (plan 1.1): with a cutting context, the
           classical rule's confidence is resolved for that context's scope so
           a false-alarm-prone regime quiets without silencing others.
        """
        external_signals = external_signals or {}

        # Context-scoped model trust: when a caller has NOT already supplied a
        # model_confidence and a real context is available, resolve the
        # feedback-driven confidence for that context (empirical-Bayes fallback
        # to the site-wide aggregate). Off when there is no context (keeps the
        # global behaviour) or when the caller injected an explicit value.
        if "model_confidence" not in external_signals and context is not None:
            ctx_key = self._context_profile_key(context)
            if ctx_key != "_global":
                try:
                    from ..model_confidence import current_model_confidence
                    external_signals = dict(external_signals)
                    external_signals["model_confidence"] = current_model_confidence(
                        self._model_confidence_path, context_key=ctx_key
                    )
                except Exception:
                    logger.debug("context-scoped model confidence lookup failed", exc_info=True)

        # Derive context-conditioned priors for transparency and for the
        # historical prior rule.  Store under BOTH the original key and the
        # normalised canonical form so that _HistoricalPriorRule lookups
        # succeed regardless of which naming convention the caller used.
        derived_priors: Dict[str, float] = {}
        historical_prior = 0.5
        effective_historical_prior = 0.5
        prior_evidence_count = 0.0
        prior_damping_factor = 1.0
        # Sub-neutral tracking (ISS-13): the most-dismissed pattern, considered
        # only when no pattern carries a boosting prior (confirms win).
        _min_effective_prior = 0.5
        _min_prior_raw = 0.5
        _min_prior_evidence = 0.0
        _min_prior_damping = 1.0
        for p in (patterns or []):
            if not getattr(p, "key", None):
                continue
            prior_val, evidence_count = self._get_pattern_prior_and_count(
                p.key,
                context=context,
            )
            derived_priors[p.key] = prior_val
            canonical = normalize_pattern_key(p.key)
            if canonical != p.key:
                derived_priors[canonical] = prior_val
            damping = self._prior_damping_factor(evidence_count)
            effective_prior = 0.5 + ((prior_val - 0.5) * damping)
            if (
                effective_prior > effective_historical_prior
                or (
                    effective_prior == effective_historical_prior
                    and prior_val > historical_prior
                )
            ):
                historical_prior = prior_val
                effective_historical_prior = effective_prior
                prior_evidence_count = evidence_count
                prior_damping_factor = damping
            if effective_prior < _min_effective_prior:
                _min_effective_prior = effective_prior
                _min_prior_raw = prior_val
                _min_prior_evidence = evidence_count
                _min_prior_damping = damping
        if (
            getattr(self.config, "prior_allow_subneutral", False)
            and effective_historical_prior <= 0.5
            and _min_effective_prior < 0.5
            and _min_prior_evidence >= float(
                getattr(self.config, "prior_subneutral_min_evidence", 3.0)
            )
        ):
            historical_prior = _min_prior_raw
            effective_historical_prior = _min_effective_prior
            prior_evidence_count = _min_prior_evidence
            prior_damping_factor = _min_prior_damping

        # --- Improvement 5: resolve context-conditioned weight profile ---
        ctx_key = self._context_profile_key(context)
        profile = self._get_weight_profile(ctx_key, context)
        profile.n_events += 1

        # Base weights from profile (or config defaults)
        eff_w_classical = profile.classical
        eff_w_harmonic = profile.harmonic
        eff_w_pattern = profile.pattern
        eff_w_anomaly = profile.anomaly
        eff_w_historical = profile.historical

        # --- Improvement 4: performance-adjusted weights ---
        if self.config.enable_rule_performance_tracking:
            eff_w_classical, eff_w_harmonic, eff_w_pattern, eff_w_anomaly, eff_w_historical = (
                self._performance_adjusted_weights(
                    eff_w_classical, eff_w_harmonic, eff_w_pattern,
                    eff_w_anomaly, eff_w_historical,
                )
            )

        # --- Improvement 7: model age decay ---
        if self.config.enable_model_age_decay and self._model_last_retrained_at is not None:
            retrained_at = self._model_last_retrained_at
            # Defensive: convert ISO string to epoch float if needed
            if isinstance(retrained_at, str):
                from datetime import datetime as _dt
                retrained_at = _dt.fromisoformat(retrained_at).timestamp()
                self._model_last_retrained_at = retrained_at
            age_hours = (time.time() - retrained_at) / 3600.0
            age_factor = max(
                self.config.model_age_decay_floor,
                1.0 - self.config.model_age_decay_rate * age_hours,
            )
            eff_w_classical *= age_factor

        base_rl_w_classical = eff_w_classical
        base_rl_w_pattern = eff_w_pattern

        # --- Improvement 1: RL agent weight adjustments ---
        rl_action_idx: Optional[int] = None
        rl_state: Optional[Any] = None
        rl_action: Optional[Any] = None
        if (
            self.config.enable_rl_weight_tuning
            and self._rl_agent is not None
            and hasattr(self._rl_agent, "get_recommended_action")
        ):
            try:
                from ..processing.classical_models import RLState
                max_prior = max(derived_priors.values()) if derived_priors else 0.5
                rl_state = RLState(
                    seed_model_score=float(external_signals.get("anomaly_detector_score", 0.0)),
                    harmonic_score=float(external_signals.get("harmonic_context_score", 0.0)),
                    pattern_score=float(max(
                        (p.confidence or 0.0) for p in (patterns or [])
                    ) if patterns else 0.0),
                    historical_prior=max_prior,
                    operating_regime=(
                        context.operating_regime.value
                        if context and context.operating_regime else "unknown"
                    ),
                    tool_type=getattr(context, "tool_type", None) or "unknown",
                    material=getattr(context, "workpiece_material", None) or "unknown",
                )
                rl_action_idx, rl_action = self._rl_agent.select_action(rl_state)
                # Safe mode: only apply exploration noise to sub-ALERT events
                # (greedy is always applied; exploration only for lower scores)
                eff_w_classical = base_rl_w_classical + rl_action.classical_weight_adj
                eff_w_pattern = base_rl_w_pattern + rl_action.pattern_weight_adj
            except Exception as e:
                logger.debug("RL weight adjustment failed: %s", e)

        # Clamp weights to positive
        eff_w_classical = max(0.05, eff_w_classical)
        eff_w_harmonic = max(0.05, eff_w_harmonic)
        eff_w_pattern = max(0.05, eff_w_pattern)
        eff_w_anomaly = max(0.05, eff_w_anomaly)
        eff_w_historical = max(0.05, eff_w_historical)
        
        # Build evaluation context for rules
        eval_ctx = _EvaluationContext(
            patterns=patterns,
            metrics=metrics,
            cutting_context=context,
            session_id=session_id,
            external_signals=external_signals,
            config=self.config,
            pattern_priors=derived_priors,
            baseline=self._get_baseline(session_id, metrics),
        )

        # Map rule name -> effective weight
        _effective_weights = {
            "classical_alert": eff_w_classical,
            "harmonic_alert": eff_w_harmonic,
            "pattern_match": eff_w_pattern,
            "anomaly_deviation": eff_w_anomaly,
            "historical_prior": eff_w_historical,
        }
        # Plugged-in rules (not in the built-in weight profile) contribute via
        # their own weight() — so a selected model rule (e.g. aad_combiner) is
        # actually counted in the fused score instead of silently dropped.
        for _rule in self._rules:
            if _rule.name not in _effective_weights:
                _effective_weights[_rule.name] = float(_rule.weight(self.config))

        # Evaluate all rules
        triggered_rules: List[str] = []
        reasons: List[str] = []
        weighted_scores: List[float] = []
        weights: List[float] = []
        # Agent P (2026-04-24): keep a rule_name -> (weighted_score, weight)
        # mapping so trace assembly doesn't rely on rule iteration order
        # matching the parallel weighted_scores/weights lists. Previously a
        # `zip(_non_hist_triggered, weighted_scores, weights)` would silently
        # misalign if the rule registry were ever reordered.
        weighted_by_rule: Dict[str, Tuple[float, float]] = {}
        rule_results: Dict[str, _RuleResult] = {}
        max_prior_score: float = 0.5  # For multiplicative mode
        protective_penalty: float = 0.0
        suppression_penalty: float = 0.0

        for rule in self._rules:
            try:
                result = rule.evaluate(eval_ctx)
                rule_results[rule.name] = result
                ew = _effective_weights.get(rule.name, rule.weight(self.config))
                if result.triggered:
                    triggered_rules.append(rule.name)
                    reasons.extend(result.reasons)
                    if result.protective_score > 0.0:
                        protective_penalty = max(
                            protective_penalty,
                            result.protective_score * self.config.weight_protective_pattern,
                        )
                    if result.suppression_score > 0.0:
                        suppression_penalty = max(
                            suppression_penalty,
                            result.suppression_score * self.config.weight_protective_pattern,
                        )

                    if rule.name == "historical_prior":
                        # Track the prior score for multiplicative mode
                        max_prior_score = effective_historical_prior
                    elif result.score > 0.0:
                        weighted_scores.append(result.score * ew)
                        weights.append(ew)
                        weighted_by_rule[rule.name] = (result.score * ew, ew)
            except Exception as e:
                logger.warning(f"Rule {rule.name} failed: {e}")

        def _combine_rule_scores(
            effective_weights: Dict[str, float],
        ) -> tuple[float, List[float], List[float], Dict[str, Tuple[float, float]], float, float]:
            local_weighted_scores: List[float] = []
            local_weights: List[float] = []
            local_weighted_by_rule: Dict[str, Tuple[float, float]] = {}
            local_max_prior_score = 0.5

            for rule_name, result in rule_results.items():
                if not result.triggered:
                    continue
                if rule_name == "historical_prior":
                    local_max_prior_score = effective_historical_prior
                    continue
                ew = effective_weights.get(rule_name, 0.0)
                if result.score > 0.0:
                    local_weighted_scores.append(result.score * ew)
                    local_weights.append(ew)
                    local_weighted_by_rule[rule_name] = (result.score * ew, ew)

            if self.config.enable_rule_agreement:
                agreement_bonus = 0.0
                for r1, r2, bonus_attr in _RULE_AGREEMENT_PAIRS:
                    if r1 in triggered_rules and r2 in triggered_rules:
                        agreement_bonus += getattr(self.config, bonus_attr, 0.05)
                local_multi_rule_bonus = min(self.config.agreement_bonus_cap, agreement_bonus)
            else:
                local_multi_rule_bonus = min(0.2, 0.05 * len(triggered_rules))

            if local_weights:
                base_score = sum(local_weighted_scores) / sum(local_weights)
                base_with_penalty = max(
                    0.0,
                    base_score + local_multi_rule_bonus - protective_penalty - suppression_penalty,
                )
                if self.config.prior_mode == "multiplicative":
                    half_range = self.config.prior_multiplier_range / 2.0
                    prior_factor = (1.0 - half_range) + self.config.prior_multiplier_range * local_max_prior_score
                    if prior_factor < 1.0:
                        # Sub-neutral damping (ISS-13): floored, and never
                        # applied when the base evidence alone already reaches
                        # the critical band — feedback must not silence
                        # critical-strength evidence.
                        prior_factor = max(
                            prior_factor,
                            float(getattr(self.config, "prior_factor_floor", 0.85)),
                        )
                        if base_with_penalty >= self.config.critical_threshold:
                            prior_factor = 1.0
                    local_final_score = min(1.0, base_with_penalty * prior_factor)
                else:
                    historical_boost = (local_max_prior_score - 0.5) * eff_w_historical
                    local_final_score = min(1.0, base_with_penalty + historical_boost)
            else:
                if self.config.prior_mode == "multiplicative":
                    local_final_score = max(
                        0.0,
                        local_max_prior_score * 0.5 - protective_penalty - suppression_penalty,
                    )
                else:
                    local_final_score = max(
                        0.0,
                        (local_max_prior_score - 0.5) * eff_w_historical - protective_penalty - suppression_penalty,
                    )

            return (
                local_final_score,
                local_weighted_scores,
                local_weights,
                local_weighted_by_rule,
                local_max_prior_score,
                local_multi_rule_bonus,
            )

        final_score, weighted_scores, weights, weighted_by_rule, max_prior_score, multi_rule_bonus = _combine_rule_scores(
            _effective_weights
        )

        # --- Improvement 1 (safe mode): revert RL exploration if score is too high ---
        safe_mode_overrode_rl = False
        if (
            self.config.enable_rl_weight_tuning
            and self._rl_agent is not None
            and rl_action_idx is not None
            and final_score >= self.config.rl_safe_mode_threshold
            and self._rl_agent._epsilon > 0
        ):
            # For high-score events, use greedy action only (no exploration risk)
            try:
                greedy_action = self._rl_agent.get_recommended_action(rl_state)
                if rl_action is not None and greedy_action != rl_action:
                    safe_weights = dict(_effective_weights)
                    safe_weights["classical_alert"] = max(
                        0.05,
                        base_rl_w_classical + greedy_action.classical_weight_adj,
                    )
                    safe_weights["pattern_match"] = max(
                        0.05,
                        base_rl_w_pattern + greedy_action.pattern_weight_adj,
                    )
                    final_score, weighted_scores, weights, weighted_by_rule, max_prior_score, multi_rule_bonus = _combine_rule_scores(
                        safe_weights
                    )
                    _effective_weights = safe_weights
                    rl_action_idx = None
                    safe_mode_overrode_rl = True
                    logger.debug("RL safe mode: reverted exploratory action for high-score event")
            except Exception:
                pass

        # --- Improvement 6: adaptive thresholds ---
        if self.config.enable_adaptive_thresholds:
            eff_alert = self._adaptive_thresholds.alert_threshold
            eff_store = self._adaptive_thresholds.store_threshold
            eff_critical = self._adaptive_thresholds.critical_threshold
        else:
            eff_alert = self.config.alert_threshold
            eff_store = self.config.store_threshold
            eff_critical = self.config.critical_threshold

        severity_adjustment = self._severity_adjustment_for_patterns(patterns)
        if severity_adjustment != 0.0:
            final_score = min(1.0, max(0.0, final_score + severity_adjustment))
            reasons.append(f"Severity calibration adjustment: {severity_adjustment:+.2f}")

        # Determine action
        if final_score >= eff_critical:
            action = SignificanceAction.CRITICAL
        elif final_score >= eff_alert:
            action = SignificanceAction.ALERT
        elif final_score >= eff_store:
            action = SignificanceAction.STORE
        else:
            action = SignificanceAction.IGNORE

        # Lone-rule alert gate: a single contributing detector below the critical
        # band caps at STORE — cuts single-model false alerts. weighted_by_rule
        # holds only detector rules that scored > 0 (historical_prior excluded),
        # so a lone rule that reaches CRITICAL still alerts.
        if (
            getattr(self.config, "require_multi_rule_alert", False)
            and action == SignificanceAction.ALERT
            and len(weighted_by_rule) < 2
        ):
            action = SignificanceAction.STORE
            reasons.append("Single-detector alert capped at store (awaiting corroboration)")

        is_significant = action in (SignificanceAction.ALERT, SignificanceAction.CRITICAL, SignificanceAction.STORE)

        # Historical boost for diagnostics (the prior's contribution)
        if self.config.prior_mode == "multiplicative":
            prior_boost = max_prior_score
            _half = self.config.prior_multiplier_range / 2.0
            prior_factor_value: Optional[float] = (1.0 - _half) + self.config.prior_multiplier_range * max_prior_score
        else:
            prior_boost = (max_prior_score - 0.5) * eff_w_historical
            prior_factor_value = None
        
        # Collect derived priors for transparency
        pattern_priors = {p.key: derived_priors.get(p.key, 0.5) for p in patterns}

        # Agent D (2026-04-24): assemble the ordered score_trace so the UI
        # waterfall and diagnostics tools can explain how the final score was
        # built. Purely additive — no rescoring — we read values already in
        # scope from the computation above.
        score_trace: List[Dict[str, Any]] = []
        for _rn, _rw in _effective_weights.items():
            score_trace.append({
                "component": f"weight:{_rn}",
                "value": float(_rw),
                "source": "effective_weights",
            })
        try:
            model_confidence = float(external_signals.get("model_confidence", 1.0))
        except (TypeError, ValueError):
            model_confidence = 1.0
        if "classical_alert" in triggered_rules and model_confidence < 1.0:
            score_trace.append({
                "component": "model_trust",
                "value": float(model_confidence),
                "source": "feedback_driven_model_confidence",
            })
        _non_hist_triggered = [
            r.name for r in self._rules
            if r.name in triggered_rules and r.name != "historical_prior"
        ]
        for _rn in _non_hist_triggered:
            _ws, _w = weighted_by_rule.get(_rn, (0.0, 0.0))
            score_trace.append({
                "component": f"rule:{_rn}",
                "value": float(_ws),
                "source": f"weighted (w={_w:.3f})",
            })
        if weights:
            _wsum = float(sum(weights))
            score_trace.append({
                "component": "base_score",
                "value": float(sum(weighted_scores) / _wsum) if _wsum > 0 else 0.0,
                "source": "weighted_avg",
            })
            score_trace.append({
                "component": "agreement_bonus",
                "value": float(multi_rule_bonus),
                "source": "rule_agreement" if self.config.enable_rule_agreement else "legacy_flat",
            })
        if protective_penalty > 0.0:
            score_trace.append({
                "component": "protective_pattern_match",
                "value": float(-protective_penalty),
                "source": f"weighted (w={self.config.weight_protective_pattern:.3f})",
            })
        if suppression_penalty > 0.0:
            score_trace.append({
                "component": "suppression_pattern_match",
                "value": float(-suppression_penalty),
                "source": f"weighted (w={self.config.weight_protective_pattern:.3f})",
            })
        if self.config.prior_mode == "multiplicative":
            _half = self.config.prior_multiplier_range / 2.0
            score_trace.append({
                "component": "prior_factor",
                "value": float((1.0 - _half) + self.config.prior_multiplier_range * max_prior_score),
                "source": (
                    f"multiplicative prior={historical_prior:.3f} "
                    f"damping={prior_damping_factor:.3f}"
                ),
            })
        else:
            score_trace.append({
                "component": "prior_boost_additive",
                "value": float((max_prior_score - 0.5) * eff_w_historical),
                "source": (
                    f"additive prior={historical_prior:.3f} "
                    f"damping={prior_damping_factor:.3f}"
                ),
            })
        if rl_action_idx is not None:
            score_trace.append({
                "component": "rl_adjustment",
                "value": float(rl_action_idx),
                "source": "rl_agent_action_idx",
            })
        if safe_mode_overrode_rl:
            score_trace.append({
                "component": "rl_safe_mode",
                "value": 1.0,
                "source": "greedy_override",
            })
        if severity_adjustment != 0.0:
            score_trace.append({
                "component": "severity_calibration",
                "value": float(severity_adjustment),
                "source": "weighted_feedback",
            })
        score_trace.append({
            "component": "final_score",
            "value": float(final_score),
            "source": f"action={action.value}",
        })

        pattern_rule_score = float(weighted_by_rule.get("pattern_match", (0.0, 0.0))[0])
        return SignificanceResult(
            is_significant=is_significant,
            score=final_score,
            action=action,
            reasons=reasons,
            triggered_rules=triggered_rules,
            pattern_priors=pattern_priors,
            prior_boost=prior_boost,
            pattern_rule_score=pattern_rule_score,
            prior_factor=prior_factor_value,
            prior_mode=self.config.prior_mode,
            historical_prior=historical_prior,
            prior_evidence_count=int(round(prior_evidence_count)),
            prior_damping_factor=prior_damping_factor,
            score_trace=score_trace,
        )

    # ------------------------------------------------------------------
    # Priors derived from feedback events (context-conditioned)
    # ------------------------------------------------------------------

    def refresh_priors(self) -> None:
        """Refresh global prior cache from feedback events (for diagnostics).

        This computes *global* priors (context-free) for any pattern keys that
        have feedback history.  If local counts are empty, reload from disk
        first so that priors saved by subprocess experiment runs are picked up
        rather than being overwritten with an empty dict.
        """
        # Reload from disk first so subprocess-written priors are visible
        if self._priors_path and self._priors_path.exists():
            if not self._local_feedback_counts:
                self._load_priors()

        if self.feedback_store and hasattr(self.feedback_store, "list_pattern_keys_with_feedback"):
            try:
                keys = list(self.feedback_store.list_pattern_keys_with_feedback() or [])
            except Exception:
                keys = []
        else:
            keys = list(self._local_feedback_counts.keys())

        if not keys:
            # Nothing to recompute — keep whatever was loaded from disk
            return

        priors: Dict[str, float] = {}
        for k in keys:
            canonical = normalize_pattern_key(str(k).strip())
            if not canonical or is_signature_pattern_key(canonical):
                continue
            priors[canonical] = float(self.get_pattern_prior(canonical, context=None))
        with self._priors_lock:
            self._pattern_priors = priors
        self._save_priors()

    def get_pattern_prior(self, pattern_key: str, context: Optional[CuttingContext] = None) -> float:
        """Return a prior in [0,1], derived from feedback counts.

        Beta(1,1) (Laplace) smoothing over *saturated* effective counts, via the
        shared :mod:`prior_math` estimator. NOTE: despite the historical name
        "recency decay", this is **not** recency weighting — only aggregate
        confirm/dismiss totals are stored (no event ordering), so the transform
        ``(1 - decay**n)/(1 - decay)`` saturates the total rather than favouring
        recent events. Its effect is a confidence ceiling (≈0.885/0.115 at
        decay=0.85), i.e. small-sample regularisation. See ``prior_math``.
        """
        prior, _evidence_count = self._get_pattern_prior_and_count(pattern_key, context=context)
        return prior

    def get_pattern_prior_record(
        self, pattern_key: str, context: Optional[CuttingContext] = None
    ) -> "PatternPrior":
        """Structured, auditable view of a pattern's learned prior.

        Read-side aggregation only: combines the derived prior
        (:meth:`get_pattern_prior`) with the confirm/dismiss feedback counts and a
        volume-based confidence into a single record. Does not change how priors
        are computed or persisted. Backs the deliverable's "auditable, traceable
        priors" (§5.8 Learned Priors).
        """
        from .pattern_prior import PatternPrior

        key = normalize_pattern_key(str(pattern_key).strip())
        prior = self.get_pattern_prior(key, context=context)
        user_id = self._feedback_scope_user_id(context)
        ctx_key: Optional[str] = None
        confirm = dismiss = 0.0
        # Resolve the most-specific context level that actually has feedback —
        # the same hierarchy the prior is derived from.
        for cand in self._candidate_context_keys(context):
            c, d = self._get_feedback_counts(key, cand, user_id)
            if (c + d) > 0:
                ctx_key, confirm, dismiss = cand, c, d
                break
        total = confirm + dismiss
        # Volume-based confidence n/(n+k): 0.0 with no evidence, rising with
        # volume. NOT _prior_damping_factor (which returns 1.0 at n=0, correct
        # for damping a neutral prior but wrong as a confidence). Reuses the
        # configured damping constant when set, else a small default.
        damping_k = max(0.0, float(getattr(self.config, "prior_evidence_damping_k", 0.0) or 0.0))
        conf_k = damping_k if damping_k > 0.0 else 5.0
        confidence = (total / (total + conf_k)) if total > 0 else 0.0
        return PatternPrior(
            pattern_key=key,
            prior_strength=float(prior),
            confidence=float(confidence),
            confirmed=float(confirm),
            dismissed=float(dismiss),
            evidence_count=float(total),
            context_key=ctx_key,
        )

    def list_pattern_prior_records(
        self, context: Optional[CuttingContext] = None
    ) -> List["PatternPrior"]:
        """All learned pattern priors as structured records (see
        :meth:`get_pattern_prior_record`)."""
        with self._priors_lock:
            keys = sorted(self._pattern_priors.keys())
        return [self.get_pattern_prior_record(k, context=context) for k in keys]

    def _get_pattern_prior_and_count(
        self,
        pattern_key: str,
        context: Optional[CuttingContext] = None,
    ) -> tuple[float, float]:
        pattern_key = normalize_pattern_key(str(pattern_key).strip())
        if not pattern_key or is_signature_pattern_key(pattern_key):
            return (0.5, 0.0)

        # Saturation factor 0.85 (shared default in prior_math): the geometric
        # series (1-0.85^N)/(1-0.85) caps the effective count at ~6.67, which
        # caps the derived prior at ~0.885/0.115. This is small-sample
        # regularisation, not recency weighting (counts are aggregate, not
        # ordered). Held here rather than in ScorerConfig because changing it
        # retroactively invalidates cached priors.
        recency_decay: float = DEFAULT_RECENCY_DECAY
        feedback_user_id = self._feedback_scope_user_id(context)

        if getattr(self.config, "context_prior_pooling", False):
            return self._pooled_pattern_prior(pattern_key, context, feedback_user_id, recency_decay)

        for ctx_key in self._candidate_context_keys(context):
            confirm, dismiss = self._get_feedback_counts(pattern_key, ctx_key, feedback_user_id)
            total = confirm + dismiss
            if total <= 0:
                continue

            prior, _ = self._prior_from_counts(confirm, dismiss, recency_decay=recency_decay)
            return (prior, float(total))

        return (0.5, 0.0)

    def _pooled_pattern_prior(
        self,
        pattern_key: str,
        context: Optional[CuttingContext],
        feedback_user_id: Optional[str],
        recency_decay: float,
    ) -> tuple[float, float]:
        """D1: empirical-Bayes partial pooling across the context hierarchy.

        Walk from the LEAST-specific (parent) to the MOST-specific context key.
        At each level shrink the local estimate toward the running parent
        estimate by kappa pseudo-counts. Sparse leaf contexts stay close to the
        parent; well-observed leaves dominate their own evidence.
        """
        kappa = max(0.0, float(getattr(self.config, "context_prior_pooling_kappa", 8.0)))
        # candidate keys are most->least specific; reverse to go parent->leaf.
        keys = list(self._candidate_context_keys(context))[::-1]
        p_parent = 0.5
        total_seen = 0.0
        for ctx_key in keys:
            confirm, dismiss = self._get_feedback_counts(pattern_key, ctx_key, feedback_user_id)
            n = confirm + dismiss
            if n <= 0:
                continue
            p_local, _ = self._prior_from_counts(confirm, dismiss, recency_decay=recency_decay)
            # shrink local toward the parent estimate by kappa pseudo-counts
            p_parent = (n * p_local + kappa * p_parent) / (n + kappa)
            total_seen = n  # most-specific observed level's true volume
        return (float(p_parent), float(total_seen))

    def _prior_damping_factor(self, evidence_count: float) -> float:
        try:
            damping_k = max(0.0, float(self.config.prior_evidence_damping_k))
        except (TypeError, ValueError):
            damping_k = 0.0
        try:
            count = max(0.0, float(evidence_count))
        except (TypeError, ValueError):
            count = 0.0
        if damping_k <= 0.0 or count <= 0.0:
            return 1.0
        return count / (count + damping_k)

    @staticmethod
    def _feedback_scope_user_id(context: Optional[CuttingContext]) -> Optional[str]:
        if context is None:
            return None
        extra = getattr(context, "extra", None) or {}
        for key in ("feedback_scope_user_id", "feedback_scope"):
            value = extra.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _candidate_context_keys(self, context: Optional[CuttingContext]) -> List[Optional[str]]:
        """Generate context keys from most-specific to least-specific."""
        if context is None:
            return [None]

        # Order matters for hierarchical fallback.
        dims: List[tuple[str, Optional[str]]] = [
            ("machine_type", getattr(context, "machine_type", None)),
            ("tool_type", getattr(context, "tool_type", None)),
            ("workpiece_material", getattr(context, "workpiece_material", None)),
            ("operating_regime", getattr(context, "operating_regime", None).value if getattr(context, "operating_regime", None) else None),
        ]

        present = [(k, str(v).strip()) for k, v in dims if v is not None and str(v).strip()]
        if not present:
            return [None]

        keys: List[Optional[str]] = []
        # Build progressively less-specific keys by dropping suffixes.
        for i in range(len(present), 0, -1):
            parts = present[:i]
            keys.append("|".join([f"{k}={v}" for k, v in parts]))

        return keys

    def _get_feedback_counts(
        self,
        pattern_key: str,
        context_key: Optional[str],
        feedback_user_id: Optional[str] = None,
    ) -> tuple[float, float]:
        """Return (confirm, dismiss) counts."""
        if self.feedback_store and hasattr(self.feedback_store, "get_feedback_counts"):
            try:
                raw_counts = self.feedback_store.get_feedback_counts(
                    pattern_key=pattern_key,
                    context_key=context_key,
                    user_id=feedback_user_id,
                )
                return (float(raw_counts[0]), float(raw_counts[1]))  # type: ignore[index]
            except Exception:
                pass

        if context_key is not None:
            scoped_patterns = self._context_feedback_counts.get(context_key) or {}
            scoped_counts = scoped_patterns.get(pattern_key) or {}
            return (
                float(scoped_counts.get("confirm", 0.0)),
                float(scoped_counts.get("dismiss", 0.0)),
            )

        local = self._local_feedback_counts.get(pattern_key) or {}
        return (float(local.get("confirm", 0.0)), float(local.get("dismiss", 0.0)))

    @staticmethod
    def _effective_feedback_count(count: float, recency_decay: float) -> float:
        # Delegates to the shared estimator (prior_math) so the live scorer and
        # the offline experiment compute identical priors. See that module's
        # docstring: this is a saturating transform, NOT recency weighting.
        return effective_feedback_count(count, recency_decay)

    def _prior_from_counts(self, confirm: float, dismiss: float, *, recency_decay: float) -> tuple[float, float]:
        return prior_from_counts(confirm, dismiss, recency_decay)
    
    def update_baseline(self, session_id: str, metrics: WindowMetrics) -> None:
        """
        Update rolling baseline for a session.
        
        [INTEGRATION_POINT] Should be called as windows are processed.
        """
        if session_id not in self._session_baselines:
            self._session_baselines[session_id] = _RollingBaseline(
                window_size=self.config.baseline_window_size
            )
        self._session_baselines[session_id].add(metrics)

    def seed_feedback_counts(
        self,
        counts_by_pattern: Dict[str, Dict[str, float]],
        context: Optional[CuttingContext] = None,
    ) -> None:
        """Replace the in-memory feedback counts for the given patterns.

        Public entry point for callers that drive the scorer from their own
        accumulated counts — notably the offline experiment bridge, which must
        seed exact confirm/dismiss volumes so ``get_pattern_prior`` derives the
        same value (and evidence damping sees the same volume) the live pipeline
        would. Overwrites existing entries; thread-safe.

        When ``context`` is provided, the counts are ALSO written under the
        least-specific (shared) context key so that a context-aware ``score``
        call resolves the prior to this evidence. This is required because the
        prior lookup (``_get_pattern_prior_and_count``) tries context-scoped
        keys only — it does NOT fall back to the global counts — so seeding the
        global map alone would leave a context-aware caller seeing a neutral
        0.5. Seeding the least-specific key (e.g. ``machine_type=...``, shared
        across every sample) preserves the aggregate-prior numerics while
        letting the scorer exercise the context-conditioned code path.

        Only effective when no feedback store with ``get_feedback_counts`` is
        attached (otherwise the store is authoritative); that is the case for
        the standalone scorer used in experiments.
        """
        ctx_keys = self._candidate_context_keys(context) if context is not None else []
        scope_key = ctx_keys[-1] if ctx_keys and ctx_keys[-1] is not None else None
        with self._priors_lock:
            for pk, counts in counts_by_pattern.items():
                key = normalize_pattern_key(str(pk).strip())
                if not key:
                    continue
                payload = {
                    "confirm": float(counts.get("confirm", 0.0)),
                    "dismiss": float(counts.get("dismiss", 0.0)),
                }
                self._local_feedback_counts[key] = dict(payload)
                if scope_key is not None:
                    self._context_feedback_counts.setdefault(scope_key, {})[key] = dict(payload)

    def update_pattern_prior(
        self,
        pattern_key: str,
        was_significant: bool,
        *,
        context: Optional[CuttingContext] = None,
        weight: float = 1.0,
        source: Optional[str] = None,
    ) -> None:
        """Record feedback about a pattern (append-only), then refresh priors.

        This intentionally does NOT do EMA updates. Priors are derived from the
        feedback event history.

        NOTE: The durable feedback-event write is handled by the *caller*
        (``MemoryFeedbackHandler._persist_feedback_event``). We only update
        the local feedback-count cache here. After recomputing the exact
        scalar prior, we mirror that value into the backing store when the
        store exposes ``sync_pattern_prior`` so graph views do not drift away
        from the scorer's source-of-truth prior state.
        """
        pattern_key = normalize_pattern_key(str(pattern_key).strip())
        if not pattern_key or is_signature_pattern_key(pattern_key):
            return

        try:
            feedback_weight = max(0.0, float(weight))
        except (TypeError, ValueError):
            feedback_weight = 1.0
        if feedback_weight <= 0.0:
            return

        action = "confirm" if bool(was_significant) else "dismiss"
        context_key = next((key for key in self._candidate_context_keys(context) if key), None)

        # P7: the only true shared-state race is the count-map mutation, so the
        # lock guards just that (and the cache writes below). The prior
        # recompute can touch a feedback store (Neo4j/SQLite), so it runs
        # OUTSIDE the lock — never hold the mutex across store I/O. The cache is
        # an observability snapshot; last-writer-wins under concurrency is fine
        # (scoring recomputes from counts and never reads the cache).
        with self._priors_lock:
            local = self._local_feedback_counts.setdefault(pattern_key, {"confirm": 0.0, "dismiss": 0.0})
            local[action] = float(local.get(action, 0.0)) + feedback_weight
            if context_key:
                scoped_patterns = self._context_feedback_counts.setdefault(context_key, {})
                scoped_counts = scoped_patterns.setdefault(pattern_key, {"confirm": 0.0, "dismiss": 0.0})
                scoped_counts[action] = float(scoped_counts.get(action, 0.0)) + feedback_weight

        # Global cache stays global (persisted as the global `pattern_priors`
        # map, read by global-only views). P3 (D2): the per-context cache holds
        # the value scoring actually uses under this context — the hierarchical
        # get_pattern_prior(context), not a recomputation from exact-context
        # counts that ignores the most-specific-with-data fallback chain.
        new_prior = self.get_pattern_prior(pattern_key, context=None)
        context_prior = self.get_pattern_prior(pattern_key, context=context) if context_key else None
        with self._priors_lock:
            self._pattern_priors[pattern_key] = float(new_prior)
            if context_key and context_prior is not None:
                self._context_pattern_priors.setdefault(context_key, {})[pattern_key] = float(context_prior)

        self._record_feedback_observability(
            pattern_key,
            weight=feedback_weight,
            was_significant=bool(was_significant),
            source=source or ("confirm_explicit" if bool(was_significant) else "dismiss_oneclick"),
        )
        self._save_priors()
        # The graph store node has no context dimension, so mirror the global
        # value there (unambiguous); context-resolved priors live in the
        # labelled _context_pattern_priors cache for context-aware consumers.
        self._sync_pattern_prior_to_store(pattern_key, float(new_prior))

    def _sync_pattern_prior_to_store(self, pattern_key: str, prior: float) -> None:
        """Mirror the current scorer-derived prior into the backing store.

        The scorer remains the source of truth for prior computation. This
        hook only propagates the already-computed scalar prior to stores that
        choose to surface priors independently (for example Neo4j graph views).
        """
        store = self.feedback_store
        if store is None or not hasattr(store, "sync_pattern_prior"):
            return

        def _write() -> None:
            try:
                store.sync_pattern_prior(pattern_key, float(prior))
            except Exception:
                logger.debug(
                    "Failed to sync pattern prior to backing store for %s",
                    pattern_key,
                    exc_info=True,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _write()
        else:
            loop.create_task(asyncio.to_thread(_write))

    @staticmethod
    def _empty_feedback_observability() -> Dict[str, Any]:
        return {
            "effective_weight_total": 0.0,
            "confirm_weight_total": 0.0,
            "dismiss_weight_total": 0.0,
            "passive_outcome_count": 0,
            "passive_outcome_weight_total": 0.0,
            "severity_correction_count": 0,
            "severity_correction_weight_total": 0.0,
            "source_weight_totals": {},
        }

    def _record_feedback_observability(
        self,
        pattern_key: str,
        *,
        weight: float,
        was_significant: Optional[bool],
        source: str,
    ) -> None:
        state = self._feedback_observability.setdefault(
            pattern_key,
            self._empty_feedback_observability(),
        )
        state["effective_weight_total"] = float(state.get("effective_weight_total", 0.0) or 0.0) + weight

        if was_significant is True:
            state["confirm_weight_total"] = float(state.get("confirm_weight_total", 0.0) or 0.0) + weight
        elif was_significant is False:
            state["dismiss_weight_total"] = float(state.get("dismiss_weight_total", 0.0) or 0.0) + weight

        if source == "passive_cycle_completed_without_intervention":
            state["passive_outcome_count"] = int(state.get("passive_outcome_count", 0) or 0) + 1
            state["passive_outcome_weight_total"] = (
                float(state.get("passive_outcome_weight_total", 0.0) or 0.0) + weight
            )
        if source == "severity_correction":
            state["severity_correction_count"] = int(state.get("severity_correction_count", 0) or 0) + 1
            state["severity_correction_weight_total"] = (
                float(state.get("severity_correction_weight_total", 0.0) or 0.0) + weight
            )

        raw_source_totals = state.get("source_weight_totals")
        if not isinstance(raw_source_totals, dict):
            raw_source_totals = {}
        raw_source_totals[str(source)] = float(raw_source_totals.get(str(source), 0.0) or 0.0) + weight
        state["source_weight_totals"] = raw_source_totals

    @staticmethod
    def _normalize_severity_label(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "ignore": "info",
            "store": "info",
            "info": "info",
            "alert": "warning",
            "warning": "warning",
            "critical": "critical",
        }.get(normalized)

    def record_severity_correction(
        self,
        pattern_key: str,
        *,
        target_severity: str,
        current_score: Optional[float] = None,
        current_severity: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        pattern_key = normalize_pattern_key(str(pattern_key).strip())
        if not pattern_key:
            return

        target_label = self._normalize_severity_label(target_severity)
        if target_label is None:
            return

        try:
            feedback_weight = max(0.0, float(weight))
        except (TypeError, ValueError):
            feedback_weight = 1.0
        if feedback_weight <= 0.0:
            return

        baseline_score = current_score
        if baseline_score is None:
            current_label = self._normalize_severity_label(current_severity)
            if current_label is not None:
                baseline_score = _SEVERITY_TARGET_SCORES[current_label]
        if baseline_score is None:
            baseline_score = _SEVERITY_TARGET_SCORES[target_label]

        target_score = _SEVERITY_TARGET_SCORES[target_label]
        max_delta = float(self.config.severity_correction_max_delta or 0.0)
        delta = max(-max_delta, min(max_delta, float(target_score - baseline_score)))

        state = self._severity_calibration.setdefault(
            pattern_key,
            {
                "delta_sum": 0.0,
                "weight_sum": 0.0,
                "targets": {label: 0.0 for label in _SEVERITY_TARGET_SCORES},
            },
        )
        state["delta_sum"] = float(state.get("delta_sum", 0.0)) + delta * feedback_weight
        state["weight_sum"] = float(state.get("weight_sum", 0.0)) + feedback_weight
        raw_targets = state.get("targets")
        if not isinstance(raw_targets, dict):
            raw_targets = {}
        targets = {
            label: float(raw_targets.get(label, 0.0) or 0.0)
            for label in _SEVERITY_TARGET_SCORES
        }
        targets[target_label] = float(targets.get(target_label, 0.0) or 0.0) + feedback_weight
        state["targets"] = targets
        self._record_feedback_observability(
            pattern_key,
            weight=feedback_weight,
            was_significant=None,
            source="severity_correction",
        )
        self._save_priors()

    def _severity_adjustment_for_patterns(self, patterns: List[PatternKey]) -> float:
        total_weight = 0.0
        weighted_delta = 0.0
        for pattern in patterns or []:
            pattern_key = normalize_pattern_key(str(getattr(pattern, "key", "")).strip())
            if not pattern_key:
                continue
            state = self._severity_calibration.get(pattern_key) or {}
            weight_sum = float(state.get("weight_sum", 0.0) or 0.0)
            if weight_sum <= 0.0:
                continue
            avg_delta = float(state.get("delta_sum", 0.0) or 0.0) / weight_sum
            total_weight += weight_sum
            weighted_delta += avg_delta * weight_sum

        if total_weight <= 0.0:
            return 0.0
        max_delta = float(self.config.severity_correction_max_delta or 0.0)
        return max(-max_delta, min(max_delta, weighted_delta / total_weight))

    def get_feedback_diagnostics(self, pattern_key: Optional[str] = None) -> Dict[str, Any]:
        def build(pattern: str) -> Dict[str, Any]:
            confirm_weight, dismiss_weight = self._get_feedback_counts(pattern, context_key=None)
            observability = self._feedback_observability.get(pattern) or {}
            severity = self._severity_calibration.get(pattern) or {}
            severity_weight = float(severity.get("weight_sum", 0.0) or 0.0)
            raw_targets = severity.get("targets") if isinstance(severity.get("targets"), dict) else {}
            targets = {
                label: float((raw_targets or {}).get(label, 0.0) or 0.0)
                for label in _SEVERITY_TARGET_SCORES
            }
            effective_weight_total = max(
                float(observability.get("effective_weight_total", 0.0) or 0.0),
                float(confirm_weight + dismiss_weight + severity_weight),
            )
            average_delta = 0.0
            if severity_weight > 0.0:
                average_delta = float(severity.get("delta_sum", 0.0) or 0.0) / severity_weight

            source_weight_totals = observability.get("source_weight_totals")
            if not isinstance(source_weight_totals, dict):
                source_weight_totals = {}

            return {
                "confirm_weight_total": float(confirm_weight),
                "dismiss_weight_total": float(dismiss_weight),
                "effective_weight_total": float(effective_weight_total),
                "passive_outcome_count": int(observability.get("passive_outcome_count", 0) or 0),
                "passive_outcome_weight_total": float(observability.get("passive_outcome_weight_total", 0.0) or 0.0),
                "severity_correction_count": int(observability.get("severity_correction_count", 0) or 0),
                "severity_correction_weight_total": float(observability.get("severity_correction_weight_total", 0.0) or 0.0),
                "severity_calibration": {
                    "average_delta": float(average_delta),
                    "weight_total": float(severity_weight),
                    "targets": targets,
                },
                "source_weight_totals": {
                    str(name): float(value or 0.0)
                    for name, value in source_weight_totals.items()
                },
            }

        if pattern_key is not None:
            normalized = normalize_pattern_key(str(pattern_key).strip())
            if not normalized:
                return {}
            return build(normalized)

        all_keys = (
            set(self._pattern_priors.keys())
            | set(self._local_feedback_counts.keys())
            | set(self._severity_calibration.keys())
            | set(self._feedback_observability.keys())
        )
        return {pattern: build(pattern) for pattern in sorted(all_keys)}
    
    def _load_priors(self) -> None:
        """Load pattern priors from disk."""
        if not self._priors_path or not self._priors_path.exists():
            logger.debug("No priors file found, starting with empty priors")
            return
        
        try:
            with open(self._priors_path, 'r') as f:
                data = json.load(f)
                # A bootstrap-seeded priors file (fleet/repo seed, not real local
                # feedback) is ignored unless explicitly opted in, so a fresh site
                # starts cold rather than inheriting seeded priors silently.
                if data.get("bootstrap_seeded") and not getattr(self.config, "bootstrap_pattern_priors", False):
                    logger.debug("Ignoring bootstrap-seeded priors (bootstrap_pattern_priors disabled)")
                    return
                raw_priors = data.get("pattern_priors", {}) or {}
                normalized_priors: Dict[str, float] = {}
                for pattern_key, prior in raw_priors.items():
                    canonical = normalize_pattern_key(str(pattern_key).strip())
                    if not canonical or is_signature_pattern_key(canonical):
                        continue
                    normalized_priors[canonical] = float(prior or 0.5)
                with self._priors_lock:
                    self._pattern_priors = normalized_priors
                raw_context_priors = data.get("pattern_priors_by_context", {}) or {}
                normalized_context_priors: Dict[str, Dict[str, float]] = {}
                for context_key, priors in raw_context_priors.items():
                    if not isinstance(priors, dict):
                        continue
                    scoped_priors: Dict[str, float] = {}
                    for pattern_key, prior in priors.items():
                        canonical = normalize_pattern_key(str(pattern_key).strip())
                        if not canonical or is_signature_pattern_key(canonical):
                            continue
                        scoped_priors[canonical] = float(prior or 0.5)
                    if scoped_priors:
                        normalized_context_priors[str(context_key)] = scoped_priors
                with self._priors_lock:
                    self._context_pattern_priors = normalized_context_priors
                raw_counts = data.get("feedback_counts", {}) or {}
                normalized_counts: Dict[str, Dict[str, float]] = {}
                for pattern_key, counts in raw_counts.items():
                    if not isinstance(counts, dict):
                        continue
                    canonical = normalize_pattern_key(str(pattern_key).strip())
                    if not canonical or is_signature_pattern_key(canonical):
                        continue
                    bucket = normalized_counts.setdefault(canonical, {"confirm": 0.0, "dismiss": 0.0})
                    bucket["confirm"] += float(counts.get("confirm", 0.0) or 0.0)
                    bucket["dismiss"] += float(counts.get("dismiss", 0.0) or 0.0)
                self._local_feedback_counts = normalized_counts
                raw_context_counts = data.get("feedback_counts_by_context", {}) or {}
                normalized_context_counts: Dict[str, Dict[str, Dict[str, float]]] = {}
                for context_key, patterns in raw_context_counts.items():
                    if not isinstance(patterns, dict):
                        continue
                    scoped_counts: Dict[str, Dict[str, float]] = {}
                    for pattern_key, counts in patterns.items():
                        if not isinstance(counts, dict):
                            continue
                        canonical = normalize_pattern_key(str(pattern_key).strip())
                        if not canonical or is_signature_pattern_key(canonical):
                            continue
                        scoped_counts[canonical] = {
                            "confirm": float(counts.get("confirm", 0.0) or 0.0),
                            "dismiss": float(counts.get("dismiss", 0.0) or 0.0),
                        }
                    if scoped_counts:
                        normalized_context_counts[str(context_key)] = scoped_counts
                self._context_feedback_counts = normalized_context_counts
                raw_severity = data.get("severity_calibration", {}) or {}
                normalized_severity: Dict[str, Dict[str, float]] = {}
                for pattern_key, state in raw_severity.items():
                    if not isinstance(state, dict):
                        continue
                    normalized_severity[str(pattern_key)] = {
                        "delta_sum": float(state.get("delta_sum", 0.0) or 0.0),
                        "weight_sum": float(state.get("weight_sum", 0.0) or 0.0),
                        "targets": {
                            label: float(((state.get("targets") or {}) if isinstance(state.get("targets"), dict) else {}).get(label, 0.0) or 0.0)
                            for label in _SEVERITY_TARGET_SCORES
                        },
                    }
                self._severity_calibration = normalized_severity
                raw_observability = data.get("feedback_observability", {}) or {}
                normalized_observability: Dict[str, Dict[str, Any]] = {}
                for pattern_key, state in raw_observability.items():
                    if not isinstance(state, dict):
                        continue
                    raw_source_totals = state.get("source_weight_totals")
                    if not isinstance(raw_source_totals, dict):
                        raw_source_totals = {}
                    normalized_observability[str(pattern_key)] = {
                        "effective_weight_total": float(state.get("effective_weight_total", 0.0) or 0.0),
                        "confirm_weight_total": float(state.get("confirm_weight_total", 0.0) or 0.0),
                        "dismiss_weight_total": float(state.get("dismiss_weight_total", 0.0) or 0.0),
                        "passive_outcome_count": int(state.get("passive_outcome_count", 0) or 0),
                        "passive_outcome_weight_total": float(state.get("passive_outcome_weight_total", 0.0) or 0.0),
                        "severity_correction_count": int(state.get("severity_correction_count", 0) or 0),
                        "severity_correction_weight_total": float(state.get("severity_correction_weight_total", 0.0) or 0.0),
                        "source_weight_totals": {
                            str(name): float(value or 0.0)
                            for name, value in raw_source_totals.items()
                        },
                    }
                self._feedback_observability = normalized_observability
                logger.info(f"Loaded {len(self._pattern_priors)} pattern priors cache from {self._priors_path}")
        except Exception as e:
            logger.warning(f"Failed to load priors from {self._priors_path}: {e}")
    
    def _save_priors(self) -> None:
        """Save pattern priors to disk."""
        if not self._priors_path:
            return
        
        try:
            # Ensure parent directory exists
            self._priors_path.parent.mkdir(parents=True, exist_ok=True)

            with self._priors_lock:
                priors_snapshot = {
                    pattern: prior
                    for pattern, prior in self._pattern_priors.items()
                    if pattern and not is_signature_pattern_key(pattern)
                }
                context_priors_snapshot = {
                    context_key: {
                        pattern: prior
                        for pattern, prior in scoped.items()
                        if pattern and not is_signature_pattern_key(pattern)
                    }
                    for context_key, scoped in self._context_pattern_priors.items()
                    if context_key
                }
            feedback_counts_snapshot = {
                pattern: counts
                for pattern, counts in self._local_feedback_counts.items()
                if pattern and not is_signature_pattern_key(pattern)
            }
            context_feedback_counts_snapshot = {
                context_key: {
                    pattern: counts
                    for pattern, counts in scoped.items()
                    if pattern and not is_signature_pattern_key(pattern)
                }
                for context_key, scoped in self._context_feedback_counts.items()
                if context_key
            }
            severity_snapshot = {
                pattern: state
                for pattern, state in self._severity_calibration.items()
                if pattern and not is_signature_pattern_key(pattern)
            }
            feedback_observability_snapshot = {
                pattern: state
                for pattern, state in self._feedback_observability.items()
                if pattern and not is_signature_pattern_key(pattern)
            }

            with open(self._priors_path, 'w') as f:
                json.dump({
                    "pattern_priors": priors_snapshot,
                    "pattern_priors_by_context": context_priors_snapshot,
                    "feedback_counts": feedback_counts_snapshot,
                    "feedback_counts_by_context": context_feedback_counts_snapshot,
                    "severity_calibration": severity_snapshot,
                    "feedback_observability": feedback_observability_snapshot,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
            logger.debug(f"Saved {len(priors_snapshot)} pattern priors to {self._priors_path}")
        except Exception as e:
            logger.warning(f"Failed to save priors to {self._priors_path}: {e}")

    def reset_feedback_state(self) -> None:
        with self._priors_lock:
            self._pattern_priors.clear()
            self._context_pattern_priors.clear()
        self._local_feedback_counts.clear()
        self._context_feedback_counts.clear()
        self._severity_calibration.clear()
        self._feedback_observability.clear()
        self._save_priors()

    # ------------------------------------------------------------------
    # Adaptive scoring: new methods (Improvements 1-7)
    # ------------------------------------------------------------------

    def set_rl_agent(self, rl_agent: Any) -> None:
        """Wire the RL agent for adaptive weight tuning (Improvement 1).

        Called by the orchestrator after construction so the scorer can
        apply RL-recommended weight adjustments per event.
        """
        self._rl_agent = rl_agent
        logger.info("RL agent wired into scorer (adaptive weights enabled)")

    def set_model_retrained_at(self, epoch_seconds: float) -> None:
        """Record when the SeedModel was last retrained (Improvement 7).

        Used for model-age decay: weight of classical rule decreases
        over time as the model goes stale, shifting reliance to pattern
        rules and priors which update in real-time from feedback.
        """
        self._model_last_retrained_at = epoch_seconds

    def set_sindit_provider(self, provider: Any) -> None:
        """Set the SINDIT context provider for enriching weight profiles."""
        self._sindit_provider = provider

    # --- Improvement 4: Performance-adjusted weights ---

    def _performance_adjusted_weights(
        self,
        w_classical: float,
        w_harmonic: float,
        w_pattern: float,
        w_anomaly: float,
        w_historical: float,
    ) -> Tuple[float, float, float, float, float]:
        """Adjust weights proportionally to each rule's F1 score.

        Rules with high precision/recall get amplified; rules that
        consistently fire on dismissed events get dampened.  Requires
        at least 10 feedback samples to activate.
        """
        rule_f1 = {}
        for rule in self._rules:
            perf = self._rule_performance.get(rule.name)
            if perf and perf.n_samples >= 10:
                rule_f1[rule.name] = max(0.1, perf.f1)  # Floor to avoid zeroing out
            else:
                rule_f1[rule.name] = 0.5  # Neutral when insufficient data

        total_f1 = sum(rule_f1.values())
        if total_f1 <= 0:
            return w_classical, w_harmonic, w_pattern, w_anomaly, w_historical

        lr = self.config.weight_adaptation_rate

        # Blend: (1-lr)*base_weight + lr*(f1_proportion * sum_base_weights)
        sum_w = w_classical + w_harmonic + w_pattern + w_anomaly + w_historical
        adjusted = {}
        mapping = {
            "classical_alert": ("w_classical", w_classical),
            "harmonic_alert": ("w_harmonic", w_harmonic),
            "pattern_match": ("w_pattern", w_pattern),
            "anomaly_deviation": ("w_anomaly", w_anomaly),
            "historical_prior": ("w_historical", w_historical),
        }
        for rule_name, (label, base_w) in mapping.items():
            f1_share = rule_f1.get(rule_name, 0.5) / total_f1
            target_w = f1_share * sum_w
            adjusted[label] = (1.0 - lr) * base_w + lr * target_w

        return (
            adjusted["w_classical"],
            adjusted["w_harmonic"],
            adjusted["w_pattern"],
            adjusted["w_anomaly"],
            adjusted["w_historical"],
        )

    def record_rule_feedback(
        self,
        triggered_rules: List[str],
        was_confirmed: bool,
    ) -> None:
        """Record feedback for per-rule precision/recall tracking (Improvement 4).

        Called by the feedback handler after confirm/dismiss.  For each rule,
        we record whether the rule fired AND whether the operator confirmed.
        """
        if not self.config.enable_rule_performance_tracking:
            return

        all_rule_names = {rule.name for rule in self._rules}
        for rule_name in all_rule_names:
            fired = rule_name in triggered_rules
            perf = self._rule_performance.get(rule_name)
            if perf is None:
                perf = RulePerformance(
                    _history=deque(maxlen=self.config.rule_performance_window)
                )
                self._rule_performance[rule_name] = perf
            perf.record(fired, was_confirmed)

    def record_model_feedback(
        self,
        *,
        triggered_rules: List[str],
        was_confirmed: bool,
        external_signals: Optional[Dict[str, Any]] = None,
        cutting_context: Optional[CuttingContext] = None,
    ) -> None:
        """Persist feedback outcomes for model-emitted signals that were evaluated.

        Records under both the global aggregate and — when a cutting context is
        available — the context scope (plan 1.1), so trust quiets selectively by
        operating regime/tool/material instead of uniformly.
        """
        from ..model_confidence import record_model_feedback_outcome

        del triggered_rules
        context_key = self._context_profile_key(cutting_context)
        if context_key == "_global":
            context_key = None  # let model_confidence use its own global scope
        signals = external_signals or {}
        thresholds = {
            "anomaly_detector_score": float(self.config.anomaly_score_threshold),
            "harmonic_context_score": 0.5,
            "breakage_prediction": 0.5,
        }

        for signal_name, threshold in thresholds.items():
            if signal_name not in signals:
                continue
            try:
                value = float(signals.get(signal_name))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue

            record_model_feedback_outcome(
                model_fired=value > threshold,
                was_confirmed=was_confirmed,
                path=self._model_confidence_paths[signal_name],
                context_key=context_key,
            )

    def get_rule_performance(self) -> Dict[str, Dict[str, Any]]:
        """Return performance stats for all rules (for diagnostics/UI)."""
        return {
            name: perf.to_dict()
            for name, perf in self._rule_performance.items()
        }

    # --- Improvement 5: Context-conditioned weight profiles ---

    @staticmethod
    def _context_profile_key(context: Optional[CuttingContext]) -> str:
        """Build a hierarchical key for context-conditioned weight profiles.

        Uses the same dimensions as prior context keys but returns a single
        string suitable for dict lookup.  Falls back to ``"_global"`` when
        no context is available.
        """
        if context is None:
            return "_global"

        parts: List[str] = []
        regime = getattr(context, "operating_regime", None)
        if regime:
            parts.append(regime.value if hasattr(regime, "value") else str(regime))
        tool = getattr(context, "tool_type", None)
        if tool:
            parts.append(str(tool).strip())
        material = getattr(context, "workpiece_material", None)
        if material:
            parts.append(str(material).strip())

        return "|".join(parts) if parts else "_global"

    def _get_weight_profile(
        self,
        ctx_key: str,
        context: Optional[CuttingContext] = None,
    ) -> WeightProfile:
        """Get or create a weight profile for a context key.

        If SINDIT context is available, enrich the profile with digital-twin
        metadata (asset IRI, machine state) for traceability.
        """
        if ctx_key not in self._weight_profiles:
            # Initialise from config defaults
            self._weight_profiles[ctx_key] = WeightProfile(
                classical=self.config.weight_classical_alert,
                harmonic=self.config.weight_harmonic_alert,
                pattern=self.config.weight_pattern_rule,
                anomaly=self.config.weight_anomaly_deviation,
                historical=self.config.weight_historical_prior,
            )

        profile = self._weight_profiles[ctx_key]

        # Enrich with SINDIT metadata if available (first time only)
        if profile.sindit_asset_iri is None and context is not None:
            # Check for SINDIT- provided fields in the context
            machine_id = getattr(context, "machine_id", None)
            machine_state = getattr(context, "machine_state", None)
            extra = getattr(context, "extra", {}) or {}
            sindit_iri = extra.get("sindit_asset_iri")
            if sindit_iri:
                profile.sindit_asset_iri = sindit_iri
            elif machine_id:
                profile.sindit_asset_iri = f"urn:lfl:asset:{machine_id}"
            if machine_state:
                profile.sindit_machine_state = str(machine_state)

        return profile

    def update_weight_profile_from_feedback(
        self,
        context: Optional[CuttingContext],
        triggered_rules: List[str],
        was_confirmed: bool,
    ) -> None:
        """Nudge the context weight profile after operator feedback (Improvement 5).

        If a rule fired and the operator confirmed → increase that rule's weight.
        If a rule fired and the operator dismissed → decrease that rule's weight.
        If a rule didn't fire and the operator confirmed → it missed — increase slightly.

        Only applies nudges when the profile has accumulated at least 5
        feedback events, to avoid volatile weight swings from sparse data
        (Issue #16 fix, 2026-04-14).
        """
        ctx_key = self._context_profile_key(context)
        profile = self._get_weight_profile(ctx_key, context)
        profile.n_feedbacks += 1

        # Guard: require minimum feedback history before adjusting weights.
        # With fewer than MIN_PROFILE_FEEDBACKS samples the profile is
        # statistically unreliable and nudges could be counter-productive.
        _MIN_PROFILE_FEEDBACKS = 5
        if profile.n_feedbacks < _MIN_PROFILE_FEEDBACKS:
            logger.debug(
                "Weight profile '%s' has only %d feedbacks (< %d) — skipping nudge",
                ctx_key, profile.n_feedbacks, _MIN_PROFILE_FEEDBACKS,
            )
            return

        lr = self.config.profile_learning_rate
        rule_weight_map = {
            "classical_alert": "classical",
            "harmonic_alert": "harmonic",
            "pattern_match": "pattern",
            "anomaly_deviation": "anomaly",
            "historical_prior": "historical",
        }

        for rule_name, attr in rule_weight_map.items():
            current = getattr(profile, attr)
            fired = rule_name in triggered_rules

            if fired and was_confirmed:
                # Rule correctly identified — increase weight
                new_val = current + lr * (1.0 - current)
            elif fired and not was_confirmed:
                # False alarm from this rule — decrease weight
                new_val = current - lr * current
            elif not fired and was_confirmed:
                # Rule missed — small increase to make it more sensitive next time
                new_val = current + (lr * 0.3) * (1.0 - current)
            else:
                # Didn't fire, wasn't needed — no change
                continue

            setattr(profile, attr, max(0.05, min(1.0, new_val)))

        # Persist updated profiles
        self._save_weight_profiles()

    def record_feedback_for_adaptive_thresholds(
        self,
        score: float,
        action: str,
        was_confirmed: bool,
        *,
        weight: float = 1.0,
    ) -> None:
        """Feed outcome to adaptive threshold tracker (Improvement 6)."""
        if self.config.enable_adaptive_thresholds:
            self._adaptive_thresholds.record_feedback(score, action, was_confirmed, weight=weight)

    def record_rl_feedback(
        self,
        feedback_action: str,
        was_alerted: bool,
        external_signals: Optional[Dict[str, Any]] = None,
        context: Optional[CuttingContext] = None,
    ) -> None:
        """Close the RL learning loop after operator feedback (Improvement 1).

        Computes reward and updates the RL agent's Q-table.
        """
        if self._rl_agent is None:
            return

        try:
            from ..processing.classical_models import RLState

            state = RLState(
                seed_model_score=float((external_signals or {}).get("anomaly_detector_score", 0.0)),
                harmonic_score=float((external_signals or {}).get("harmonic_context_score", 0.0)),
                operating_regime=(
                    context.operating_regime.value
                    if context and context.operating_regime else "unknown"
                ),
                tool_type=getattr(context, "tool_type", None) or "unknown",
                material=getattr(context, "workpiece_material", None) or "unknown",
            )

            reward = self._rl_agent.compute_reward(feedback_action, was_alerted)
            action_idx, _ = self._rl_agent.select_action(state)
            self._rl_agent.update(state, action_idx, reward)

            # Persist RL agent state
            if hasattr(self._rl_agent, "save"):
                rl_path = None
                if self._priors_path:
                    rl_path = self._priors_path.parent / "rl_agent.json"
                if rl_path:
                    self._rl_agent.save(rl_path)
        except Exception as e:
            logger.debug("RL feedback update failed: %s", e)

    # --- Weight profile persistence ---

    def _load_weight_profiles(self) -> None:
        """Load context-conditioned weight profiles from disk."""
        if not self._profiles_path or not self._profiles_path.exists():
            return
        try:
            with open(self._profiles_path, 'r') as f:
                data = json.load(f)
            for key, profile_dict in data.get("profiles", {}).items():
                self._weight_profiles[key] = WeightProfile.from_dict(profile_dict)
            logger.info(
                "Loaded %d weight profiles from %s",
                len(self._weight_profiles), self._profiles_path,
            )
        except Exception as e:
            logger.warning("Failed to load weight profiles: %s", e)

    def _save_weight_profiles(self) -> None:
        """Save context-conditioned weight profiles to disk."""
        if not self._profiles_path:
            return
        try:
            self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._profiles_path, 'w') as f:
                json.dump({
                    "profiles": {
                        k: v.to_dict() for k, v in self._weight_profiles.items()
                    },
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save weight profiles: %s", e)

    def get_weight_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Return all weight profiles (for diagnostics/UI)."""
        return {k: v.to_dict() for k, v in self._weight_profiles.items()}

    def get_adaptive_thresholds(self) -> Dict[str, Any]:
        """Return current adaptive threshold state (for diagnostics/UI)."""
        return self._adaptive_thresholds.to_dict()

    def get_model_trust_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics for the classical model-confidence feedback state."""
        from ..model_confidence import get_model_confidence_diagnostics

        return get_model_confidence_diagnostics(self._model_confidence_path)

    def get_scoring_diagnostics(self) -> Dict[str, Any]:
        """Return full diagnostics for the adaptive scoring system."""
        return {
            "rule_performance": self.get_rule_performance(),
            "weight_profiles": self.get_weight_profiles(),
            "adaptive_thresholds": self.get_adaptive_thresholds(),
            "model_feedback_paths": {k: str(v) for k, v in self._model_confidence_paths.items()},
            "rl_agent_active": self._rl_agent is not None,
            "rl_agent_stats": (
                self._rl_agent.stats if self._rl_agent and hasattr(self._rl_agent, "stats") else None
            ),
            "model_age_decay_active": self._model_last_retrained_at is not None,
            "prior_mode": self.config.prior_mode,
            "sindit_provider_active": self._sindit_provider is not None,
        }

    # --- Graph-aware: query Neo4j for related context profiles ---

    def _query_graph_for_context_weights(
        self, ctx_key: str, context: Optional[CuttingContext]
    ) -> Optional[WeightProfile]:
        """Query the knowledge graph for weight data from similar contexts.

        When a new operating context is encountered for the first time,
        we check if the Neo4j graph has Pattern nodes with feedback in
        similar contexts (e.g., same tool type but different material).
        If found, we use those as a warm-start for the new profile.
        """
        if not self.feedback_store or not hasattr(self.feedback_store, "_run"):
            return None  # Only available with Neo4j backend

        if context is None:
            return None

        try:
            # Find patterns that have been scored in similar contexts
            regime = (
                context.operating_regime.value
                if context and context.operating_regime else None
            )
            tool = getattr(context, "tool_type", None)

            if not regime and not tool:
                return None

            # Look for existing profiles that share partial context
            for existing_key, existing_profile in self._weight_profiles.items():
                if existing_key == ctx_key or existing_key == "_global":
                    continue
                # Check for partial overlap
                parts = existing_key.split("|")
                if regime and regime in parts:
                    return WeightProfile(
                        classical=existing_profile.classical,
                        harmonic=existing_profile.harmonic,
                        pattern=existing_profile.pattern,
                        anomaly=existing_profile.anomaly,
                        historical=existing_profile.historical,
                    )
                if tool and tool in parts:
                    return WeightProfile(
                        classical=existing_profile.classical,
                        harmonic=existing_profile.harmonic,
                        pattern=existing_profile.pattern,
                        anomaly=existing_profile.anomaly,
                        historical=existing_profile.historical,
                    )
        except Exception as e:
            logger.debug("Graph context weight query failed: %s", e)

        return None

    # ------------------------------------------------------------------
    # Prior sandboxing — isolate experiment runs from production priors
    # ------------------------------------------------------------------

    def snapshot_priors(self, run_id: str) -> Path:
        """Copy current priors to a per-experiment snapshot and redirect writes.

        Returns the snapshot path.  The live scorer will operate on the copy
        until ``restore_priors()`` is called, so experiment feedback doesn't
        contaminate the production priors file.
        """
        if not self._priors_path:
            raise RuntimeError("Cannot snapshot priors: no priors_path configured")

        snapshot_dir = self._priors_path.parent / "experiment_snapshots" / run_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / self._priors_path.name

        # Copy current priors (or create empty) to snapshot location
        if self._priors_path.exists():
            shutil.copy2(self._priors_path, snapshot_path)
        else:
            with open(snapshot_path, "w") as f:
                json.dump({
                    "pattern_priors": {},
                    "feedback_counts": {},
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)

        # Store the original path so we can restore later
        if not hasattr(self, "_original_priors_path"):
            self._original_priors_path: Optional[Path] = None
        if self._original_priors_path is None:
            self._original_priors_path = self._priors_path

        # Redirect to sandbox
        self._priors_path = snapshot_path
        self._load_priors()
        logger.info("Priors sandboxed for run %s → %s", run_id, snapshot_path)
        return snapshot_path

    def restore_priors(self) -> None:
        """Restore the scorer to point at the original (production) priors file.

        Call this after an experiment finishes to stop writing to the snapshot.
        """
        original = getattr(self, "_original_priors_path", None)
        if original is None:
            logger.debug("restore_priors called but no original path stored — no-op")
            return

        self._priors_path = original
        self._original_priors_path = None
        self._load_priors()
        logger.info("Priors restored to production path → %s", self._priors_path)

    def get_sandbox_diff(self, run_id: str) -> Dict[str, Any]:
        """Compare the sandbox snapshot for *run_id* against production priors.

        Returns ``{"added": {...}, "changed": {...}, "removed": [...]}`` where
        *changed* maps pattern keys to ``{"before": float, "after": float}``.
        """
        if not self._priors_path:
            return {"added": {}, "changed": {}, "removed": []}

        original = getattr(self, "_original_priors_path", None) or self._priors_path
        snapshot_path = original.parent / "experiment_snapshots" / run_id / original.name

        def _load(p: Path) -> Dict[str, float]:
            if not p.exists():
                return {}
            try:
                with open(p) as f:
                    return json.load(f).get("pattern_priors", {})
            except Exception:
                return {}

        before = _load(original)
        after = _load(snapshot_path)

        added = {k: v for k, v in after.items() if k not in before}
        removed = [k for k in before if k not in after]
        changed = {}
        for k in set(before) & set(after):
            if abs(before[k] - after[k]) > 1e-6:
                changed[k] = {"before": before[k], "after": after[k]}

        return {"added": added, "changed": changed, "removed": removed}

    @property
    def is_sandboxed(self) -> bool:
        """Return True if priors are currently redirected to a sandbox."""
        return getattr(self, "_original_priors_path", None) is not None

    def _get_baseline(
        self, 
        session_id: Optional[str],
        current_metrics: Optional[WindowMetrics]
    ) -> Optional["_RollingBaseline"]:
        """Get or initialize baseline for session."""
        if not session_id:
            return None
        
        if session_id not in self._session_baselines:
            self._session_baselines[session_id] = _RollingBaseline(
                window_size=self.config.baseline_window_size
            )
        
        baseline = self._session_baselines[session_id]
        
        # Auto-update baseline if metrics provided
        if current_metrics:
            baseline.add(current_metrics)
        
        return baseline


# [PROTOTYPE_LLM_MEMORY_V1] - Internal data structures

@dataclass
class _EvaluationContext:
    """Context passed to rule evaluators."""
    patterns: List[PatternKey]
    metrics: Optional[WindowMetrics]
    cutting_context: Optional[CuttingContext]
    session_id: Optional[str]
    external_signals: Dict[str, Any]
    config: SignificanceConfig
    pattern_priors: Dict[str, float]
    baseline: Optional["_RollingBaseline"]


@dataclass
class _RuleResult:
    """Result from a single rule evaluation."""
    triggered: bool
    score: float  # 0.0 - 1.0
    reasons: List[str]
    protective_score: float = 0.0
    suppression_score: float = 0.0


class _RollingBaseline:
    """
    Rolling statistics for baseline computation.
    
    [PROTOTYPE_LLM_MEMORY_V1] - Simple buffer-based implementation.
    Could be replaced with online algorithms (Welford's) for efficiency.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buffer: List[np.ndarray] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
    
    def add(self, metrics: WindowMetrics) -> None:
        """Add metrics to baseline buffer."""
        vec = metrics.to_vector()
        self._buffer.append(vec)
        
        # Trim to window size
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]
        
        # Recompute stats
        if len(self._buffer) >= 5:  # Minimum samples
            arr = np.stack(self._buffer)
            self._mean = np.mean(arr, axis=0)
            self._std = np.std(arr, axis=0) + 1e-8  # Avoid division by zero
    
    def z_score(self, metrics: WindowMetrics) -> Optional[np.ndarray]:
        """Compute z-score of metrics relative to baseline."""
        if self._mean is None or self._std is None:
            return None
        vec = metrics.to_vector()
        return (vec - self._mean) / self._std
    
    def max_deviation(self, metrics: WindowMetrics) -> Optional[float]:
        """Maximum absolute z-score across all features."""
        z = self.z_score(metrics)
        if z is None:
            return None
        return float(np.max(np.abs(z)))
    
    @property
    def is_ready(self) -> bool:
        """Whether baseline has enough data."""
        return self._mean is not None


# [PROTOTYPE_LLM_MEMORY_V1] - Rule implementations

class _SignificanceRule:
    """Base class for significance rules."""
    name: str = "base_rule"
    
    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        raise NotImplementedError
    
    def weight(self, config: SignificanceConfig) -> float:
        raise NotImplementedError


class _ClassicalAlertRule(_SignificanceRule):
    """
    Rule: Check for external classical model alerts.
    
    External signals expected:
    - breakage_prediction: float (0-1 probability)
    - tool_wear_estimate: float (0-1 remaining life)
    - anomaly_detector_score: float (0-1)
    - model_confidence: float (0-1) — scales the rule's contribution
    - chatter_severity: float (0-1) — chatter detection confidence
    - slip_likelihood: float (0-1) — workpiece slip probability
    
    When model_confidence is provided the rule output is scaled so that
    a cold (low confidence) model contributes less to the composite score.
    This avoids a poorly-calibrated seed model overwhelming the pattern +
    prior signals.
    """
    name = "classical_alert"
    
    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        signals = ctx.external_signals
        triggered = False
        score = 0.0
        reasons = []
        
        # Model confidence (default 1.0 = fully trusted)
        model_confidence = float(signals.get("model_confidence", 1.0))
        
        # Breakage prediction
        if signals.get("breakage_prediction", 0.0) > 0.5:
            triggered = True
            score = max(score, signals["breakage_prediction"])
            reasons.append(f"Breakage prediction: {signals['breakage_prediction']:.2f}")
        
        # Tool wear
        if signals.get("tool_wear_estimate", 1.0) < 0.2:
            triggered = True
            score = max(score, 1.0 - signals["tool_wear_estimate"])
            reasons.append(f"Low tool life remaining: {signals['tool_wear_estimate']:.2f}")
        
        # Generic anomaly detector
        if signals.get("anomaly_detector_score", 0.0) > ctx.config.anomaly_score_threshold:
            triggered = True
            score = max(score, signals["anomaly_detector_score"])
            reasons.append(f"Anomaly detector score: {signals['anomaly_detector_score']:.2f}")

        # Chatter severity (from external vibration analysis)
        if signals.get("chatter_severity", 0.0) > 0.5:
            triggered = True
            score = max(score, signals["chatter_severity"])
            reasons.append(f"Chatter severity: {signals['chatter_severity']:.2f}")

        # Workpiece slip likelihood
        if signals.get("slip_likelihood", 0.0) > 0.5:
            triggered = True
            score = max(score, signals["slip_likelihood"])
            reasons.append(f"Workpiece slip likelihood: {signals['slip_likelihood']:.2f}")
        
        # Scale by model confidence — cold model contributes less
        if triggered and model_confidence < 1.0:
            score *= model_confidence
            reasons.append(f"Model confidence: {model_confidence:.2f}")
        
        return _RuleResult(triggered=triggered, score=score, reasons=reasons)
    
    def weight(self, config: SignificanceConfig) -> float:
        return config.weight_classical_alert


class _HarmonicAlertRule(_SignificanceRule):
    """Rule: check harmonic scorer output independently from the classical bucket."""

    name = "harmonic_alert"

    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        hc_score = ctx.external_signals.get("harmonic_context_score", 0.0)
        hc_threshold = ctx.external_signals.get("harmonic_context_threshold", 0.5)
        numeric_threshold = float(hc_threshold) if isinstance(hc_threshold, (int, float)) else 0.5
        if isinstance(hc_score, (int, float)) and hc_score >= numeric_threshold:
            return _RuleResult(
                triggered=True,
                score=float(hc_score),
                reasons=[f"Harmonic context score: {hc_score:.2f} (threshold {numeric_threshold:.2f})"],
            )
        return _RuleResult(triggered=False, score=0.0, reasons=[])

    def weight(self, config: SignificanceConfig) -> float:
        return config.weight_harmonic_alert


class _PatternMatchRule(_SignificanceRule):
    """
    Rule: Check for significant pattern types.
    
    Matches both generic anomaly patterns and physics-based machining
    fault patterns (tool breakage, chatter, chip adhesion, workpiece slip).
    """
    name = "pattern_match"
    
    # Patterns that are always significant
    CRITICAL_PATTERNS = {
        PatternType.ANOMALY,
        PatternType.FAULT,
    }
    
    # Pattern key substrings that indicate significance
    SIGNIFICANT_SUBSTRINGS = [
        "CHATTER",
        "BREAKAGE",  # backward compat: live pipeline may still emit these
        "SPIKE_RATE:>10",
        # Fault-specific patterns (physics-based)
        "fault:tool_breakage",
        "fault:chatter",
        "fault:chip_adhesion",
        "fault:workpiece_slip",
        # Supporting spectral/temporal indicators
        "spectral:hf_burst",
        "temporal:impulsive_burst",
        "temporal:periodicity_loss",
        "spectral:modulated_vibration",
        "spectral:irregular_tooth_passing",
        "spectral:spindle_freq_shift",
        "spectral:tp_harmonic",
        # Renamed observable patterns (stoppage experiment)
        "SPINDLE_POWER_SURGE",
        "VIBRATION_REGIME_SHIFT",
        "FEED_OVERRIDE_DROP",
        "SENSOR_DECORRELATION",
        "SPINDLE_LOAD_RAMP",
        "FEED_STALL",
        # Legacy names (backward compat with live pipeline)
        "BREAKAGE_POWER_SPIKE",
        "BREAKAGE_VIB_SHIFT",
        "BREAKAGE_FEED_OVERRIDE_DROP",
        "BREAKAGE_DECORRELATION",
    ]

    # Single-feature "supporting" indicators that must not alert alone (plan 1.7).
    # Declared here for the CANONICAL keys the live pipeline emits, because the
    # experiment registry is keyed by raw name (HF_ENERGY_BURST) and does not
    # resolve the canonical form (spectral:hf_burst) — same reason FAULT_SEVERITY
    # below carries canonical keys. Keep this in sync with the registry's
    # requires_corroboration=True defs.
    CORROBORATION_REQUIRED: set = {
        "spectral:hf_burst",
        "temporal:impulsive_burst",
    }

    # Severity mapping for fault patterns
    FAULT_SEVERITY: Dict[str, float] = {
        "fault:tool_breakage": 0.95,
        "fault:chatter": 0.80,
        "fault:chip_adhesion": 0.65,
        "fault:workpiece_slip": 0.75,
        # Supporting single-feature indicators (plan 1.7 / 1.11): alert-band,
        # corroboration-gated. Live emits these canonical keys directly.
        "spectral:hf_burst": 0.65,
        "temporal:impulsive_burst": 0.65,
        # Renamed observable patterns (stoppage experiment)
        "SPINDLE_POWER_SURGE": 0.90,
        "VIBRATION_REGIME_SHIFT": 0.85,
        "FEED_OVERRIDE_DROP": 0.75,
        "SENSOR_DECORRELATION": 0.80,
        "SPINDLE_LOAD_RAMP": 0.70,
        "FEED_STALL": 0.65,
        # Legacy names (backward compat with live pipeline)
        "BREAKAGE_POWER_SPIKE": 0.90,
        "BREAKAGE_VIB_SHIFT": 0.85,
        "BREAKAGE_FEED_OVERRIDE_DROP": 0.75,
        "BREAKAGE_DECORRELATION": 0.80,
    }
    
    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        triggered = False
        score = 0.0
        protective_score = 0.0
        suppression_score = 0.0
        reasons = []
        registry = get_pattern_registry()

        # Co-occurrence gating (plan 1.7): collect each "significant" pattern's
        # (severity, requires_corroboration) so a lone corroboration-requiring
        # indicator can be capped to STORE-band after the loop, while ≥2 distinct
        # patterns (or a corroborating rule, counted separately in the fused
        # score) let it contribute full severity.
        sig_hits: List[Tuple[float, bool]] = []

        for pattern in ctx.patterns:
            canonical_key = normalize_pattern_key(pattern.key)
            pdef = registry.get(canonical_key) or registry.get(pattern.key)
            polarity = getattr(pdef, "polarity", "fault_supporting") if pdef else "fault_supporting"
            requires_corr = bool(getattr(pdef, "requires_corroboration", False)) if pdef else False
            # Also honor the canonical-key declaration (live pipeline emits
            # canonical keys the experiment registry can't resolve).
            if canonical_key in self.CORROBORATION_REQUIRED:
                requires_corr = True
            severity = getattr(pdef, "severity", None) if pdef else None
            if severity is None:
                severity = self.FAULT_SEVERITY.get(pattern.key, self.FAULT_SEVERITY.get(canonical_key, 0.7))

            if canonical_key.startswith("suppressed:") or pattern.key.startswith("suppressed:"):
                triggered = True
                suppression_score = max(
                    suppression_score,
                    float(pattern.confidence if pattern.confidence is not None else severity),
                )
                reasons.append(f"Suppression pattern: {pattern.key}")
                continue

            if polarity == "protective":
                triggered = True
                protective_score = max(protective_score, float(severity))
                reasons.append(f"Protective pattern: {canonical_key}")
                continue

            if is_signature_pattern_key(canonical_key):
                triggered = True
                hit = float(pattern.confidence if pattern.confidence is not None else severity)
                sig_hits.append((hit, requires_corr))
                reasons.append(f"Significant pattern: {canonical_key}")
                continue

            # Check pattern type
            if pattern.pattern_type in self.CRITICAL_PATTERNS:
                triggered = True
                score = max(score, 0.8)
                reasons.append(f"Critical pattern type: {pattern.pattern_type.value}")

            # Check pattern key substrings — against the canonical key too, so
            # legacy keys routed through data/pattern_aliases.json (e.g.
            # HF_ENERGY_BURST -> spectral:hf_burst) are recognized the same as
            # their canonical form.
            for substr in self.SIGNIFICANT_SUBSTRINGS:
                if substr.lower() in pattern.key.lower() or substr.lower() in canonical_key.lower():
                    triggered = True
                    # Use fault severity if available, otherwise default
                    sig_hits.append((float(severity), requires_corr))
                    reasons.append(f"Significant pattern: {pattern.key}")
                    break
            
            # Check force ratio (chatter indicator) — any bucket whose lower bound
            # reaches the chatter threshold (e.g. "RATIO_Fx_Fy:>5" or "RATIO_Fx_Fy:5-10").
            if "RATIO" in pattern.key:
                lower_bound = _parse_ratio_bucket_lower_bound(pattern.key)
                if lower_bound is not None and lower_bound >= ctx.config.chatter_ratio_threshold:
                    triggered = True
                    score = max(score, 0.6)
                    reasons.append(f"High force ratio: {pattern.key}")

        # Resolve the significant-pattern contribution with co-occurrence gating.
        if sig_hits:
            top_score = max(h[0] for h in sig_hits)
            n_distinct = len(sig_hits)
            gate_on = getattr(ctx.config, "enable_cooccurrence_gating", True)
            top_needs_corr = max(sig_hits, key=lambda h: h[0])[1]
            if gate_on and n_distinct == 1 and top_needs_corr:
                # Lone supporting indicator — cap to STORE-band. It still escalates
                # when the classical/anomaly rule agrees (their weighted scores add
                # to the fused total) or when a 2nd pattern co-fires (n_distinct>1).
                cap = float(getattr(ctx.config, "supporting_uncorroborated_cap", 0.5))
                gated = min(top_score, cap)
                if gated < top_score:
                    reasons.append("Supporting pattern uncorroborated — capped to STORE-band")
                top_score = gated
            score = max(score, top_score)

        return _RuleResult(
            triggered=triggered,
            score=score,
            reasons=reasons,
            protective_score=protective_score,
            suppression_score=suppression_score,
        )
    
    def weight(self, config: SignificanceConfig) -> float:
        return config.weight_pattern_rule


class _AnomalyDeviationRule(_SignificanceRule):
    """
    Rule: Statistical deviation from session baseline.
    """
    name = "anomaly_deviation"
    
    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        if ctx.metrics is None or ctx.baseline is None or not ctx.baseline.is_ready:
            return _RuleResult(triggered=False, score=0.0, reasons=[])
        
        max_z = ctx.baseline.max_deviation(ctx.metrics)
        if max_z is None:
            return _RuleResult(triggered=False, score=0.0, reasons=[])
        
        z_thr = ctx.config.anomaly_z_threshold
        triggered = max_z > z_thr
        # Normalize so a *barely*-triggered anomaly scores ~0 (was 0.5), ramping
        # to 1.0 at 2× threshold: z=thr -> 0.0, z=2·thr -> 1.0. Stops ordinary
        # threshold-grazing blips from crossing the store threshold on their own.
        score = min(1.0, max(0.0, (max_z - z_thr) / z_thr))
        
        reasons = []
        if triggered:
            reasons.append(f"Statistical anomaly: max z-score = {max_z:.2f}")
        
        return _RuleResult(triggered=triggered, score=score, reasons=reasons)
    
    def weight(self, config: SignificanceConfig) -> float:
        return config.weight_anomaly_deviation


class _HistoricalPriorRule(_SignificanceRule):
    """
    Rule: Use learned priors from historical feedback.
    
    [PROTOTYPE_LLM_MEMORY_V1] - Uses in-memory priors.
    Should integrate with PatternKnowledgeStore.
    """
    name = "historical_prior"
    
    def evaluate(self, ctx: _EvaluationContext) -> _RuleResult:
        if not ctx.patterns or not ctx.pattern_priors:
            return _RuleResult(triggered=False, score=0.0, reasons=[])
        
        # Get highest (and lowest) prior among patterns
        max_prior = 0.0
        max_pattern = None
        min_prior = 1.0
        min_pattern = None

        for pattern in ctx.patterns:
            canonical = normalize_pattern_key(pattern.key)
            if is_signature_pattern_key(canonical):
                continue
            prior = ctx.pattern_priors.get(canonical, ctx.pattern_priors.get(pattern.key, 0.5))
            if prior > max_prior:
                max_prior = prior
                max_pattern = pattern.key
            if prior < min_prior:
                min_prior = prior
                min_pattern = pattern.key

        # Trigger if any pattern has learned significance above neutral
        # Lower threshold (0.52) means even a single confirmation can contribute.
        # With sub-neutral damping enabled (ISS-13), also trigger on a clearly
        # below-neutral prior so the multiplicative damping path can apply
        # (evidence-volume gating happens in the prior derivation, and
        # confirmations always win over dismissals there).
        allow_subneutral = bool(getattr(ctx.config, "prior_allow_subneutral", False))
        triggered = max_prior > 0.51 or (allow_subneutral and min_prior < 0.49)
        score = max_prior

        reasons = []
        if max_prior > 0.51:
            reasons.append(f"Historical significance: {max_pattern} (prior={max_prior:.2f})")
        elif triggered:
            reasons.append(f"Historical damping: {min_pattern} (prior={min_prior:.2f})")
        
        return _RuleResult(triggered=triggered, score=score, reasons=reasons)
    
    def weight(self, config: SignificanceConfig) -> float:
        return config.weight_historical_prior


class _AADCombinerRule(_SignificanceRule):
    """Surfaces the feedback-trained AAD combiner's probability into scoring.

    The combiner (``processing/aad_combiner.py``) runs as a signal model and
    writes ``aad_score`` into external_signals; this rule contributes it to the
    fused score. Opt in via ``enabled_rules=[... "aad_combiner"]``.
    """
    name = "aad_combiner"

    def evaluate(self, ctx: "_EvaluationContext") -> _RuleResult:
        p = float(ctx.external_signals.get("aad_score", 0.0) or 0.0)
        triggered = p > 0.5
        return _RuleResult(
            triggered=triggered,
            score=p,
            reasons=[f"AAD combiner p={p:.2f}"] if triggered else [],
        )

    def weight(self, config: SignificanceConfig) -> float:
        return getattr(config, "weight_aad", 0.5)


# Register the built-in scoring rules (== the default pluggable model set).
# Third-party / experimental models register themselves the same way and are
# opted in via SignificanceConfig.enabled_rules.
register_scoring_rule("classical_alert", _ClassicalAlertRule)
register_scoring_rule("aad_combiner", _AADCombinerRule)
register_scoring_rule("harmonic_alert", _HarmonicAlertRule)
register_scoring_rule("pattern_match", _PatternMatchRule)
register_scoring_rule("anomaly_deviation", _AnomalyDeviationRule)
register_scoring_rule("historical_prior", _HistoricalPriorRule)


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function for easy instantiation
def create_scorer(config: Optional[SignificanceConfig] = None) -> SignificanceScorer:
    """Create a configured SignificanceScorer instance."""
    return SignificanceScorer(config)

"""Experiment configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Domain-driven defaults
# ---------------------------------------------------------------------------
# These constants are loaded from the active DomainConfig at import time.
# If the domain config module is unavailable (e.g. running standalone), we
# fall back to the original hardcoded CNC values.

try:
    from backend.agents.domain_config import get_active_domain as _get_domain
    _domain = _get_domain()
except Exception:  # pragma: no cover
    _domain = None  # type: ignore[assignment]

try:
    from .pattern_registry import get_registry as _get_pattern_registry
except Exception:  # pragma: no cover
    _get_pattern_registry = None  # type: ignore[assignment]


def _merge_pattern_keys(base_keys: List[str], registry_keys: List[str]) -> List[str]:
    """Return ordered union preserving the seeded key order first."""
    merged: List[str] = []
    seen = set()
    for key in list(base_keys) + list(registry_keys):
        if not isinstance(key, str) or not key or key in seen:
            continue
        merged.append(key)
        seen.add(key)
    return merged


def _registry_defaults() -> Tuple[List[str], Dict[str, float]]:
    """Return enabled registry pattern names and severities."""
    if _get_pattern_registry is None:
        return [], {}
    try:
        patterns = _get_pattern_registry().list_patterns(enabled_only=True)
    except Exception:  # pragma: no cover
        return [], {}
    return [p.name for p in patterns], {
        p.name: float(getattr(p, "severity", 0.8)) for p in patterns
    }

_BASE_PATTERN_KEYS: List[str] = (
    _domain.pattern_keys if _domain and _domain.pattern_keys else [
        # Builtin (original 4 renamed)
        "SPINDLE_POWER_SURGE",
        "VIBRATION_REGIME_SHIFT",
        "FEED_OVERRIDE_DROP",
        "SENSOR_DECORRELATION",
        # Domain-derived
        "SPINDLE_LOAD_RAMP",
        "FEED_STALL",
    ]
)
_REGISTRY_PATTERN_KEYS, _REGISTRY_FAULT_SEVERITY = _registry_defaults()

# Pattern keys tracked across phases (observable phenomena, not diagnoses).
# Preserve the original seeded order and append enabled registry-only names.
PATTERN_KEYS: List[str] = _merge_pattern_keys(_BASE_PATTERN_KEYS, _REGISTRY_PATTERN_KEYS)

# Backward-compat alias (deprecated — use PATTERN_KEYS)
BREAKAGE_PATTERN_KEYS = PATTERN_KEYS

# Fault severity values (from domain config or production defaults)
_BASE_FAULT_SEVERITY: Dict[str, float] = (
    _domain.fault_severity if _domain else {
        "SPINDLE_POWER_SURGE": 0.90,
        "VIBRATION_REGIME_SHIFT": 0.85,
        "FEED_OVERRIDE_DROP": 0.75,
        "SENSOR_DECORRELATION": 0.80,
        "SPINDLE_LOAD_RAMP": 0.70,
        "FEED_STALL": 0.65,
    }
)
FAULT_SEVERITY: Dict[str, float] = dict(_BASE_FAULT_SEVERITY)
for _pattern_key, _severity in _REGISTRY_FAULT_SEVERITY.items():
    FAULT_SEVERITY.setdefault(_pattern_key, _severity)
for _pattern_key in PATTERN_KEYS:
    FAULT_SEVERITY.setdefault(_pattern_key, 0.80)

# Columns that encode the stop event itself (data leakage)
LEAKY_COLUMNS: List[str] = (
    _domain.leaky_columns if _domain and _domain.leaky_columns else [
        "event_stop_duration_s",
        "event_spindle_rpm",
        "event_feed_rate",
        "event_feed_override",
        "feed_actual_min",
        "spindle_actual_min",
        "feed_override_min",
        "spindle_override_min",
    ]
)

# Metadata columns (not features)
METADATA_COLUMNS: List[str] = (
    _domain.metadata_columns if _domain and _domain.metadata_columns else [
        "sample_id", "label", "operation_id", "tool_number",
        "event_timestamp", "severity", "stop_type", "window_seconds",
        "gap_seconds", "sample_rate_hz", "window_entries",
        "trim_seconds_removed",
        # Site_a_line2 / breakage-specific metadata
        "timestamp", "session", "condition",
    ]
)

# A3: any column that is label-derived or metadata must never enter a model
# feature set. `severity`/`stop_type` in particular are computed FROM the stop
# and would leak the label perfectly. The set is the union of leaky + metadata;
# `assert_features_safe` is the hard guard called wherever a feature list is
# finalised, so an accidental widening fails loudly instead of leaking silently.
FORBIDDEN_FEATURE_COLUMNS: frozenset = frozenset(LEAKY_COLUMNS) | frozenset(METADATA_COLUMNS)


def assert_features_safe(feature_cols: "List[str]", *, where: str = "") -> None:
    """Raise if any selected feature is a forbidden (leaky/metadata) column."""
    leaked = sorted(set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise ValueError(
            f"Leaky/metadata columns in model feature set"
            f"{(' (' + where + ')') if where else ''}: {leaked}. "
            f"These are label-derived or metadata and must be excluded."
        )


# All four operations available in the casedata
ALL_OPERATIONS: List[str] = ["OF00001", "OF00002", "OF00003", "OF00004"]

# Leave-one-out rotation schemes  (train, test, eval)
ROTATION_SCHEMES: List[Tuple[List[str], str, str]] = [
    (["OF00001", "OF00002"], "OF00003", "OF00004"),
    (["OF00001", "OF00002"], "OF00004", "OF00003"),
    (["OF00003", "OF00004"], "OF00001", "OF00002"),
    (["OF00003", "OF00004"], "OF00002", "OF00001"),
]


@dataclass
class ExperimentConfig:
    """Configuration for a three-phase stoppage prediction experiment."""

    # --- Data paths -------------------------------------------------------
    # Computed dynamically from window_size_s and prediction_gap_s in
    # __post_init__.  The factory default covers the legacy 60 s / gap=0 case.
    features_csv: Path = field(
        default_factory=lambda: ROOT / "data" / "breakage_patterns" / "stoppage_features.csv"
    )
    output_dir: Path = field(
        default_factory=lambda: ROOT / "data" / "breakage_patterns" / "stoppage_experiment"
    )

    # --- Split assignment -------------------------------------------------
    train_ops: List[str] = field(default_factory=lambda: ["OF00001", "OF00002"])
    test_op: str = "OF00003"
    eval_op: str = "OF00004"

    # --- Phase 1: training ------------------------------------------------
    seed_model_filename: str = "stoppage_seed_train.joblib"
    baseline_priors_filename: str = "stoppage_priors_baseline.json"
    train_meta_filename: str = "stoppage_train_meta.json"

    # --- Scoring config ---------------------------------------------------
    store_threshold: float = 0.3
    alert_threshold: float = 0.6
    critical_threshold: float = 0.85

    # Scoring formula weights (match production SignificanceScorer)
    weight_classical_alert: float = 0.40
    weight_pattern_rule: float = 0.25
    weight_protective_pattern: float = 0.20
    weight_anomaly_deviation: float = 0.20
    weight_historical_prior: float = 0.30  # Aligned with backend scorer for faster convergence

    # Prior boost: MULTIPLICATIVE on base_score, not additive.
    # New formula: final = base_score * (1 + prior_boost_weight * (max_prior - 0.5))
    # With prior_boost_weight=2.0 and max_prior=0.8:
    #   multiplier = 1 + 2.0 * 0.3 = 1.6x  (60% boost to base score)
    # With prior_boost_weight=2.0 and max_prior=0.3 (dismissed):
    #   multiplier = 1 + 2.0 * (-0.2) = 0.6x  (40% penalty)
    prior_boost_weight: float = 2.0

    # Anomaly deviation rule: z-score threshold for triggering
    anomaly_z_threshold: float = 3.0

    # --- Pattern detection thresholds ---------------------------------
    # If calibrate_patterns_from_data=True (default), these are OVERRIDDEN
    # by percentile-based thresholds computed from the training set.
    # The hardcoded values below serve as fallbacks only.
    calibrate_patterns_from_data: bool = True
    pattern_calibration_normal_percentile: float = 95.0  # fires above p95 of normal
    pattern_power_spindle_delta_max: float = 15.0   # % of rated power
    pattern_power_y_delta_max: float = 10.0          # % of rated power
    pattern_vib_severity_x_delta_max: float = 0.8    # mm/s^2
    pattern_chatter_freq_x_slope_abs: float = 5.0    # Hz/s
    pattern_feed_override_delta_mean: float = -10.0   # % drop
    pattern_feed_override_min: float = 50.0           # % below this = suspicious
    pattern_corr_spindle_power_vib_x_low: float = 0.3 # normally > 0.6

    # Domain-expert pattern thresholds (calibrated from data when enabled)
    pattern_power_spindle_slope: float = 5.0            # SPINDLE_LOAD_RAMP
    pattern_feed_actual_range_ratio: float = 3.0        # FEED_STALL (range/mean ratio)
    pattern_power_xy_asymmetry: float = 0.6             # POWER_ASYMMETRY (|px-py|/(px+py))
    pattern_energy_total_slope: float = 3.0             # ENERGY_ACCUMULATION
    # Time-series derived pattern thresholds
    pattern_vib_severity_x_std: float = 1.0             # VARIANCE_EXPLOSION (vib)
    pattern_power_spindle_std: float = 5.0              # VARIANCE_EXPLOSION (power)
    pattern_power_spindle_slope_tr: float = 1.0         # TREND_REVERSAL
    pattern_vib_severity_x_iqr: float = 0.5             # AUTOCORRELATION_BREAK

    # --- Online model retraining on feedback ---------------------------
    # After N confirmed anomalies, retrain the SeedModel augmenting the
    # normal set with dismissed samples (known normal from operator) and
    # adjusting contamination to account for confirmed anomalies.
    online_retrain_enabled: bool = True
    online_retrain_min_feedback: int = 5  # min feedback events before first retrain
    online_retrain_interval: int = 5      # retrain every N feedback events after min

    # --- Supervised model (runs in parallel with unsupervised) ----------
    use_supervised_model: bool = True
    supervised_model_type: str = "random_forest"  # "random_forest" | "gradient_boosting"
    supervised_n_estimators: int = 100
    supervised_max_depth: int = 5
    supervised_model_filename: str = "stoppage_supervised.joblib"
    # Composite weighting: how much the supervised vs unsupervised model
    # contributes to the final score. Feedback shifts these weights.
    weight_supervised: float = 0.5
    weight_unsupervised: float = 0.5
    # When feedback confirms an anomaly, shift +weight_shift toward supervised;
    # when feedback dismisses, shift +weight_shift toward unsupervised.
    model_weight_shift_per_feedback: float = 0.02
    model_weight_shift_max: float = 0.30  # max cumulative shift from 50/50
    # Exclude leaky features that encode the stop event itself
    exclude_leaky_features: bool = True

    # --- Tool-level priors (lightweight knowledge graph) ------------------
    use_tool_priors: bool = True
    tool_prior_weight: float = 0.15  # weight in composite scoring
    tool_prior_default: float = 0.5  # prior for unseen tools
    tool_priors_filename: str = "stoppage_tool_priors.json"

    # --- Feedback ---------------------------------------------------------
    feedback_mode: str = "auto"  # "auto" | "interactive"
    feedback_user_id: str = "experiment"

    # Missed-event feedback: simulate operator catching FN.
    # Triggers on actual false negatives (label=positive AND not predicted)
    # independent of model score.  Set to 0.0 to disable.
    missed_event_feedback_rate: float = 0.5   # probability of feedback on FN
    secondary_threshold_offset: float = 0.30  # secondary = threshold - offset

    # Threshold adaptation: shift threshold based on accumulated feedback
    threshold_adaptation_rate: float = 0.01   # per feedback event
    threshold_adaptation_max: float = 0.15    # max cumulative shift

    # --- Feedback realism -------------------------------------------------
    # Delay buffer: feedback is applied N samples after the detection event.
    # 0 = instant (oracle, default), >0 = realistic delay.
    feedback_delay_samples: int = 0

    # Alert fatigue: response probability decays with cumulative alert count.
    # response_prob = feedback_response_rate * exp(-fatigue_decay * alert_count)
    # Set fatigue_decay=0.0 for constant rate (oracle, default).
    feedback_response_rate: float = 1.0       # base probability of responding
    feedback_fatigue_decay: float = 0.0       # exponential decay per alert

    # Confidence-dependent noise: borderline samples get noisier feedback.
    # noise = noise_rate_base + noise_rate_ambiguity * ambiguity
    # where ambiguity ∈ [0,1] = closeness to threshold.
    # Set both to 0.0 for perfect oracle (default).
    noise_rate_base: float = 0.0              # irreducible error rate
    noise_rate_ambiguity: float = 0.0         # additional error near threshold

    # Per-pattern feedback weighting: scale prior updates by pattern severity.
    # When True, co-firing patterns with higher severity get larger updates.
    feedback_per_pattern_weighting: bool = False

    # --- Variants ---------------------------------------------------------
    eval_variant: str = "cold"  # "cold" | "warm"
    noise_rate: float = 0.0     # DEPRECATED: use noise_rate_base instead (kept for compat)
    feedback_every_n: int = 1   # DEPRECATED: use feedback_response_rate instead (kept for compat)

    # --- Rotation ---------------------------------------------------------
    rotate: bool = False  # run all 4 rotation schemes

    # --- Knowledge Graph / SINDIT integration -------------------------
    use_memory_store: bool = True       # store scored events in MemoryStore
    use_co_occurrence: bool = True      # track pattern co-occurrence
    co_occurrence_decay: float = 0.3    # propagation decay per hop
    use_sindit_simulation: bool = True  # simulate SINDIT context from CSV

    # --- API mode (route events through live backend API) ----------------
    api_mode: bool = False              # when True, POST events to backend API
    api_mode_strict: bool = False       # fail the run if the backend API path is unavailable
    api_base_url: str = "http://localhost:8000"  # backend base URL
    sindit_live: bool = False           # use real SINDIT instead of CSV simulation
    api_generate_explanations: Optional[bool] = None  # None = use backend default for this run
    experiment_fast_path: bool = True   # skip storage/retrieval in API orchestrator
    api_use_server_patterns: bool = False  # let backend derive patterns from raw metrics
    api_batch_size: int = 64            # events per batch POST (1 = legacy per-sample)
    api_request_timeout: float = 120.0  # seconds; per-request HTTP timeout for the experiment API client
    parallel_folds: int = 1             # >1 runs LOOCV folds in parallel threads

    # Persisting back into the shared production priors file breaks run isolation.
    persist_shared_priors: bool = False

    # --- Prediction gap (0 = detection, >0 = true prediction) -----------
    prediction_gap_s: float = 0.0  # seconds between feature window end and stop

    # --- Window size (must match extraction --window parameter) ----------
    window_size_s: float = 60.0  # extraction window duration in seconds
    sample_rate_hz: float = 1.0  # data sampling frequency (1 Hz for casedata CSVs)

    # --- Pattern quality gate (Fix 4b) ------------------------------------
    min_discrimination_ratio: float = 1.5  # below this, weak patterns become protective or uninformative
    # Patterns whose fire rate on pre_stoppage / fire rate on normal < this value
    # are not fault-specific (they fire equally or more on healthy cutting).
    # Set to 0.0 to disable this gate.

    # --- Negative-sampling feedback (DISABLED — no real-world analogue) -----
    # Negative sampling was an artificial mechanism to prevent CONFIRM-only
    # prior drift.  It has no operator analogue (operators don't review
    # events the model didn't flag).  Disabled by default.
    negative_sampling_enabled: bool = False
    negative_sampling_rate: float = 0.3   # P(implicit DISMISS on true-negative)

    # Pluggable model selection: which scoring rules (== models) the unified
    # scorer runs. None -> the default set. e.g. ["classical_alert",
    # "pattern_match", "historical_prior"] to ablate, or add a registered model
    # rule by name. See docs/PLUGGABLE_MODELS_2026-06-15.md.
    enabled_rules: Optional[List[str]] = None

    # Remove the one-class seed (anomaly) model entirely: not loaded or scored,
    # raw_model_score forced to 0, online retraining skipped. Default OFF — the seed
    # model was found to be near-random and to *worsen* results, so the modular
    # pipeline runs without it (patterns + selected models only). Set True to re-add
    # it for an ablation. The detector set is otherwise chosen via ``enabled_rules``.
    use_seed_model: bool = False

    # --- System under test (A2 decision: faithful by default) -------------
    # The deployed pipeline (MemoryEventOrchestrator.process_event) scores with
    # the unsupervised 4-rule scorer + context priors and a fixed operating
    # threshold. It does NOT apply the supervised-RF blend, the separate
    # tool-prior multiplier, or threshold adaptation. Best practice is that the
    # evaluation measures the deployed system, so these experiment-only layers
    # are OFF by default. Set faithful_pipeline=False for an ABLATION run that
    # re-enables them to measure their marginal contribution.
    faithful_pipeline: bool = True

    # --- Misc -------------------------------------------------------------
    downsample_max: int = 0  # 0 = no downsampling
    random_seed: int = 42
    verbose: bool = True

    # ---- Path coercion (handles str inputs from Streamlit cache) ---------
    def __post_init__(self):
        if isinstance(self.features_csv, str):
            self.features_csv = Path(self.features_csv)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # A2: enforce the faithful (deployed) system under test. Disable the
        # experiment-only decision layers so the headline measures exactly the
        # production scorer path. Ablation runs pass faithful_pipeline=False.
        if self.faithful_pipeline:
            self.use_supervised_model = False
            self.weight_supervised = 0.0
            self.weight_unsupervised = 1.0
            self.use_tool_priors = False
            self.threshold_adaptation_rate = 0.0
            self.threshold_adaptation_max = 0.0

        # When the backend orchestrator (or this run's explicit override) has
        # explanations enabled and we're in API mode, disable fast_path so
        # events go through the full scoring → alert → explanation pipeline.
        # Fast-path skips the alert dispatch gate which also gates explanation
        # generation.
        if self.api_mode and self.experiment_fast_path:
            try:
                import os as _os
                explanations_enabled = self.api_generate_explanations
                if explanations_enabled is None:
                    explanations_enabled = _os.environ.get("GENERATE_EXPLANATIONS", "false").lower() == "true"
                if explanations_enabled:
                    self.experiment_fast_path = False
            except Exception:
                pass

        if self.api_mode and self.api_use_server_patterns and self.api_batch_size != 1:
            self.api_batch_size = 1

        # Re-derive features_csv when window, gap, or Hz differs from defaults
        # so each (window, gap, hz) combination resolves to a distinct file.
        # Only override if the caller did NOT supply an explicit path.
        default_csv = ROOT / "data" / "breakage_patterns" / "stoppage_features.csv"
        if self.features_csv == default_csv:
            win_tag = f"_w{int(self.window_size_s)}s" if self.window_size_s != 60.0 else ""
            hz_tag = f"_{int(self.sample_rate_hz)}hz" if self.sample_rate_hz != 1.0 else ""
            gap_tag = f"_gap{int(self.prediction_gap_s)}s" if self.prediction_gap_s > 0 else ""
            self.features_csv = (
                ROOT / "data" / "breakage_patterns"
                / f"stoppage_features{win_tag}{hz_tag}{gap_tag}.csv"
            )

    # ---- Derived paths ---------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """Directory for a specific split's outputs."""
        base = f"train_{'_'.join(self.train_ops)}_test_{self.test_op}_eval_{self.eval_op}"
        if self.window_size_s != 60.0:
            base += f"_w{int(self.window_size_s)}s"
        if self.sample_rate_hz != 1.0:
            base += f"_{int(self.sample_rate_hz)}hz"
        if self.prediction_gap_s > 0:
            base += f"_gap{int(self.prediction_gap_s)}s"
        return self.output_dir / base

    @property
    def window_entries(self) -> int:
        """Number of data-points in one window: window_size_s × sample_rate_hz."""
        return int(self.window_size_s * self.sample_rate_hz)

    @property
    def raw_series_npz(self) -> Path:
        """Path to the raw time-series NPZ for this window size and Hz."""
        win_tag = f"_w{int(self.window_size_s)}s" if self.window_size_s != 60.0 else ""
        hz_tag = f"_{int(self.sample_rate_hz)}hz" if self.sample_rate_hz != 1.0 else ""
        return (
            ROOT / "data" / "breakage_patterns"
            / f"stoppage_raw_series{win_tag}{hz_tag}.npz"
        )

    @property
    def seed_model_path(self) -> Path:
        return self.run_dir / self.seed_model_filename

    @property
    def baseline_priors_path(self) -> Path:
        return self.run_dir / self.baseline_priors_filename

    @property
    def train_meta_path(self) -> Path:
        return self.run_dir / self.train_meta_filename

    @property
    def supervised_model_path(self) -> Path:
        return self.run_dir / self.supervised_model_filename

    @property
    def tool_priors_path(self) -> Path:
        return self.run_dir / self.tool_priors_filename

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

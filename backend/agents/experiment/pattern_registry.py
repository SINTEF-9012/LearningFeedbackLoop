"""Extensible pattern registry for stoppage prediction.

Provides a central, plugin-friendly way to define and manage patterns
that fire on CNC sensor features.  Patterns can come from three sources:

  1. **Built-in**  – hard-coded from prior experiments (the original 4).
  2. **Domain-expert**  – added at runtime via ``register()`` with
     human-readable rules distilled from machinist knowledge.
  3. **Time-series derived** – detected automatically from statistical
     properties of the raw signal (trend reversal, variance explosion, …).

Usage
-----
>>> from backend.agents.experiment.pattern_registry import get_registry
>>> reg = get_registry()              # singleton, pre-loaded with defaults
>>> reg.list_patterns(category="fault")
>>> reg.detect_all(features_dict)     # → ["POWER_SPIKE", ...]
>>> reg.register(PatternDefinition(   # add your own at any time
...     name="MY_CUSTOM_PATTERN",
...     description="Custom vibration rule",
...     category="domain",
...     detector=lambda f, _: f.get("vib_severity_x_mean", 0) > 2.0,
... ))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definition
# ---------------------------------------------------------------------------

DetectorFn = Callable[[Dict[str, float], Dict[str, Any]], bool]
"""Signature: (features, thresholds) → fires?

``features``   – flat dict of column→value from one sample row.
``thresholds`` – calibrated threshold dict (from trainer) keyed by
                 config attribute name.  Detectors that don't need
                 calibrated thresholds can ignore this argument.
"""


@dataclass
class PatternDefinition:
    """One atomic pattern that can fire on a feature vector."""

    name: str
    description: str = ""
    category: str = "fault"            # "fault" | "process" | "domain" | "ts_derived"
    severity: float = 0.5              # 0–1, used for prior seeding
    detector: DetectorFn = field(default=lambda f, t: False, repr=False)
    columns: List[str] = field(default_factory=list)  # CSV columns used
    default_prior: float = 0.5
    enabled: bool = True
    source: str = "builtin"            # "builtin" | "domain_expert" | "time_series"
    discrimination_ratio: Optional[float] = None  # filled during training
    discrimination_score: Optional[float] = None
    fire_rate_normal: Optional[float] = None
    fire_rate_event: Optional[float] = None
    polarity: str = "fault_supporting"  # fault_supporting | protective | uninformative
    # Co-occurrence gating (plan 1.7): a single weak indicator that should not
    # raise an alert on its own — it needs a corroborating pattern or model
    # agreement. The scorer caps its lone contribution to STORE-band.
    requires_corroboration: bool = False


# ---------------------------------------------------------------------------
# Built-in detector functions
# ---------------------------------------------------------------------------

def _power_spike(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Spindle or Y-axis power delta exceeds calibrated threshold."""
    th_sp = _tval(t, "pattern_power_spindle_delta_max", 15.0)
    th_y  = _tval(t, "pattern_power_y_delta_max", 10.0)
    return f.get("power_spindle_delta_max", 0) > th_sp or \
           f.get("power_y_delta_max", 0) > th_y


def _vib_shift(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Vibration severity delta or chatter frequency slope is anomalous."""
    th_vib = _tval(t, "pattern_vib_severity_x_delta_max", 0.8)
    th_ch  = _tval(t, "pattern_chatter_freq_x_slope_abs", 5.0)
    return f.get("vib_severity_x_delta_max", 0) > th_vib or \
           abs(f.get("chatter_freq_x_slope", 0)) > th_ch


def _feed_override_drop(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Feed override drops (delta < threshold) or falls into a low band."""
    th_delta = _tval(t, "pattern_feed_override_delta_mean", -10.0)
    th_min   = _tval(t, "pattern_feed_override_min", 50.0)
    fo_delta = f.get("feed_override_delta_mean", 0)
    fo_min   = f.get("feed_override_min", 0)
    return fo_delta < th_delta or (fo_min > 0 and fo_min < th_min)


def _decorrelation(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Spindle-power-vibration correlation drops into anomalous band."""
    th_low = _tval(t, "pattern_corr_spindle_power_vib_x_low", 0.3)
    corr = f.get("corr_spindle_power_vib_x", 0)
    return 0.0 < abs(corr) < th_low


# --- Domain-knowledge patterns (new) ---

def _spindle_load_ramp(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """Sustained spindle power increase (tool dulling / chip packing)."""
    return f.get("power_spindle_slope", 0) > 5.0


def _feed_stall(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """Feed actual drops far below commanded without override change.

    Indicates the machine is slowing involuntarily — CNC adaptive feed
    control reducing speed due to load, or mechanical resistance.
    """
    feed_mean = f.get("feed_actual_mean", 0)
    feed_range = f.get("feed_actual_range", 0)
    # Large range + low mean = feed is unstable / stalling
    return feed_mean > 0 and feed_range > 3.0 * feed_mean


def _power_asymmetry(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """X vs Y axis power diverges (uneven cutting load → chatter risk).

    Both axes must actually be present: a *missing* axis channel (0) is not asymmetry.
    This guard stops the rule firing on every row of datasets that don't map X/Y power
    (e.g. Site_a_line2), where it was degenerate (|0-py|/(0+py)=1 → always fired).
    """
    px = f.get("power_x_mean", 0.0)
    py = f.get("power_y_mean", 0.0)
    if px <= 0.0 or py <= 0.0 or (px + py) < 1.0:
        return False
    return abs(px - py) / (px + py) > _tval(t, "pattern_power_asymmetry_ratio", 0.6)


def _energy_accumulation(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """Energy total ramps faster than baseline (tool wear)."""
    return f.get("energy_total_slope", 0) > 3.0


# --- Time-series derived patterns ---

def _variance_explosion(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """Rolling std of vibration or power jumps >3× its own mean-std.

    Indicates sudden instability in a signal that was previously stable.
    """
    vib_std = f.get("vib_severity_x_std", 0)
    vib_mean = f.get("vib_severity_x_mean", 0)
    if vib_mean > 0 and vib_std > 3.0 * vib_mean:
        return True
    pwr_std = f.get("power_spindle_std", 0)
    pwr_mean = f.get("power_spindle_mean", 0)
    if pwr_mean > 0 and pwr_std > 3.0 * pwr_mean:
        return True
    return False


def _trend_reversal(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """Delta-mean vs overall slope have opposite signs (regime change)."""
    slope = f.get("power_spindle_slope", 0)
    delta = f.get("power_spindle_delta_mean", 0)
    # Both must be significant and opposite
    if abs(slope) < 1.0 or abs(delta) < 1.0:
        return False
    return (slope > 0) != (delta > 0)


def _autocorrelation_break(f: Dict[str, float], _t: Dict[str, Any]) -> bool:
    """IQR of vibration is very large relative to its range ⇒ non-stationary."""
    iqr = f.get("vib_severity_x_iqr", 0)
    rng = f.get("vib_severity_x_range", 0)
    if rng < 0.01:
        return False
    # IQR / range > 0.7 means signal is spread across the full range
    # (non-stationary, autocorrelation has broken down)
    return iqr / rng > 0.7


# --- Dimensionless impulse indicators (SOTA: robust to cutting parameters / speed) ---

def _impulse_burst(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Crest factor (peak/RMS) high — sharp impulsive vibration (impact / breakage).

    Dimensionless and largely insensitive to cutting parameters, which is why the
    tool-breakage literature favours it under time-varying conditions. Fires on HIGH
    values, so a missing feature (0) does not trigger it."""
    return f.get("impulse_crest_factor", 0.0) > _tval(t, "pattern_crest_factor", 5.0)


def _kurtosis_spike(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """Excess kurtosis (>4; Gaussian≈3) — impulsive, non-Gaussian vibration
    (chipping / breakage). Dimensionless; speed-robust."""
    return f.get("kurtosis_max", 0.0) > _tval(t, "pattern_kurtosis", 4.0)


def _hf_energy_burst(f: Dict[str, float], t: Dict[str, Any]) -> bool:
    """High-frequency energy ratio spike — broadband HF burst (the 'hf_burst' half of
    the tool-breakage signature). Dimensionless energy ratio."""
    return f.get("hf_energy_ratio", 0.0) > _tval(t, "pattern_hf_energy_ratio", 0.5)


# --- Helper ---

def _tval(thresholds: Dict[str, Any], key: str, default: float) -> float:
    """Extract a threshold value, supporting both flat dicts and nested."""
    v = thresholds.get(key, default)
    if isinstance(v, dict):
        return v.get("value", default)
    return float(v) if v is not None else default


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PatternRegistry:
    """Central registry of all available pattern definitions.

    Patterns can be registered at import time (built-in), by domain
    experts at configuration time, or derived from time-series analysis
    during training.  Each pattern has a ``detector`` callable that
    accepts a flat feature dict and a calibrated-threshold dict and
    returns whether the pattern fires.
    """

    def __init__(self) -> None:
        self._patterns: Dict[str, PatternDefinition] = {}

    # -- Mutation ----------------------------------------------------------

    def register(self, pattern: PatternDefinition) -> None:
        """Add or replace a pattern definition."""
        if pattern.name in self._patterns:
            logger.info("Replacing pattern %s (source: %s → %s)",
                        pattern.name, self._patterns[pattern.name].source, pattern.source)
        self._patterns[pattern.name] = pattern

    def disable(self, name: str) -> None:
        if name in self._patterns:
            self._patterns[name].enabled = False
            logger.info("Disabled pattern %s", name)

    def enable(self, name: str) -> None:
        if name in self._patterns:
            self._patterns[name].enabled = True

    def remove(self, name: str) -> None:
        self._patterns.pop(name, None)

    # -- Queries -----------------------------------------------------------

    def get(self, name: str) -> Optional[PatternDefinition]:
        return self._patterns.get(name)

    def list_patterns(
        self,
        category: Optional[str] = None,
        enabled_only: bool = False,
        source: Optional[str] = None,
        polarity: Optional[str] = None,
    ) -> List[PatternDefinition]:
        """List patterns, optionally filtered."""
        out = list(self._patterns.values())
        if category:
            out = [p for p in out if p.category == category]
        if enabled_only:
            out = [p for p in out if p.enabled]
        if source:
            out = [p for p in out if p.source == source]
        if polarity:
            out = [p for p in out if p.polarity == polarity]
        return sorted(out, key=lambda p: p.name)

    def pattern_names(self, enabled_only: bool = True) -> List[str]:
        """Return sorted list of pattern names."""
        return [p.name for p in self.list_patterns(enabled_only=enabled_only)]

    def __len__(self) -> int:
        return len(self._patterns)

    def __contains__(self, name: str) -> bool:
        return name in self._patterns

    # -- Detection ---------------------------------------------------------

    def detect_all(
        self,
        features: Dict[str, float],
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Run all enabled detectors and return names of those that fire."""
        thresholds = thresholds or {}
        fired: List[str] = []
        for name, pdef in self._patterns.items():
            if not pdef.enabled:
                continue
            try:
                if pdef.detector(features, thresholds):
                    fired.append(name)
            except Exception as exc:
                logger.debug("Pattern %s detector error: %s", name, exc)
        return fired

    def detect_with_details(
        self,
        features: Dict[str, float],
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Like detect_all but returns full details for each fired pattern."""
        thresholds = thresholds or {}
        results = []
        for name, pdef in self._patterns.items():
            if not pdef.enabled:
                continue
            try:
                fires = pdef.detector(features, thresholds)
            except Exception:
                fires = False
            if fires:
                results.append({
                    "name": name,
                    "category": pdef.category,
                    "severity": pdef.severity,
                    "source": pdef.source,
                    "description": pdef.description,
                    "discrimination_ratio": pdef.discrimination_ratio,
                    "discrimination_score": pdef.discrimination_score,
                    "fire_rate_normal": pdef.fire_rate_normal,
                    "fire_rate_event": pdef.fire_rate_event,
                    "polarity": pdef.polarity,
                })
        return results

    # -- Discrimination gate -----------------------------------------------

    def classify_patterns(
        self,
        min_ratio: float,
        ratios: Dict[str, float],
        *,
        polarities: Optional[Dict[str, str]] = None,
        discrimination_scores: Optional[Dict[str, float]] = None,
        fire_rate_normal: Optional[Dict[str, float]] = None,
        fire_rate_event: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:
        """Classify patterns by discrimination quality and polarity."""
        buckets: Dict[str, List[str]] = {
            "fault_supporting": [],
            "protective": [],
            "uninformative": [],
        }
        default_polarities = polarities or {}
        score_by_name = discrimination_scores or {}
        normal_rates = fire_rate_normal or {}
        event_rates = fire_rate_event or {}
        for name, ratio in ratios.items():
            pdef = self._patterns.get(name)
            if not pdef:
                continue
            polarity = default_polarities.get(name)
            if not polarity:
                polarity = "fault_supporting" if min_ratio <= 0 or ratio >= min_ratio else "uninformative"
            pdef.discrimination_ratio = ratio
            pdef.discrimination_score = score_by_name.get(name)
            pdef.fire_rate_normal = normal_rates.get(name)
            pdef.fire_rate_event = event_rates.get(name)
            pdef.polarity = polarity
            pdef.enabled = polarity != "uninformative"
            buckets.setdefault(polarity, []).append(name)
        return buckets

    def apply_discrimination_gate(
        self,
        min_ratio: float,
        ratios: Dict[str, float],
    ) -> List[str]:
        """Disable patterns whose discrimination ratio is below the gate.

        Returns names of disabled patterns.
        """
        buckets = self.classify_patterns(min_ratio, ratios)
        disabled = buckets.get("uninformative", [])
        for name in disabled:
            ratio = ratios.get(name, 0.0)
            logger.warning(
                "Pattern %s disabled: discrimination_ratio=%.2f < %.2f",
                name, ratio, min_ratio,
            )
        return disabled

    # -- Factory -----------------------------------------------------------

    @classmethod
    def default(cls) -> "PatternRegistry":
        """Build a registry pre-loaded with ALL patterns (built-in + domain + ts)."""
        reg = cls()
        for pdef in _ALL_DEFAULTS:
            reg.register(pdef)
        return reg

    # -- Serialisation (for experiment metadata) ---------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: {
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
                "description": p.description,
            }
            for name, p in self._patterns.items()
        }


# ---------------------------------------------------------------------------
# Default pattern catalogue
# ---------------------------------------------------------------------------

_ALL_DEFAULTS: List[PatternDefinition] = [
    # === BUILT-IN (original 4 — observable phenomena, not diagnoses) ===
    PatternDefinition(
        name="SPINDLE_POWER_SURGE",
        description="Spindle or Y-axis power delta exceeds normal p95",
        category="fault",
        severity=0.90,
        detector=_power_spike,
        columns=["power_spindle_delta_max", "power_y_delta_max"],
        default_prior=0.5,
        source="builtin",
    ),
    PatternDefinition(
        name="VIBRATION_REGIME_SHIFT",
        description="Vibration severity or chatter frequency anomaly",
        category="fault",
        severity=0.85,
        detector=_vib_shift,
        columns=["vib_severity_x_delta_max", "chatter_freq_x_slope"],
        default_prior=0.5,
        source="builtin",
    ),
    PatternDefinition(
        name="FEED_OVERRIDE_DROP",
        description="Feed override delta drops or enters low band",
        category="fault",
        severity=0.75,
        detector=_feed_override_drop,
        columns=["feed_override_delta_mean", "feed_override_min"],
        default_prior=0.5,
        source="builtin",
    ),
    PatternDefinition(
        name="SENSOR_DECORRELATION",
        description="Spindle-power-vibration correlation drops (decoupling)",
        category="fault",
        severity=0.80,
        detector=_decorrelation,
        columns=["corr_spindle_power_vib_x"],
        default_prior=0.5,
        source="builtin",
    ),

    # === DOMAIN-EXPERT (new — from machinist knowledge) ===
    PatternDefinition(
        name="SPINDLE_LOAD_RAMP",
        description="Sustained spindle power increase — tool dulling or chip packing",
        category="domain",
        severity=0.70,
        detector=_spindle_load_ramp,
        columns=["power_spindle_slope"],
        default_prior=0.5,
        source="domain_expert",
    ),
    PatternDefinition(
        name="FEED_STALL",
        description="Feed actual drops far below commanded — mechanical resistance",
        category="domain",
        severity=0.75,
        detector=_feed_stall,
        columns=["feed_actual_mean", "feed_actual_range"],
        default_prior=0.5,
        source="domain_expert",
    ),
    PatternDefinition(
        name="POWER_ASYMMETRY",
        description="X vs Y axis power diverges — uneven cutting load / chatter risk",
        category="domain",
        severity=0.65,
        detector=_power_asymmetry,
        columns=["power_x_mean", "power_y_mean"],
        default_prior=0.5,
        source="domain_expert",
    ),
    PatternDefinition(
        name="ENERGY_ACCUMULATION",
        description="Energy total ramps faster than baseline — tool wear indicator",
        category="domain",
        severity=0.60,
        detector=_energy_accumulation,
        columns=["energy_total_slope"],
        default_prior=0.5,
        source="domain_expert",
    ),

    # === TIME-SERIES DERIVED (new — from statistical signal properties) ===
    PatternDefinition(
        name="VARIANCE_EXPLOSION",
        description="Vibration or power std jumps >3× mean — sudden instability",
        category="ts_derived",
        # Single-feature supporting indicator: alert-band (0.65), NOT critical.
        # Was 0.80/0.85 == critical threshold, which made single-pattern events
        # immune to feedback damping. At 0.65 a lone spike is a demotable alert;
        # the ~20%%-of-windows flood is addressed separately by co-occurrence
        # gating (plan 1.7), not by severity alone. Plan 1.11, 2026-07-07.
        severity=0.65,
        detector=_variance_explosion,
        columns=["vib_severity_x_std", "vib_severity_x_mean",
                  "power_spindle_std", "power_spindle_mean"],
        default_prior=0.5,
        source="time_series",
        requires_corroboration=True,  # plan 1.7
    ),
    PatternDefinition(
        name="TREND_REVERSAL",
        description="Delta-mean vs slope have opposite signs — regime change",
        category="ts_derived",
        severity=0.65,
        detector=_trend_reversal,
        columns=["power_spindle_slope", "power_spindle_delta_mean"],
        default_prior=0.5,
        source="time_series",
    ),
    PatternDefinition(
        name="AUTOCORRELATION_BREAK",
        description="Vibration IQR/range > 0.7 — signal is non-stationary",
        category="ts_derived",
        severity=0.70,
        detector=_autocorrelation_break,
        columns=["vib_severity_x_iqr", "vib_severity_x_range"],
        default_prior=0.5,
        source="time_series",
    ),
    # Dimensionless impulse indicators (SOTA, speed-robust). Promote existing features
    # to discriminative patterns for impact/chipping/breakage.
    PatternDefinition(
        name="IMPULSE_BURST",
        description="Crest factor (peak/RMS) high — impulsive impact (breakage)",
        category="fault",
        # Single-feature supporting indicator: alert-band (0.65), not critical;
        # demotable by feedback damping. Flood handled by co-occurrence (1.7). Plan 1.11.
        severity=0.65,
        detector=_impulse_burst,
        columns=["impulse_crest_factor"],
        default_prior=0.5,
        source="builtin",
        requires_corroboration=True,  # plan 1.7
    ),
    PatternDefinition(
        name="KURTOSIS_SPIKE",
        description="Excess kurtosis (>4) — impulsive non-Gaussian vibration (chipping)",
        category="fault",
        # Single-feature supporting indicator: alert-band (0.65), not critical;
        # demotable by feedback damping. Flood handled by co-occurrence (1.7). Plan 1.11.
        severity=0.65,
        detector=_kurtosis_spike,
        columns=["kurtosis_max"],
        default_prior=0.5,
        source="builtin",
        requires_corroboration=True,  # plan 1.7
    ),
    PatternDefinition(
        name="HF_ENERGY_BURST",
        description="High-frequency energy ratio spike — broadband HF burst (breakage)",
        category="fault",
        # Single-feature supporting indicator: alert-band (0.65), not critical;
        # demotable by feedback damping. Flood handled by co-occurrence (1.7). Plan 1.11.
        severity=0.65,
        detector=_hf_energy_burst,
        columns=["hf_energy_ratio"],
        default_prior=0.5,
        source="builtin",
        requires_corroboration=True,  # plan 1.7
    ),
]


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_REGISTRY: Optional[PatternRegistry] = None


def get_registry() -> PatternRegistry:
    """Return the module-level singleton (created on first call)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PatternRegistry.default()
    return _REGISTRY


def reset_registry() -> PatternRegistry:
    """Reset and return a fresh default registry."""
    global _REGISTRY
    _REGISTRY = PatternRegistry.default()
    return _REGISTRY

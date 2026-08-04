from backend.agents.experiment.config import FAULT_SEVERITY, PATTERN_KEYS
from backend.agents.experiment.trainer import _build_initial_priors_payload


def test_pattern_keys_include_registry_only_canonical_patterns() -> None:
    expected_seeded_prefix = [
        "SPINDLE_POWER_SURGE",
        "VIBRATION_REGIME_SHIFT",
        "FEED_OVERRIDE_DROP",
        "SENSOR_DECORRELATION",
        "SPINDLE_LOAD_RAMP",
        "FEED_STALL",
    ]

    assert PATTERN_KEYS[: len(expected_seeded_prefix)] == expected_seeded_prefix
    assert "POWER_ASYMMETRY" in PATTERN_KEYS
    assert "ENERGY_ACCUMULATION" in PATTERN_KEYS
    assert "VARIANCE_EXPLOSION" in PATTERN_KEYS
    assert "TREND_REVERSAL" in PATTERN_KEYS
    assert "AUTOCORRELATION_BREAK" in PATTERN_KEYS
    assert "CHATTER_ONSET" not in PATTERN_KEYS
    assert "THERMAL_DRIFT" not in PATTERN_KEYS
    assert "ANOMALY_HIGH_POWER" not in PATTERN_KEYS
    assert "ANOMALY_HIGH_VIBRATION" not in PATTERN_KEYS
    assert "ANOMALY_FEED_DEVIATION" not in PATTERN_KEYS
    assert len(PATTERN_KEYS) == len(set(PATTERN_KEYS))


def test_fault_severity_includes_registry_only_defaults() -> None:
    assert FAULT_SEVERITY["POWER_ASYMMETRY"] == 0.65
    assert FAULT_SEVERITY["ENERGY_ACCUMULATION"] == 0.60
    # Recalibrated to 0.65 (single-feature supporting indicator, alert-band not
    # critical) — plan 1.11, 2026-07-07.
    assert FAULT_SEVERITY["VARIANCE_EXPLOSION"] == 0.65
    assert FAULT_SEVERITY["TREND_REVERSAL"] == 0.65
    assert FAULT_SEVERITY["AUTOCORRELATION_BREAK"] == 0.70


def test_build_initial_priors_payload_seeds_registry_only_patterns() -> None:
    payload = _build_initial_priors_payload({"SPINDLE_POWER_SURGE": {"value": 12.3}})

    assert payload["pattern_priors"]["POWER_ASYMMETRY"] == 0.5
    assert payload["pattern_priors"]["VARIANCE_EXPLOSION"] == 0.5
    assert payload["feedback_counts"]["AUTOCORRELATION_BREAK"] == {"confirm": 0, "dismiss": 0}
    assert "CHATTER_ONSET" not in payload["pattern_priors"]
    assert "THERMAL_DRIFT" not in payload["pattern_priors"]
    assert "ANOMALY_HIGH_POWER" not in payload["pattern_priors"]
    assert payload["calibrated_pattern_thresholds"] == {"SPINDLE_POWER_SURGE": {"value": 12.3}}
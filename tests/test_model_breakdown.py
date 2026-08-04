"""Tests for Agent C model_breakdown (2026-04-24)."""
from __future__ import annotations

from backend.agents.memory.model_breakdown import build_model_breakdown


def test_empty_signals_returns_all_none_and_no_available():
    out = build_model_breakdown(None)
    assert out["available"] == []
    for section in ("classical", "harmonic", "stoppage", "online"):
        assert section in out
        assert all(v is None for v in out[section].values())


def test_classical_section_populated_from_aliases():
    signals = {
        "anomaly_detector_score": 0.83,
        "model_confidence": 0.9,
        "isolation_forest_score": 0.75,
        "lof_score": 0.6,
        "ensemble_score": 0.78,
        "breakage_prediction": 0.42,
    }
    out = build_model_breakdown(signals)
    c = out["classical"]
    assert c["anomaly_score"] == 0.83
    assert c["model_confidence"] == 0.9
    assert c["isolation_forest"] == 0.75
    assert c["lof"] == 0.6
    assert c["ensemble"] == 0.78
    assert c["breakage_probability"] == 0.42
    assert "classical" in out["available"]


def test_harmonic_section_picks_score_and_source():
    signals = {
        "harmonic_context_score": 0.55,
        "harmonic_context_source": "site_a_line2_breakage_preset",
    }
    out = build_model_breakdown(signals)
    assert out["harmonic"]["score"] == 0.55
    assert out["harmonic"]["source"] == "site_a_line2_breakage_preset"
    assert "harmonic" in out["available"]


def test_stoppage_section_picks_aliased_keys():
    signals = {
        "stoppage_prob": 0.22,
        "eta_s": 12.5,
        "stoppage_label": "pre_break",
    }
    out = build_model_breakdown(signals)
    assert out["stoppage"]["probability"] == 0.22
    assert out["stoppage"]["eta_s"] == 12.5
    assert out["stoppage"]["label"] == "pre_break"
    assert "stoppage" in out["available"]


def test_online_section_probability_and_running_flag():
    signals = {"online_probability": 0.31, "online_running": True}
    out = build_model_breakdown(signals)
    assert out["online"]["probability"] == 0.31
    assert out["online"]["running"] is True
    assert "online" in out["available"]


def test_booleans_are_not_picked_as_numbers():
    # A stray bool must not be coerced into the numeric anomaly_score slot.
    signals = {"anomaly_detector_score": True, "model_confidence": 0.5}
    out = build_model_breakdown(signals)
    assert out["classical"]["anomaly_score"] is None
    assert out["classical"]["model_confidence"] == 0.5


def test_non_numeric_and_non_string_values_are_ignored():
    signals = {
        "anomaly_detector_score": "high",  # string → rejected
        "harmonic_context_source": 42,       # int in string slot → rejected
        "harmonic_context_score": None,
    }
    out = build_model_breakdown(signals)
    assert out["classical"]["anomaly_score"] is None
    assert out["harmonic"]["source"] is None
    assert out["harmonic"]["score"] is None
    assert out["available"] == []


def test_optional_feature_schema_metadata_passthrough():
    out = build_model_breakdown(
        {"anomaly_detector_score": 0.1},
        feature_schema_version=3,
        feature_count=28,
    )
    assert out["feature_schema_version"] == 3
    assert out["feature_count"] == 28


def test_partial_signals_produce_partial_available():
    signals = {"harmonic_context_score": 0.4}
    out = build_model_breakdown(signals)
    assert out["available"] == ["harmonic"]
    # All classical fields stay None
    assert all(v is None for v in out["classical"].values())

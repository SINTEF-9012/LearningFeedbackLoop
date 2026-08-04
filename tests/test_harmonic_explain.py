"""Tests for the pure harmonic-explanation helper (Agent O).

These tests do NOT require PyTorch — they only exercise
``build_harmonic_explanation`` with hand-crafted score-result dicts.
"""

from __future__ import annotations

from backend.agents.processing.harmonic_explain import build_harmonic_explanation


def test_unavailable_when_available_false():
    out = build_harmonic_explanation(
        None, None, None, available=False, reason="torch missing",
    )
    assert out["available"] is False
    assert out["reason"] == "torch missing"
    assert out["score"] is None
    assert out["contributions"] == []
    assert out["top_weighted"] == []


def test_unavailable_when_score_result_none():
    out = build_harmonic_explanation(None, [1.0, 2.0], ["a", "b"])
    assert out["available"] is False
    assert out["contributions"] == []


def test_happy_path_computes_contribution_and_top_k():
    score = {
        "harmonic_context_score": 0.73,
        "context_weights": [0.1, 0.9, 0.3, 0.5],
        "model_source": "trained",
    }
    values = [1.0, 2.0, 3.0, 4.0]
    labels = ["X·H1", "X·H2", "Y·H1", "Y·H2"]
    out = build_harmonic_explanation(score, values, labels, dataset_name="casedata", top_k=2)
    assert out["available"] is True
    assert out["score"] == 0.73
    assert out["model_source"] == "trained"
    assert out["dataset"] == "casedata"
    # contributions = weight * value
    by_label = {c["label"]: c for c in out["contributions"]}
    assert by_label["X·H2"]["contribution"] == 1.8      # 0.9 * 2.0
    assert by_label["Y·H2"]["contribution"] == 2.0      # 0.5 * 4.0
    assert by_label["X·H1"]["contribution"] == 0.1      # 0.1 * 1.0
    # top_k=2 → two largest |contribution|: Y·H2 (2.0), X·H2 (1.8)
    assert len(out["top_weighted"]) == 2
    assert out["top_weighted"][0]["label"] == "Y·H2"
    assert out["top_weighted"][1]["label"] == "X·H2"


def test_top_k_sorted_by_abs_contribution():
    # Negative weight × positive value gives a strongly-negative contribution
    # that should still rank at the top by magnitude.
    score = {
        "harmonic_context_score": 0.5,
        "context_weights": [-2.0, 0.1, 0.2],
        "model_source": "trained",
    }
    out = build_harmonic_explanation(score, [1.0, 1.0, 1.0], ["a", "b", "c"], top_k=1)
    assert out["top_weighted"][0]["label"] == "a"
    assert out["top_weighted"][0]["contribution"] == -2.0


def test_weights_values_length_mismatch_padded():
    score = {"harmonic_context_score": 0.5, "context_weights": [0.1, 0.2, 0.3]}
    out = build_harmonic_explanation(score, [1.0], None, top_k=5)
    # Values padded to length-3 with zeros; labels default to h1..h3
    assert len(out["contributions"]) == 3
    assert out["harmonic_values"] == [1.0, 0.0, 0.0]
    assert out["feature_labels"] == ["h1", "h2", "h3"]


def test_missing_labels_uses_hN_placeholders():
    score = {"harmonic_context_score": 0.0, "context_weights": [0.0, 0.0]}
    out = build_harmonic_explanation(score, [0.0, 0.0], ["only_one"])
    assert out["feature_labels"] == ["only_one", "h2"]


def test_non_float_score_coerced_to_none():
    score = {"harmonic_context_score": "not-a-number", "context_weights": [0.1]}
    out = build_harmonic_explanation(score, [1.0], ["a"])
    assert out["score"] is None
    assert out["available"] is True


def test_top_k_zero_returns_empty_top_weighted():
    score = {"harmonic_context_score": 0.5, "context_weights": [0.1, 0.2]}
    out = build_harmonic_explanation(score, [1.0, 1.0], ["a", "b"], top_k=0)
    assert out["top_weighted"] == []
    assert len(out["contributions"]) == 2

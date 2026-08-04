import pandas as pd

from backend.agents.core.schemas import PatternKey, PatternType
from backend.agents.memory.experiment_results import _extract_pattern_polarity_counts
from backend.agents.memory.scorer import SignificanceConfig, SignificanceScorer
from backend.agents.patterns.registry import get_registry, list_patterns_dict, reset_registry
from backend.agents.experiment.config import ExperimentConfig
from backend.agents.experiment.evaluator import _create_unified_scorer
from backend.agents.experiment.trainer import (
    _classify_pattern_polarity,
    calibrate_pattern_thresholds,
)


def test_classify_pattern_polarity_distinguishes_fault_protective_and_weak() -> None:
    assert _classify_pattern_polarity(2.0, 0.02, 0.05, 1.5) == "fault_supporting"
    assert _classify_pattern_polarity(0.0, 0.10, 0.0, 1.5) == "protective"
    assert _classify_pattern_polarity(1.1, 0.05, 0.055, 1.5) == "uninformative"


def test_calibration_marks_normal_supporting_pattern_as_protective() -> None:
    reset_registry()

    rows = []
    for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60]:
        rows.append({
            "label": "normal",
            "vib_severity_x_delta_max": value,
            "chatter_freq_x_slope": 0.0,
        })
    for value in [0.5] * 12:
        rows.append({
            "label": "pre_stoppage",
            "vib_severity_x_delta_max": value,
            "chatter_freq_x_slope": 0.0,
        })

    df = pd.DataFrame(rows)
    calibrated = calibrate_pattern_thresholds(
        df,
        ExperimentConfig(min_discrimination_ratio=1.5),
        percentile=90.0,
    )

    threshold_info = calibrated["VIBRATION_REGIME_SHIFT"]["thresholds"]["pattern_vib_severity_x_delta_max"]
    registry_pattern = get_registry().get("VIBRATION_REGIME_SHIFT")

    assert threshold_info["polarity"] == "protective"
    assert threshold_info.get("disabled", False) is False
    assert registry_pattern is not None
    assert registry_pattern.enabled is True
    assert registry_pattern.polarity == "protective"


def test_list_patterns_dict_filters_by_protective_polarity() -> None:
    reset_registry()
    pattern = get_registry().get("VIBRATION_REGIME_SHIFT")
    assert pattern is not None

    pattern.enabled = True
    pattern.polarity = "protective"

    items = list_patterns_dict(polarity="protective")
    assert any(item["name"] == "VIBRATION_REGIME_SHIFT" for item in items)
    assert all(item["polarity"] == "protective" for item in items)


def test_protective_pattern_reduces_score_and_emits_negative_trace() -> None:
    reset_registry()
    pattern = get_registry().get("VIBRATION_REGIME_SHIFT")
    assert pattern is not None

    pattern.enabled = True
    pattern.severity = 0.85

    scorer = SignificanceScorer(SignificanceConfig(weight_protective_pattern=0.2, prior_mode="additive"))
    key = PatternKey(pattern_type=PatternType.CUSTOM, key="VIBRATION_REGIME_SHIFT")

    pattern.polarity = "fault_supporting"
    positive = scorer.score([key], external_signals={})

    pattern.polarity = "protective"
    protective = scorer.score([key], external_signals={})

    assert positive.score > protective.score
    assert protective.score == 0.0
    assert "Protective pattern: VIBRATION_REGIME_SHIFT" in protective.reasons
    assert any(
        entry["component"] == "protective_pattern_match" and entry["value"] < 0
        for entry in protective.score_trace
    )


def test_suppression_pattern_reduces_score_and_emits_negative_trace() -> None:
    scorer = SignificanceScorer(SignificanceConfig(weight_protective_pattern=0.2, prior_mode="additive"))
    suppressed = PatternKey(
        pattern_type=PatternType.CLUSTER,
        key="suppressed:power_spindle_mean_H+vib_severity_x_mean_H",
        confidence=0.7,
    )

    baseline = scorer.score([], external_signals={"breakage_prediction": 0.9})
    suppressed_result = scorer.score([suppressed], external_signals={"breakage_prediction": 0.9})

    assert suppressed_result.score < baseline.score
    assert "Suppression pattern: suppressed:power_spindle_mean_H+vib_severity_x_mean_H" in suppressed_result.reasons
    assert any(
        entry["component"] == "suppression_pattern_match" and entry["value"] < 0
        for entry in suppressed_result.score_trace
    )


def test_unified_scorer_uses_experiment_protective_weight() -> None:
    scorer = _create_unified_scorer(ExperimentConfig(weight_protective_pattern=0.35))

    assert scorer.config.weight_protective_pattern == 0.35


def test_extract_pattern_polarity_counts_normalizes_mixed_thresholds() -> None:
    counts = _extract_pattern_polarity_counts({
        "train_phase": {
            "calibrated_pattern_thresholds": {
                "A": {"polarity": "fault_supporting"},
                "B": {"polarity": "protective"},
                "C": {
                    "thresholds": {
                        "x": {"polarity": "fault_supporting"},
                        "y": {"polarity": "protective"},
                    }
                },
                "D": {"polarity": "uninformative"},
            }
        }
    })

    assert counts == {
        "fault_supporting": 1,
        "protective": 1,
        "uninformative": 1,
        "mixed": 1,
    }
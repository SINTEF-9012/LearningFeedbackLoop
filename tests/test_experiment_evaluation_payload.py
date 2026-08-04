import asyncio

from backend.agents.memory.breakage_experiment_runner import _serialize_breakage_sample
from backend.agents.memory.experiment_results import (
    _build_breakage_evaluation,
    _build_stoppage_evaluation_from_json,
)
from backend.agents.memory.experiment_routes import (
    evaluate_experiment_run,
    get_experiment_run,
)
from backend.agents.experiment.evaluator import PhaseResult, SampleResult


def test_phase_result_serializes_memory_model_breakdown_fields():
    phase = PhaseResult(
        phase="eval",
        operation="OF00003",
        sample_results=[
            SampleResult(
                sample_id="sample-1",
                label="pre_stoppage",
                operation_id="OF00003",
                tool_number="7",
                memory_id="mem-123",
                model_breakdown={
                    "available": ["harmonic"],
                    "harmonic": {"score": 0.82, "source": "harmonic_context"},
                },
            )
        ],
    )

    payload = phase.to_dict()
    sample = payload["sample_results"][0]

    assert sample["memory_id"] == "mem-123"
    assert sample["model_breakdown"]["harmonic"]["score"] == 0.82


def test_cached_stoppage_evaluation_maps_memory_and_model_breakdown():
    data = {
        "test": {
            "sample_results": [
                {
                    "sample_id": "sample-1",
                    "label": "normal",
                    "operation_id": "OF00002",
                    "tool_number": "3",
                    "memory_id": "mem-001",
                    "model_breakdown": {
                        "available": ["classical", "harmonic"],
                        "classical": {"ensemble": 0.41},
                        "harmonic": {"score": 0.63, "source": "harmonic_context"},
                    },
                }
            ],
        },
        "eval": {
            "sample_results": [
                {
                    "sample_id": "sample-2",
                    "label": "pre_stoppage",
                    "operation_id": "OF00003",
                    "tool_number": "5",
                    "memory_id": "mem-002",
                    "model_breakdown": {
                        "available": ["stoppage"],
                        "stoppage": {"probability": 0.77, "eta_s": 18.0},
                    },
                }
            ],
        },
    }

    mapped = _build_stoppage_evaluation_from_json("run-1", data)

    assert mapped is not None
    assert mapped["test"]["samples"][0]["memory_id"] == "mem-001"
    assert mapped["test"]["samples"][0]["model_breakdown"]["harmonic"]["score"] == 0.63
    assert mapped["eval"]["samples"][0]["memory_id"] == "mem-002"
    assert mapped["eval"]["samples"][0]["model_breakdown"]["stoppage"]["probability"] == 0.77


def test_breakage_evaluation_maps_feedback_audit_and_sample_evidence():
    data = {
        "config": {"test_op": "OF00013"},
        "comparison": {
            "test": {"n_samples": 1},
            "eval": {"n_samples": 2},
        },
        "folds": [
            {
                "test_operation": "OF00013",
                "co_occurrence_graph": {"A|B": 2},
                "pattern_fire_counts": {"VIBRATION_REGIME_SHIFT": 3},
                "memories_stored": 1,
                "test": {
                    "n_samples": 1,
                    "sample_results": [
                        {
                            "sample_id": "test-sample",
                            "label": "normal",
                            "operation_id": "OF00013",
                            "tool_number": "5",
                            "significance_score": 0.11,
                            "action": "IGNORE",
                            "predicted_positive": False,
                        }
                    ],
                },
                "eval": {
                    "n_samples": 1,
                    "n_predictions_flipped": 1,
                    "n_model_retrains": 1,
                    "n_propagation_events": 1,
                    "all_propagated_deltas": [{"FEED_STALL": 0.0312}],
                    "n_discovered_patterns": 1,
                    "n_suppression_patterns": 0,
                    "discovered_pattern_keys": ["discovered:alpha"],
                    "feedback_events": [
                        {
                            "source_sample_id": "eval-sample-1",
                            "feedback_action": "CONFIRM",
                            "detected_patterns": ["VIBRATION_REGIME_SHIFT"],
                            "propagated_prior_deltas": {"FEED_STALL": 0.0312},
                            "pattern_updates": [
                                {
                                    "pattern_key": "VIBRATION_REGIME_SHIFT",
                                    "polarity": "protective",
                                    "new_prior": 0.62,
                                    "delta": 0.12,
                                }
                            ],
                        }
                    ],
                    "pattern_feedback_summary": {
                        "VIBRATION_REGIME_SHIFT": {
                            "polarity": "protective",
                            "n_feedback_events": 1,
                            "n_confirms": 1,
                            "n_dismissals": 0,
                            "total_prior_delta": 0.12,
                            "mean_prior_delta": 0.12,
                            "max_abs_prior_delta": 0.12,
                            "last_prior": 0.62,
                        }
                    },
                    "sample_results": [
                        {
                            "sample_id": "eval-sample-1",
                            "label": "pre_stoppage",
                            "operation_id": "OF00013",
                            "tool_number": "5",
                            "memory_id": "mem-101",
                            "significance_score": 0.91,
                            "action": "ALERT",
                            "predicted_positive": True,
                            "raw_model_score": 0.44,
                            "pattern_rule_score": 0.33,
                            "anomaly_z_score": 2.8,
                            "prior_boost": 0.12,
                            "multi_rule_bonus": 0.05,
                            "n_rules_triggered": 2,
                            "detected_patterns": ["VIBRATION_REGIME_SHIFT"],
                            "supervised_score": 0.72,
                            "unsupervised_score": 0.61,
                            "combined_score": 0.68,
                            "tool_prior": 0.5,
                            "tool_multiplier": 1.0,
                            "weight_supervised": 0.4,
                            "weight_unsupervised": 0.6,
                            "score_trace": [{"component": "protective_pattern_match", "value": -0.17, "source": "weighted"}],
                            "model_breakdown": {"harmonic": {"score": 0.63}},
                            "feedback_given": True,
                            "feedback_action": "CONFIRM",
                            "feedback_source": "flagged",
                            "counterfactual_score": 0.73,
                            "prediction_flipped": True,
                            "prior_snapshot": {"VIBRATION_REGIME_SHIFT": 0.62},
                            "stored_in_memory": True,
                            "co_occurring_pairs": [["A", "B"]],
                            "propagated_prior_deltas": {"FEED_STALL": 0.0312},
                            "sindit_context": {"machine_state": "degraded"},
                            "explanation": "Investigate vibration spike.",
                            "explanation_source": "llm",
                            "alert_line": "High vibration with protective penalty.",
                            "alert_line_source": "fallback",
                        }
                    ],
                },
            },
            {
                "test_operation": "OF00015",
                "co_occurrence_graph": {},
                "pattern_fire_counts": {},
                "memories_stored": 0,
                "test": {"n_samples": 0, "sample_results": []},
                "eval": {
                    "n_samples": 1,
                    "n_predictions_flipped": 0,
                    "n_model_retrains": 0,
                    "n_propagation_events": 0,
                    "all_propagated_deltas": [],
                    "n_discovered_patterns": 0,
                    "n_suppression_patterns": 1,
                    "discovered_pattern_keys": ["suppressed:beta"],
                    "feedback_events": [
                        {
                            "source_sample_id": "eval-sample-2",
                            "feedback_action": "DISMISS",
                            "detected_patterns": ["VIBRATION_REGIME_SHIFT"],
                            "propagated_prior_deltas": {},
                            "pattern_updates": [
                                {
                                    "pattern_key": "VIBRATION_REGIME_SHIFT",
                                    "polarity": "protective",
                                    "new_prior": 0.57,
                                    "delta": -0.05,
                                }
                            ],
                        }
                    ],
                    "pattern_feedback_summary": {
                        "VIBRATION_REGIME_SHIFT": {
                            "polarity": "protective",
                            "n_feedback_events": 1,
                            "n_confirms": 0,
                            "n_dismissals": 1,
                            "total_prior_delta": -0.05,
                            "mean_prior_delta": -0.05,
                            "max_abs_prior_delta": 0.05,
                            "last_prior": 0.57,
                        }
                    },
                    "sample_results": [],
                },
            },
        ],
    }

    mapped = _build_breakage_evaluation("breakage-run-1", data)

    sample = mapped["eval"]["samples"][0]
    summary = mapped["eval"]["pattern_feedback_summary"]["VIBRATION_REGIME_SHIFT"]

    assert sample["memory_id"] == "mem-101"
    assert sample["score_trace"][0]["component"] == "protective_pattern_match"
    assert sample["model_breakdown"]["harmonic"]["score"] == 0.63
    assert sample["alert_line_source"] == "fallback"
    assert mapped["eval"]["feedback_events"][0]["source_sample_id"] == "eval-sample-1"
    assert summary["n_feedback_events"] == 2
    assert summary["n_confirms"] == 1
    assert summary["n_dismissals"] == 1
    assert summary["total_prior_delta"] == 0.07
    assert mapped["eval"]["n_propagation_events"] == 1
    assert mapped["eval"]["n_discovered_patterns"] == 1
    assert mapped["eval"]["n_suppression_patterns"] == 1
    assert mapped["eval"]["discovered_pattern_keys"] == ["discovered:alpha", "suppressed:beta"]


def test_get_experiment_run_sanitizes_non_finite_values(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._resolve_run_dir",
        lambda run_id: "/tmp/fake-run",
    )
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._load_run_json",
        lambda run_dir: {
            "comparison": {"threshold": float("inf")},
            "nested": [{"score": float("nan")}],
        },
    )
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._extract_pattern_polarity_counts",
        lambda data: {"fault": 1},
    )

    payload = asyncio.run(get_experiment_run("run-1"))

    assert payload["run_id"] == "run-1"
    assert payload["comparison"]["threshold"] is None
    assert payload["nested"][0]["score"] is None
    assert payload["polarity_counts"] == {"fault": 1}


def test_evaluate_experiment_run_sanitizes_cached_payload(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._resolve_run_dir",
        lambda run_id: "/tmp/fake-run",
    )
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._load_run_json",
        lambda run_dir: {"experiment_type": "stoppage", "config": {}},
    )
    monkeypatch.setattr(
        "backend.agents.memory.experiment_routes._build_stoppage_evaluation_from_json",
        lambda run_id, data: {
            "run_id": run_id,
            "eval": {"samples": [{"combined_score": float("inf")}]},
            "test": {"samples": [{"combined_score": float("nan")}]},
        },
    )

    payload = asyncio.run(evaluate_experiment_run("run-1"))

    assert payload["eval"]["samples"][0]["combined_score"] is None
    assert payload["test"]["samples"][0]["combined_score"] is None


def test_breakage_sample_serializer_preserves_evidence_fields():
    sample = SampleResult(
        sample_id="sample-serialize-1",
        label="pre_stoppage",
        operation_id="OF00013",
        tool_number="5",
        memory_id="mem-serialize-1",
        significance_score=0.91,
        action="ALERT",
        predicted_positive=True,
        prior_snapshot={"VIBRATION_REGIME_SHIFT": 0.625},
        co_occurring_pairs=[("A", "B")],
        propagated_prior_deltas={"FEED_STALL": 0.03125},
        score_trace=[{"component": "protective_pattern_match", "value": -0.17, "source": "weighted"}],
        model_breakdown={"harmonic": {"score": 0.63}},
        explanation="Investigate vibration spike.",
        explanation_source="llm",
        alert_line="High vibration with protective penalty.",
        alert_line_source="fallback",
        counterfactual_score=0.731,
        stored_in_memory=True,
    )

    payload = _serialize_breakage_sample(sample)

    assert payload["memory_id"] == "mem-serialize-1"
    assert payload["score_trace"][0]["component"] == "protective_pattern_match"
    assert payload["model_breakdown"]["harmonic"]["score"] == 0.63
    assert payload["prior_snapshot"] == {"VIBRATION_REGIME_SHIFT": 0.625}
    assert payload["co_occurring_pairs"] == [["A", "B"]]
    assert payload["propagated_prior_deltas"] == {"FEED_STALL": 0.0312}
    assert payload["counterfactual_score"] == 0.731
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.patterns.signatures import WORKPIECE_SLIP_SIGNATURE
from backend.agents.core.schemas import TimeRange
from backend.inference_streamer import _build_memory_feature_payload
from backend.agents.memory.feature_stream_bridge import _merge_stoppage_signals, create_memory_event_from_feature
from backend.agents.memory.orchestrator import MemoryEvent, MemoryEventOrchestrator, OrchestratorConfig


def test_merge_stoppage_signals_exposes_flat_keys_and_nested_debug_block():
    event = MemoryEvent(
        session_id="session-1",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
    )
    prediction = SimpleNamespace(
        probability=0.42,
        label="pre_break",
        is_stop_predicted=True,
    )

    _merge_stoppage_signals(event, prediction, gap_s=12.5)

    assert event.external_signals["stoppage_probability"] == 0.42
    assert event.external_signals["stoppage_label"] == "pre_break"
    assert event.external_signals["stoppage_eta_s"] == 12.5
    assert event.external_signals["stoppage_predictor"] == {
        "probability": 0.42,
        "label": "pre_break",
        "is_stop_predicted": True,
        "gap_s": 12.5,
    }


def test_build_metrics_for_alert_includes_model_and_harmonic_provenance():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
        )
    )
    event = MemoryEvent(
        session_id="session-2",
        time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
        external_signals={
            "anomaly_detector_score": 0.8,
            "model_source": "seed_model_v1",
            "harmonic_context_score": 0.55,
            "harmonic_context_source": "harmonic_cnn_v1",
            "stoppage_probability": 0.2,
            "stoppage_label": "pre_break",
            "stoppage_eta_s": 9.0,
        },
        raw_metrics={"Feed_Rate_Actual": 120.0},
    )
    significance = SimpleNamespace(score=0.61, prior_boost=0.12, triggered_rules=["a", "b"])

    metrics = orchestrator._build_metrics_for_alert(event, significance)

    assert metrics["Feed_Rate_Actual"] == 120.0
    assert metrics["anomaly_detector_score"] == 0.8
    assert metrics["model_source"] == "seed_model_v1"
    assert metrics["harmonic_context_score"] == 0.55
    assert metrics["harmonic_context_source"] == "harmonic_cnn_v1"
    assert metrics["sample_rate_hz"] == 1.0
    assert metrics["fs"] == 1.0
    assert metrics["significance_score"] == 0.61
    assert metrics["prior_boost"] == 0.12
    assert metrics["prior_damping_factor"] == 1.0
    assert metrics["prior_evidence_count"] == 0
    assert metrics["n_rules_triggered"] == 2


def test_create_memory_event_from_feature_preserves_raw_metrics_and_metadata():
    payload = {
        "position": 10,
        "window_seconds": 10.0,
        "raw_metrics": {
            "power_spindle_mean": 12.5,
            "feed_rate_mean": 140.0,
            "ignored": "x",
        },
        "external_signals": {
            "anomaly_detector_score": 0.82,
            "model_source": "seed_model_v1",
        },
        "metadata": {
            "source": "simulated_casedata",
            "sample_frequency": 1.0,
            "batch": {
                "batch_id": "batch-1",
                "unit_index": 2,
                "recipe_id": "recipe-9",
            },
            "harmonic_runtime": {
                "scorer_kind": "context",
                "dataset": "casedata",
            },
            "harmonic_context": {
                "source": "harmonic_context_v1",
                "feature_labels": ["X·H1"],
                "feature_values": [1.23],
                "context_weights": [0.25],
            },
        },
    }

    event = create_memory_event_from_feature(
        "session-raw",
        payload,
        {
            "fs": 1.0,
            "source": "simulated_casedata",
            "sample_frequency": 1.0,
            "batch": {"batch_id": "batch-1", "unit_count": 8},
        },
    )

    assert event.raw_metrics == {
        "power_spindle_mean": 12.5,
        "feed_rate_mean": 140.0,
    }
    assert event.external_signals["anomaly_detector_score"] == 0.82
    assert event.batch is not None
    assert event.batch.batch_id == "batch-1"
    assert event.batch.unit_index == 2
    assert event.batch.unit_count == 8
    assert event.batch.recipe_id == "recipe-9"
    assert event.metadata == {
        "source": "simulated_casedata",
        "sample_frequency": 1.0,
        "batch": {
            "batch_id": "batch-1",
            "unit_index": 2,
            "recipe_id": "recipe-9",
        },
        "harmonic_runtime": {
            "scorer_kind": "context",
            "dataset": "casedata",
        },
        "harmonic_context": {
            "source": "harmonic_context_v1",
            "feature_labels": ["X·H1"],
            "feature_values": [1.23],
            "context_weights": [0.25],
        },
    }


def test_create_memory_event_from_feature_derives_patterns_from_raw_metrics(monkeypatch):
    captured = {}

    def fake_detect_patterns(features, thresholds=None, include_details=False):
        captured["features"] = dict(features)
        captured["include_details"] = include_details
        return {
            "fired": ["SPINDLE_POWER_SURGE"],
            "details": [
                {
                    "name": "SPINDLE_POWER_SURGE",
                    "category": "power",
                    "severity": 0.82,
                    "source": "registry",
                    "description": "Surging spindle power",
                }
            ],
        }

    monkeypatch.setattr("backend.agents.memory.feature_stream_bridge.detect_patterns", fake_detect_patterns)

    event = create_memory_event_from_feature(
        "session-raw-patterns",
        {
            "position": 10,
            "window_seconds": 10.0,
            "raw_metrics": {
                "power_spindle_mean": 12.5,
                "feed_rate_mean": 140.0,
            },
            "external_signals": {
                "anomaly_detector_score": 0.82,
            },
        },
        {"fs": 1.0, "source": "simulated_casedata"},
    )

    assert captured["features"] == {
        "power_spindle_mean": 12.5,
        "feed_rate_mean": 140.0,
    }
    assert captured["include_details"] is True
    assert [pattern.key for pattern in event.patterns] == ["SPINDLE_POWER_SURGE"]


@pytest.mark.asyncio
async def test_live_harmonic_feature_payload_persists_harmonic_context_on_memory():
    session_meta = {
        "fs": 1.0,
        "source": "simulated_casedata",
        "sample_frequency": 1.0,
        "batch": {
            "batch_id": "batch-h1",
            "unit_index": 0,
            "unit_count": 3,
            "recipe_id": "recipe-h1",
        },
        "harmonic_scorer_kind": "context",
        "harmonic_dataset": "casedata",
    }
    payload = _build_memory_feature_payload(
        session_id="session-harmonic-live",
        metadata=session_meta,
        fs=1.0,
        win_start=0,
        win_end=4,
        window_seconds=4.0,
        feature_dict={
            "power_spindle_mean": 12.5,
            "Vibration_Harmonic_1_X_Amplitude": 1.23,
        },
        ensemble_score=None,
        z_anomaly=0.5,
        harmonic_score_val=0.81,
        harmonic_labels=["X·H1"],
        harmonic_values=[1.23],
        harmonic_weights=[0.25],
        ground_truth=None,
        fault_indicators={},
        model_confidence=0.0,
        model_source=None,
    )

    event = create_memory_event_from_feature(
        "session-harmonic-live",
        payload,
        session_meta,
    )

    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
            generate_explanations=False,
        )
    )
    result = await orchestrator.process_event(event)
    memory = orchestrator.get_memory(result.memory_id)

    assert result.processed is True
    assert result.memory_id is not None
    assert memory.metadata["external_signals"]["harmonic_context_score"] == pytest.approx(0.81)
    assert memory.metadata["external_signals"]["harmonic_context_source"] == "harmonic_context_v1"
    assert memory.metadata["batch"] == {
        "batch_id": "batch-h1",
        "unit_index": 0,
        "unit_count": 3,
        "recipe_id": "recipe-h1",
    }
    assert memory.metadata["harmonic_runtime"] == {
        "scorer_kind": "context",
        "dataset": "casedata",
    }
    assert memory.metadata["harmonic_context"] == {
        "source": "harmonic_context_v1",
        "feature_labels": ["X·H1"],
        "feature_values": [1.23],
        "context_weights": [0.25],
    }


def test_create_memory_event_from_feature_preserves_hypothesis_confidence():
    payload = {
        "position": 10,
        "window_seconds": 10.0,
        "metrics": {
            "dominant_frequencies": [100.0],
            "channel_rms": [0.8],
            "channel_crest_factors": [5.0],
            "phase_differences": [0.0],
            "snr_estimate": 30.0,
        },
        "metadata": {
            "sample_frequency": 1.0,
            "casedata": {
                "spindle_speed": 6000.0,
            },
        },
    }

    event = create_memory_event_from_feature(
        "session-hypothesis",
        payload,
        {
            "fs": 1.0,
            "sample_frequency": 1.0,
            "casedata": {"spindle_speed": 6000.0},
        },
    )

    hypothesis = next(pattern for pattern in event.patterns if pattern.key == WORKPIECE_SLIP_SIGNATURE)

    assert hypothesis.confidence == pytest.approx(0.5, rel=1e-6)
    assert hypothesis.additional is not None
    assert hypothesis.additional["indicators_present"] == 2
    assert hypothesis.additional["indicators_required"] == 4
    assert "spectral:spindle_freq_shift" in hypothesis.additional["supporting_patterns"]


def test_create_memory_event_from_feature_threads_observation_strength_metadata():
    payload = {
        "position": 10,
        "window_seconds": 10.0,
        "metrics": {
            "dominant_frequencies": [800.0],
            "channel_rms": [0.8],
            "channel_crest_factors": [3.0],
            "channel_kurtosis": [0.5],
            "spectral_bandwidths": [200.0],
            "spectral_centroids": [750.0],
            "snr_estimate": 35.0,
            "total_energy": 500.0,
        },
        "metadata": {
            "sample_frequency": 1.0,
        },
    }

    event = create_memory_event_from_feature(
        "session-observation",
        payload,
        {
            "fs": 1.0,
            "sample_frequency": 1.0,
        },
    )

    loud = next(pattern for pattern in event.patterns if pattern.key == "amp:loud")

    assert 0.0 < loud.confidence <= 1.0
    assert loud.source_metric == "channel_rms"
    assert loud.additional is not None
    assert "RMS amplitude" in loud.additional["reason"]
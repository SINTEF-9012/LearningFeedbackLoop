from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.fft_streamer import fft_stream_task
from backend.inference_streamer import inference_stream_task


@pytest.mark.asyncio
async def test_inference_stream_task_waits_for_live_data_and_scores_once_available(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    class StubModel:
        is_trained = True
        n_training_samples = 500

        def score_detailed_dict(self, feature_dict):
            assert feature_dict
            return {
                "ensemble": 0.81,
                "isolation_forest": 0.73,
                "lof": 0.69,
            }

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    from backend import inference_streamer as streamer

    original_seed_model = streamer._cached_model
    streamer._cached_model = StubModel()
    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)

    session = {
        "session_id": "live-inference",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "operation_id": "OF0001",
                "case_dir": Path("/tmp/demo-case"),
            },
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        streamer._cached_model = original_seed_model

    assert payload["window"] == [0, 4]
    assert payload["i_center"] == 2
    assert payload["scores"]["ensemble"] == pytest.approx(0.81)
    assert published and published[0][0] == "live-inference"
    assert published[0][1]["position"] == 4
    assert published[0][1]["window_seconds"] == pytest.approx(4.0)
    assert published[0][1]["external_signals"]["anomaly_detector_score"] == pytest.approx(0.81)
    metadata = published[0][1]["metadata"]
    assert metadata["sample_frequency"] == pytest.approx(1.0)
    assert metadata["source"] == "simulated_casedata"
    assert metadata["casedata"] == {
        "operation_id": "OF0001",
        "case_dir": "/tmp/demo-case",
    }
    assert published[0][1]["raw_metrics"]


@pytest.mark.asyncio
async def test_inference_stream_task_stays_idle_while_session_is_paused(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    class StubModel:
        is_trained = True
        n_training_samples = 500

        def score_detailed_dict(self, feature_dict):
            assert feature_dict
            return {
                "ensemble": 0.81,
                "isolation_forest": 0.73,
                "lof": 0.69,
            }

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    from backend import inference_streamer as streamer

    original_seed_model = streamer._cached_model
    streamer._cached_model = StubModel()
    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)

    session = {
        "session_id": "paused-live-inference",
        "data": {"A": [1.0, 2.0, 3.0, 4.0]},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 4,
        "running": True,
        "paused": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    try:
        await asyncio.sleep(0.15)
        assert queue.empty()
        assert published == []

        session["paused"] = False
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        streamer._cached_model = original_seed_model
        session["running_inference"] = False
        session["running"] = False

    assert payload["window"] == [0, 4]
    assert payload["scores"]["ensemble"] == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_inference_stream_task_uses_session_time_axis_for_plot_clock(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()

    class StubModel:
        is_trained = True
        n_training_samples = 500

        def score_detailed_dict(self, feature_dict):
            assert feature_dict
            return {
                "ensemble": 0.81,
                "isolation_forest": 0.73,
                "lof": 0.69,
            }

    from backend import inference_streamer as streamer

    original_seed_model = streamer._cached_model
    streamer._cached_model = StubModel()

    session = {
        "session_id": "timed-live-inference",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "time_axis_unix": [1700000000.0, 1700000001.0, 1700000002.0, 1700000003.0],
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        streamer._cached_model = original_seed_model
        session["running_inference"] = False
        session["running"] = False

    assert payload["t"] == pytest.approx(1700000003.0)
    assert payload["t0"] == pytest.approx(1700000000.0)
    assert payload["t1"] == pytest.approx(1700000003.0)
    assert payload["t_center"] == pytest.approx(1700000001.0)


@pytest.mark.asyncio
async def test_inference_stream_task_starts_from_current_session_position(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    class StubModel:
        is_trained = True
        n_training_samples = 500

        def score_detailed_dict(self, feature_dict):
            assert feature_dict
            return {
                "ensemble": 0.81,
                "isolation_forest": 0.73,
                "lof": 0.69,
            }

    from backend import inference_streamer as streamer

    original_seed_model = streamer._cached_model
    streamer._cached_model = StubModel()

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)

    session = {
        "session_id": "seeked-live-inference",
        "data": {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 6,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))

    try:
        await asyncio.sleep(0.05)
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        streamer._cached_model = original_seed_model
        session["running_inference"] = False
        session["running"] = False

    assert payload["window"] == [2, 6]
    assert payload["i_center"] == 4
    assert payload["scores"]["ensemble"] == pytest.approx(0.81)
    assert published and published[0][0] == "seeked-live-inference"


@pytest.mark.asyncio
async def test_inference_stream_task_warns_and_skips_model_on_feature_schema_mismatch(monkeypatch, caplog):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    class StubModel:
        is_trained = True
        n_training_samples = 500
        feature_names = ["legacy_feature"]

        def score_detailed_dict(self, _feature_dict):
            raise AssertionError("score_detailed_dict should not be called when schema mismatches")

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    from backend import inference_streamer as streamer

    original_seed_model = streamer._cached_model
    streamer._cached_model = StubModel()
    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)

    session = {
        "session_id": "mismatch-inference",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(inference_stream_task(session))
        await asyncio.sleep(0.05)
        session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
        session["position"] = 4

        try:
            payload = await asyncio.wait_for(queue.get(), timeout=2.0)
            session["running_inference"] = False
            session["running"] = False
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            streamer._cached_model = original_seed_model

    assert payload["scores"]["ensemble"] == pytest.approx(0.5)
    assert payload["scores"]["isolation_forest"] == pytest.approx(0.5)
    assert payload["scores"]["lof"] == pytest.approx(0.5)
    assert published == []
    assert "starting sid=mismatch-inference source=simulated_casedata" in caplog.text
    assert "feature_keys=" in caplog.text
    assert "seed model schema mismatch sid=mismatch-inference" in caplog.text


@pytest.mark.asyncio
async def test_inference_stream_task_emits_harmonic_outputs_with_weights(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    class StubHarmonicScorer:
        config = SimpleNamespace(
            context_param_keys=["spindle_speed", "feed_rate"],
            context_param_sources={
                "spindle_speed": "spindle_speed_mean",
                "feed_rate": "feed_rate_mean",
            },
            context_param_stats={},
            harmonic_mode="pre_extracted",
            harmonic_columns=["Vibration_Harmonic_1_X_Amplitude"],
            cnn_window=1,
        )

        def is_available(self):
            return True

        def score(self, harmonics, params):
            assert harmonics.shape == (1, 1)
            assert params.shape == (2,)
            return {
                "harmonic_context_score": 0.66,
                "context_weights": [0.25],
                "feature_labels": ["X·H1"],
                "harmonic_values": [1.23],
                "model_source": "harmonic_context_test",
            }

    from backend import inference_streamer as streamer

    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)
    monkeypatch.setattr(streamer, "_get_seed_model", lambda: None)
    monkeypatch.setattr(streamer, "_get_harmonic_scorer", lambda: StubHarmonicScorer())
    monkeypatch.setattr(
        streamer,
        "_features_from_channels",
        lambda *args, **kwargs: {
            "Vibration_Harmonic_1_X_Amplitude": 1.23,
            "spindle_speed_mean": 6000.0,
            "feed_rate_mean": 120.0,
        },
    )

    session = {
        "session_id": "harmonic-inference",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        session["running_inference"] = False
        session["running"] = False

    assert payload["scores"]["harmonic_context_score"] == pytest.approx(0.66)
    assert payload["harmonic_context_weights"] == [0.25]
    assert payload["harmonic_feature_labels"] == ["X·H1"]
    assert payload["harmonic_values"] == pytest.approx([1.23])
    assert published and published[0][1]["external_signals"]["harmonic_context_score"] == pytest.approx(0.66)
    assert published[0][1]["metadata"]["harmonic_context"] == {
        "source": "harmonic_context_v1",
        "feature_labels": ["X·H1"],
        "feature_values": [1.23],
        "context_weights": [0.25],
    }


@pytest.mark.asyncio
async def test_inference_stream_task_passes_pre_extracted_harmonic_window_columns_into_raw_metrics(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    class StubHarmonicScorer:
        config = SimpleNamespace(
            context_param_keys=["spindle_speed", "feed_rate"],
            context_param_sources={
                "spindle_speed": "spindle_speed_mean",
                "feed_rate": "feed_rate_mean",
            },
            context_param_stats={},
            harmonic_mode="pre_extracted",
            harmonic_columns=["Vibration_Harmonic_1_X_Amplitude"],
            cnn_window=1,
        )

        def is_available(self):
            return True

        def score(self, harmonics, params):
            np.testing.assert_allclose(harmonics, np.array([[1.23]], dtype=np.float32))
            np.testing.assert_allclose(params, np.array([6000.0, 120.0], dtype=np.float32))
            return {
                "harmonic_context_score": 0.72,
                "context_weights": [0.4],
                "feature_labels": ["X·H1"],
                "harmonic_values": [1.23],
                "model_source": "harmonic_context_test",
            }

    from backend import inference_streamer as streamer

    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)
    monkeypatch.setattr(streamer, "_get_seed_model", lambda: None)
    monkeypatch.setattr(streamer, "_get_harmonic_scorer", lambda *args, **kwargs: StubHarmonicScorer())
    monkeypatch.setattr(
        streamer,
        "_features_from_channels",
        lambda *args, **kwargs: {
            "spindle_speed_mean": 6000.0,
            "feed_rate_mean": 120.0,
        },
    )

    session = {
        "session_id": "harmonic-pass-through",
        "data": {
            "A": [],
            "Vibration_Harmonic_1_X_Amplitude": [],
        },
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {
            "channels": ["A", "Vibration_Harmonic_1_X_Amplitude"],
            "speed": 1000.0,
        },
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["data"]["Vibration_Harmonic_1_X_Amplitude"].extend([0.1, 0.2, 0.3, 1.23])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        session["running_inference"] = False
        session["running"] = False

    assert payload["scores"]["harmonic_context_score"] == pytest.approx(0.72)
    assert "Vibration_Harmonic_1_X_Amplitude" not in payload["features"]
    assert published and published[0][1]["raw_metrics"]["Vibration_Harmonic_1_X_Amplitude"] == pytest.approx(1.23)


@pytest.mark.asyncio
async def test_inference_stream_task_passes_pair_fft_window_columns_into_raw_metrics(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    published: list[tuple[str, dict]] = []

    async def fake_publish_feature(session_id: str, payload: dict):
        published.append((session_id, payload))

    class StubPairScorer:
        config = SimpleNamespace(
            scorer_kind="pair",
            context_param_keys=["spindle_speed", "feed_rate", "teeth_count"],
            context_param_sources={
                "spindle_speed": "CNC_parameters_Programed_spindle_speed",
                "feed_rate": "Axis_FeedRate_commanded",
                "teeth_count": "CNC_parameters_teeth_num",
            },
            context_param_stats={},
            harmonic_mode="pre_extracted",
            pair_frequency_column_patterns=[r"Accel_FFT_Acc\d+_range\d+_Frequencies_\d+"],
            pair_amplitude_column_patterns=[r"Accel_FFT_Acc\d+_range\d+_Amplitudes_\d+"],
            harmonic_columns=["Acc1·P0", "Acc2·P0"],
            k_peaks=1,
            f_max_rel=12.0,
            cnn_window=1,
        )

        def is_available(self):
            return True

        def score(self, pairs, params):
            np.testing.assert_allclose(
                pairs,
                np.array([[[[1.0, 3.0]], [[2.0, 4.0]]]], dtype=np.float32),
            )
            np.testing.assert_allclose(params, np.array([1200.0, 150.0, 9.0], dtype=np.float32))
            return {
                "harmonic_context_score": 0.77,
                "feature_labels": ["Acc1·P0", "Acc2·P0"],
                "harmonic_values": [3.0, 4.0],
                "model_source": "harmonic_pair_test",
            }

    from backend import inference_streamer as streamer

    monkeypatch.setattr(streamer, "publish_feature", fake_publish_feature)
    monkeypatch.setattr(streamer, "_get_seed_model", lambda: None)
    monkeypatch.setattr(streamer, "_get_harmonic_scorer", lambda *args, **kwargs: StubPairScorer())
    monkeypatch.setattr(streamer, "_features_from_channels", lambda *args, **kwargs: {})

    session = {
        "session_id": "pair-pass-through",
        "data": {
            "A": [],
            "CNC_parameters_Programed_spindle_speed": [],
            "Axis_FeedRate_commanded": [],
            "CNC_parameters_teeth_num": [],
            "Accel_FFT_Acc1_range1_Frequencies_0": [],
            "Accel_FFT_Acc1_range1_Amplitudes_0": [],
            "Accel_FFT_Acc2_range1_Frequencies_0": [],
            "Accel_FFT_Acc2_range1_Amplitudes_0": [],
        },
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
        },
        "config": {
            "channels": [
                "A",
                "CNC_parameters_Programed_spindle_speed",
                "Axis_FeedRate_commanded",
                "CNC_parameters_teeth_num",
                "Accel_FFT_Acc1_range1_Frequencies_0",
                "Accel_FFT_Acc1_range1_Amplitudes_0",
                "Accel_FFT_Acc2_range1_Frequencies_0",
                "Accel_FFT_Acc2_range1_Amplitudes_0",
            ],
            "speed": 1000.0,
        },
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["data"]["CNC_parameters_Programed_spindle_speed"].extend([1200.0] * 4)
    session["data"]["Axis_FeedRate_commanded"].extend([150.0] * 4)
    session["data"]["CNC_parameters_teeth_num"].extend([9.0] * 4)
    session["data"]["Accel_FFT_Acc1_range1_Frequencies_0"].extend([10.0, 10.0, 10.0, 20.0])
    session["data"]["Accel_FFT_Acc1_range1_Amplitudes_0"].extend([1.0, 1.0, 1.0, 3.0])
    session["data"]["Accel_FFT_Acc2_range1_Frequencies_0"].extend([20.0, 20.0, 20.0, 40.0])
    session["data"]["Accel_FFT_Acc2_range1_Amplitudes_0"].extend([2.0, 2.0, 2.0, 4.0])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        session["running_inference"] = False
        session["running"] = False

    assert payload["scores"]["harmonic_context_score"] == pytest.approx(0.77)
    assert published and published[0][1]["raw_metrics"]["Accel_FFT_Acc1_range1_Frequencies_0"] == pytest.approx(20.0)
    assert published[0][1]["raw_metrics"]["Accel_FFT_Acc2_range1_Amplitudes_0"] == pytest.approx(4.0)
    assert published[0][1]["raw_metrics"]["CNC_parameters_teeth_num"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_inference_stream_task_records_runtime_tool_observation_from_live_features(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()
    observations: list[tuple[str, dict]] = []

    class StubToolAuditModule:
        @staticmethod
        def record_tool_observation(session_id: str, cutting_context: dict):
            observations.append((session_id, cutting_context))

    from backend import inference_streamer as streamer
    from backend.agents.sindit import tool_audit as tool_audit_module

    monkeypatch.setattr(tool_audit_module, "record_tool_observation", StubToolAuditModule.record_tool_observation)
    monkeypatch.setattr(streamer, "publish_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamer, "_get_seed_model", lambda: None)
    monkeypatch.setattr(streamer, "_get_harmonic_scorer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        streamer,
        "_features_from_channels",
        lambda *args, **kwargs: {
            "spindle_speed_mean": 1800.0,
            "feed_rate_mean": 900.0,
        },
    )

    session = {
        "session_id": "harmonic-observation",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "operation_id": "OF0055",
                "tool_id": "T55",
                "cutting_context": {
                    "machine_id": "Site_b - MACHINE_B1 - CASE_B1",
                    "tool_id": "T55",
                    "tool_diameter": 20.0,
                    "num_teeth": 4,
                    "extra": {
                        "machine_family": "builder_b12",
                        "tool_number": 55,
                        "sindit_tool_iri": "urn:lfl:tool:builder_b12-t55",
                    },
                },
            },
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    try:
        await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        session["running_inference"] = False
        session["running"] = False

    assert observations and observations[0][0] == "harmonic-observation"
    assert observations[0][1]["spindle_speed"] == pytest.approx(1800.0)
    assert observations[0][1]["feed_rate"] == pytest.approx(900.0)
    assert observations[0][1]["tool_diameter"] == pytest.approx(20.0)
    assert observations[0][1]["num_teeth"] == 4


@pytest.mark.asyncio
async def test_inference_stream_task_emits_harmonic_outputs_without_trained_scorer(monkeypatch):
    queue: asyncio.Queue = asyncio.Queue()

    from backend import inference_streamer as streamer

    monkeypatch.setattr(streamer, "_get_seed_model", lambda: None)
    monkeypatch.setattr(streamer, "_get_harmonic_scorer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        streamer,
        "_features_from_channels",
        lambda *args, **kwargs: {
            "Vibration_Harmonic_1_X_Amplitude": 1.23,
            "spindle_speed_mean": 6000.0,
            "feed_rate_mean": 120.0,
        },
    )

    session = {
        "session_id": "harmonic-outputs-only",
        "data": {"A": []},
        "metadata": {
            "sample_frequency": 1.0,
            "source": "simulated_casedata",
            "casedata": {
                "operation_id": "OF0010",
            },
        },
        "config": {"channels": ["A"], "speed": 1000.0},
        "inference_config": {"window_samples": 4, "window_seconds": 4.0, "stride_samples": 2},
        "position": 0,
        "running": True,
        "running_inference": True,
        "inference_subscribers": [queue],
        "inference_task": None,
    }

    task = asyncio.create_task(inference_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    try:
        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        session["running_inference"] = False
        session["running"] = False
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        session["running_inference"] = False
        session["running"] = False

    assert payload["harmonic_feature_labels"] == ["X·H1"]
    assert payload["harmonic_values"] == pytest.approx([1.23])
    assert "harmonic_context_score" not in payload["scores"]
    assert "harmonic_context_weights" not in payload


@pytest.mark.asyncio
async def test_fft_stream_task_waits_for_live_data_and_emits_when_window_ready():
    queue: asyncio.Queue = asyncio.Queue()
    session = {
        "session_id": "live-fft",
        "data": {"A": []},
        "metadata": {"sample_frequency": 1.0},
        "config": {"channels": ["A"], "speed": 1000.0},
        "fft_config": {"nfft": 4, "overlap": 0.5, "inherit_speed": True},
        "position": 0,
        "running": True,
        "running_fft": True,
        "fft_subscribers": [queue],
        "fft_task": None,
    }

    task = asyncio.create_task(fft_stream_task(session))
    await asyncio.sleep(0.05)
    session["data"]["A"].extend([1.0, 2.0, 3.0, 4.0])
    session["position"] = 4

    payload = await asyncio.wait_for(queue.get(), timeout=2.0)
    session["running_fft"] = False
    session["running"] = False
    await asyncio.wait_for(task, timeout=2.0)

    assert payload["nfft"] == 4
    assert payload["i_center"] == 2

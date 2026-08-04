from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.agents.core.schemas import TimeRange
from backend.agents.memory import orchestrator as memory_orchestrator_module
from backend.agents.memory.enrichment import enrich_with_harmonic_score
from backend.agents.processing import harmonic_config as harmonic_config_module
from backend.agents.processing import harmonic_pair_model as harmonic_pair_model_module
from backend.agents.processing import harmonic_peak_pairs as harmonic_peak_pairs_module
from backend.agents.processing import harmonic_pair_trainer as harmonic_pair_trainer_module
from backend.agents.processing import harmonic_runtime as harmonic_runtime_module
from backend.agents.processing import tool_lookup as tool_lookup_module
from backend.agents.processing.harmonic_features import extract_context_params
from backend.agents.processing.harmonic_trainer import TrainResult
from backend.inference_streamer import (
    _augment_harmonic_runtime_features,
    _compute_harmonic_window,
    _harmonic_scorer_candidates,
)
from backend.agents.memory.orchestrator import MemoryEvent, MemoryEventOrchestrator, OrchestratorConfig
from backend.routers import harmonic as harmonic_router
from backend.routers import sessions as sessions_router
from backend.routers.dependencies import PlaybackConfigUpdate


def _write_casedata_pair_operation(
    root: Path,
    case_dir: str,
    operation_id: str,
    timestamps: list[str],
) -> None:
    op_dir = root / case_dir / operation_id
    op_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "timestamp": timestamps,
            "Vibration_Severity_X": [0.1] * len(timestamps),
            "Vibration_Severity_Y": [0.2] * len(timestamps),
            "Vibration_Peak_1_X_Frequency": [40.0] * len(timestamps),
            "Vibration_Peak_1_X_Amplitude": [1.0 + idx for idx in range(len(timestamps))],
            "Vibration_Peak_1_Y_Frequency": [50.0] * len(timestamps),
            "Vibration_Peak_1_Y_Amplitude": [1.5 + idx for idx in range(len(timestamps))],
        }
    ).to_csv(op_dir / f"{operation_id}_7DTZHE.csv", index=False)

    pd.DataFrame(
        {
            "timestamp": timestamps,
            "Spindle_Speed_Actual": [1200.0] * len(timestamps),
            "Feed_Rate_Actual": [100.0] * len(timestamps),
            "Tool_Number": [7.0] * len(timestamps),
        }
    ).to_csv(op_dir / f"{operation_id}_TYZBPS.csv", index=False)


def test_load_casedata_requires_labelled_feature_file(tmp_path, monkeypatch):
    monkeypatch.setattr(harmonic_router, "_FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(harmonic_router, "_CASEDATA_DIR", tmp_path / "casedata")

    with pytest.raises(FileNotFoundError, match="breakage_features.csv"):
        harmonic_router._load_casedata(str(tmp_path / "missing-dataset"))


def test_load_stoppage_1hz_flattens_npz_rows(tmp_path):
    np.savez(
        tmp_path / "stoppage_raw_series.npz",
        data=np.array(
            [
                [
                    [1.0, 2.0, 3.0],
                    [100.0, 101.0, 102.0],
                    [10.0, 11.0, 12.0],
                ]
            ],
            dtype=np.float32,
        ),
        sample_ids=np.array(["sample-1"]),
        labels=np.array(["pre_stoppage"]),
        channel_names=np.array(
            [
                "Vibration_Harmonic_1_X_Amplitude",
                "Spindle_Speed_Actual",
                "Feed_Rate_Actual",
            ]
        ),
        sample_rate_hz=np.array(1.0),
    )

    df = harmonic_router._load_stoppage_1hz(str(tmp_path))

    assert len(df) == 3
    assert set(df["operation_id"]) == {"sample-1"}
    assert set(df["label"]) == {"pre_stoppage"}
    assert list(df["Vibration_Harmonic_1_X_Amplitude"]) == pytest.approx([1.0, 2.0, 3.0])


def test_load_pair_casedata_backfills_case_dir_from_event_timestamp(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    casedata_dir = tmp_path / "casedata"
    features_dir.mkdir()
    casedata_dir.mkdir()

    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "label": "normal",
                "operation_id": "OF00001",
                "tool_number": 7.0,
                "event_timestamp": "2025-04-29T00:30:00Z",
                "window_seconds": 60.0,
                "gap_seconds": 0.0,
                "trim_seconds_removed": 0.0,
            }
        ]
    ).to_csv(features_dir / "stoppage_features.csv", index=False)

    _write_casedata_pair_operation(
        casedata_dir,
        "Site_b - MACHINE_B1 - CASE_B1",
        "OF00001",
        [
            "2025-03-03T00:29:10Z",
            "2025-03-03T00:29:40Z",
            "2025-03-03T00:30:10Z",
        ],
    )
    _write_casedata_pair_operation(
        casedata_dir,
        "SITE_C - MACHINE_C1 - CASE_C1",
        "OF00001",
        [
            "2025-04-29T00:29:10Z",
            "2025-04-29T00:29:40Z",
            "2025-04-29T00:30:10Z",
        ],
    )

    monkeypatch.setattr(harmonic_router, "_FEATURES_DIR", features_dir)
    monkeypatch.setattr(harmonic_router, "_CASEDATA_DIR", casedata_dir)

    df = harmonic_router._load_pair_casedata()

    assert set(df["case_dir"]) == {"SITE_C - MACHINE_C1 - CASE_C1"}
    assert set(df["raw_operation_id"]) == {"OF00001"}
    assert set(df["operation_id"]) == {"sample-1"}


def test_load_pair_casedata_enriches_lfl_context_columns(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    casedata_dir = tmp_path / "casedata"
    features_dir.mkdir()
    casedata_dir.mkdir()

    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "label": "normal",
                "operation_id": "OF00001",
                "tool_number": 7.0,
                "case_dir": "SITE_C - MACHINE_C1 - CASE_C1",
                "event_timestamp": "2025-04-29T00:30:00Z",
                "window_seconds": 60.0,
                "gap_seconds": 0.0,
                "trim_seconds_removed": 0.0,
            }
        ]
    ).to_csv(features_dir / "stoppage_features.csv", index=False)

    _write_casedata_pair_operation(
        casedata_dir,
        "SITE_C - MACHINE_C1 - CASE_C1",
        "OF00001",
        [
            "2025-04-29T00:29:10Z",
            "2025-04-29T00:29:40Z",
            "2025-04-29T00:30:10Z",
        ],
    )

    monkeypatch.setattr(harmonic_router, "_FEATURES_DIR", features_dir)
    monkeypatch.setattr(harmonic_router, "_CASEDATA_DIR", casedata_dir)
    monkeypatch.setattr(tool_lookup_module, "resolve_machine_family", lambda machine_id, path=None: "press_c-20-0482-010")
    monkeypatch.setattr(
        tool_lookup_module,
        "resolve_tool_context",
        lambda machine_family, tool_number, **kwargs: {
            "machine_family": machine_family,
            "tool_number": int(tool_number),
            "tool_id": f"T{int(tool_number)}",
            "tool_diameter": 80.0,
            "num_teeth": 6,
        },
    )

    df = harmonic_router._load_pair_casedata()

    assert set(df["machine_family"].dropna()) == {"press_c-20-0482-010"}
    assert set(df["tool_id"].dropna()) == {"T7"}
    assert df["tool_diameter"].dropna().iloc[0] == pytest.approx(80.0)
    assert df["num_teeth"].dropna().iloc[0] == pytest.approx(6.0)
    assert df["feed_per_tooth"].dropna().iloc[0] == pytest.approx(100.0 / (6.0 * 1200.0))


def test_load_pair_casedata_requires_case_dir_when_timestamp_is_ambiguous(tmp_path, monkeypatch):
    features_dir = tmp_path / "features"
    casedata_dir = tmp_path / "casedata"
    features_dir.mkdir()
    casedata_dir.mkdir()

    pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "label": "normal",
                "operation_id": "OF00001",
                "tool_number": 7.0,
                "event_timestamp": "2025-04-29T00:30:00Z",
                "window_seconds": 60.0,
                "gap_seconds": 0.0,
                "trim_seconds_removed": 0.0,
            }
        ]
    ).to_csv(features_dir / "stoppage_features.csv", index=False)

    _write_casedata_pair_operation(
        casedata_dir,
        "Site_b - MACHINE_B1 - CASE_B1",
        "OF00001",
        [
            "2025-04-29T00:29:10Z",
            "2025-04-29T00:29:40Z",
            "2025-04-29T00:30:10Z",
        ],
    )
    _write_casedata_pair_operation(
        casedata_dir,
        "SITE_C - MACHINE_C1 - CASE_C1",
        "OF00001",
        [
            "2025-04-29T00:29:10Z",
            "2025-04-29T00:29:40Z",
            "2025-04-29T00:30:10Z",
        ],
    )

    monkeypatch.setattr(harmonic_router, "_FEATURES_DIR", features_dir)
    monkeypatch.setattr(harmonic_router, "_CASEDATA_DIR", casedata_dir)

    with pytest.raises(ValueError, match="require case_dir"):
        harmonic_router._load_pair_casedata()


def test_orchestrator_waits_for_full_harmonic_window_before_scoring():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
        )
    )

    scorer = SimpleNamespace(
        config=SimpleNamespace(
            context_param_keys=["spindle_speed", "feed_rate"],
            context_param_sources={},
            context_param_stats={},
            harmonic_columns=["Vibration_Harmonic_1_X_Amplitude"],
            cnn_window=3,
        ),
        calls=[],
    )

    def _is_available():
        return True

    def _score(harmonics, params):
        scorer.calls.append((np.asarray(harmonics), np.asarray(params)))
        return {
            "harmonic_context_score": 0.81,
            "model_source": "harmonic_context_test",
            "decision_threshold": 0.65,
        }

    scorer.is_available = _is_available
    scorer.score = _score
    orchestrator.harmonic_scorer = scorer

    def _event(value: float) -> MemoryEvent:
        return MemoryEvent(
            session_id="session-1",
            time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
            raw_metrics={
                "Vibration_Harmonic_1_X_Amplitude": value,
                "spindle_speed": 1200.0,
                "feed_rate": 150.0,
            },
        )

    def _score(event):
        return enrich_with_harmonic_score(
            event,
            harmonic_scorer=orchestrator.harmonic_scorer,
            row_history=orchestrator._harmonic_row_history,
        )

    assert _score(_event(1.0)) == {}
    assert _score(_event(2.0)) == {}

    third = _score(_event(3.0))

    assert third == {
        "harmonic_context_score": 0.81,
        "harmonic_context_source": "harmonic_context_test",
        "harmonic_context_threshold": 0.65,
        "harmonic_context_triggered": True,
    }
    assert len(scorer.calls) == 1
    np.testing.assert_allclose(scorer.calls[0][0], np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.testing.assert_allclose(scorer.calls[0][1], np.array([1200.0, 150.0], dtype=np.float32))


def test_inference_streamer_pair_candidate_is_opt_in():
    default_candidates = [name for name, _ in _harmonic_scorer_candidates({})]
    pair_candidates = [
        name for name, _ in _harmonic_scorer_candidates({"harmonic_scorer_kind": "pair"})
    ]
    pair_lfl_candidates = [
        name
        for name, _ in _harmonic_scorer_candidates(
            {"harmonic_scorer_kind": "pair", "harmonic_dataset": "pair_lfl"}
        )
    ]
    pair_with_context_dataset = [
        name
        for name, _ in _harmonic_scorer_candidates(
            {"harmonic_scorer_kind": "pair", "harmonic_dataset": "casedata"}
        )
    ]
    pair_with_casedata_metadata = [
        name
        for name, _ in _harmonic_scorer_candidates(
            {
                "harmonic_scorer_kind": "pair",
                "source": "simulated_casedata",
                "casedata": {"case_dir": "SITE_C - MACHINE_C1 - CASE_C1"},
            }
        )
    ]
    context_with_pair_dataset = [
        name
        for name, _ in _harmonic_scorer_candidates(
            {"harmonic_scorer_kind": "context", "harmonic_dataset": "pair_raw"}
        )
    ]

    assert "pair_raw" not in default_candidates
    assert pair_candidates == ["pair_raw"]
    assert pair_lfl_candidates == ["pair_lfl"]
    assert pair_with_context_dataset == ["pair_lfl"]
    assert pair_with_casedata_metadata == ["pair_lfl"]
    assert context_with_pair_dataset[0] != "pair_raw"


def test_augment_harmonic_runtime_features_backfills_lfl_context_from_metadata():
    cfg = SimpleNamespace(
        context_param_stats={},
        context_param_sources={
            "d": "tool_diameter",
            "z": "num_teeth",
            "n": "spindle_speed_mean",
            "f": "feed_per_tooth",
            "vf": "feed_rate_mean",
        },
        harmonic_mode="raw",
    )

    features = _augment_harmonic_runtime_features(
        {"spindle_speed_mean": 1200.0, "feed_rate_mean": 720.0},
        {},
        cfg,
        {
            "casedata": {
                "cutting_context": {
                    "tool_diameter": 80.0,
                    "num_teeth": 6,
                }
            }
        },
    )

    assert features["tool_diameter"] == pytest.approx(80.0)
    assert features["num_teeth"] == pytest.approx(6.0)
    assert features["feed_per_tooth"] == pytest.approx(720.0 / (6.0 * 1200.0))


def test_extract_context_params_uses_training_means_for_missing_lfl_values():
    ctx_sources = {
        "d": "tool_diameter",
        "z": "num_teeth",
        "n": "spindle_speed_mean",
        "f": "feed_per_tooth",
        "vf": "feed_rate_mean",
    }
    ctx_stats = {
        "d": {"mean": 100.0, "std": 10.0, "source_column": "tool_diameter"},
        "z": {"mean": 6.0, "std": 1.0, "source_column": "num_teeth"},
        "n": {"mean": 1600.0, "std": 400.0, "source_column": "spindle_speed_mean"},
        "f": {"mean": 0.04, "std": 0.01, "source_column": "feed_per_tooth"},
        "vf": {"mean": 200.0, "std": 50.0, "source_column": "feed_rate_mean"},
    }

    ctx_vec = extract_context_params(
        {
            "tool_diameter": 125.0,
            "spindle_speed_mean": 750.0,
            "feed_rate_mean": 195.0,
        },
        ["d", "z", "n", "f", "vf"],
        ctx_sources,
        ctx_stats,
        normalize=False,
    )

    assert ctx_vec.tolist() == pytest.approx(
        [
            125.0,
            ctx_stats["z"]["mean"],
            750.0,
            ctx_stats["f"]["mean"],
            195.0,
        ]
    )


def test_inference_streamer_site_a_candidates_prefer_casedata_family():
    names = [
        name
        for name, _ in _harmonic_scorer_candidates(
            {
                "source": "simulated_casedata",
                "casedata": {"case_dir": "Site_a - MACHINE_A1 - CASE_A1"},
            }
        )
    ]

    assert names[:4] == ["casedata_peaks", "casedata", "stoppage_1hz", "site_a_line2"]


def test_pair_trainer_split_holds_out_multiple_operations_per_label():
    trainer = harmonic_pair_trainer_module.HarmonicPairTrainer(SimpleNamespace())
    sample_ops = []
    sample_labels = []

    for idx in range(5):
        sample_ops.extend([f"normal-{idx}"] * 3)
        sample_labels.extend([0, 0, 0])

    for idx in range(5):
        sample_ops.extend([f"positive-{idx}"] * 2)
        sample_labels.extend([1, 1])

    train_idx, val_idx = trainer._split_train_val(sample_ops, sample_labels, 0.2)

    train_ops = {sample_ops[i] for i in train_idx}
    val_ops = {sample_ops[i] for i in val_idx}
    val_labels = {sample_labels[i] for i in val_idx}

    assert len(val_ops) == 2
    assert train_ops.isdisjoint(val_ops)
    assert val_labels == {0, 1}


def test_pair_casedata_trains_one_trailing_window_per_operation():
    config = harmonic_config_module.pair_casedata_preset(cnn_window=2, k_peaks=1)
    trainer = harmonic_pair_trainer_module.HarmonicPairTrainer(config)
    df = pd.DataFrame(
        [
            {
                "operation_id": "normal-1",
                "label": "normal",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 1.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 1.5,
            },
            {
                "operation_id": "normal-1",
                "label": "normal",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 2.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 2.5,
            },
            {
                "operation_id": "normal-1",
                "label": "normal",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 3.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 3.5,
            },
            {
                "operation_id": "positive-1",
                "label": "pre_stoppage",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 7.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 7.5,
            },
            {
                "operation_id": "positive-1",
                "label": "pre_stoppage",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 8.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 8.5,
            },
            {
                "operation_id": "positive-1",
                "label": "pre_stoppage",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 9.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 9.5,
            },
        ]
    )

    specs = harmonic_peak_pairs_module.discover_peak_pair_columns(
        list(df.columns),
        frequency_patterns=config.pair_frequency_column_patterns,
        amplitude_patterns=config.pair_amplitude_column_patterns,
        k_peaks=config.k_peaks,
    )
    ctx_cols = {"spindle_speed": "spindle_speed_mean", "feed_rate": "feed_rate_mean"}
    ctx_stats = trainer._compute_context_stats(df, ctx_cols)
    labels = df[config.target_label].isin(config.positive_labels).astype(int).to_numpy()

    _, _, sample_labels, sample_ops = trainer._build_samples(
        df,
        specs,
        ctx_cols,
        ctx_stats,
        labels,
        "operation_id",
    )

    assert config.pair_sample_mode == "trailing_window"
    assert sample_ops == ["normal-1", "positive-1"]
    assert sample_labels == [0, 1]


def test_update_config_syncs_harmonic_selection_into_session_metadata():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sessions={
                    "session-1": {
                        "session_id": "session-1",
                        "config": {"speed": 1.0, "samples_per_tick": 32},
                        "metadata": {},
                        "data": {},
                        "running": False,
                    }
                }
            )
        )
    )

    response = sessions_router.update_config(
        "session-1",
        PlaybackConfigUpdate(harmonic_scorer_kind="pair", harmonic_dataset="auto"),
        request,
    )

    assert response["config"]["harmonic_scorer_kind"] == "pair"
    assert response["config"]["harmonic_dataset"] == "pair_raw"
    assert request.app.state.sessions["session-1"]["metadata"] == {
        "pause_on_alert": False,
        "harmonic_scorer_kind": "pair",
        "harmonic_dataset": "pair_raw",
    }


def test_update_config_prefers_pair_lfl_for_casedata_sessions():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sessions={
                    "session-1": {
                        "session_id": "session-1",
                        "config": {"speed": 1.0, "samples_per_tick": 32},
                        "metadata": {
                            "source": "simulated_casedata",
                            "casedata": {
                                "case_dir": "SITE_C - MACHINE_C1 - CASE_C1",
                                "operation_id": "OF00001",
                            },
                        },
                        "data": {},
                        "running": False,
                    }
                }
            )
        )
    )

    response = sessions_router.update_config(
        "session-1",
        PlaybackConfigUpdate(harmonic_scorer_kind="pair", harmonic_dataset="auto"),
        request,
    )

    assert response["config"]["harmonic_dataset"] == "pair_lfl"
    assert request.app.state.sessions["session-1"]["metadata"]["harmonic_dataset"] == "pair_lfl"


def test_update_config_defaults_site_c_sessions_to_pair_lfl():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sessions={
                    "session-1": {
                        "session_id": "session-1",
                        "config": {
                            "speed": 1.0,
                            "samples_per_tick": 32,
                            "harmonic_scorer_kind": "pair",
                            "harmonic_dataset": "pair_casedata",
                        },
                        "metadata": {
                            "source": "simulated_casedata",
                            "casedata": {
                                "case_dir": "SITE_C - MACHINE_C1 - CASE_C1",
                                "operation_id": "OF00001",
                            },
                            "harmonic_scorer_kind": "pair",
                            "harmonic_dataset": "pair_casedata",
                        },
                        "data": {},
                        "running": False,
                    }
                }
            )
        )
    )

    response = sessions_router.update_config(
        "session-1",
        PlaybackConfigUpdate(harmonic_scorer_kind="context", harmonic_dataset="casedata"),
        request,
    )

    assert response["config"]["harmonic_scorer_kind"] == "pair"
    assert response["config"]["harmonic_dataset"] == "pair_lfl"
    assert request.app.state.sessions["session-1"]["metadata"]["harmonic_scorer_kind"] == "pair"
    assert request.app.state.sessions["session-1"]["metadata"]["harmonic_dataset"] == "pair_lfl"


def test_update_config_respects_explicit_pair_casedata_for_site_c_sessions():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sessions={
                    "session-1": {
                        "session_id": "session-1",
                        "config": {
                            "speed": 1.0,
                            "samples_per_tick": 32,
                            "harmonic_scorer_kind": "pair",
                            "harmonic_dataset": "pair_casedata",
                        },
                        "metadata": {
                            "source": "simulated_casedata",
                            "casedata": {
                                "case_dir": "SITE_C - MACHINE_C1 - CASE_C1",
                                "operation_id": "OF00001",
                            },
                            "harmonic_scorer_kind": "pair",
                            "harmonic_dataset": "pair_casedata",
                        },
                        "data": {},
                        "running": False,
                    }
                }
            )
        )
    )

    response = sessions_router.update_config(
        "session-1",
        PlaybackConfigUpdate(harmonic_scorer_kind="pair", harmonic_dataset="pair_casedata"),
        request,
    )

    assert response["config"]["harmonic_scorer_kind"] == "pair"
    assert response["config"]["harmonic_dataset"] == "pair_casedata"
    assert request.app.state.sessions["session-1"]["metadata"]["harmonic_scorer_kind"] == "pair"
    assert request.app.state.sessions["session-1"]["metadata"]["harmonic_dataset"] == "pair_casedata"


def test_orchestrator_pair_mode_waits_for_full_window_before_scoring():
    orchestrator = MemoryEventOrchestrator(
        config=OrchestratorConfig(
            use_classical_models=False,
            enable_harmonic_scorer=False,
            dispatch_alerts=False,
        )
    )

    scorer = SimpleNamespace(
        config=SimpleNamespace(
            scorer_kind="pair",
            context_param_keys=["spindle_speed", "feed_rate", "teeth_count"],
            context_param_sources={
                "spindle_speed": "spindle_speed",
                "feed_rate": "feed_rate",
                "teeth_count": "teeth_count",
            },
            context_param_stats={},
            pair_frequency_column_patterns=[r"Accel_FFT_Acc\d+_range\d+_Frequencies_\d+"],
            pair_amplitude_column_patterns=[r"Accel_FFT_Acc\d+_range\d+_Amplitudes_\d+"],
            harmonic_columns=["Acc1·P0", "Acc2·P0"],
            k_peaks=1,
            f_max_rel=12.0,
            cnn_window=2,
        ),
        calls=[],
    )

    def _is_available():
        return True

    def _score(pairs, params):
        scorer.calls.append((np.asarray(pairs), np.asarray(params)))
        return {
            "harmonic_context_score": 0.77,
            "model_source": "harmonic_pair_test",
            "decision_threshold": 0.92,
        }

    scorer.is_available = _is_available
    scorer.score = _score
    orchestrator.harmonic_scorer = scorer

    def _event(freq1: float, amp1: float, freq2: float, amp2: float) -> MemoryEvent:
        return MemoryEvent(
            session_id="pair-session",
            time_range=TimeRange(i0=0, i1=1, t0=0.0, t1=1.0, fs=1.0),
            raw_metrics={
                "spindle_speed": 1200.0,
                "feed_rate": 150.0,
                "teeth_count": 9.0,
                "Accel_FFT_Acc1_range1_Frequencies_0": freq1,
                "Accel_FFT_Acc1_range1_Amplitudes_0": amp1,
                "Accel_FFT_Acc2_range1_Frequencies_0": freq2,
                "Accel_FFT_Acc2_range1_Amplitudes_0": amp2,
            },
        )

    def _score(event):
        return enrich_with_harmonic_score(
            event,
            harmonic_scorer=orchestrator.harmonic_scorer,
            row_history=orchestrator._harmonic_row_history,
        )

    assert _score(_event(20.0, 1.0, 40.0, 2.0)) == {}

    second = _score(_event(40.0, 3.0, 60.0, 4.0))

    assert second == {
        "harmonic_context_score": 0.77,
        "harmonic_context_source": "harmonic_pair_test",
        "harmonic_context_threshold": 0.92,
        "harmonic_context_triggered": False,
    }
    assert len(scorer.calls) == 1
    np.testing.assert_allclose(
        scorer.calls[0][0],
        np.array(
            [
                [[[1.0, 1.0]], [[2.0, 2.0]]],
                [[[2.0, 3.0]], [[3.0, 4.0]]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        scorer.calls[0][1],
        np.array([1200.0, 150.0, 9.0], dtype=np.float32),
    )


def test_inference_streamer_pair_mode_scores_before_full_history():
    scorer = SimpleNamespace(
        config=SimpleNamespace(
            scorer_kind="pair",
            harmonic_mode="pre_extracted",
            context_param_keys=["spindle_speed", "feed_rate"],
            context_param_sources={
                "spindle_speed": "spindle_speed_mean",
                "feed_rate": "feed_rate_mean",
            },
            context_param_stats={},
            pair_frequency_column_patterns=[r"Vibration_Peak_\d+_[XY]_Frequency"],
            pair_amplitude_column_patterns=[r"Vibration_Peak_\d+_[XY]_Amplitude"],
            harmonic_columns=[],
            k_peaks=1,
            f_max_rel=12.0,
            cnn_window=16,
            decision_threshold=0.7048,
        ),
        calls=[],
    )

    def _is_available():
        return True

    def _score(pairs, params):
        scorer.calls.append((np.asarray(pairs), np.asarray(params)))
        return {
            "harmonic_context_score": 0.91,
            "model_source": "harmonic_pair_test",
            "decision_threshold": 0.7048,
            "feature_labels": ["X·P1", "Y·P1"],
            "harmonic_values": [1.1, 0.9],
        }

    scorer.is_available = _is_available
    scorer.score = _score

    feature_dict = {
        "spindle_speed_mean": 5000.0,
        "feed_rate_mean": 120.0,
        "Vibration_Peak_1_X_Frequency": 40.0,
        "Vibration_Peak_1_X_Amplitude": 1.1,
        "Vibration_Peak_1_Y_Frequency": 55.0,
        "Vibration_Peak_1_Y_Amplitude": 0.9,
    }
    win_data = {
        key: np.array([value], dtype=np.float32)
        for key, value in feature_dict.items()
    }

    result = _compute_harmonic_window(
        scorer=scorer,
        cfg=scorer.config,
        feature_dict=feature_dict,
        win_data=win_data,
        fs=1.0,
        row_history=deque(maxlen=scorer.config.cnn_window),
    )

    assert result["score"] == pytest.approx(0.91)
    assert result["decision_threshold"] == pytest.approx(0.7048)
    assert result["labels"] == ["X·P1", "Y·P1"]
    assert result["values"] == pytest.approx([1.1, 0.9])
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0].shape == (1, 2, 1, 2)
    np.testing.assert_allclose(
        scorer.calls[0][1],
        np.array([5000.0, 120.0], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_harmonic_status_reports_checkpoint_matrix(tmp_path, monkeypatch):
    def _config(dataset_name: str, model_name: str, scorer_kind: str = "context"):
        return SimpleNamespace(
            dataset_name=dataset_name,
            model_save_path=str(tmp_path / model_name),
            scorer_kind=scorer_kind,
            n_harm_features=4,
            n_params=2,
            harmonic_mode="pre_extracted",
            cnn_window=3,
            trained_at=None,
            training_metrics={},
        )

    configs = {
        "default": _config("default", "harmonic_context.pt"),
        "casedata": _config("casedata", "harmonic_context_casedata.pt"),
        "pair_casedata": _config("pair_casedata", "harmonic_pair_casedata.pt", scorer_kind="pair"),
        "stoppage_1hz": _config("stoppage_1hz", "harmonic_context_1hz.pt"),
        "site_a_line2": _config("site_a_line2", "harmonic_context_site_a_line2.pt"),
        "raw_accelerometer": _config("raw_accelerometer", "harmonic_context_raw.pt"),
        "pair_raw": _config("pair_raw", "harmonic_pair_raw.pt", scorer_kind="pair"),
    }

    for name in ("harmonic_context_casedata.pt", "harmonic_context_site_a_line2.pt", "harmonic_context_1hz.pt"):
        (tmp_path / name).write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(harmonic_config_module, "HarmonicContextConfig", lambda: configs["default"])
    monkeypatch.setattr(harmonic_config_module, "casedata_stoppage_preset", lambda: configs["casedata"])
    monkeypatch.setattr(harmonic_config_module, "pair_casedata_preset", lambda: configs["pair_casedata"])
    monkeypatch.setattr(harmonic_config_module, "stoppage_1hz_preset", lambda: configs["stoppage_1hz"])
    monkeypatch.setattr(harmonic_config_module, "site_a_line2_breakage_preset", lambda: configs["site_a_line2"])
    monkeypatch.setattr(harmonic_config_module, "raw_accelerometer_preset", lambda: configs["raw_accelerometer"])
    monkeypatch.setattr(harmonic_config_module, "pair_raw_preset", lambda: configs["pair_raw"])

    class StubScorer:
        def __init__(self, config):
            self.config = config

        def _ensure_model(self):
            return Path(self.config.model_save_path).is_file()

        def get_model_info(self):
            return {
                "available": True,
                "torch_installed": True,
                "model_loaded": False,
                "dataset_name": self.config.dataset_name,
                "scorer_kind": self.config.scorer_kind,
                "n_harm_features": self.config.n_harm_features,
                "n_params": self.config.n_params,
                "harmonic_mode": self.config.harmonic_mode,
                "cnn_window": self.config.cnn_window,
                "trained_at": self.config.trained_at,
                "training_metrics": self.config.training_metrics,
                "model_save_path": self.config.model_save_path,
            }

    monkeypatch.setattr(harmonic_runtime_module, "harmonic_torch_available", lambda config=None: True)
    monkeypatch.setattr(
        harmonic_runtime_module,
        "build_harmonic_scorer",
        lambda config=None: StubScorer(config),
    )

    out = await harmonic_router.harmonic_status(dataset="casedata")

    assert out.dataset_name == "casedata"
    assert out.model_loaded is True
    assert out.model_path_exists is True
    assert out.checkpoint_statuses["casedata"]["model_loaded"] is True
    assert out.checkpoint_statuses["stoppage_1hz"]["model_loaded"] is True
    assert out.checkpoint_statuses["site_a_line2"]["model_loaded"] is True
    assert out.checkpoint_statuses["raw_accelerometer"]["model_loaded"] is False
    assert out.checkpoint_statuses["raw_accelerometer"]["model_path_exists"] is False
    assert out.checkpoint_statuses["pair_casedata"]["scorer_kind"] == "pair"
    assert out.checkpoint_statuses["pair_raw"]["scorer_kind"] == "pair"


@pytest.mark.asyncio
async def test_harmonic_explain_falls_back_to_stored_harmonic_context(monkeypatch):
    memory = SimpleNamespace(
        metadata={
            "external_signals": {
                "harmonic_context_score": 0.66,
                "harmonic_context_source": "harmonic_context_v1",
            },
            "harmonic_context": {
                "source": "harmonic_context_v1",
                "feature_labels": ["X·H1"],
                "feature_values": [1.23],
                "context_weights": [0.25],
            },
        }
    )
    orchestrator = SimpleNamespace(
        harmonic_scorer=None,
        get_memory=lambda memory_id: memory,
    )

    monkeypatch.setattr(memory_orchestrator_module, "get_orchestrator", lambda: orchestrator)

    out = await harmonic_router.harmonic_explain("memory-1", top_k=1)

    assert out["available"] is True
    assert out["score"] == pytest.approx(0.66)
    assert out["model_source"] == "harmonic_context_v1"
    assert out["feature_labels"] == ["X·H1"]
    assert out["harmonic_values"] == [1.23]
    assert out["top_weighted"][0]["contribution"] == pytest.approx(0.3075)


@pytest.mark.asyncio
async def test_harmonic_evaluate_reports_runtime_pair_metrics(tmp_path, monkeypatch):
    model_path = tmp_path / "harmonic_pair_casedata.pt"
    model_path.write_text("checkpoint", encoding="utf-8")

    config = SimpleNamespace(
        dataset_name="pair_casedata",
        scorer_kind="pair",
        model_save_path=str(model_path),
        target_label="label",
        positive_labels=["pre_stoppage"],
        pair_frequency_column_patterns=[r"Vibration_Peak_\d+_[XY]_Frequency"],
        pair_amplitude_column_patterns=[r"Vibration_Peak_\d+_[XY]_Amplitude"],
        k_peaks=1,
        f_max_rel=12.0,
        context_param_keys=["spindle_speed", "feed_rate"],
        context_param_sources={
            "spindle_speed": "spindle_speed_mean",
            "feed_rate": "feed_rate_mean",
        },
        context_param_stats={
            "spindle_speed": {"mean": 0.0, "std": 1.0, "source_column": "spindle_speed_mean"},
            "feed_rate": {"mean": 0.0, "std": 1.0, "source_column": "feed_rate_mean"},
        },
        training_metrics={"best_val_acc": 0.99},
    )

    df = pd.DataFrame(
        [
            {
                "operation_id": "normal-1",
                "label": "normal",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 1.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 1.5,
            },
            {
                "operation_id": "normal-2",
                "label": "normal",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 2.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 2.5,
            },
            {
                "operation_id": "positive-1",
                "label": "pre_stoppage",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 8.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 8.5,
            },
            {
                "operation_id": "positive-2",
                "label": "pre_stoppage",
                "spindle_speed_mean": 1200.0,
                "feed_rate_mean": 100.0,
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 9.0,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 9.5,
            },
        ]
    )

    class StubScorer:
        def __init__(self, config):
            self.config = config

        def load(self, path=None):
            return True

        def score(self, pairs, params):
            amp = float(np.asarray(pairs, dtype=np.float32)[-1, 0, 0, 1])
            return {
                "harmonic_context_score": round(max(0.0, min(1.0, amp / 10.0)), 4),
                "model_source": "stub_pair_runtime",
            }

    monkeypatch.setattr(harmonic_config_module, "pair_casedata_preset", lambda **kwargs: config)
    monkeypatch.setattr(harmonic_router, "_load_pair_casedata", lambda data_dir, positive_labels=None: df.copy())
    monkeypatch.setattr(harmonic_pair_model_module, "HarmonicPairScorer", StubScorer)

    out = await harmonic_router.harmonic_evaluate(
        harmonic_router.HarmonicEvaluationRequest(dataset="pair_casedata")
    )

    assert out.success is True
    assert out.dataset_name == "pair_casedata"
    assert out.model_loaded is True
    assert out.n_windows == 4
    assert out.n_positive == 2
    assert out.n_negative == 2
    assert out.accuracy == pytest.approx(1.0)
    assert out.balanced_accuracy == pytest.approx(1.0)
    assert out.confusion == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}
    assert out.score_summary["normal"]["mean"] == pytest.approx(0.15)
    assert out.score_summary["pre_stoppage"]["mean"] == pytest.approx(0.85)
    assert out.recommended_threshold is not None


def test_run_training_attaches_runtime_evaluation_for_pair_models(monkeypatch):
    config = SimpleNamespace(
        dataset_name="pair_casedata",
        scorer_kind="pair",
        positive_labels=["pre_stoppage"],
        model_save_path="data/models/harmonic_pair_casedata.pt",
        random_seed=17,
    )
    df = pd.DataFrame([{"operation_id": "sample-1", "label": "normal"}])
    captured_overrides = {}

    class StubPairTrainer:
        def __init__(self, cfg):
            self.config = cfg

        def train_from_dataframe(self, frame):
            return TrainResult(
                success=True,
                model_path=str(config.model_save_path),
                n_samples=1,
                n_positive=0,
                n_negative=1,
                best_val_loss=0.2,
                best_val_acc=0.8,
            )

    monkeypatch.setattr(harmonic_router, "_last_train_result", None)
    monkeypatch.setattr(harmonic_router, "_training_in_progress", False)

    def _pair_casedata_preset(**kwargs):
        captured_overrides.update(kwargs)
        return config

    monkeypatch.setattr(harmonic_config_module, "pair_casedata_preset", _pair_casedata_preset)
    monkeypatch.setattr(harmonic_router, "_load_pair_casedata", lambda data_dir, positive_labels=None: df.copy())
    monkeypatch.setattr(harmonic_pair_trainer_module, "HarmonicPairTrainer", StubPairTrainer)
    monkeypatch.setattr(harmonic_router, "_refresh_runtime_harmonic_scorers", lambda cfg: None)
    monkeypatch.setattr(
        harmonic_router,
        "_evaluate_pair_runtime",
        lambda cfg, frame, **kwargs: {
            "success": True,
            "dataset_name": "pair_casedata",
            "scorer_kind": "pair",
            "n_windows": 1,
            "accuracy": 1.0,
        },
    )

    harmonic_router._run_training(harmonic_router.TrainRequest(dataset="pair_casedata", random_seed=17))

    assert harmonic_router._last_train_result is not None
    assert harmonic_router._last_train_result["status"] == "completed"
    assert harmonic_router._last_train_result["random_seed"] == 17
    assert captured_overrides["random_seed"] == 17
    assert harmonic_router._last_train_result["runtime_checkpoint_activated"] is False
    assert harmonic_router._last_train_result["model_path"] == "data/models/harmonic_pair_casedata.seed17.pt"
    assert harmonic_router._last_train_result["runtime_evaluation"] == {
        "success": True,
        "dataset_name": "pair_casedata",
        "scorer_kind": "pair",
        "n_windows": 1,
        "accuracy": 1.0,
    }


def test_training_config_overrides_prefers_explicit_random_seed() -> None:
    merged = harmonic_router._training_config_overrides(
        {"model_save_path": "x.pt", "random_seed": 99},
        random_seed=7,
    )

    assert merged == {
        "model_save_path": "x.pt",
        "random_seed": 7,
    }
    assert harmonic_router._training_config_overrides(None, random_seed=None) is None


def test_resolve_training_model_save_path_uses_safe_seed_suffix() -> None:
    assert harmonic_config_module.resolve_training_model_save_path(
        "data/models/harmonic_pair_lfl.pt",
        random_seed=17,
    ) == "data/models/harmonic_pair_lfl.seed17.pt"
    assert harmonic_config_module.resolve_training_model_save_path(
        "data/models/harmonic_pair_lfl.pt",
        checkpoint_suffix="trial/a",
    ) == "data/models/harmonic_pair_lfl.trial-a.pt"
    assert harmonic_config_module.resolve_training_model_save_path(
        "data/models/harmonic_pair_lfl.pt",
        random_seed=17,
        replace_checkpoint=True,
    ) == "data/models/harmonic_pair_lfl.pt"
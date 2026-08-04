from __future__ import annotations

import numpy as np
import pandas as pd

from backend.agents.processing.harmonic_config import pair_lfl_preset, pair_raw_preset
from backend.agents.processing.harmonic_peak_pairs import (
    discover_peak_pair_columns,
    extract_peak_pairs_from_df,
)
from backend.agents.processing.harmonic_pair_model import (
    HarmonicPairBreakNet,
    HarmonicPairBreakNetLfl,
    HarmonicPairScorer,
)
from backend.agents.processing.harmonic_pair_trainer import HarmonicPairTrainer, TORCH_AVAILABLE
from backend.routers.harmonic import _load_pair_raw

if TORCH_AVAILABLE:
    import torch


def test_extract_peak_pairs_from_df_builds_expected_tensor() -> None:
    df = pd.DataFrame(
        {
            "CNC_parameters_Programed_spindle_speed": [600.0, 600.0],
            "Accel_FFT_Acc1_range1_Frequencies_0": [10.0, 20.0],
            "Accel_FFT_Acc1_range1_Amplitudes_0": [1.0, 2.0],
            "Accel_FFT_Acc2_range1_Frequencies_0": [30.0, 40.0],
            "Accel_FFT_Acc2_range1_Amplitudes_0": [3.0, 4.0],
        }
    )

    specs = discover_peak_pair_columns(
        df.columns,
        frequency_patterns=[r"Accel_FFT_Acc\d+_range\d+_Frequencies_\d+"],
        amplitude_patterns=[r"Accel_FFT_Acc\d+_range\d+_Amplitudes_\d+"],
        k_peaks=1,
    )
    pairs = extract_peak_pairs_from_df(
        df,
        specs,
        spindle_speed_col="CNC_parameters_Programed_spindle_speed",
        k_peaks=1,
    )

    assert len(specs) == 2
    assert pairs.shape == (2, 2, 1, 2)
    assert pairs[0, 0, 0].tolist() == [1.0, 1.0]
    assert pairs[0, 1, 0].tolist() == [3.0, 3.0]


def test_load_pair_raw_filters_to_trainable_labels(tmp_path) -> None:
    normal = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "engagement_idx": [0, 0],
            "label": ["normal", "normal"],
            "CNC_parameters_Programed_spindle_speed": [600.0, 600.0],
            "Axis_FeedRate_commanded": [1200.0, 1200.0],
            "CNC_parameters_teeth_num": [9.0, 9.0],
            "Accel_FFT_Acc1_range1_Frequencies_0": [10.0, 11.0],
            "Accel_FFT_Acc1_range1_Amplitudes_0": [1.0, 1.1],
        }
    )
    pre_break = pd.DataFrame(
        {
            "session": ["s2", "s2"],
            "engagement_idx": [1, 1],
            "label": ["pre_break", "pre_break"],
            "CNC_parameters_Programed_spindle_speed": [600.0, 600.0],
            "Axis_FeedRate_commanded": [1200.0, 1200.0],
            "CNC_parameters_teeth_num": [9.0, 9.0],
            "Accel_FFT_Acc1_range1_Frequencies_0": [20.0, 21.0],
            "Accel_FFT_Acc1_range1_Amplitudes_0": [2.0, 2.1],
        }
    )
    exclude = pd.DataFrame(
        {
            "session": ["s3"],
            "engagement_idx": [2],
            "label": ["exclude"],
            "CNC_parameters_Programed_spindle_speed": [600.0],
            "Axis_FeedRate_commanded": [1200.0],
            "CNC_parameters_teeth_num": [9.0],
            "Accel_FFT_Acc1_range1_Frequencies_0": [30.0],
            "Accel_FFT_Acc1_range1_Amplitudes_0": [3.0],
        }
    )

    normal.to_parquet(tmp_path / "normal.parquet")
    pre_break.to_parquet(tmp_path / "pre_break.parquet")
    exclude.to_parquet(tmp_path / "exclude.parquet")

    df = _load_pair_raw(str(tmp_path))

    assert sorted(df["label"].unique().tolist()) == ["normal", "pre_break"]
    assert set(df["operation_id"].unique().tolist()) == {"s1:0:normal", "s2:1:pre_break"}


def test_pair_trainer_smoke(tmp_path) -> None:
    if not TORCH_AVAILABLE:
        return

    np.random.seed(0)
    rows = []
    for idx in range(12):
        rows.append(
            {
                "row_id": idx,
                "CNC_parameters_Programed_spindle_speed": 600.0,
                "Axis_FeedRate_commanded": 1200.0,
                "CNC_parameters_teeth_num": 9.0,
                "Accel_FFT_Acc1_range1_Frequencies_0": 10.0 + idx,
                "Accel_FFT_Acc1_range1_Amplitudes_0": 1.0 + 0.1 * idx,
                "Accel_FFT_Acc2_range1_Frequencies_0": 20.0 + idx,
                "Accel_FFT_Acc2_range1_Amplitudes_0": 2.0 + 0.1 * idx,
                "label": "pre_break" if idx >= 8 else "normal",
            }
        )
    df = pd.DataFrame(rows)

    cfg = pair_raw_preset(
        cnn_window=2,
        k_peaks=1,
        batch_size=2,
        val_split=0.34,
        n_windows_per_sample=1,
        early_stopping_patience=1,
        learning_rate_schedule=[{"lr": 1e-3, "epochs": 1}],
        model_save_path=str(tmp_path / "pair_smoke.pt"),
    )
    trainer = HarmonicPairTrainer(cfg)

    result = trainer.train_from_dataframe(df, operation_col="row_id")

    assert result.success is True
    assert result.model_path.endswith("pair_smoke.pt")


def test_pair_lfl_preset_matches_original_lfl_shape_contract() -> None:
    cfg = pair_lfl_preset()
    restored = type(cfg).from_dict(cfg.to_dict())

    assert cfg.model_kind == "lfl_v2"
    assert restored.model_kind == "lfl_v2"
    assert cfg.context_param_keys == ["d", "z", "n", "f", "vf"]
    assert cfg.context_param_sources == {
        "d": "tool_diameter",
        "z": "num_teeth",
        "n": "spindle_speed_mean",
        "f": "feed_per_tooth",
        "vf": "feed_rate_mean",
    }
    assert cfg.cnn_window == 12
    assert cfg.conv_channels == [16, 16]
    assert cfg.kernel_size == 5
    assert restored.random_seed == cfg.random_seed == 0
    assert cfg.model_save_path.endswith("harmonic_pair_lfl.pt")


def test_pair_scorer_loads_legacy_and_lfl_model_kinds(tmp_path) -> None:
    if not TORCH_AVAILABLE:
        return

    legacy_cfg = pair_raw_preset(
        model_save_path=str(tmp_path / "legacy_pair.pt"),
        k_peaks=1,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        cnn_window=4,
        kernel_size=3,
    )
    legacy_scorer = HarmonicPairScorer(legacy_cfg)
    legacy_scorer._model = HarmonicPairBreakNet(
        n_channels=2,
        k_peaks=1,
        n_params=len(legacy_cfg.context_param_keys),
        cnn_window=4,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        ks=3,
    )
    legacy_scorer.save()

    loaded_legacy = HarmonicPairScorer(legacy_cfg)
    assert loaded_legacy.load() is True
    assert isinstance(loaded_legacy._model, HarmonicPairBreakNet)
    assert loaded_legacy.get_model_info()["model_kind"] == "legacy_v1"

    lfl_cfg = pair_lfl_preset(
        model_save_path=str(tmp_path / "lfl_pair.pt"),
        k_peaks=1,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        cnn_window=4,
        kernel_size=3,
    )
    lfl_scorer = HarmonicPairScorer(lfl_cfg)
    lfl_model = HarmonicPairBreakNetLfl(
        n_channels=2,
        k_peaks=1,
        n_params=len(lfl_cfg.context_param_keys),
        cnn_window=4,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        ks=3,
    )
    lfl_model.set_param_stats(torch.arange(5, dtype=torch.float32), torch.ones(5, dtype=torch.float32) * 2.0)
    lfl_scorer._model = lfl_model
    lfl_scorer.save()

    loaded_lfl = HarmonicPairScorer(lfl_cfg)
    assert loaded_lfl.load() is True
    assert isinstance(loaded_lfl._model, HarmonicPairBreakNetLfl)
    assert loaded_lfl.get_model_info()["model_kind"] == "lfl_v2"
    assert torch.allclose(loaded_lfl._model.param_mean, torch.arange(5, dtype=torch.float32))
    assert torch.allclose(loaded_lfl._model.param_std, torch.ones(5, dtype=torch.float32) * 2.0)


def test_pair_lfl_trainer_smoke(tmp_path) -> None:
    if not TORCH_AVAILABLE:
        return

    np.random.seed(0)
    rows = []
    for idx in range(12):
        rows.append(
            {
                "row_id": idx,
                "tool_diameter": 80.0,
                "num_teeth": 6.0,
                "spindle_speed_mean": 1200.0 + idx,
                "feed_rate_mean": 720.0 + idx,
                "feed_per_tooth": (720.0 + idx) / (6.0 * (1200.0 + idx)),
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 1.0 + 0.2 * idx,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 2.0 + 0.2 * idx,
                "label": "pre_stoppage" if idx >= 8 else "normal",
            }
        )
    df = pd.DataFrame(rows)

    cfg = pair_lfl_preset(
        cnn_window=4,
        k_peaks=1,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        kernel_size=3,
        batch_size=2,
        val_split=0.34,
        n_windows_per_sample=1,
        early_stopping_patience=1,
        learning_rate_schedule=[{"lr": 1e-3, "epochs": 1}],
        model_save_path=str(tmp_path / "pair_lfl_smoke.pt"),
    )
    trainer = HarmonicPairTrainer(cfg)

    result = trainer.train_from_dataframe(df, operation_col="row_id")

    assert result.success is True
    scorer = HarmonicPairScorer(cfg)
    assert scorer.load() is True
    assert isinstance(scorer._model, HarmonicPairBreakNetLfl)
    expected_mean = torch.tensor(
        [80.0, 6.0, df["spindle_speed_mean"].mean(), df["feed_per_tooth"].mean(), df["feed_rate_mean"].mean()],
        dtype=torch.float32,
    )
    assert torch.allclose(scorer._model.param_mean, expected_mean, atol=1e-4)


def test_pair_lfl_trainer_same_seed_is_reproducible(tmp_path) -> None:
    if not TORCH_AVAILABLE:
        return

    rows = []
    for idx in range(12):
        rows.append(
            {
                "row_id": idx,
                "tool_diameter": 80.0,
                "num_teeth": 6.0,
                "spindle_speed_mean": 1200.0 + idx,
                "feed_rate_mean": 720.0 + idx,
                "feed_per_tooth": (720.0 + idx) / (6.0 * (1200.0 + idx)),
                "Vibration_Peak_1_X_Frequency": 40.0,
                "Vibration_Peak_1_X_Amplitude": 1.0 + 0.2 * idx,
                "Vibration_Peak_1_Y_Frequency": 50.0,
                "Vibration_Peak_1_Y_Amplitude": 2.0 + 0.2 * idx,
                "label": "pre_stoppage" if idx >= 8 else "normal",
            }
        )
    df = pd.DataFrame(rows)

    common_cfg = dict(
        cnn_window=4,
        k_peaks=1,
        pair_embed_dim=4,
        conv_channels=[4],
        fc_hidden=4,
        kernel_size=3,
        batch_size=2,
        val_split=0.34,
        n_windows_per_sample=1,
        early_stopping_patience=2,
        learning_rate_schedule=[{"lr": 1e-3, "epochs": 2}],
        random_seed=123,
    )

    cfg_a = pair_lfl_preset(
        model_save_path=str(tmp_path / "pair_lfl_seed_a.pt"),
        **common_cfg,
    )
    cfg_b = pair_lfl_preset(
        model_save_path=str(tmp_path / "pair_lfl_seed_b.pt"),
        **common_cfg,
    )

    result_a = HarmonicPairTrainer(cfg_a).train_from_dataframe(df, operation_col="row_id")
    result_b = HarmonicPairTrainer(cfg_b).train_from_dataframe(df, operation_col="row_id")

    assert result_a.success is True
    assert result_b.success is True
    assert np.isclose(result_a.best_val_loss, result_b.best_val_loss)
    assert np.isclose(result_a.best_val_acc, result_b.best_val_acc)
    assert np.allclose(result_a.train_loss_history, result_b.train_loss_history)
    assert np.allclose(result_a.val_loss_history, result_b.val_loss_history)

    scorer_a = HarmonicPairScorer(cfg_a)
    scorer_b = HarmonicPairScorer(cfg_b)
    assert scorer_a.load() is True
    assert scorer_b.load() is True
    assert scorer_a.config.training_metrics.get("random_seed") == 123
    assert scorer_b.config.training_metrics.get("random_seed") == 123

    state_a = scorer_a._model.state_dict()
    state_b = scorer_b._model.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
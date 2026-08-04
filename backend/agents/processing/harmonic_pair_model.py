"""Pair-input harmonic model and scorer.

Supports two pair-model families:

- ``legacy_v1``: the currently shipped DeepSets-style pair encoder with a
    temporal CNN and FC-head param concatenation.
- ``lfl_v2``: the original LFL parameter-conditioned per-pair encoder where
    the cutting-parameter vector reshapes the first pair-reading layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...json_utils import json_safe

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not installed — harmonic pair model unavailable")


def _pair_model_kind(config: Optional[Any] = None, checkpoint: Optional[Dict[str, Any]] = None) -> str:
    requested = None
    if isinstance(checkpoint, dict):
        requested = checkpoint.get("model_kind")
    if requested is None and config is not None:
        requested = getattr(config, "model_kind", None)
    kind = str(requested or "legacy_v1").strip().lower()
    return "lfl_v2" if kind == "lfl_v2" else "legacy_v1"


if TORCH_AVAILABLE:

    class ParamConditionedPairEncoder(nn.Module):
        """Original LFL parameter-conditioned per-pair encoder."""

        def __init__(self, pair_in_dim: int, pair_embed_dim: int, n_params: int):
            super().__init__()
            self.pair_in_dim = pair_in_dim
            self.pair_embed_dim = pair_embed_dim
            self.n_params = n_params

            self.W0 = nn.Parameter(
                torch.randn(pair_embed_dim, pair_in_dim) * (1.0 / pair_in_dim**0.5)
            )
            self.M = nn.Parameter(torch.randn(pair_embed_dim, n_params, pair_in_dim) * 0.01)
            self.b1 = nn.Parameter(torch.zeros(pair_embed_dim))

            self.linear2 = nn.Linear(pair_embed_dim, pair_embed_dim)
            self.linear3 = nn.Linear(pair_embed_dim, pair_embed_dim)

        def forward(self, pairs: "torch.Tensor", params_std: "torch.Tensor") -> "torch.Tensor":
            w_eff = torch.einsum("bp,dpi->bdi", params_std, self.M) + self.W0
            hidden = torch.einsum("bdi,btcki->btckd", w_eff, pairs) + self.b1
            hidden = F.relu(hidden)
            hidden = self.linear2(hidden)
            hidden = F.relu(hidden)
            hidden = self.linear3(hidden)
            return hidden

    class HarmonicPairBreakNet(nn.Module):
        """Legacy pair-input model used by existing integrated checkpoints."""

        def __init__(
            self,
            *,
            n_channels: int = 2,
            k_peaks: int = 5,
            n_params: int = 3,
            cnn_window: int = 16,
            pair_embed_dim: int = 16,
            conv_channels: Optional[List[int]] = None,
            fc_hidden: int = 32,
            ks: int = 3,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [32, 32]

            self.n_channels = n_channels
            self.k_peaks = k_peaks
            self.n_params = n_params
            self.cnn_window = cnn_window
            self.pair_embed_dim = pair_embed_dim
            self.conv_channels_cfg = conv_channels

            self.pair_mlp = nn.Sequential(
                nn.Linear(2, pair_embed_dim),
                nn.ReLU(),
                nn.Linear(pair_embed_dim, pair_embed_dim),
                nn.ReLU(),
            )

            in_channels = max(1, n_channels * pair_embed_dim)
            pad = ks // 2
            layers: list[nn.Module] = []
            ch = in_channels
            for ch_out in conv_channels:
                layers += [
                    nn.Conv1d(ch, ch_out, ks, padding=pad, padding_mode="replicate"),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(),
                    nn.AvgPool1d(2, ceil_mode=True),
                    nn.Dropout(0.3),
                ]
                ch = ch_out
            self.conv = nn.Sequential(*layers)

            t_out = cnn_window
            for _ in conv_channels:
                t_out = max(1, (t_out + 1) // 2)

            self.fc1 = nn.Linear(ch * t_out + n_params, fc_hidden)
            self.bn1 = nn.BatchNorm1d(fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, fc_hidden)
            self.bn2 = nn.BatchNorm1d(fc_hidden)
            self.fc3 = nn.Linear(fc_hidden, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.5)

        def forward(self, pairs: "torch.Tensor", params: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                pairs: ``(B, T, C, K, 2)`` pair tensor.
                params: ``(B, n_params)`` z-scored context parameters.
            """
            batch_size, time_steps, n_channels, k_peaks, _ = pairs.shape
            x = pairs.reshape(batch_size * time_steps * n_channels * k_peaks, 2)
            x = self.pair_mlp(x)
            x = x.reshape(batch_size, time_steps, n_channels, k_peaks, self.pair_embed_dim)
            x = x.sum(dim=3)
            x = x.reshape(batch_size, time_steps, n_channels * self.pair_embed_dim)
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = x.flatten(1)
            x = torch.cat([x, params], dim=1)
            x = self.drop(self.relu(self.bn1(self.fc1(x))))
            x = self.drop(self.relu(self.bn2(self.fc2(x))))
            return self.fc3(x).squeeze(-1)


    class HarmonicPairBreakNetLfl(nn.Module):
        """Original LFL parameter-conditioned pair model."""

        def __init__(
            self,
            *,
            n_channels: int = 2,
            k_peaks: int = 5,
            pair_in_dim: int = 2,
            pair_embed_dim: int = 16,
            n_params: int = 5,
            cnn_window: int = 12,
            conv_channels: Optional[List[int]] = None,
            fc_hidden: int = 32,
            ks: int = 5,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [16, 16]

            self.n_channels = n_channels
            self.k_peaks = k_peaks
            self.pair_in_dim = pair_in_dim
            self.pair_embed_dim = pair_embed_dim
            self.n_params = n_params
            self.cnn_window = cnn_window
            self.conv_channels_cfg = conv_channels

            self.register_buffer("param_mean", torch.zeros(n_params))
            self.register_buffer("param_std", torch.ones(n_params))

            self.pair_encoder = ParamConditionedPairEncoder(
                pair_in_dim=pair_in_dim,
                pair_embed_dim=pair_embed_dim,
                n_params=n_params,
            )

            per_step_dim = n_channels * pair_embed_dim
            pad = ks // 2
            layers: list[nn.Module] = []
            ch = per_step_dim
            for ch_out in conv_channels:
                layers += [
                    nn.Conv1d(ch, ch_out, ks, padding=pad, padding_mode="replicate"),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(),
                    nn.AvgPool1d(2),
                    nn.Dropout(0.3),
                ]
                ch = ch_out
            self.conv = nn.Sequential(*layers)

            t_out = cnn_window
            for _ in conv_channels:
                t_out = t_out // 2
            flat = ch * t_out

            self.fc1 = nn.Linear(flat, fc_hidden)
            self.bn1 = nn.BatchNorm1d(fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, fc_hidden)
            self.bn2 = nn.BatchNorm1d(fc_hidden)
            self.fc3 = nn.Linear(fc_hidden, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.5)

        def set_param_stats(self, mean: "torch.Tensor", std: "torch.Tensor") -> None:
            adjusted_std = std.clone()
            adjusted_std[adjusted_std < 1e-8] = 1.0
            self.param_mean.copy_(mean.to(self.param_mean.device))
            self.param_std.copy_(adjusted_std.to(self.param_std.device))

        def forward(self, pairs: "torch.Tensor", params: "torch.Tensor") -> "torch.Tensor":
            batch_size, time_steps, n_channels, k_peaks, feature_dim = pairs.shape
            del time_steps, n_channels, k_peaks, feature_dim

            params_std = (params - self.param_mean) / self.param_std
            encoded = self.pair_encoder(pairs, params_std)
            encoded = encoded.sum(dim=3)
            encoded = encoded.reshape(batch_size, pairs.shape[1], self.n_channels * self.pair_embed_dim)
            encoded = encoded.transpose(1, 2)
            encoded = self.conv(encoded)
            encoded = encoded.flatten(1)
            encoded = self.drop(self.relu(self.bn1(self.fc1(encoded))))
            encoded = self.drop(self.relu(self.bn2(self.fc2(encoded))))
            return self.fc3(encoded).squeeze(-1)


def _build_pair_model(
    *,
    model_kind: str,
    n_channels: int,
    k_peaks: int,
    n_params: int,
    cnn_window: int,
    pair_embed_dim: int,
    conv_channels: Optional[List[int]],
    fc_hidden: int,
    ks: int,
) -> Any:
    if model_kind == "lfl_v2":
        return HarmonicPairBreakNetLfl(
            n_channels=n_channels,
            k_peaks=k_peaks,
            n_params=n_params,
            cnn_window=cnn_window,
            pair_embed_dim=pair_embed_dim,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            ks=ks,
        )
    return HarmonicPairBreakNet(
        n_channels=n_channels,
        k_peaks=k_peaks,
        n_params=n_params,
        cnn_window=cnn_window,
        pair_embed_dim=pair_embed_dim,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        ks=ks,
    )


class HarmonicPairScorer:
    """Sklearn-style scorer wrapper for the pair-input harmonic model."""

    def __init__(self, config: Optional[Any] = None):
        from .harmonic_config import HarmonicContextConfig

        self.config = config or HarmonicContextConfig(scorer_kind="pair")
        self._model: Optional[Any] = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    def is_available(self) -> bool:
        return TORCH_AVAILABLE and self._is_loaded and self._model is not None

    def _ensure_model(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if self._model is not None:
            return True

        model_path = Path(self.config.model_save_path)
        if model_path.exists():
            return self.load(model_path)
        return False

    def load(self, path: Optional[Path] = None) -> bool:
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load harmonic pair model: PyTorch not installed")
            return False

        path = path or Path(self.config.model_save_path)
        if not path.exists():
            logger.debug("Harmonic pair model file not found: %s", path)
            return False

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

            if "config" in checkpoint:
                from .harmonic_config import HarmonicContextConfig

                self.config = HarmonicContextConfig.from_dict(checkpoint["config"])

            n_channels = checkpoint.get("n_channels", 1)
            k_peaks = checkpoint.get("k_peaks", self.config.k_peaks)
            n_params = checkpoint.get("n_params", self.config.n_params)
            cnn_window = checkpoint.get("cnn_window", self.config.cnn_window)
            conv_channels = checkpoint.get("conv_channels", self.config.conv_channels)
            pair_embed_dim = checkpoint.get("pair_embed_dim", self.config.pair_embed_dim)
            fc_hidden = checkpoint.get("fc_hidden", self.config.fc_hidden)
            ks = checkpoint.get("kernel_size", self.config.kernel_size)
            model_kind = _pair_model_kind(self.config, checkpoint)

            self._model = _build_pair_model(
                model_kind=model_kind,
                n_channels=n_channels,
                k_peaks=k_peaks,
                n_params=n_params,
                cnn_window=cnn_window,
                pair_embed_dim=pair_embed_dim,
                conv_channels=conv_channels,
                fc_hidden=fc_hidden,
                ks=ks,
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            self._is_loaded = True

            logger.info(
                "Loaded harmonic pair model from %s (channels=%d, k=%d, n_params=%d, dataset=%s)",
                path,
                n_channels,
                k_peaks,
                n_params,
                self.config.dataset_name,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load harmonic pair model from %s: %s", path, exc)
            self._is_loaded = False
            return False

    def save(self, path: Optional[Path] = None) -> None:
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("Cannot save harmonic pair model: model not available")
            return

        path = path or Path(self.config.model_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config.to_dict(),
            "model_kind": _pair_model_kind(self.config),
            "n_channels": self._model.n_channels,
            "k_peaks": self._model.k_peaks,
            "n_params": self._model.n_params,
            "cnn_window": self._model.cnn_window,
            "pair_embed_dim": self._model.pair_embed_dim,
            "conv_channels": self._model.conv_channels_cfg,
            "fc_hidden": self.config.fc_hidden,
            "kernel_size": self.config.kernel_size,
        }
        torch.save(checkpoint, path)
        logger.info("Saved harmonic pair model to %s", path)

    def score(self, pairs: np.ndarray, params: np.ndarray) -> Dict[str, Any]:
        decision_threshold = float(getattr(self.config, "decision_threshold", 0.5) or 0.5)
        if not self.is_available():
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "decision_threshold": decision_threshold,
                "model_source": "harmonic_pair_unavailable",
            }

        try:
            arr = np.asarray(pairs, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"Expected pair tensor (T, C, K, 2), got shape {arr.shape}")

            time_steps = arr.shape[0]
            cw = int(self.config.cnn_window)
            if time_steps < cw:
                pad_shape = (cw - time_steps,) + arr.shape[1:]
                pad = np.zeros(pad_shape, dtype=np.float32)
                arr = np.concatenate([pad, arr], axis=0)
                time_steps = cw

            window = arr[time_steps - cw : time_steps]
            pairs_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            params_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            feature_labels = self.get_feature_labels()
            harmonic_values = np.asarray(window[-1, :, :, 1], dtype=np.float32).reshape(-1).tolist()

            with torch.no_grad():
                self._model.eval()
                logit = self._model(pairs_t, params_t)
                prob = float(torch.sigmoid(logit).item())

            return {
                "harmonic_context_score": round(prob, 4),
                "context_weights": [],
                "feature_labels": feature_labels,
                "harmonic_values": [round(float(v), 4) for v in harmonic_values],
                "decision_threshold": decision_threshold,
                "threshold_triggered": bool(prob >= decision_threshold),
                "model_source": f"harmonic_pair_{self.config.dataset_name}",
            }
        except Exception as exc:
            logger.warning("Harmonic pair scoring failed: %s", exc)
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "decision_threshold": decision_threshold,
                "model_source": "harmonic_pair_error",
            }

    def score_from_raw(
        self,
        signals: np.ndarray,
        context_params: np.ndarray,
        fg: float,
        sample_rate: float,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "decision_threshold": float(getattr(self.config, "decision_threshold", 0.5) or 0.5),
                "model_source": "harmonic_pair_unavailable",
            }

        from .harmonic_peak_pairs import compute_peak_pairs

        pairs = compute_peak_pairs(
            signals,
            fg=fg,
            sample_rate=sample_rate,
            k_peaks=self.config.k_peaks,
            fft_win=self.config.fft_window,
            fft_step=self.config.fft_step,
            f_max_rel=self.config.f_max_rel,
        )
        if pairs.shape[0] == 0:
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "decision_threshold": float(getattr(self.config, "decision_threshold", 0.5) or 0.5),
                "model_source": "harmonic_pair_insufficient_data",
            }
        return self.score(pairs, context_params)

    def get_feature_labels(self) -> List[str]:
        labels = list(getattr(self.config, "harmonic_columns", []) or [])
        if labels:
            return labels

        n_channels = 0
        if self._model is not None:
            n_channels = int(getattr(self._model, "n_channels", 0))
        n_channels = max(1, n_channels)
        k_peaks = max(1, int(getattr(self.config, "k_peaks", 1)))

        return [
            f"Acc{channel_idx + 1}·P{peak_idx}"
            for channel_idx in range(n_channels)
            for peak_idx in range(k_peaks)
        ]

    def get_model_info(self) -> Dict[str, Any]:
        return json_safe({
            "available": self.is_available(),
            "torch_installed": TORCH_AVAILABLE,
            "model_loaded": self._is_loaded,
            "dataset_name": self.config.dataset_name,
            "n_harm_features": self.config.n_harm_features,
            "n_params": self.config.n_params,
            "context_param_keys": self.config.context_param_keys,
            "harmonic_mode": self.config.harmonic_mode,
            "cnn_window": self.config.cnn_window,
            "trained_at": self.config.trained_at,
            "training_metrics": self.config.training_metrics,
            "model_save_path": self.config.model_save_path,
            "scorer_kind": self.config.scorer_kind,
            "model_kind": _pair_model_kind(self.config),
            "decision_threshold": self.config.decision_threshold,
            "k_peaks": self.config.k_peaks,
            "pair_embed_dim": self.config.pair_embed_dim,
        })
"""
Harmonic Context-Weighted CNN — Model and Scorer.

Adapted from classical/lfl/backend/model.py.  The core architecture is a CNN
that learns context-conditioned weights for harmonic features via a W matrix.
Made domain-agnostic: ``n_harm_features`` and ``n_params`` are configurable.

All PyTorch imports are guarded — the module degrades gracefully when torch
is not installed.

Tag: [HARMONIC_CONTEXT_V1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...json_utils import json_safe

logger = logging.getLogger(__name__)

# ── Optional PyTorch import ───────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not installed — harmonic context model unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# HarmonicContextNet (PyTorch Module)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class HarmonicContextNet(nn.Module):
        """CNN classifier that learns to weight harmonic features via context parameters.

        Architecture (adapted from HarmonicBreakNet):
          1. Learnable W matrix: params (B, n_params) → w (B, n_features)
             used to weight-combine harmonic features per time step.
          2. Conv1d blocks with BatchNorm, ReLU, AvgPool(2), Dropout.
          3. Two FC hidden layers with BatchNorm and Dropout → logit output.

        The W matrix is the key domain-agnostic element: it learns which
        harmonic-channel combinations matter for given cutting/context params.
        """

        def __init__(
            self,
            n_harm_features: int = 21,
            n_params: int = 2,
            cnn_window: int = 16,
            conv_channels: Optional[List[int]] = None,
            fc_hidden: int = 32,
            ks: int = 5,
        ):
            super().__init__()
            if conv_channels is None:
                conv_channels = [16, 16]

            self.n_harm_features = n_harm_features
            self.n_params = n_params
            self.cnn_window = cnn_window
            self.conv_channels_cfg = conv_channels

            # Learnable context→weight projection
            self.W = nn.Parameter(torch.randn(n_harm_features, n_params) * 0.01)

            # Conv1d stack
            pad = ks // 2
            layers: list[nn.Module] = []
            ch = 1
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

            # Temporal dimension after pooling
            t_out = cnn_window
            for _ in conv_channels:
                t_out = t_out // 2

            self.fc1 = nn.Linear(ch * t_out, fc_hidden)
            self.bn1 = nn.BatchNorm1d(fc_hidden)
            self.fc2 = nn.Linear(fc_hidden, fc_hidden)
            self.bn2 = nn.BatchNorm1d(fc_hidden)
            self.fc3 = nn.Linear(fc_hidden, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.5)

        def forward(
            self, harmonics: "torch.Tensor", params: "torch.Tensor"
        ) -> "torch.Tensor":
            """Forward pass.

            Args:
                harmonics: (B, T, n_harm_features) — harmonic magnitudes per timestep.
                params: (B, n_params) — context parameters (normalised).

            Returns:
                logits: (B,) — pass through sigmoid for probability.
            """
            # Context-conditioned weighting: params → per-feature weights
            w = params @ self.W.T  # (B, n_harm_features)
            # Weighted combination of harmonic features → scalar per timestep
            x = torch.einsum("btc,bc->bt", harmonics, w)  # (B, T)
            x = x.unsqueeze(1)  # (B, 1, T) — single-channel for Conv1d
            x = self.conv(x)  # (B, ch, T')
            x = x.flatten(1)
            x = self.drop(self.relu(self.bn1(self.fc1(x))))
            x = self.drop(self.relu(self.bn2(self.fc2(x))))
            return self.fc3(x).squeeze(-1)

        def get_context_weights(self, params: "torch.Tensor") -> "torch.Tensor":
            """Get the learned weight vector for given context params.

            Useful for interpretability — shows which harmonic-channel features
            the model considers most important for the current cutting context.

            Args:
                params: (1, n_params) or (n_params,) — single context vector.

            Returns:
                (n_harm_features,) weight vector.
            """
            if params.dim() == 1:
                params = params.unsqueeze(0)
            return (params @ self.W.T).squeeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# HarmonicContextScorer — sklearn-like interface wrapping the CNN
# ══════════════════════════════════════════════════════════════════════════════


class HarmonicContextScorer:
    """Domain-agnostic scorer wrapping HarmonicContextNet.

    Provides a simple interface aligned with the existing SeedModel pattern:
    - ``is_available()`` — checks torch + trained model exist
    - ``score()`` — score pre-computed harmonics + params → dict
    - ``score_from_raw()`` — end-to-end: raw signals → harmonics → score
    - ``load()`` / ``save()`` — persist model + config

    All torch operations are guarded — calling any method when torch is not
    installed returns safe defaults.
    """

    def __init__(
        self,
        config: Optional[Any] = None,  # HarmonicContextConfig
    ):
        from .harmonic_config import HarmonicContextConfig

        self.config = config or HarmonicContextConfig()
        self._model: Optional[Any] = None  # HarmonicContextNet
        self._device: str = "cpu"
        self._is_loaded: bool = False

    def is_available(self) -> bool:
        """Check if the scorer can run (torch installed + model loaded)."""
        return TORCH_AVAILABLE and self._is_loaded and self._model is not None

    def _ensure_model(self) -> bool:
        """Create or load the model if not yet initialised."""
        if not TORCH_AVAILABLE:
            return False
        if self._model is not None:
            return True

        # Try loading from configured path
        model_path = Path(self.config.model_save_path)
        if model_path.exists():
            return self.load(model_path)
        return False

    def load(self, path: Optional[Path] = None) -> bool:
        """Load a trained model + config from disk.

        The saved file contains:
        - ``model_state_dict``: PyTorch state dict
        - ``config``: HarmonicContextConfig as dict
        - ``n_harm_features``, ``n_params``, ``cnn_window``: architecture params
        """
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load harmonic model: PyTorch not installed")
            return False

        path = path or Path(self.config.model_save_path)
        if not path.exists():
            logger.debug("Harmonic model file not found: %s", path)
            return False

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=False)

            # Restore config
            if "config" in checkpoint:
                from .harmonic_config import HarmonicContextConfig
                self.config = HarmonicContextConfig.from_dict(checkpoint["config"])

            # Reconstruct model
            n_harm = checkpoint.get("n_harm_features", self.config.n_harm_features)
            n_params = checkpoint.get("n_params", self.config.n_params)
            cnn_window = checkpoint.get("cnn_window", self.config.cnn_window)
            conv_channels = checkpoint.get("conv_channels", self.config.conv_channels)
            fc_hidden = checkpoint.get("fc_hidden", self.config.fc_hidden)
            ks = checkpoint.get("kernel_size", self.config.kernel_size)

            self._model = HarmonicContextNet(
                n_harm_features=n_harm,
                n_params=n_params,
                cnn_window=cnn_window,
                conv_channels=conv_channels,
                fc_hidden=fc_hidden,
                ks=ks,
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            self._is_loaded = True

            logger.info(
                "Loaded harmonic context model from %s "
                "(n_harm=%d, n_params=%d, cnn_window=%d, dataset=%s)",
                path, n_harm, n_params, cnn_window,
                self.config.dataset_name,
            )
            return True

        except Exception as e:
            logger.warning("Failed to load harmonic model from %s: %s", path, e)
            self._is_loaded = False
            return False

    def save(self, path: Optional[Path] = None) -> None:
        """Save the model + config to disk."""
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("Cannot save: model not available")
            return

        path = path or Path(self.config.model_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config.to_dict(),
            "n_harm_features": self._model.n_harm_features,
            "n_params": self._model.n_params,
            "cnn_window": self._model.cnn_window,
            "conv_channels": self._model.conv_channels_cfg,
            "fc_hidden": self.config.fc_hidden,
            "kernel_size": self.config.kernel_size,
        }
        torch.save(checkpoint, path)
        logger.info("Saved harmonic context model to %s", path)

    def score(
        self,
        harmonics: np.ndarray,
        params: np.ndarray,
    ) -> Dict[str, Any]:
        """Score pre-computed harmonic features + context params.

        Args:
            harmonics: (T, n_harm_features) — one sample's harmonic time series.
            params: (n_params,) — context parameter vector (normalised).

        Returns:
            Dict with:
            - ``harmonic_context_score``: float in [0, 1] (probability)
            - ``context_weights``: list[float] — learned weights for this context
            - ``model_source``: str
        """
        if not self.is_available():
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "model_source": "harmonic_context_unavailable",
            }

        try:
            T, F = harmonics.shape
            cw = self.config.cnn_window

            # If not enough time steps, pad with zeros
            if T < cw:
                pad = np.zeros((cw - T, F), dtype=np.float32)
                harmonics = np.vstack([pad, harmonics])
                T = cw

            # Take the latest cnn_window steps
            window = harmonics[T - cw: T]

            # Convert to tensors
            h_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, cw, F)
            p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)  # (1, n_params)

            with torch.no_grad():
                self._model.eval()
                logit = self._model(h_t, p_t)
                prob = float(torch.sigmoid(logit).item())
                ctx_weights = self._model.get_context_weights(p_t.squeeze(0))

            return {
                "harmonic_context_score": round(prob, 4),
                "context_weights": [round(float(w), 4) for w in ctx_weights.tolist()],
                "model_source": f"harmonic_context_{self.config.dataset_name}",
            }

        except Exception as e:
            logger.warning("Harmonic context scoring failed: %s", e)
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "model_source": "harmonic_context_error",
            }

    def score_from_raw(
        self,
        signals: np.ndarray,
        context_params: np.ndarray,
        fg: float,
        sample_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """End-to-end scoring: raw signals → harmonics → CNN → score.

        Only works with ``harmonic_mode='raw_fft'``.

        Args:
            signals: (N, C) raw multi-channel time series.
            context_params: (n_params,) normalised context vector.
            fg: Spindle frequency in Hz (RPM / 60).
            sample_rate: Sample rate in Hz.  Passed to ``compute_harmonics``
                for correct FFT bin mapping.

        Returns:
            Same format as ``score()``.
        """
        if not self.is_available():
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "model_source": "harmonic_context_unavailable",
            }

        from .harmonic_features import compute_harmonics

        harmonics = compute_harmonics(
            signals,
            fg=fg,
            harm_mults=self.config.harmonic_multipliers,
            fft_win=self.config.fft_window,
            fft_step=self.config.fft_step,
            sample_rate=sample_rate,
        )

        if harmonics.shape[0] == 0:
            return {
                "harmonic_context_score": 0.5,
                "context_weights": [],
                "model_source": "harmonic_context_insufficient_data",
            }

        return self.score(harmonics, context_params)

    def get_feature_labels(self) -> List[str]:
        """Derive human-readable labels for each context weight.

        Returns labels in the format ``"Group·Harmonic"`` so the UI can
        split on ``·`` to group weights by channel/axis (matching the
        original HarmonicBreakNet bar-chart layout).

        For raw_fft mode:  ``["X·1×fg", "X·2×fg", ..., "Z·10×fg"]``
        For pre_extracted:  ``["X·H1", "X·H2", ..., "Y·H1", ...]``
        """
        import re as _re

        cfg = self.config

        if cfg.harmonic_columns:
            labels: List[str] = []
            for col in cfg.harmonic_columns:
                # Vibration_Harmonic_N_X_Amplitude → "X·H{N}"
                m = _re.match(
                    r'.*[Hh]armonic[_\s]*(\d+).*[_\s]([XYZ])[_\s]', col,
                )
                if m:
                    labels.append(f"{m.group(2).upper()}·H{m.group(1)}")
                    continue
                # ..._Acc_N_... → "A{N}·H{M}"
                m = _re.match(
                    r'.*[Hh]armonic[_\s]*(\d+).*[Aa]cc[_\s]*(\d+)', col,
                )
                if m:
                    labels.append(f"A{m.group(2)}·H{m.group(1)}")
                    continue
                # Fallback: shorten
                short = (
                    col.replace('Vibration_', '')
                    .replace('_Amplitude', '')
                    .replace('Harmonic_', 'H')
                )
                labels.append(short[:15])
            return labels

        # Raw FFT mode: Channel × Harmonic multiplier
        n_mults = len(cfg.harmonic_multipliers) if cfg.harmonic_multipliers else 1
        n_ch = (
            max(1, cfg.n_harm_features // n_mults)
            if cfg.n_harm_features > 0
            else 1
        )
        labels = []
        for ch_i in range(n_ch):
            if cfg.input_columns and ch_i < len(cfg.input_columns):
                cn = cfg.input_columns[ch_i].lower()
                if '_x' in cn or cn.endswith('x'):
                    ch_label = 'X'
                elif '_y' in cn or cn.endswith('y'):
                    ch_label = 'Y'
                elif '_z' in cn or cn.endswith('z'):
                    ch_label = 'Z'
                else:
                    ch_label = cfg.input_columns[ch_i][:4]
            else:
                ch_label = (
                    ['X', 'Y', 'Z'][ch_i] if ch_i < 3 else f'Ch{ch_i + 1}'
                )
            for mult in cfg.harmonic_multipliers:
                labels.append(f"{ch_label}·{mult}×fg")
        return labels

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model (for diagnostics/UI)."""
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
        })

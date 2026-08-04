"""
Harmonic Context-Weighted CNN — Training Pipeline.

Adapted from classical/lfl/backend/trainer.py.  Trains HarmonicContextNet on
labelled DataFrames from any dataset (casedata, Site_a_line2, or generic).

The pipeline:
  1. Discover harmonic + context columns from config patterns.
  2. Build (harmonics, params, label) samples with sliding windows.
  3. Split train/val respecting operation boundaries.
  4. Train with BCEWithLogitsLoss, pos_weight, staged LR schedule.
  5. Early stopping on validation loss.
  6. Persist model + config (including normalisation stats).

Tag: [HARMONIC_CONTEXT_V1]
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...json_utils import json_safe

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainResult:
    """Output of a training run."""

    success: bool = False
    model_path: str = ""
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    n_harm_features: int = 0
    n_params: int = 0
    harmonic_columns: List[str] = field(default_factory=list)
    context_param_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    epochs_trained: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "success": self.success,
            "model_path": self.model_path,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_harm_features": self.n_harm_features,
            "n_params": self.n_params,
            "harmonic_columns": self.harmonic_columns,
            "context_param_stats": self.context_param_stats,
            "epochs_trained": self.epochs_trained,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "duration_s": round(self.duration_s, 1),
            "error": self.error,
        })


# ══════════════════════════════════════════════════════════════════════════════
# Column discovery
# ══════════════════════════════════════════════════════════════════════════════


def discover_harmonic_columns(
    df: DataFrame,
    patterns: List[str],
    explicit_columns: Optional[List[str]] = None,
) -> List[str]:
    """Find harmonic feature columns in a DataFrame by regex patterns.

    Args:
        df: The DataFrame to search.
        patterns: Regex patterns to match against column names.
        explicit_columns: If non-empty, use these instead of pattern matching.

    Returns:
        Sorted list of discovered column names.
    """
    if explicit_columns:
        found = [c for c in explicit_columns if c in df.columns]
        if found:
            return sorted(found)

    matched: set[str] = set()
    for pat in patterns:
        regex = re.compile(pat)
        for col in df.columns:
            if regex.search(col):
                matched.add(col)

    result = sorted(matched)
    logger.info("Discovered %d harmonic columns from %d patterns", len(result), len(patterns))
    return result


def discover_context_columns(
    df: DataFrame,
    param_keys: List[str],
    param_sources: Dict[str, str],
) -> Dict[str, str]:
    """Resolve context param keys to actual DataFrame column names.

    Args:
        df: The DataFrame.
        param_keys: Logical param names (e.g., "spindle_speed").
        param_sources: key → column name mapping from config.

    Returns:
        Dict mapping param_key → actual column name (only those present).
    """
    resolved: Dict[str, str] = {}
    for key in param_keys:
        col = param_sources.get(key, key)
        if col in df.columns:
            resolved[key] = col
        else:
            # Fuzzy: try case-insensitive match
            lower_map = {c.lower(): c for c in df.columns}
            if col.lower() in lower_map:
                resolved[key] = lower_map[col.lower()]
            else:
                logger.warning(
                    "Context param '%s' (column '%s') not found in DataFrame", key, col
                )
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════


if TORCH_AVAILABLE:

    class HarmonicWindowDataset(Dataset):
        """Dataset of (harmonics_window, params, label) samples.

        Each sample is a contiguous window of ``cnn_window`` time steps from
        the harmonic feature matrix, paired with the context param vector
        and a binary label.

        During training ``n_windows_per_sample`` random windows are drawn
        from each sample's valid range (data augmentation).
        """

        def __init__(
            self,
            harmonic_matrices: List[np.ndarray],  # Each (T_i, n_features)
            param_vectors: List[np.ndarray],       # Each (n_params,)
            labels: List[int],                     # 0/1
            cnn_window: int = 16,
            n_windows: int = 2,
        ):
            self.cnn_window = cnn_window
            self.entries: List[Tuple[np.ndarray, np.ndarray, int]] = []

            for h_mat, p_vec, label in zip(harmonic_matrices, param_vectors, labels):
                T = h_mat.shape[0]
                if T < cnn_window:
                    # Pad short sequences
                    pad = np.zeros((cnn_window - T, h_mat.shape[1]), dtype=np.float32)
                    h_mat = np.vstack([pad, h_mat])
                    T = cnn_window

                # Random window sampling
                valid_starts = max(1, T - cnn_window + 1)
                if valid_starts == 1:
                    starts = [0] * n_windows
                else:
                    starts = [
                        np.random.randint(0, valid_starts) for _ in range(n_windows)
                    ]

                for s in starts:
                    window = h_mat[s: s + cnn_window].astype(np.float32)
                    self.entries.append((window, p_vec.astype(np.float32), label))

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int):
            window, params, label = self.entries[idx]
            return (
                torch.tensor(window, dtype=torch.float32),
                torch.tensor(params, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════


class HarmonicContextTrainer:
    """Train a HarmonicContextNet from a labelled DataFrame.

    Usage::

        from backend.agents.processing.harmonic_config import casedata_stoppage_preset
        from backend.agents.processing.harmonic_trainer import HarmonicContextTrainer

        config = casedata_stoppage_preset()
        trainer = HarmonicContextTrainer(config)
        result = trainer.train_from_dataframe(df)
    """

    def __init__(self, config: Any):
        """
        Args:
            config: HarmonicContextConfig instance.
        """
        from .harmonic_config import HarmonicContextConfig

        self.config: HarmonicContextConfig = config
        self._progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_progress_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Set a callback for training progress updates (e.g., WebSocket)."""
        self._progress_callback = cb

    def _emit_progress(self, data: Dict[str, Any]) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(data)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train_from_dataframe(
        self,
        df: DataFrame,
        operation_col: str = "operation_id",
    ) -> TrainResult:
        """Train the harmonic context model from a labelled DataFrame.

        Args:
            df: DataFrame with harmonic features, context params, and labels.
                Must have a column matching ``config.target_label`` with values
                from ``config.positive_labels`` (positive class) and anything
                else (negative class).
            operation_col: Column used for train/val splitting (respects
                operation boundaries so no data leakage).

        Returns:
            TrainResult with metrics and paths.
        """
        if not TORCH_AVAILABLE:
            return TrainResult(success=False, error="PyTorch not installed")

        if pd is None:
            return TrainResult(success=False, error="pandas not installed")

        t0 = time.time()
        result = TrainResult()

        try:
            # 1. Discover columns
            ctx_cols = discover_context_columns(
                df, self.config.context_param_keys, self.config.context_param_sources
            )
            if not ctx_cols:
                return TrainResult(
                    success=False,
                    error=f"No context param columns found for keys: "
                          f"{self.config.context_param_keys}",
                )

            harm_data: Optional[np.ndarray] = None
            feature_source = str(
                getattr(self.config, "pre_extracted_feature_source", "harmonic_columns") or "harmonic_columns"
            ).strip().lower()
            if feature_source == "peak_bins":
                from .harmonic_features import extract_peak_binned_harmonic_matrix_from_df

                harm_data, harm_cols = extract_peak_binned_harmonic_matrix_from_df(
                    df,
                    frequency_patterns=list(getattr(self.config, "pair_frequency_column_patterns", []) or []),
                    amplitude_patterns=list(getattr(self.config, "pair_amplitude_column_patterns", []) or []),
                    spindle_speed_col=ctx_cols.get("spindle_speed"),
                    harmonic_bins=list(getattr(self.config, "peak_harmonic_bins", []) or []),
                    k_peaks=int(getattr(self.config, "k_peaks", 5)),
                    f_max_rel=float(getattr(self.config, "f_max_rel", 12.0)),
                    tolerance=float(getattr(self.config, "peak_bin_tolerance", 0.35)),
                )
                if harm_data.shape[1] == 0 or not harm_cols:
                    return TrainResult(
                        success=False,
                        error="No peak-derived harmonic features could be built from the DataFrame",
                    )
            else:
                harm_cols = discover_harmonic_columns(
                    df, self.config.harmonic_column_patterns, self.config.harmonic_columns
                )
                if not harm_cols:
                    return TrainResult(
                        success=False,
                        error=f"No harmonic columns found matching patterns: "
                              f"{self.config.harmonic_column_patterns}",
                    )

            n_harm = int(harm_data.shape[1]) if harm_data is not None else len(harm_cols)
            n_params = len(ctx_cols)
            logger.info(
                "Harmonic training: %d harmonic cols, %d context params, %d rows",
                n_harm, n_params, len(df),
            )

            # 2. Compute context param normalisation stats
            ctx_stats = self._compute_context_stats(df, ctx_cols)

            # 3. Build labels
            label_col = self.config.target_label
            if label_col not in df.columns:
                return TrainResult(
                    success=False,
                    error=f"Label column '{label_col}' not in DataFrame",
                )

            positive_mask = df[label_col].isin(self.config.positive_labels)
            labels = positive_mask.astype(int).values
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)
            logger.info("Labels: %d positive, %d negative", n_pos, n_neg)

            if n_pos == 0:
                return TrainResult(
                    success=False,
                    error=f"No positive samples found (looked for labels: "
                          f"{self.config.positive_labels})",
                )

            # 4. Build per-sample data structures
            harm_matrices, param_vectors, sample_labels, sample_ops = (
                self._build_samples(df, harm_cols, ctx_cols, ctx_stats, labels, operation_col)
            )

            result.n_samples = len(sample_labels)
            result.n_positive = sum(sample_labels)
            result.n_negative = result.n_samples - result.n_positive
            result.n_harm_features = n_harm
            result.n_params = n_params
            result.harmonic_columns = harm_cols
            result.context_param_stats = ctx_stats

            # 5. Train/val split
            train_idx, val_idx = self._split_train_val(
                sample_ops, sample_labels, self.config.val_split
            )

            if len(train_idx) == 0 or len(val_idx) == 0:
                return TrainResult(
                    success=False,
                    error="Train/val split resulted in empty set",
                )

            # 6. Create datasets + dataloaders
            train_ds = HarmonicWindowDataset(
                [harm_matrices[i] for i in train_idx],
                [param_vectors[i] for i in train_idx],
                [sample_labels[i] for i in train_idx],
                cnn_window=self.config.cnn_window,
                n_windows=self.config.n_windows_per_sample,
            )
            val_ds = HarmonicWindowDataset(
                [harm_matrices[i] for i in val_idx],
                [param_vectors[i] for i in val_idx],
                [sample_labels[i] for i in val_idx],
                cnn_window=self.config.cnn_window,
                n_windows=1,  # No augmentation for validation
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, shuffle=False
            )

            logger.info(
                "Datasets: train=%d windows, val=%d windows",
                len(train_ds), len(val_ds),
            )

            # 7. Build model
            from .harmonic_model import HarmonicContextNet

            model = HarmonicContextNet(
                n_harm_features=n_harm,
                n_params=n_params,
                cnn_window=self.config.cnn_window,
                conv_channels=self.config.conv_channels,
                fc_hidden=self.config.fc_hidden,
                ks=self.config.kernel_size,
            )

            # 8. Train
            train_result = self._train_loop(model, train_loader, val_loader)
            result.epochs_trained = train_result["epochs"]
            result.best_val_loss = train_result["best_val_loss"]
            result.best_val_acc = train_result["best_val_acc"]
            result.train_loss_history = train_result["train_losses"]
            result.val_loss_history = train_result["val_losses"]

            # 9. Save
            self._save_model(model, n_harm, n_params, harm_cols, ctx_stats, result)
            result.success = True

        except Exception as e:
            logger.exception("Harmonic context training failed")
            result.error = str(e)

        result.duration_s = time.time() - t0
        logger.info(
            "Training %s in %.1fs: %s",
            "succeeded" if result.success else "failed",
            result.duration_s,
            result.error or f"val_loss={result.best_val_loss:.4f}, acc={result.best_val_acc:.3f}",
        )
        return result

    def train_from_sequence_samples(
        self,
        *,
        harmonic_matrices: List[np.ndarray],
        param_vectors: List[np.ndarray],
        labels: List[int],
        harmonic_columns: List[str],
        sample_groups: Optional[List[str]] = None,
    ) -> TrainResult:
        """Train the harmonic context model from pre-extracted per-sample sequences.

        Unlike :meth:`train_from_dataframe`, the harmonic matrices, context-param
        vectors, integer labels and (optional) grouping keys are supplied directly —
        one entry per sample — so no DataFrame extraction is performed. Grouping keys
        drive a leakage-safe train/val split (defaulting to per-sample groups).
        Context-param normalisation stats are computed from the supplied vectors, with
        source columns taken from ``config.context_param_sources``.
        """
        if not TORCH_AVAILABLE:
            return TrainResult(success=False, error="PyTorch not installed")

        t0 = time.time()
        result = TrainResult()
        try:
            n = len(harmonic_matrices)
            if not (n == len(param_vectors) == len(labels)):
                return TrainResult(success=False, error="sequence inputs have mismatched lengths")
            if n == 0:
                return TrainResult(success=False, error="no samples provided")
            if sample_groups is None:
                sample_groups = [str(i) for i in range(n)]
            elif len(sample_groups) != n:
                return TrainResult(success=False, error="sample_groups length mismatch")

            n_params_expected = len(self.config.context_param_keys or [])
            n_params = int(np.asarray(param_vectors[0]).shape[0])
            if n_params_expected and n_params != n_params_expected:
                return TrainResult(
                    success=False,
                    error=f"Expected {n_params_expected} context params, got {n_params}",
                )

            sample_labels = [int(x) for x in labels]
            sample_ops = list(sample_groups)
            n_harm = int(np.asarray(harmonic_matrices[0]).shape[1])
            harm_cols = list(harmonic_columns)
            if sum(sample_labels) == 0:
                return TrainResult(success=False, error="No positive samples found")

            ctx_stats = self._context_stats_from_vectors(param_vectors)

            result.n_samples = n
            result.n_positive = sum(sample_labels)
            result.n_negative = n - result.n_positive
            result.n_harm_features = n_harm
            result.n_params = n_params
            result.harmonic_columns = harm_cols
            result.context_param_stats = ctx_stats

            train_idx, val_idx = self._split_train_val(
                sample_ops, sample_labels, self.config.val_split
            )
            if len(train_idx) == 0 or len(val_idx) == 0:
                return TrainResult(success=False, error="Train/val split resulted in empty set")

            harm_mats = [np.asarray(m, dtype=np.float32) for m in harmonic_matrices]
            param_vecs = [np.asarray(p, dtype=np.float32) for p in param_vectors]
            train_ds = HarmonicWindowDataset(
                [harm_mats[i] for i in train_idx],
                [param_vecs[i] for i in train_idx],
                [sample_labels[i] for i in train_idx],
                cnn_window=self.config.cnn_window,
                n_windows=self.config.n_windows_per_sample,
            )
            val_ds = HarmonicWindowDataset(
                [harm_mats[i] for i in val_idx],
                [param_vecs[i] for i in val_idx],
                [sample_labels[i] for i in val_idx],
                cnn_window=self.config.cnn_window,
                n_windows=1,
            )
            train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

            from .harmonic_model import HarmonicContextNet
            model = HarmonicContextNet(
                n_harm_features=n_harm,
                n_params=n_params,
                cnn_window=self.config.cnn_window,
                conv_channels=self.config.conv_channels,
                fc_hidden=self.config.fc_hidden,
                ks=self.config.kernel_size,
            )

            train_result = self._train_loop(model, train_loader, val_loader)
            result.epochs_trained = train_result["epochs"]
            result.best_val_loss = train_result["best_val_loss"]
            result.best_val_acc = train_result["best_val_acc"]
            result.train_loss_history = train_result["train_losses"]
            result.val_loss_history = train_result["val_losses"]

            self._save_model(model, n_harm, n_params, harm_cols, ctx_stats, result)
            result.success = True
        except Exception as e:
            logger.exception("Harmonic sequence training failed")
            result.error = str(e)

        result.duration_s = time.time() - t0
        return result

    def _context_stats_from_vectors(
        self, param_vectors: List[np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Z-score stats per context param from supplied vectors; columns from config."""
        keys = list(self.config.context_param_keys or [])
        sources = dict(self.config.context_param_sources or {})
        arr = np.asarray(param_vectors, dtype=float)
        stats: Dict[str, Dict[str, float]] = {}
        for i, key in enumerate(keys):
            col = arr[:, i] if (arr.ndim == 2 and i < arr.shape[1]) else np.array([])
            mean_val = float(np.mean(col)) if col.size else 0.0
            std_val = float(np.std(col)) if col.size else 1.0
            if std_val < 1e-10:
                std_val = 1.0
            stats[key] = {"mean": mean_val, "std": std_val, "source_column": sources.get(key, key)}
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_context_stats(
        self, df: DataFrame, ctx_cols: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute z-score normalisation stats for context params."""
        stats: Dict[str, Dict[str, float]] = {}
        for key, col in ctx_cols.items():
            vals = df[col].dropna().values.astype(float)
            mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std_val = float(np.std(vals)) if len(vals) > 0 else 1.0
            if std_val < 1e-10:
                std_val = 1.0
            stats[key] = {"mean": mean_val, "std": std_val, "source_column": col}
            logger.info(
                "  Context param '%s' (%s): mean=%.2f, std=%.2f",
                key, col, mean_val, std_val,
            )
        return stats

    def _build_samples(
        self,
        df: DataFrame,
        harm_cols: List[str],
        ctx_cols: Dict[str, str],
        ctx_stats: Dict[str, Dict[str, float]],
        labels: np.ndarray,
        operation_col: str,
            harm_data: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str]]:
        """Build per-window samples from the DataFrame.

        For pre_extracted mode, each row is one time step.  We group by
        operation and slide a window over the time axis.

        Returns:
            (harmonic_matrices, param_vectors, labels, operation_ids)
        """
        cw = self.config.cnn_window
        harm_matrices: List[np.ndarray] = []
        param_vectors: List[np.ndarray] = []
        sample_labels: List[int] = []
        sample_ops: List[str] = []

        # Reset index to guarantee positional alignment with numpy arrays
        df = df.reset_index(drop=True)

        # Extract and normalise context params for entire df
        norm_params = np.zeros((len(df), len(ctx_cols)), dtype=np.float32)
        for i, (key, col) in enumerate(ctx_cols.items()):
            vals = df[col].fillna(ctx_stats[key]["mean"]).values.astype(np.float32)
            norm_params[:, i] = (vals - ctx_stats[key]["mean"]) / ctx_stats[key]["std"]

        # Extract harmonic feature matrix
        if harm_data is None:
            harm_data = df[harm_cols].fillna(0).values.astype(np.float32)
        else:
            harm_data = np.asarray(harm_data, dtype=np.float32)

        # Group by operation for proper windowing
        if operation_col in df.columns:
            groups = df.groupby(operation_col)
        else:
            # Single group
            groups = [(self.config.dataset_name, df)]

        for op_id, group_df in groups:
            # Use integer positions (safe after reset_index)
            idx_arr = group_df.index

            group_harm = harm_data[idx_arr]
            group_params = norm_params[idx_arr]
            group_labels = labels[idx_arr]
            T = len(group_harm)

            if T < cw:
                # Use the entire short sequence as one sample
                mean_params = np.mean(group_params, axis=0)
                majority_label = int(np.round(np.mean(group_labels)))
                harm_matrices.append(group_harm)
                param_vectors.append(mean_params)
                sample_labels.append(majority_label)
                sample_ops.append(str(op_id))
                continue

            # Sliding window (step = cnn_window // 2 for overlap)
            step = max(1, cw // 2)
            for start in range(0, T - cw + 1, step):
                end = start + cw
                window_harm = group_harm[start:end]
                window_params = np.mean(group_params[start:end], axis=0)
                window_labels = group_labels[start:end]

                # Label: positive if ANY row in window is positive
                label = int(window_labels.max())

                harm_matrices.append(window_harm)
                param_vectors.append(window_params)
                sample_labels.append(label)
                sample_ops.append(str(op_id))

        logger.info(
            "Built %d windowed samples (%d positive, %d negative) from %d rows",
            len(sample_labels),
            sum(sample_labels),
            len(sample_labels) - sum(sample_labels),
            len(df),
        )
        return harm_matrices, param_vectors, sample_labels, sample_ops

    def _split_train_val(
        self,
        sample_ops: List[str],
        sample_labels: List[int],
        val_split: float,
    ) -> Tuple[List[int], List[int]]:
        """Split into train/val respecting operation boundaries.

        Tries to hold out one operation for validation.  Falls back to
        random split if there's only one operation.
        """
        unique_ops = sorted(set(sample_ops))

        if len(unique_ops) >= 2:
            # Hold out the last operation as validation
            val_op = unique_ops[-1]
            train_idx = [i for i, op in enumerate(sample_ops) if op != val_op]
            val_idx = [i for i, op in enumerate(sample_ops) if op == val_op]

            # Check balance: val should have some positives
            val_pos = sum(sample_labels[i] for i in val_idx)
            if val_pos == 0 and len(unique_ops) >= 3:
                # Try a different val op
                for candidate in reversed(unique_ops[:-1]):
                    candidate_idx = [i for i, op in enumerate(sample_ops) if op == candidate]
                    candidate_pos = sum(sample_labels[i] for i in candidate_idx)
                    if candidate_pos > 0:
                        val_op = candidate
                        train_idx = [i for i, op in enumerate(sample_ops) if op != val_op]
                        val_idx = [i for i, op in enumerate(sample_ops) if op == val_op]
                        break

            logger.info(
                "Operation-based split: train=%d (%s), val=%d (op=%s)",
                len(train_idx), [o for o in unique_ops if o != val_op],
                len(val_idx), val_op,
            )
            return train_idx, val_idx

        # Fallback: random split preserving label ratio
        n = len(sample_labels)
        indices = list(range(n))
        np.random.shuffle(indices)
        n_val = max(1, int(n * val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        logger.info(
            "Random split: train=%d, val=%d (single operation)", len(train_idx), len(val_idx)
        )
        return train_idx, val_idx

    def _train_loop(
        self,
        model: Any,  # HarmonicContextNet
        train_loader: "DataLoader",
        val_loader: "DataLoader",
    ) -> Dict[str, Any]:
        """Run the staged LR training loop with early stopping.

        Returns:
            Dict with epochs, best_val_loss, best_val_acc, train_losses, val_losses.
        """
        device = "cpu"
        model.to(device)

        pw = self.config.pos_weight
        if pw is not None:
            pos_weight = torch.tensor([pw], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        total_epochs = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        for stage in self.config.learning_rate_schedule:
            lr = stage["lr"]
            n_epochs = stage["epochs"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            logger.info("Training stage: lr=%.1e, %d epochs", lr, n_epochs)

            for epoch in range(n_epochs):
                # Train
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for harmonics, params, labels_batch in train_loader:
                    harmonics = harmonics.to(device)
                    params = params.to(device)
                    labels_batch = labels_batch.to(device)

                    optimizer.zero_grad()
                    logits = model(harmonics, params)
                    loss = criterion(logits, labels_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_train_loss)

                # Validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for harmonics, params, labels_batch in val_loader:
                        harmonics = harmonics.to(device)
                        params = params.to(device)
                        labels_batch = labels_batch.to(device)

                        logits = model(harmonics, params)
                        loss_val = criterion(logits, labels_batch)
                        val_loss += loss_val.item()

                        preds = (torch.sigmoid(logits) > 0.5).float()
                        val_correct += (preds == labels_batch).sum().item()
                        val_total += len(labels_batch)

                n_val_batches = max(len(val_loader), 1)
                avg_val_loss = val_loss / n_val_batches
                val_acc = val_correct / max(val_total, 1)
                val_losses.append(avg_val_loss)
                total_epochs += 1

                # Progress
                self._emit_progress({
                    "stage": "training",
                    "epoch": total_epochs,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(avg_val_loss, 4),
                    "val_acc": round(val_acc, 3),
                    "lr": lr,
                })

                if total_epochs % 5 == 0 or total_epochs <= 3:
                    logger.info(
                        "  Epoch %d: train_loss=%.4f, val_loss=%.4f, val_acc=%.3f",
                        total_epochs, avg_train_loss, avg_val_loss, val_acc,
                    )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d)",
                            total_epochs, self.config.early_stopping_patience,
                        )
                        break

            # Check if early stopping triggered
            if patience_counter >= self.config.early_stopping_patience:
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "epochs": total_epochs,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def _save_model(
        self,
        model: Any,
        n_harm: int,
        n_params: int,
        harm_cols: List[str],
        ctx_stats: Dict[str, Dict[str, float]],
        result: TrainResult,
    ) -> None:
        """Persist model + config to disk."""
        from .harmonic_model import HarmonicContextScorer

        # Update config with training results
        self.config.n_harm_features = n_harm
        self.config.n_params = n_params
        self.config.harmonic_columns = harm_cols
        self.config.context_param_stats = json_safe(ctx_stats)
        self.config.trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.config.training_metrics = json_safe({
            "best_val_loss": result.best_val_loss,
            "best_val_acc": result.best_val_acc,
            "n_samples": result.n_samples,
            "n_positive": result.n_positive,
            "epochs_trained": result.epochs_trained,
        })

        # Save via scorer
        scorer = HarmonicContextScorer(config=self.config)
        scorer._model = model
        scorer._is_loaded = True
        save_path = Path(self.config.model_save_path)
        scorer.save(save_path)
        result.model_path = str(save_path)

        logger.info("Model saved to %s", save_path)

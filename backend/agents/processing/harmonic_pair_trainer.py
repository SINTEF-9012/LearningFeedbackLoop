"""Training pipeline for the pair-input harmonic model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...json_utils import json_safe
from .harmonic_features import resolve_spindle_speed_source_column
from .harmonic_peak_pairs import (
    build_pair_feature_labels,
    discover_peak_pair_columns,
    extract_peak_pairs_from_df,
)
from .harmonic_trainer import TrainResult, discover_context_columns

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


if TORCH_AVAILABLE:

    class PairWindowDataset(Dataset):
        """Dataset of ``(pair_window, params, label)`` samples."""

        def __init__(
            self,
            pair_matrices: List[np.ndarray],
            param_vectors: List[np.ndarray],
            labels: List[int],
            *,
            cnn_window: int = 16,
            n_windows: int = 2,
        ):
            self.entries: List[Tuple[np.ndarray, np.ndarray, int]] = []

            for pair_mat, p_vec, label in zip(pair_matrices, param_vectors, labels):
                time_steps = pair_mat.shape[0]
                if time_steps < cnn_window:
                    pad_shape = (cnn_window - time_steps,) + pair_mat.shape[1:]
                    pad = np.zeros(pad_shape, dtype=np.float32)
                    pair_mat = np.concatenate([pad, pair_mat], axis=0)
                    time_steps = cnn_window

                valid_starts = max(1, time_steps - cnn_window + 1)
                if valid_starts == 1:
                    starts = [0] * n_windows
                else:
                    starts = [np.random.randint(0, valid_starts) for _ in range(n_windows)]

                for start in starts:
                    window = pair_mat[start : start + cnn_window].astype(np.float32)
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


class HarmonicPairTrainer:
    """Train a pair-input harmonic model from labelled FFT-peak DataFrames."""

    def __init__(self, config: Any):
        from .harmonic_config import HarmonicContextConfig

        self.config: HarmonicContextConfig = config
        self._progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def _training_seed(self) -> int:
        return int(getattr(self.config, "random_seed", 0) or 0)

    def set_progress_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self._progress_callback = cb

    def _emit_progress(self, data: Dict[str, Any]) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(data)
            except Exception:
                pass

    def train_from_dataframe(
        self,
        df: DataFrame,
        operation_col: str = "operation_id",
    ) -> TrainResult:
        if not TORCH_AVAILABLE:
            return TrainResult(success=False, error="PyTorch not installed")
        if pd is None:
            return TrainResult(success=False, error="pandas not installed")

        from .harmonic_pair_model import _build_pair_model

        t0 = time.time()
        result = TrainResult()
        seed = self._training_seed()

        try:
            specs = discover_peak_pair_columns(
                list(df.columns),
                frequency_patterns=self.config.pair_frequency_column_patterns,
                amplitude_patterns=self.config.pair_amplitude_column_patterns,
                k_peaks=self.config.k_peaks,
            )
            if not specs:
                return TrainResult(
                    success=False,
                    error="No FFT peak frequency/amplitude column pairs found for pair-input training",
                )

            ctx_cols = discover_context_columns(
                df,
                self.config.context_param_keys,
                self.config.context_param_sources,
            )
            if not ctx_cols:
                return TrainResult(
                    success=False,
                    error=f"No context param columns found for keys: {self.config.context_param_keys}",
                )

            ctx_stats = self._compute_context_stats(df, ctx_cols)
            label_col = self.config.target_label
            if label_col not in df.columns:
                return TrainResult(
                    success=False,
                    error=f"Label column '{label_col}' not in DataFrame",
                )

            positive_mask = df[label_col].isin(self.config.positive_labels)
            labels = positive_mask.astype(int).to_numpy()
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)
            logger.info("Pair labels: %d positive, %d negative", n_pos, n_neg)
            if n_pos == 0:
                return TrainResult(
                    success=False,
                    error=f"No positive samples found (looked for labels: {self.config.positive_labels})",
                )

            pair_matrices, param_vectors, sample_labels, sample_ops = self._build_samples(
                df,
                specs,
                ctx_cols,
                ctx_stats,
                labels,
                operation_col,
            )
            if not sample_labels:
                return TrainResult(success=False, error="No pair-input samples could be built")

            n_channels = pair_matrices[0].shape[1]
            n_params = len(ctx_cols)

            result.n_samples = len(sample_labels)
            result.n_positive = int(sum(sample_labels))
            result.n_negative = result.n_samples - result.n_positive
            result.n_harm_features = int(n_channels * self.config.k_peaks * 2)
            result.n_params = n_params
            result.harmonic_columns = build_pair_feature_labels(specs)
            result.context_param_stats = ctx_stats

            train_idx, val_idx = self._split_train_val(
                sample_ops,
                sample_labels,
                self.config.val_split,
                rng=np.random.default_rng(seed),
            )
            if len(train_idx) == 0 or len(val_idx) == 0:
                return TrainResult(success=False, error="Train/val split resulted in empty set")

            train_ds = PairWindowDataset(
                [pair_matrices[i] for i in train_idx],
                [param_vectors[i] for i in train_idx],
                [sample_labels[i] for i in train_idx],
                cnn_window=self.config.cnn_window,
                n_windows=self.config.n_windows_per_sample,
            )
            val_ds = PairWindowDataset(
                [pair_matrices[i] for i in val_idx],
                [param_vectors[i] for i in val_idx],
                [sample_labels[i] for i in val_idx],
                cnn_window=self.config.cnn_window,
                n_windows=1,
            )

            train_batch_size = max(1, min(int(self.config.batch_size), len(train_ds)))
            drop_last = len(train_ds) > train_batch_size and (len(train_ds) % train_batch_size == 1)
            val_batch_size = max(1, min(int(self.config.batch_size), len(val_ds)))

            train_loader = DataLoader(
                train_ds,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=drop_last,
                generator=self._train_loader_generator(seed),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=val_batch_size,
                shuffle=False,
            )

            model_kind = str(getattr(self.config, "model_kind", "legacy_v1") or "legacy_v1").strip().lower()
            torch.manual_seed(seed)
            model = _build_pair_model(
                model_kind=model_kind,
                n_channels=n_channels,
                k_peaks=self.config.k_peaks,
                n_params=n_params,
                cnn_window=self.config.cnn_window,
                pair_embed_dim=self.config.pair_embed_dim,
                conv_channels=self.config.conv_channels,
                fc_hidden=self.config.fc_hidden,
                ks=self.config.kernel_size,
            )
            if model_kind == "lfl_v2" and hasattr(model, "set_param_stats"):
                ctx_order = list(ctx_cols.keys())
                param_mean = torch.tensor(
                    [ctx_stats[key]["mean"] for key in ctx_order],
                    dtype=torch.float32,
                )
                param_std = torch.tensor(
                    [ctx_stats[key]["std"] for key in ctx_order],
                    dtype=torch.float32,
                )
                model.set_param_stats(param_mean, param_std)

            train_result = self._train_loop(model, train_loader, val_loader)
            result.epochs_trained = train_result["epochs"]
            result.best_val_loss = train_result["best_val_loss"]
            result.best_val_acc = train_result["best_val_acc"]
            result.train_loss_history = train_result["train_losses"]
            result.val_loss_history = train_result["val_losses"]

            self._save_model(model, n_channels, n_params, result.harmonic_columns, ctx_stats, result)
            result.success = True
        except Exception as exc:
            logger.exception("Harmonic pair training failed")
            result.error = str(exc)

        result.duration_s = time.time() - t0
        logger.info(
            "Pair training %s in %.1fs: %s",
            "succeeded" if result.success else "failed",
            result.duration_s,
            result.error or f"val_loss={result.best_val_loss:.4f}, acc={result.best_val_acc:.3f}",
        )
        return result

    def _compute_context_stats(
        self,
        df: DataFrame,
        ctx_cols: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for key, col in ctx_cols.items():
            values = df[col].dropna().to_numpy(dtype=float)
            mean_val = float(np.mean(values)) if len(values) > 0 else 0.0
            std_val = float(np.std(values)) if len(values) > 0 else 1.0
            if std_val < 1e-10:
                std_val = 1.0
            stats[key] = {"mean": mean_val, "std": std_val, "source_column": col}
        return stats

    def _build_samples(
        self,
        df: DataFrame,
        specs: List[Any],
        ctx_cols: Dict[str, str],
        ctx_stats: Dict[str, Dict[str, float]],
        labels: np.ndarray,
        operation_col: str,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str]]:
        cw = self.config.cnn_window
        sample_mode = str(getattr(self.config, "pair_sample_mode", "sliding_windows") or "sliding_windows")
        pair_matrices: List[np.ndarray] = []
        param_vectors: List[np.ndarray] = []
        sample_labels: List[int] = []
        sample_ops: List[str] = []
        model_kind = str(getattr(self.config, "model_kind", "legacy_v1") or "legacy_v1").strip().lower()

        df = df.reset_index(drop=True).copy()
        if operation_col not in df.columns:
            if {"session", "engagement_idx"}.issubset(df.columns):
                df[operation_col] = (
                    df["session"].astype(str)
                    + ":"
                    + df["engagement_idx"].astype(str)
                    + ":"
                    + df[self.config.target_label].astype(str)
                )
            else:
                df[operation_col] = self.config.dataset_name or "pair_raw"

        prepared_params = np.zeros((len(df), len(ctx_cols)), dtype=np.float32)
        for idx, (key, col) in enumerate(ctx_cols.items()):
            values = df[col].fillna(ctx_stats[key]["mean"]).to_numpy(dtype=np.float32)
            if model_kind == "lfl_v2":
                prepared_params[:, idx] = values
            else:
                prepared_params[:, idx] = (values - ctx_stats[key]["mean"]) / ctx_stats[key]["std"]

        resolved_spindle_speed_col = resolve_spindle_speed_source_column(self.config)
        spindle_speed_col = (
            ctx_cols.get("spindle_speed")
            or ctx_cols.get("n")
            or (resolved_spindle_speed_col if resolved_spindle_speed_col in df.columns else None)
        )
        groups = df.groupby(operation_col) if operation_col in df.columns else [(self.config.dataset_name, df)]

        for op_id, group_df in groups:
            idx_arr = group_df.index.to_numpy()
            group_pairs = extract_peak_pairs_from_df(
                group_df,
                specs,
                spindle_speed_col=spindle_speed_col,
                k_peaks=self.config.k_peaks,
                f_max_rel=self.config.f_max_rel,
            )
            if group_pairs.shape[0] == 0:
                continue

            group_params = prepared_params[idx_arr]
            group_labels = labels[idx_arr]
            time_steps = group_pairs.shape[0]

            if time_steps < cw:
                pair_matrices.append(group_pairs)
                param_vectors.append(np.mean(group_params, axis=0).astype(np.float32))
                sample_labels.append(int(np.max(group_labels)))
                sample_ops.append(str(op_id))
                continue

            if sample_mode == "trailing_window":
                pair_matrices.append(group_pairs[time_steps - cw : time_steps])
                param_vectors.append(np.mean(group_params[time_steps - cw : time_steps], axis=0).astype(np.float32))
                sample_labels.append(int(np.max(group_labels)))
                sample_ops.append(str(op_id))
                continue

            step = max(1, cw // 2)
            for start in range(0, time_steps - cw + 1, step):
                end = start + cw
                pair_matrices.append(group_pairs[start:end])
                param_vectors.append(np.mean(group_params[start:end], axis=0).astype(np.float32))
                sample_labels.append(int(np.max(group_labels[start:end])))
                sample_ops.append(str(op_id))

        logger.info(
            "Built %d pair samples (%d positive, %d negative) from %d rows",
            len(sample_labels),
            sum(sample_labels),
            len(sample_labels) - sum(sample_labels),
            len(df),
        )
        return pair_matrices, param_vectors, sample_labels, sample_ops

    def _split_train_val(
        self,
        sample_ops: List[str],
        sample_labels: List[int],
        val_split: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], List[int]]:
        rng = rng or np.random.default_rng(self._training_seed())
        op_to_indices: Dict[str, List[int]] = {}
        for idx, op_id in enumerate(sample_ops):
            op_to_indices.setdefault(str(op_id), []).append(idx)

        unique_ops = sorted(op_to_indices)
        if len(unique_ops) >= 2:
            positive_ops: List[str] = []
            negative_ops: List[str] = []
            for op_id in unique_ops:
                op_label = max(sample_labels[i] for i in op_to_indices[op_id])
                if op_label > 0:
                    positive_ops.append(op_id)
                else:
                    negative_ops.append(op_id)

            rng.shuffle(positive_ops)
            rng.shuffle(negative_ops)

            target_val_ops = max(1, min(len(unique_ops) - 1, int(round(len(unique_ops) * val_split))))

            def _max_holdout_count(ops: List[str]) -> int:
                if len(ops) <= 1:
                    return 0
                return len(ops) - 1

            n_val_pos = int(round(len(positive_ops) / max(len(unique_ops), 1) * target_val_ops))
            n_val_neg = target_val_ops - n_val_pos

            n_val_pos = max(0, min(n_val_pos, _max_holdout_count(positive_ops)))
            n_val_neg = max(0, min(n_val_neg, _max_holdout_count(negative_ops)))

            if target_val_ops >= 2:
                if len(positive_ops) > 1 and n_val_pos == 0:
                    n_val_pos = 1
                if len(negative_ops) > 1 and n_val_neg == 0:
                    n_val_neg = 1

            while n_val_pos + n_val_neg > target_val_ops:
                if n_val_neg > n_val_pos and n_val_neg > 0:
                    n_val_neg -= 1
                elif n_val_pos > 0:
                    n_val_pos -= 1
                else:
                    break

            while n_val_pos + n_val_neg < target_val_ops:
                can_add_pos = n_val_pos < _max_holdout_count(positive_ops)
                can_add_neg = n_val_neg < _max_holdout_count(negative_ops)
                if can_add_neg and (not can_add_pos or (len(negative_ops) - n_val_neg) >= (len(positive_ops) - n_val_pos)):
                    n_val_neg += 1
                elif can_add_pos:
                    n_val_pos += 1
                else:
                    break

            val_ops = set(positive_ops[:n_val_pos] + negative_ops[:n_val_neg])
            if val_ops:
                train_idx = [i for i, op in enumerate(sample_ops) if str(op) not in val_ops]
                val_idx = [i for i, op in enumerate(sample_ops) if str(op) in val_ops]
                if train_idx and val_idx:
                    return train_idx, val_idx

        n = len(sample_labels)
        indices = list(range(n))
        rng.shuffle(indices)
        n_val = max(1, int(n * val_split))
        return indices[n_val:], indices[:n_val]

    def _train_loader_generator(self, seed: int) -> "torch.Generator":
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _train_loop(self, model: Any, train_loader: "DataLoader", val_loader: "DataLoader") -> Dict[str, Any]:
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

            for _ in range(n_epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for pairs, params, labels_batch in train_loader:
                    pairs = pairs.to(device)
                    params = params.to(device)
                    labels_batch = labels_batch.to(device)

                    optimizer.zero_grad()
                    logits = model(pairs, params)
                    loss = criterion(logits, labels_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train_loss = epoch_loss / max(n_batches, 1)
                train_losses.append(avg_train_loss)

                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for pairs, params, labels_batch in val_loader:
                        pairs = pairs.to(device)
                        params = params.to(device)
                        labels_batch = labels_batch.to(device)

                        logits = model(pairs, params)
                        loss_val = criterion(logits, labels_batch)
                        val_loss += loss_val.item()

                        preds = (torch.sigmoid(logits) > 0.5).float()
                        val_correct += (preds == labels_batch).sum().item()
                        val_total += len(labels_batch)

                avg_val_loss = val_loss / max(len(val_loader), 1)
                val_acc = val_correct / max(val_total, 1)
                val_losses.append(avg_val_loss)
                total_epochs += 1

                self._emit_progress({
                    "stage": "training",
                    "epoch": total_epochs,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(avg_val_loss, 4),
                    "val_acc": round(val_acc, 3),
                    "lr": lr,
                })

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        break

            if patience_counter >= self.config.early_stopping_patience:
                break

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
        n_channels: int,
        n_params: int,
        pair_labels: List[str],
        ctx_stats: Dict[str, Dict[str, float]],
        result: TrainResult,
    ) -> None:
        from .harmonic_pair_model import HarmonicPairScorer

        self.config.n_harm_features = int(n_channels * self.config.k_peaks * 2)
        self.config.n_params = n_params
        self.config.harmonic_columns = pair_labels
        self.config.context_param_stats = json_safe(ctx_stats)
        self.config.trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.config.training_metrics = json_safe({
            "best_val_loss": result.best_val_loss,
            "best_val_acc": result.best_val_acc,
            "n_samples": result.n_samples,
            "n_positive": result.n_positive,
            "epochs_trained": result.epochs_trained,
            "random_seed": self._training_seed(),
        })

        scorer = HarmonicPairScorer(config=self.config)
        scorer._model = model
        scorer._is_loaded = True
        save_path = Path(self.config.model_save_path)
        scorer.save(save_path)
        result.model_path = str(save_path)
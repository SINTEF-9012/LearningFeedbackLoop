"""Feedback-buffered retraining for harmonic context and pair models.

This mirrors the classical feedback retrainer pattern, but it reuses the
existing harmonic DataFrame trainers instead of inventing a separate training
pipeline. Confirmed and dismissed memories become labeled per-event rows,
bucketed by harmonic preset, and can later be retrained on demand.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model_confidence import (
    fingerprint_model_artifact,
    reset_model_confidence_state,
    resolve_model_confidence_path,
)

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


_PAIR_COLUMN_RE = re.compile(
    r"(Accel_FFT_.*_(Frequencies|Amplitudes)_\d+|Vibration_Peak_\d+_[XY]_(Amplitude|Frequency))$",
    re.IGNORECASE,
)


def _resolve_harmonic_model_confidence_path(path: Optional[str]) -> Any:
    base_path = resolve_model_confidence_path(path)
    suffix = base_path.suffix or ".json"
    return base_path.with_name(f"{base_path.stem}_harmonic_context{suffix}")


def _config_for_feedback_bucket(
    dataset_name: Optional[str],
    scorer_kind: Optional[str],
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Any]:
    from .harmonic_config import (
        HarmonicContextConfig,
        casedata_stoppage_preset,
        pair_casedata_preset,
        pair_lfl_preset,
        pair_raw_preset,
        raw_accelerometer_preset,
        stoppage_1hz_preset,
        site_a_line2_breakage_preset,
    )

    dataset_key = str(dataset_name or "default").strip().lower() or "default"
    kind = str(scorer_kind or "context").strip().lower() or "context"
    overrides = dict(config_overrides or {})

    pair_factories = {
        "pair_raw": pair_raw_preset,
        "pair_casedata": pair_casedata_preset,
        "pair_lfl": pair_lfl_preset,
    }

    if dataset_key in pair_factories or kind == "pair":
        factory = pair_factories.get(dataset_key, pair_raw_preset)
        config = factory(**overrides)
        resolved_dataset = str(getattr(config, "dataset_name", "") or dataset_key or "pair_raw")
        return resolved_dataset, "pair", config

    factories = {
        "default": HarmonicContextConfig,
        "casedata": casedata_stoppage_preset,
        "stoppage_1hz": stoppage_1hz_preset,
        "site_a_line2": site_a_line2_breakage_preset,
        "raw_accelerometer": raw_accelerometer_preset,
    }
    factory = factories.get(dataset_key, HarmonicContextConfig)
    config = factory(**overrides)
    resolved_dataset = str(getattr(config, "dataset_name", "") or dataset_key or "default")
    return resolved_dataset, "context", config


def _bucket_key(dataset_name: str, scorer_kind: str) -> str:
    return f"{scorer_kind}:{dataset_name}"


@dataclass
class HarmonicFeedbackSample:
    dataset_name: str
    scorer_kind: str
    row: Dict[str, Any]
    is_significant: bool
    memory_id: str = ""
    session_id: str = ""
    timestamp: float = 0.0
    harmonic_feature_labels: List[str] = field(default_factory=list)


@dataclass
class HarmonicFeedbackRetrainResult:
    success: bool = False
    message: str = ""
    bucket_key: str = ""
    dataset_name: str = ""
    scorer_kind: str = "context"
    n_samples_used: int = 0
    n_confirmed: int = 0
    n_dismissed: int = 0
    model_path: str = ""
    best_val_loss: Optional[float] = None
    best_val_acc: Optional[float] = None
    duration_s: float = 0.0
    training_result: Dict[str, Any] = field(default_factory=dict)
    config: Any = None


class HarmonicFeedbackRetrainer:
    """Manage per-preset feedback buffers for harmonic model retraining."""

    def __init__(
        self,
        *,
        model_confidence_path: Optional[str] = None,
        feedback_threshold: int = 20,
        min_positive_samples: int = 3,
        min_negative_samples: int = 3,
    ):
        self.model_confidence_path = _resolve_harmonic_model_confidence_path(model_confidence_path)
        self.feedback_threshold = int(feedback_threshold)
        self.min_positive_samples = int(min_positive_samples)
        self.min_negative_samples = int(min_negative_samples)

        self._buffers: Dict[str, List[HarmonicFeedbackSample]] = {}
        self._total_feedback_count: Dict[str, int] = {}
        self._last_retrain_count: Dict[str, int] = {}
        self._retrain_history: Dict[str, List[HarmonicFeedbackRetrainResult]] = {}
        self._lock = Lock()

    def record_feedback(
        self,
        *,
        was_significant: bool,
        raw_metrics: Optional[Dict[str, float]],
        harmonic_context: Optional[Dict[str, Any]],
        harmonic_runtime: Optional[Dict[str, Any]],
        cutting_context: Optional[Dict[str, Any]],
        source: Optional[str],
        casedata: Optional[Dict[str, Any]],
        memory_id: str,
        session_id: str,
    ) -> bool:
        dataset_name, scorer_kind = self._resolve_runtime(
            harmonic_runtime=harmonic_runtime,
            raw_metrics=raw_metrics,
            source=source,
            casedata=casedata,
        )
        dataset_name, scorer_kind, config = _config_for_feedback_bucket(dataset_name, scorer_kind)

        operation_id = self._derive_operation_id(
            session_id=session_id,
            memory_id=memory_id,
            casedata=casedata,
        )
        row, harmonic_labels = self._build_feedback_row(
            config=config,
            was_significant=was_significant,
            raw_metrics=raw_metrics,
            harmonic_context=harmonic_context,
            cutting_context=cutting_context,
            operation_id=operation_id,
        )
        if row is None:
            return False

        sample = HarmonicFeedbackSample(
            dataset_name=dataset_name,
            scorer_kind=scorer_kind,
            row=row,
            is_significant=was_significant,
            memory_id=memory_id,
            session_id=session_id,
            timestamp=float(row.get("_feedback_timestamp", time.time())),
            harmonic_feature_labels=harmonic_labels,
        )
        bucket = _bucket_key(dataset_name, scorer_kind)

        with self._lock:
            self._buffers.setdefault(bucket, []).append(sample)
            self._total_feedback_count[bucket] = self._total_feedback_count.get(bucket, 0) + 1

        logger.debug(
            "Recorded harmonic feedback sample bucket=%s total=%d",
            bucket,
            self._total_feedback_count.get(bucket, 0),
        )
        return True

    def should_retrain(
        self,
        *,
        dataset_name: Optional[str] = None,
        scorer_kind: Optional[str] = None,
    ) -> bool:
        bucket = self._select_bucket(dataset_name=dataset_name, scorer_kind=scorer_kind)
        if bucket is not None:
            return self._bucket_ready(bucket)
        return any(self._bucket_ready(key) for key in self._buffers.keys())

    def retrain(
        self,
        *,
        dataset_name: Optional[str] = None,
        scorer_kind: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        model_save_path: Optional[str] = None,
        checkpoint_suffix: Optional[str] = None,
        replace_checkpoint: bool = False,
    ) -> HarmonicFeedbackRetrainResult:
        from .harmonic_pair_trainer import HarmonicPairTrainer
        from .harmonic_config import resolve_training_model_save_path
        from .harmonic_trainer import HarmonicContextTrainer

        t0 = time.time()
        result = HarmonicFeedbackRetrainResult(duration_s=0.0)
        bucket = self._select_bucket(dataset_name=dataset_name, scorer_kind=scorer_kind)
        if bucket is None:
            result.message = (
                "No harmonic feedback bucket selected."
                if not self._buffers
                else "Multiple harmonic feedback buckets available; specify dataset and scorer_kind"
            )
            result.duration_s = time.time() - t0
            return result

        with self._lock:
            buffer = list(self._buffers.get(bucket, []))

        if not buffer:
            result.message = "No harmonic feedback samples available for retraining"
            result.bucket_key = bucket
            result.duration_s = time.time() - t0
            return result

        confirmed = [sample for sample in buffer if sample.is_significant]
        dismissed = [sample for sample in buffer if not sample.is_significant]
        result.bucket_key = bucket
        result.dataset_name = buffer[0].dataset_name
        result.scorer_kind = buffer[0].scorer_kind
        result.n_confirmed = len(confirmed)
        result.n_dismissed = len(dismissed)
        result.n_samples_used = len(buffer)

        if len(confirmed) < self.min_positive_samples:
            result.message = (
                f"Insufficient confirmed samples ({len(confirmed)} < {self.min_positive_samples})"
            )
            result.duration_s = time.time() - t0
            return result
        if len(dismissed) < self.min_negative_samples:
            result.message = (
                f"Insufficient dismissed samples ({len(dismissed)} < {self.min_negative_samples})"
            )
            result.duration_s = time.time() - t0
            return result

        if pd is None:
            result.message = "pandas not installed"
            result.duration_s = time.time() - t0
            return result

        resolved_dataset, resolved_kind, config = _config_for_feedback_bucket(
            result.dataset_name,
            result.scorer_kind,
            config_overrides=config_overrides,
        )
        config.model_save_path = resolve_training_model_save_path(
            getattr(config, "model_save_path", ""),
            model_save_path=model_save_path,
            checkpoint_suffix=checkpoint_suffix,
            random_seed=random_seed,
            replace_checkpoint=replace_checkpoint,
        )
        result.dataset_name = resolved_dataset
        result.scorer_kind = resolved_kind
        result.config = config

        try:
            rows = [sample.row for sample in buffer]
            df = pd.DataFrame(rows)
            if "_feedback_timestamp" in df.columns and "operation_id" in df.columns:
                df = df.sort_values(["operation_id", "_feedback_timestamp"]).reset_index(drop=True)

            if resolved_kind != "pair" and not list(getattr(config, "harmonic_columns", []) or []):
                explicit_cols: List[str] = []
                for sample in buffer:
                    for label in sample.harmonic_feature_labels:
                        if label in df.columns and label not in explicit_cols:
                            explicit_cols.append(label)
                if explicit_cols:
                    config.harmonic_columns = explicit_cols

            trainer = HarmonicPairTrainer(config) if resolved_kind == "pair" else HarmonicContextTrainer(config)
            training_result = trainer.train_from_dataframe(df, operation_col="operation_id")

            result.training_result = training_result.to_dict()
            result.success = bool(training_result.success)
            result.model_path = str(training_result.model_path or "")
            result.best_val_loss = (
                None if training_result.best_val_loss == float("inf") else float(training_result.best_val_loss)
            )
            result.best_val_acc = float(training_result.best_val_acc) if training_result.success else None
            result.message = (
                f"Harmonic model retrained for {resolved_kind}:{resolved_dataset}"
                if training_result.success
                else str(training_result.error or "Harmonic retraining failed")
            )

            if training_result.success and training_result.model_path:
                try:
                    model_fingerprint = fingerprint_model_artifact(training_result.model_path)
                    reset_model_confidence_state(
                        path=self.model_confidence_path,
                        model_fingerprint=model_fingerprint,
                        reason="model_retrained",
                    )
                except Exception as exc:
                    logger.warning(
                        "Harmonic retrained but model-confidence reset failed for %s: %s",
                        self.model_confidence_path,
                        exc,
                    )

                with self._lock:
                    self._last_retrain_count[bucket] = self._total_feedback_count.get(bucket, 0)

        except Exception as exc:
            logger.exception("Harmonic feedback retraining failed")
            result.success = False
            result.message = f"Harmonic retraining failed: {exc}"

        result.duration_s = time.time() - t0
        with self._lock:
            self._retrain_history.setdefault(bucket, []).append(result)
        return result

    def get_status(
        self,
        *,
        dataset_name: Optional[str] = None,
        scorer_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_bucket = self._select_bucket(dataset_name=dataset_name, scorer_kind=scorer_kind)
        buckets: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            keys = sorted(self._buffers.keys())

        for bucket in keys:
            samples = list(self._buffers.get(bucket, []))
            if not samples:
                continue
            sample = samples[0]
            if dataset_name is not None and sample.dataset_name != str(dataset_name).strip().lower():
                continue
            if scorer_kind is not None and sample.scorer_kind != str(scorer_kind).strip().lower():
                continue

            n_pos = sum(1 for entry in samples if entry.is_significant)
            n_neg = len(samples) - n_pos
            total_feedback = self._total_feedback_count.get(bucket, len(samples))
            since_last_retrain = total_feedback - self._last_retrain_count.get(bucket, 0)
            _, _, config = _config_for_feedback_bucket(sample.dataset_name, sample.scorer_kind)
            history = self._retrain_history.get(bucket, [])

            buckets[bucket] = {
                "dataset_name": sample.dataset_name,
                "scorer_kind": sample.scorer_kind,
                "total_feedback": total_feedback,
                "since_last_retrain": since_last_retrain,
                "retrain_threshold": self.feedback_threshold,
                "buffer_size": len(samples),
                "confirmed_in_buffer": n_pos,
                "dismissed_in_buffer": n_neg,
                "should_retrain": self._bucket_ready(bucket),
                "retrain_count": len(history),
                "last_retrain": history[-1].message if history else None,
                "model_save_path": str(getattr(config, "model_save_path", "") or ""),
            }

        return {
            "total_feedback": sum(self._total_feedback_count.values()),
            "active_bucket": active_bucket,
            "buckets": buckets,
        }

    def reset_feedback(
        self,
        *,
        dataset_name: Optional[str] = None,
        scorer_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        bucket = self._select_bucket(dataset_name=dataset_name, scorer_kind=scorer_kind)
        if bucket is None:
            return {
                "bucket_key": "",
                "removed_buffer_size": 0,
                "removed_total_feedback": 0,
            }

        with self._lock:
            removed_buffer = len(self._buffers.pop(bucket, []))
            removed_total_feedback = int(self._total_feedback_count.pop(bucket, removed_buffer))
            self._last_retrain_count.pop(bucket, None)
            self._retrain_history.pop(bucket, None)

        return {
            "bucket_key": bucket,
            "removed_buffer_size": removed_buffer,
            "removed_total_feedback": removed_total_feedback,
        }

    def _select_bucket(
        self,
        *,
        dataset_name: Optional[str] = None,
        scorer_kind: Optional[str] = None,
    ) -> Optional[str]:
        if dataset_name is not None or scorer_kind is not None:
            resolved_dataset, resolved_kind, _ = _config_for_feedback_bucket(dataset_name, scorer_kind)
            return _bucket_key(resolved_dataset, resolved_kind)

        keys = sorted(self._buffers.keys())
        if len(keys) == 1:
            return keys[0]

        ready = [key for key in keys if self._bucket_ready(key)]
        if len(ready) == 1:
            return ready[0]
        return None

    def _bucket_ready(self, bucket: str) -> bool:
        samples = self._buffers.get(bucket, [])
        total_feedback = self._total_feedback_count.get(bucket, len(samples))
        since_last_retrain = total_feedback - self._last_retrain_count.get(bucket, 0)
        if since_last_retrain < self.feedback_threshold:
            return False

        n_pos = sum(1 for sample in samples if sample.is_significant)
        n_neg = len(samples) - n_pos
        return n_pos >= self.min_positive_samples and n_neg >= self.min_negative_samples

    def _resolve_runtime(
        self,
        *,
        harmonic_runtime: Optional[Dict[str, Any]],
        raw_metrics: Optional[Dict[str, float]],
        source: Optional[str],
        casedata: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        runtime = harmonic_runtime if isinstance(harmonic_runtime, dict) else {}
        scorer_kind = str(runtime.get("scorer_kind") or "").strip().lower()
        dataset_name = str(runtime.get("dataset") or "").strip().lower()
        raw = raw_metrics or {}
        source_str = str(source or "").strip().lower()
        casedata_meta = casedata if isinstance(casedata, dict) else {}
        source_hints = " ".join(
            part
            for part in (
                source_str,
                str(casedata_meta.get("root") or "").strip().lower(),
                str(casedata_meta.get("case_dir") or "").strip().lower(),
                str(casedata_meta.get("machine_id") or "").strip().lower(),
            )
            if part
        )

        if not scorer_kind:
            has_pair_columns = any(_PAIR_COLUMN_RE.match(str(key)) for key in raw.keys())
            scorer_kind = "pair" if has_pair_columns else "context"

        if not dataset_name:
            if scorer_kind == "pair":
                dataset_name = "pair_raw"
            elif isinstance(casedata, dict) and casedata:
                dataset_name = "casedata"
            elif "olddata" in source_hints or "stoppage" in source_hints:
                dataset_name = "stoppage_1hz"
            elif "site_a_line2" in source_hints:
                dataset_name = "site_a_line2"
            elif "site_a" in source_hints:
                dataset_name = "casedata"
            elif "casedata" in source_hints or "site_b" in source_hints:
                dataset_name = "casedata"
            elif "raw" in source_hints or "accelerometer" in source_hints:
                dataset_name = "raw_accelerometer"
            else:
                dataset_name = "default"

        return dataset_name, scorer_kind

    def _build_feedback_row(
        self,
        *,
        config: Any,
        was_significant: bool,
        raw_metrics: Optional[Dict[str, float]],
        harmonic_context: Optional[Dict[str, Any]],
        cutting_context: Optional[Dict[str, Any]],
        operation_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        row: Dict[str, Any] = {}
        harmonic_feature_labels: List[str] = []

        for key, value in (raw_metrics or {}).items():
            if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
                row[str(key)] = float(value)

        raw_harmonic_context = harmonic_context if isinstance(harmonic_context, dict) else {}
        labels = raw_harmonic_context.get("feature_labels")
        values = raw_harmonic_context.get("feature_values")
        if isinstance(labels, list) and isinstance(values, list):
            for label, value in zip(labels, values):
                if label is None:
                    continue
                if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
                    label_str = str(label)
                    row.setdefault(label_str, float(value))
                    harmonic_feature_labels.append(label_str)

        context_dict = cutting_context if isinstance(cutting_context, dict) else {}
        context_sources = dict(getattr(config, "context_param_sources", {}) or {})
        if context_sources:
            for logical_key, source_col in context_sources.items():
                if not isinstance(source_col, str) or not source_col:
                    continue
                if source_col in row:
                    continue
                fallback_value = row.get(logical_key)
                if fallback_value is None:
                    fallback_value = context_dict.get(logical_key)
                if isinstance(fallback_value, (int, float, np.floating, np.integer)) and np.isfinite(fallback_value):
                    row[source_col] = float(fallback_value)
        else:
            for logical_key in getattr(config, "context_param_keys", []) or []:
                if logical_key in row:
                    continue
                fallback_value = context_dict.get(logical_key)
                if isinstance(fallback_value, (int, float, np.floating, np.integer)) and np.isfinite(fallback_value):
                    row[logical_key] = float(fallback_value)

        scorer_kind = str(getattr(config, "scorer_kind", "context") or "context").strip().lower()
        if scorer_kind == "pair":
            has_pair_features = any(_PAIR_COLUMN_RE.match(key) for key in row.keys())
            if not has_pair_features:
                return None, harmonic_feature_labels
        else:
            has_pattern_harmonics = False
            for pattern in getattr(config, "harmonic_column_patterns", []) or []:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    continue
                if any(regex.search(column) for column in row.keys()):
                    has_pattern_harmonics = True
                    break
            explicit_harmonics = [column for column in getattr(config, "harmonic_columns", []) or [] if column in row]
            if not has_pattern_harmonics and not explicit_harmonics and not harmonic_feature_labels:
                return None, harmonic_feature_labels

        row[getattr(config, "target_label", "label")] = (
            list(getattr(config, "positive_labels", ["positive"]) or ["positive"])[0]
            if was_significant else "feedback_negative"
        )
        row["operation_id"] = operation_id
        row["_feedback_timestamp"] = time.time()
        return row, harmonic_feature_labels

    @staticmethod
    def _derive_operation_id(
        *,
        session_id: str,
        memory_id: str,
        casedata: Optional[Dict[str, Any]],
    ) -> str:
        if isinstance(casedata, dict):
            operation_id = casedata.get("operation_id")
            if isinstance(operation_id, str) and operation_id:
                return operation_id
        if isinstance(session_id, str) and session_id:
            return session_id
        return memory_id or "feedback"


_harmonic_feedback_retrainer: Optional[HarmonicFeedbackRetrainer] = None


def get_harmonic_feedback_retrainer(
    model_confidence_path: Optional[str] = None,
) -> HarmonicFeedbackRetrainer:
    global _harmonic_feedback_retrainer
    if _harmonic_feedback_retrainer is None:
        _harmonic_feedback_retrainer = HarmonicFeedbackRetrainer(
            model_confidence_path=model_confidence_path,
        )
    elif model_confidence_path is not None:
        _harmonic_feedback_retrainer.model_confidence_path = _resolve_harmonic_model_confidence_path(
            model_confidence_path
        )
    return _harmonic_feedback_retrainer
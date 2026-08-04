"""Model retraining loop — triggered by accumulated operator feedback.

When operators confirm or dismiss alerts, the system accumulates labeled
feature vectors.  After a configurable threshold (default: 20 feedback
events), the classical anomaly model is retrained on the augmented dataset.

This closes the loop between feedback and model quality:
  Operator feedback → prior updates (immediate) → model retrain (deferred, batch)

The retrainer can be triggered:
  1. Automatically by the FeedbackHandler after N feedbacks
  2. Manually via ``POST /agent/memory/retrain``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..model_confidence import (
    fingerprint_model_artifact,
    reset_model_confidence_state,
    resolve_model_confidence_path,
)

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """A single labeled sample from operator feedback."""
    features: np.ndarray
    is_significant: bool  # True = confirmed, False = dismissed
    pattern_keys: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    memory_id: str = ""


@dataclass
class RetrainResult:
    """Output of a retraining run."""
    success: bool = False
    message: str = ""
    n_samples_used: int = 0
    n_confirmed: int = 0
    n_dismissed: int = 0
    model_path: Optional[str] = None
    previous_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    duration_s: float = 0.0


class ModelRetrainer:
    """Manages the feedback → retrain loop for the classical anomaly model.

    Usage::

        retrainer = ModelRetrainer(model_path=Path("data/models/seed_model.joblib"))

        # Called from FeedbackHandler after each confirm/dismiss:
        retrainer.record_feedback(features, is_significant=True)

        # After N feedbacks, check if retraining is needed:
        if retrainer.should_retrain():
            result = retrainer.retrain()
    """

    def __init__(
        self,
        model_path: Path = Path("data/models/seed_model.joblib"),
        model_confidence_path: Optional[Path] = None,
        feedback_threshold: int = 20,
        min_positive_samples: int = 3,
        min_negative_samples: int = 3,
        base_normal_features: Optional[np.ndarray] = None,
    ):
        self.model_path = Path(model_path)
        self.model_confidence_path = resolve_model_confidence_path(model_confidence_path)
        self.feedback_threshold = feedback_threshold
        self.min_positive_samples = min_positive_samples
        self.min_negative_samples = min_negative_samples
        self.base_normal_features = base_normal_features

        self._feedback_buffer: List[FeedbackSample] = []
        self._total_feedback_count: int = 0
        self._last_retrain_count: int = 0
        self._retrain_history: List[RetrainResult] = []
        self._lock = Lock()

        logger.info(
            "ModelRetrainer initialized (threshold=%d, model=%s)",
            feedback_threshold, model_path,
        )

    @property
    def feedback_count(self) -> int:
        return self._total_feedback_count

    @property
    def feedbacks_since_last_retrain(self) -> int:
        return self._total_feedback_count - self._last_retrain_count

    @property
    def retrain_history(self) -> List[RetrainResult]:
        return list(self._retrain_history)

    def record_feedback(
        self,
        features: Optional[np.ndarray],
        is_significant: bool,
        pattern_keys: Optional[List[str]] = None,
        memory_id: str = "",
    ) -> None:
        """Record a feedback sample for future retraining."""
        if features is None or len(features) == 0:
            return

        with self._lock:
            self._feedback_buffer.append(FeedbackSample(
                features=np.asarray(features).ravel(),
                is_significant=is_significant,
                pattern_keys=pattern_keys or [],
                timestamp=time.time(),
                memory_id=memory_id,
            ))
            self._total_feedback_count += 1

        logger.debug(
            "Recorded feedback sample (%s), total=%d, since_last_retrain=%d",
            "confirmed" if is_significant else "dismissed",
            self._total_feedback_count,
            self.feedbacks_since_last_retrain,
        )

    def should_retrain(self) -> bool:
        """Check whether enough feedback has accumulated to warrant retraining."""
        if self.feedbacks_since_last_retrain < self.feedback_threshold:
            return False

        # Also require minimum class balance
        with self._lock:
            n_pos = sum(1 for s in self._feedback_buffer if s.is_significant)
            n_neg = len(self._feedback_buffer) - n_pos

        if n_pos < self.min_positive_samples or n_neg < self.min_negative_samples:
            logger.debug(
                "Enough feedbacks (%d) but insufficient class balance: %d pos, %d neg",
                self.feedbacks_since_last_retrain, n_pos, n_neg,
            )
            return False

        return True

    def retrain(self, base_normal_features: Optional[np.ndarray] = None) -> RetrainResult:
        """Retrain the classical model from accumulated feedback + base features.

        Steps:
        1. Extract confirmed samples → positive class (breakage indicators)
        2. Extract dismissed samples + base normal features → negative class
        3. Retrain the SeedModel
        4. Save the updated model
        """
        from backend.agents.processing.classical_models import (
            SeedModel,
            SeedModelConfig,
            features_from_dict,
        )

        t0 = time.time()
        result = RetrainResult()

        with self._lock:
            buffer = list(self._feedback_buffer)

        if not buffer:
            result.message = "No feedback samples available for retraining"
            return result

        confirmed = [s for s in buffer if s.is_significant]
        dismissed = [s for s in buffer if not s.is_significant]
        result.n_confirmed = len(confirmed)
        result.n_dismissed = len(dismissed)

        # Need features from both classes
        if len(confirmed) < self.min_positive_samples:
            result.message = f"Insufficient confirmed samples ({len(confirmed)} < {self.min_positive_samples})"
            return result
        if len(dismissed) < self.min_negative_samples:
            result.message = f"Insufficient dismissed samples ({len(dismissed)} < {self.min_negative_samples})"
            return result

        try:
            # Build training arrays
            pos_features = np.stack([s.features for s in confirmed])
            neg_features = np.stack([s.features for s in dismissed])

            # Include base normal features if available
            base = base_normal_features if base_normal_features is not None else self.base_normal_features
            if base is not None and len(base) > 0:
                neg_features = np.vstack([neg_features, base])

            # The SeedModel is an Isolation Forest — it's trained on normal data.
            # After feedback, we know which samples are truly normal (dismissed)
            # and which are anomalous (confirmed).
            # Strategy: retrain the Isolation Forest on the normal data (dismissed),
            # then validate against confirmed as anomalies.

            # Load existing model for comparison
            old_model = SeedModel(config=SeedModelConfig())
            if self.model_path.exists():
                old_model.load(self.model_path)
                # Score the feedback data with the old model for comparison
                old_scores_pos = np.array([old_model.score(f) for f in pos_features])
                old_scores_neg = np.array([old_model.score(f) for f in neg_features])
                # Accuracy: confirmed should score high, dismissed should score low
                old_correct = np.sum(old_scores_pos > 0.5) + np.sum(old_scores_neg <= 0.5)
                old_total = len(old_scores_pos) + len(old_scores_neg)
                result.previous_accuracy = float(old_correct / old_total) if old_total > 0 else None

            # Create and train new model
            new_model = SeedModel(config=SeedModelConfig())
            new_model.train(neg_features)  # Train on normal data

            # Evaluate new model
            new_scores_pos = np.array([new_model.score(f) for f in pos_features])
            new_scores_neg = np.array([new_model.score(f) for f in neg_features])
            new_correct = np.sum(new_scores_pos > 0.5) + np.sum(new_scores_neg <= 0.5)
            new_total = len(new_scores_pos) + len(new_scores_neg)
            result.new_accuracy = float(new_correct / new_total) if new_total > 0 else None

            # Only save if new model is at least as good (or we have no baseline)
            should_save = True
            if result.previous_accuracy is not None and result.new_accuracy is not None:
                if result.new_accuracy < result.previous_accuracy - 0.05:
                    should_save = False
                    result.message = (
                        f"New model accuracy ({result.new_accuracy:.2%}) is worse than "
                        f"previous ({result.previous_accuracy:.2%}), skipping save"
                    )
                    logger.warning(result.message)

            if should_save:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                new_model.save(self.model_path)

                try:
                    model_fingerprint = fingerprint_model_artifact(self.model_path)
                    reset_model_confidence_state(
                        path=self.model_confidence_path,
                        model_fingerprint=model_fingerprint,
                        reason="model_retrained",
                    )
                except Exception as exc:
                    logger.warning(
                        "Model retrained but model-confidence reset failed for %s: %s",
                        self.model_confidence_path,
                        exc,
                    )

                result.success = True
                result.model_path = str(self.model_path)
                result.message = (
                    f"Model retrained: {result.new_accuracy:.2%} accuracy "
                    f"({result.n_confirmed} confirmed, {result.n_dismissed} dismissed"
                    f"{f', {len(base)} base normal' if base is not None else ''})"
                )
                logger.info(result.message)

                # Update internal state
                with self._lock:
                    self._last_retrain_count = self._total_feedback_count
                    # Keep buffer — future retrains use all accumulated data

        except Exception as e:
            result.message = f"Retraining failed: {e}"
            logger.exception("Model retraining failed")

        result.n_samples_used = result.n_confirmed + result.n_dismissed
        result.duration_s = time.time() - t0
        self._retrain_history.append(result)

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current retrainer status."""
        with self._lock:
            n_pos = sum(1 for s in self._feedback_buffer if s.is_significant)
            n_neg = len(self._feedback_buffer) - n_pos

        return {
            "total_feedback": self._total_feedback_count,
            "since_last_retrain": self.feedbacks_since_last_retrain,
            "retrain_threshold": self.feedback_threshold,
            "buffer_size": len(self._feedback_buffer),
            "confirmed_in_buffer": n_pos,
            "dismissed_in_buffer": n_neg,
            "should_retrain": self.should_retrain(),
            "retrain_count": len(self._retrain_history),
            "last_retrain": (
                self._retrain_history[-1].message
                if self._retrain_history else None
            ),
        }


# Module-level singleton (lazily created)
_retrainer: Optional[ModelRetrainer] = None


def get_retrainer(
    model_path: Optional[Path] = None,
    model_confidence_path: Optional[Path] = None,
) -> ModelRetrainer:
    """Get or create the global ModelRetrainer singleton."""
    global _retrainer
    if _retrainer is None:
        _retrainer = ModelRetrainer(
            model_path=model_path or Path("data/models/seed_model.joblib"),
            model_confidence_path=model_confidence_path,
        )
    else:
        if model_path is not None:
            _retrainer.model_path = Path(model_path)
        if model_confidence_path is not None:
            _retrainer.model_confidence_path = resolve_model_confidence_path(model_confidence_path)
    return _retrainer

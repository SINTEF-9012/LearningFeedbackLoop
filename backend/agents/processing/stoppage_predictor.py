"""
Stoppage Predictor Agent — Runs trained RF model for real-time stop prediction.

Loads the supervised Random Forest trained by the stoppage experiment and
provides a ``predict()`` method that maps a feature dict to a
(label, probability, patterns) tuple.

Integration points:
  - Memory bridge: called from ``_process_features`` in bridge.py when a
    feature window arrives, adding ``stoppage_prediction`` to external_signals.
  - Agent dispatch: registered as ``"stoppage"`` agent in the router so it
    can be invoked via ``POST /agent/dispatch/{session_id}``.
  - Experiment evaluator: can be used instead of inline RF scoring.

Usage:
    from backend.agents.processing.stoppage_predictor import StoppagePredictor

    predictor = StoppagePredictor.from_experiment()
    label, prob, patterns = predictor.predict(feature_dict)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.schemas import PatternKey, PatternType

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
_DEFAULT_EXPERIMENT_BASE = Path("data/breakage_patterns/stoppage_experiment")
_DEFAULT_MODEL_NAME = "stoppage_supervised.joblib"

# Columns that must NEVER be used as features (leaky / metadata)
_LEAKY_COLUMNS = frozenset([
    "event_stop_duration_s", "event_spindle_rpm", "event_feed_rate",
    "event_feed_override", "feed_actual_min", "spindle_actual_min",
    "feed_override_min", "spindle_override_min",
])
_METADATA_COLUMNS = frozenset([
    "sample_id", "label", "operation_id", "tool_number", "event_timestamp",
    "severity", "stop_type", "window_seconds", "gap_seconds",
    "trim_seconds_removed",
])
_EXCLUDED = _LEAKY_COLUMNS | _METADATA_COLUMNS


@dataclass
class PredictionResult:
    """Result of a stoppage prediction."""
    label: str                     # "pre_break" or "normal"
    probability: float             # P(pre_break)
    threshold: float               # decision threshold used
    is_stop_predicted: bool        # probability >= threshold
    pattern_keys: List[PatternKey] # patterns generated from the prediction
    feature_importances_top: Dict[str, float] = field(default_factory=dict)


class StoppagePredictor:
    """Loads a trained RF model and provides real-time stop prediction.

    The predictor is deliberately stateless per-call: it loads the model
    once and then ``predict()`` is a pure function of the input features.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: List[str],
        threshold: float = 0.5,
        prediction_gap_s: float = 0.0,
        model_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.prediction_gap_s = prediction_gap_s
        self.model_path = model_path
        self._n_features = len(feature_columns)
        logger.info(
            "StoppagePredictor initialized: %d features, threshold=%.3f, gap=%.0fs",
            self._n_features, threshold, prediction_gap_s,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_experiment(
        cls,
        experiment_dir: Optional[Path] = None,
        gap_s: float = 10.0,
    ) -> "StoppagePredictor":
        """Load the predictor from the latest (or specified) experiment run.

        Parameters
        ----------
        experiment_dir : Path, optional
            Explicit experiment directory.  If *None*, the latest run matching
            the given gap is selected automatically.
        gap_s : float
            Prediction gap in seconds.  Used to locate the correct run
            directory when *experiment_dir* is not given.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required: pip install joblib")

        if experiment_dir is None:
            experiment_dir = cls._find_latest_run(gap_s)

        model_path = experiment_dir / _DEFAULT_MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run: python scripts/run_stoppage_experiment.py --single --gap "
                f"{int(gap_s)}"
            )

        model_data = joblib.load(model_path)

        # The joblib file is a dict: {model, feature_cols, model_type}
        if isinstance(model_data, dict):
            model = model_data["model"]
            feature_columns = model_data.get("feature_cols", [])
            if not feature_columns:
                logger.warning("No feature_cols in model dict, using generic names")
                feature_columns = [f"feature_{i}" for i in range(model.n_features_in_)]
        else:
            # Legacy: bare model object
            model = model_data
            feature_columns = [f"feature_{i}" for i in range(model.n_features_in_)]

        # Load threshold from calibration (try train meta first, then results)
        threshold = 0.5
        meta_path = experiment_dir / "stoppage_train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            threshold = meta.get("calibration", {}).get("threshold", threshold)
        else:
            cal_path = experiment_dir / "experiment_results.json"
            if cal_path.exists():
                results = json.loads(cal_path.read_text())
                train = results.get("train_phase", {})
                threshold = train.get("calibration", {}).get("threshold", threshold)

        return cls(
            model=model,
            feature_columns=feature_columns,
            threshold=threshold,
            prediction_gap_s=gap_s,
            model_path=model_path,
        )

    @staticmethod
    def _find_latest_run(gap_s: float) -> Path:
        """Find the most recent experiment directory for a given gap."""
        suffix = f"_gap{int(gap_s)}s" if gap_s > 0 else ""
        candidates = sorted(
            _DEFAULT_EXPERIMENT_BASE.glob(f"train_*{suffix}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No experiment runs found for gap={gap_s}s in "
                f"{_DEFAULT_EXPERIMENT_BASE}"
            )
        return candidates[0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Run the RF model on a feature dict and return a PredictionResult.

        Parameters
        ----------
        features : dict
            Flat dict of ``{column_name: float_value}``.  Missing columns
            are filled with 0.0.  Leaky / metadata columns are ignored.

        Returns
        -------
        PredictionResult
        """
        # Build ordered feature vector, dropping excluded columns
        X = np.array(
            [features.get(col, 0.0) for col in self.feature_columns],
            dtype=np.float64,
        ).reshape(1, -1)

        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        prob = float(self.model.predict_proba(X)[0, 1])
        is_stop = prob >= self.threshold
        label = "pre_break" if is_stop else "normal"

        # Build pattern keys
        patterns = self._build_patterns(prob, is_stop, features)

        # Top feature importances for explainability
        top_feats = self._top_importances(X[0])

        return PredictionResult(
            label=label,
            probability=prob,
            threshold=self.threshold,
            is_stop_predicted=is_stop,
            pattern_keys=patterns,
            feature_importances_top=top_feats,
        )

    def predict_batch(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[PredictionResult]:
        """Run predictions on multiple samples at once."""
        return [self.predict(fd) for fd in feature_dicts]

    # ------------------------------------------------------------------
    # Agent dispatch interface
    # ------------------------------------------------------------------

    def handle_request(
        self,
        session_id: str,
        action: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Agent dispatch handler (sync).

        Expected args:
            features: dict of {column: value}
        """
        args = args or {}
        features = args.get("features", {})
        if not features:
            return {"error": "No features provided", "ok": False}

        result = self.predict(features)
        return {
            "ok": True,
            "label": result.label,
            "probability": result.probability,
            "threshold": result.threshold,
            "is_stop_predicted": result.is_stop_predicted,
            "pattern_keys": [pk.key for pk in result.pattern_keys],
            "top_features": result.feature_importances_top,
            "gap_s": self.prediction_gap_s,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_patterns(
        self, prob: float, is_stop: bool, features: Dict[str, float]
    ) -> List[PatternKey]:
        """Generate PatternKey objects based on prediction outcome."""
        patterns: List[PatternKey] = []

        if is_stop:
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key="STOPPAGE_PREDICTED",
                fault_type="tool_breakage",
                confidence=prob,
                source_metric="rf_model",
            ))
            # Add severity-based pattern
            if prob >= 0.8:
                severity = "CRITICAL"
            elif prob >= 0.6:
                severity = "HIGH"
            else:
                severity = "MEDIUM"
            patterns.append(PatternKey(
                pattern_type=PatternType.FAULT,
                key=f"STOPPAGE_SEVERITY_{severity}",
                fault_type="tool_breakage",
                confidence=prob,
            ))
            # Check for specific physical indicators
            if features.get("power_spindle_slope", 0) > 5.0:
                patterns.append(PatternKey(
                    pattern_type=PatternType.FAULT,
                    key="BREAKAGE_POWER_RAMP",
                    fault_type="tool_breakage",
                    source_metric="power_spindle_slope",
                ))
            if features.get("vib_severity_x_slope", 0) > 0.5:
                patterns.append(PatternKey(
                    pattern_type=PatternType.FAULT,
                    key="BREAKAGE_VIB_RAMP",
                    fault_type="tool_breakage",
                    source_metric="vib_severity_x_slope",
                ))
        else:
            # Low-confidence "watch" pattern when probability is elevated
            if prob >= 0.3:
                patterns.append(PatternKey(
                    pattern_type=PatternType.ANOMALY,
                    key="STOPPAGE_WATCH",
                    confidence=prob,
                    source_metric="rf_model",
                ))

        return patterns

    def _top_importances(self, x: np.ndarray, k: int = 5) -> Dict[str, float]:
        """Return top-k feature importances weighted by the input values."""
        if not hasattr(self.model, "feature_importances_"):
            return {}
        imp = self.model.feature_importances_
        # Contribution = importance * |feature_value|
        contrib = imp * np.abs(x)
        top_idx = np.argsort(contrib)[-k:][::-1]
        return {
            self.feature_columns[i]: round(float(contrib[i]), 6)
            for i in top_idx
            if contrib[i] > 0
        }


# ============================================================================
# Module-level singleton for lazy loading
# ============================================================================

_predictor: Optional[StoppagePredictor] = None


def get_predictor(gap_s: float = 10.0) -> StoppagePredictor:
    """Get or create the singleton StoppagePredictor.

    Lazily loads the model on first call.  Thread-safe for read-only
    usage (the model itself is immutable after training).
    """
    global _predictor
    if _predictor is None:
        try:
            _predictor = StoppagePredictor.from_experiment(gap_s=gap_s)
        except (FileNotFoundError, ImportError) as e:
            logger.warning("StoppagePredictor not available: %s", e)
            raise
    return _predictor


def reset_predictor() -> None:
    """Clear the cached predictor (for testing / model reloading)."""
    global _predictor
    _predictor = None

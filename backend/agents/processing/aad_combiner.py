"""AAD combiner — a feedback-trained linear model over the detectors.

Active Anomaly Discovery (Das et al., ICDM'16/'19; Siddiqui et al., KDD'18): rather
than nudging a per-pattern prior multiplier, learn a weight over the detector
ensemble from operator feedback so confirmed anomalies are pushed above threshold.
Here the ensemble is the set of fired pattern keys plus the anomaly score; the
combiner is an online logistic regression updated on each confirm/dismiss.

This solves the credit-assignment problem the per-pattern prior cannot: a gradient
step attributes blame to the specific patterns that distinguish real stops, instead
of bumping every co-firing pattern equally.

Plugs in two ways:
  - as a signal model (``score`` -> {"aad_score": p}) via the model registry, and
  - surfaced into scoring by the ``aad_combiner`` rule (reads ``aad_score``).
The owner (e.g. the experiment loop) calls ``update`` on feedback.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .model_registry import register_model


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class AADCombiner:
    """Online logistic regression over fired-pattern indicators + anomaly score.

    score = sigmoid(bias + Σ w[p] for fired p + w_anom * anomaly)
    update: gradient step on log-loss toward the feedback label, with L2 decay.
    """

    name = "aad_combiner"

    def __init__(
        self,
        learning_rate: float = 0.3,
        l2: float = 1e-3,
        anomaly_key: str = "anomaly",
        init_bias: float = 0.0,
    ):
        self.lr = float(learning_rate)
        self.l2 = float(l2)
        self.anomaly_key = anomaly_key
        self.bias = float(init_bias)
        self.w: Dict[str, float] = {}        # per-pattern weight
        self.w_anom = 0.0                    # weight on the anomaly score
        self.n_updates = 0

    # ---- ScoringModel protocol -------------------------------------------
    def score(self, features: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, float]:
        """`features` is a flat dict; expects ``pattern_keys`` (list) and an
        anomaly score under ``anomaly_key``. Returns {"aad_score": p}."""
        pattern_keys = features.get("pattern_keys") or []
        anomaly = float(features.get(self.anomaly_key, 0.0) or 0.0)
        return {"aad_score": self._prob(pattern_keys, anomaly)}

    def _prob(self, pattern_keys: List[str], anomaly: float) -> float:
        z = self.bias + self.w_anom * anomaly + sum(self.w.get(str(p), 0.0) for p in pattern_keys)
        return _sigmoid(z)

    def update(self, pattern_keys: List[str], anomaly: float, label: bool) -> float:
        """One online log-loss step toward ``label`` (True=confirm, False=dismiss).
        Returns the post-update probability for the same input."""
        y = 1.0 if label else 0.0
        p = self._prob(pattern_keys, anomaly)
        grad = (y - p)  # d(-logloss)/dz for logistic
        step = self.lr * grad
        self.bias += step
        self.w_anom += step * anomaly - self.lr * self.l2 * self.w_anom
        for pk in pattern_keys:
            k = str(pk)
            self.w[k] = self.w.get(k, 0.0) + step - self.lr * self.l2 * self.w.get(k, 0.0)
        self.n_updates += 1
        return self._prob(pattern_keys, anomaly)

    def top_weights(self, k: int = 8) -> List[tuple]:
        """Most influential patterns (for interpretability / audit)."""
        return sorted(self.w.items(), key=lambda kv: -abs(kv[1]))[:k]


# Register as a pluggable signal model.
register_model("aad_combiner", lambda **kw: AADCombiner(**kw))

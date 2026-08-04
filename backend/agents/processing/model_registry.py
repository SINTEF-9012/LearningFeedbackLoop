"""Pluggable model registry — register models, select a SET to run.

Two kinds of "model" plug into the LFL scoring pipeline:

1. **Scoring rules** (fusion layer) — register via
   ``backend.agents.memory.scorer.register_scoring_rule`` and select with
   ``SignificanceConfig.enabled_rules``. A rule consumes signals + patterns and
   contributes a weighted score.

2. **Signal models** (this module) — anomaly detectors, supervised classifiers,
   an AAD combiner, a temporal model, etc. Each maps features -> one or more
   named signals (e.g. ``anomaly_detector_score``, ``harmonic_alert_score``,
   ``aad_score``) that the scorer reads from ``external_signals``. Register here,
   pick a set per run, and ``run_models`` merges their outputs.

This makes models swappable without editing the scorer: add a class implementing
``ScoringModel``, register it, and name it in the run config.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ScoringModel(Protocol):
    """Duck-typed interface every plug-in signal model implements.

    ``name``    : unique key used for registration/selection.
    ``fit``     : optional; train from features/labels (no-op for stateless models).
    ``score``   : map a flat feature dict (+ optional context) to a dict of
                  named signals to merge into ``external_signals``.
    """

    name: str

    def score(self, features: Dict[str, float], context: Optional[Any] = None) -> Dict[str, float]:
        ...


MODEL_REGISTRY: Dict[str, Callable[..., ScoringModel]] = {}


def register_model(name: str, factory: Callable[..., ScoringModel]) -> None:
    """Register a signal-model factory under ``name``."""
    MODEL_REGISTRY[name] = factory


def available_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def select_models(names: List[str], **factory_kwargs) -> List[ScoringModel]:
    """Instantiate the chosen SET of models. Unknown names raise (fail loud)."""
    models: List[ScoringModel] = []
    for n in names:
        if n not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{n}'. Registered: {available_models()}")
        models.append(MODEL_REGISTRY[n](**factory_kwargs))
    return models


def run_models(
    models: List[ScoringModel],
    features: Dict[str, float],
    context: Optional[Any] = None,
    *,
    on_error: str = "skip",
) -> Dict[str, float]:
    """Run a set of models and merge their signals into one dict.

    Later models do not overwrite earlier signals of the same name (first wins),
    so ordering encodes precedence. ``on_error='skip'`` drops a failing model
    (degrade gracefully, the repo convention); ``'raise'`` propagates.
    """
    merged: Dict[str, float] = {}
    for m in models:
        try:
            out = m.score(features, context) or {}
        except Exception:
            if on_error == "raise":
                raise
            continue
        for k, v in out.items():
            if k not in merged and isinstance(v, (int, float)):
                merged[k] = float(v)
    return merged

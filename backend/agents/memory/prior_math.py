"""Single source of truth for the pattern-significance prior estimator.

Both the live scoring path (:class:`~backend.agents.memory.scorer.SignificanceScorer`)
and the offline stoppage experiment (:mod:`backend.agents.experiment.evaluator`)
must derive a pattern's prior from feedback counts using *exactly* the same
function. Before this module existed they diverged — production used the
``effective_feedback_count`` estimator below while the experiment used a plain
``alpha / (alpha + beta)`` Beta mean — so a prior learned offline did not match
the one applied online. Keep this the only implementation.

Semantics (be precise — the name "decay" is historically misleading):

``effective_feedback_count`` does **not** weight recent feedback more than old
feedback. Only aggregate confirm/dismiss totals are available here (no event
ordering), so ``recency_decay ** count`` is a *saturating* transform of the
total, not a time/recency weighting:

    eff(n) = (1 - decay ** n) / (1 - decay)        for 0 < decay < 1

It is bounded by ``1 / (1 - decay)`` (≈ 6.67 at decay=0.85), which caps how
confident the prior can ever become — a deliberate regularisation, but a
confidence ceiling, not forgetting. Changing ``decay`` retroactively
invalidates cached priors, so callers should not vary it casually.
"""

from __future__ import annotations

#: Default saturation factor. Geometric series ``(1 - d**n)/(1 - d)`` caps the
#: effective count at ``1/(1-d)`` (≈ 6.67 at 0.85), which in turn caps the
#: derived prior at ≈ 0.885 / 0.115. See module docstring.
DEFAULT_RECENCY_DECAY: float = 0.85


def effective_feedback_count(count: float, recency_decay: float = DEFAULT_RECENCY_DECAY) -> float:
    """Saturating transform of a raw confirm/dismiss total.

    Returns ``count`` unchanged when ``recency_decay >= 1.0`` (no saturation),
    ``0.0`` for non-positive counts, otherwise ``(1 - decay**count)/(1 - decay)``.
    """
    if count <= 0:
        return 0.0
    if recency_decay >= 1.0:
        return float(count)
    return (1.0 - recency_decay ** count) / (1.0 - recency_decay)


def prior_from_counts(
    confirm: float,
    dismiss: float,
    recency_decay: float = DEFAULT_RECENCY_DECAY,
) -> tuple[float, float]:
    """Derive a pattern prior in ``[0, 1]`` from confirm/dismiss counts.

    Beta(1, 1) (Laplace) smoothing over the saturated effective counts::

        prior = (eff_confirm + 1) / (eff_confirm + eff_dismiss + 2)

    Returns ``(prior, raw_total)`` where ``raw_total = confirm + dismiss`` is the
    *un-saturated* evidence count (used downstream for evidence damping, so it
    must reflect true volume, not the saturated value).
    """
    eff_confirm = effective_feedback_count(confirm, recency_decay)
    eff_dismiss = effective_feedback_count(dismiss, recency_decay)
    prior = float((eff_confirm + 1) / (eff_confirm + eff_dismiss + 2))
    return (prior, float(confirm + dismiss))

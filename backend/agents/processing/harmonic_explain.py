"""
Agent O — build a per-harmonic contribution breakdown for a memory.

Given a scorer result (`score` + `context_weights`) and the harmonic
feature row that was fed into it, produce an explainable breakdown:

    {
        "available": bool,
        "reason": str,                 # present when available=False
        "score": float,
        "model_source": str,
        "dataset": str,
        "context_weights": [float],
        "feature_labels": [str],
        "harmonic_values": [float],
        "contributions": [             # |value * weight|, signed
            {"label": str, "weight": float, "value": float, "contribution": float}
        ],
        "top_weighted": [              # first k by |contribution|
            {"label": str, "weight": float, "value": float, "contribution": float}
        ],
    }

All fields are safe to serialize as plain JSON. The helper does NOT call
PyTorch directly — it relies on whatever ``HarmonicContextScorer``-like
object is injected (real or fake), so tests can run without torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _as_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _default_labels(n: int) -> List[str]:
    return [f"h{i+1}" for i in range(n)]


def build_harmonic_explanation(
    score_result: Optional[Dict[str, Any]],
    harmonic_row: Optional[Sequence[float]],
    feature_labels: Optional[Sequence[str]],
    *,
    dataset_name: str = "",
    top_k: int = 5,
    available: bool = True,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose the explain payload.

    Args:
        score_result: The dict returned by ``HarmonicContextScorer.score``
            (expected keys: ``harmonic_context_score``, ``context_weights``,
            ``model_source``). ``None`` → produce an unavailable response.
        harmonic_row: (F,) harmonic-feature values for the current sample
            (the latest row used in scoring). ``None`` / empty → zeros.
        feature_labels: Optional human-readable label per harmonic feature,
            parallel to ``context_weights`` / ``harmonic_row``. Missing
            entries are filled with ``hN`` placeholders.
        dataset_name: Echoed through for UI grouping.
        top_k: How many entries the ``top_weighted`` list should hold.
        available: Force an "unavailable" payload (overrides positive data).
        reason: Optional human-readable reason used when ``available=False``
            or when ``score_result`` is ``None``.
    """
    if not available or score_result is None:
        return {
            "available": False,
            "reason": reason or "harmonic model unavailable",
            "score": None,
            "model_source": "",
            "dataset": dataset_name,
            "context_weights": [],
            "feature_labels": [],
            "harmonic_values": [],
            "contributions": [],
            "top_weighted": [],
        }

    weights = _as_float_list(score_result.get("context_weights"))
    values = _as_float_list(harmonic_row) if harmonic_row is not None else []
    n = max(len(weights), len(values))

    # Pad to common length so UI rendering never indexes out of range.
    if len(weights) < n:
        weights = weights + [0.0] * (n - len(weights))
    if len(values) < n:
        values = values + [0.0] * (n - len(values))

    labels: List[str]
    if feature_labels is None:
        labels = _default_labels(n)
    else:
        labels = [str(s) for s in feature_labels]
        if len(labels) < n:
            # Continue numbering from the existing count so tests and UIs
            # see a stable "h{N+1}, h{N+2}, …" sequence.
            labels = labels + [f"h{i+1}" for i in range(len(labels), n)]
        elif len(labels) > n:
            labels = labels[:n]

    contributions: List[Dict[str, Any]] = []
    for lbl, w, v in zip(labels, weights, values):
        contrib = w * v
        contributions.append({
            "label": lbl,
            "weight": round(w, 6),
            "value": round(v, 6),
            "contribution": round(contrib, 6),
        })

    top_k = max(0, int(top_k))
    top_weighted = sorted(
        contributions, key=lambda d: abs(d["contribution"]), reverse=True
    )[:top_k]

    score = score_result.get("harmonic_context_score")
    try:
        score_val: Optional[float] = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = None

    return {
        "available": True,
        "reason": reason,
        "score": score_val,
        "model_source": str(score_result.get("model_source", "")),
        "dataset": dataset_name,
        "context_weights": [round(w, 6) for w in weights],
        "feature_labels": labels,
        "harmonic_values": [round(v, 6) for v in values],
        "contributions": contributions,
        "top_weighted": top_weighted,
    }

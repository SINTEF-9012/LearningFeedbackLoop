"""Feedback-driven confidence for the classical model path.

The live scorer already tracks whether the classical rule fired on events that
operators later confirm or dismiss. This module persists a compact summary of
that feedback so both the inference streamer and the online anomaly detector can
emit the same model confidence value instead of using a static training-sample
heuristic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MODEL_CONFIDENCE_PATH

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()

# Scope key for the site-wide (context-free) aggregate. Kept as a reserved key
# inside the scoped store; also mirrored at the top level of the persisted JSON
# so pre-scoping readers still see a valid flat state.
GLOBAL_SCOPE = "__global__"

# Empirical-Bayes shrinkage: a context scope leans on the global aggregate until
# it has accumulated its own evidence. Half-weight at this many context events.
_SCOPE_SHRINK_K = 3.0


@dataclass
class ModelConfidenceState:
    """Persisted feedback summary for the classical model path."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    updated_at: Optional[str] = None
    model_fingerprint: Optional[str] = None
    last_reset_reason: Optional[str] = None
    version: int = 2

    @property
    def evidence_count(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives

    @property
    def feedback_count(self) -> int:
        return self.evidence_count + self.true_negatives

    def smoothed_precision(self) -> float:
        return (self.true_positives + 1.0) / (self.true_positives + self.false_positives + 2.0)

    def smoothed_recall(self) -> float:
        return (self.true_positives + 1.0) / (self.true_positives + self.false_negatives + 2.0)

    def smoothed_f1(self) -> float:
        precision = self.smoothed_precision()
        recall = self.smoothed_recall()
        return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    def smoothed_fbeta(self, beta: float = 0.5) -> float:
        precision = self.smoothed_precision()
        recall = self.smoothed_recall()
        beta_sq = float(beta) * float(beta)
        denom = (beta_sq * precision) + recall
        if denom <= 0.0:
            return 0.0
        return (1.0 + beta_sq) * precision * recall / denom

    def feedback_confidence(self, neutral: float = 0.5, damping_k: float = 5.0) -> float:
        evidence = float(self.evidence_count)
        if damping_k <= 0.0:
            damping = 1.0
        elif evidence <= 0.0:
            damping = 0.0
        else:
            damping = evidence / (evidence + float(damping_k))

        # Precision matters slightly more than recall for alert trust because
        # repeated false positives make operators stop trusting the model.
        confidence_basis = self.smoothed_fbeta(beta=0.5)
        confidence = float(neutral) + (confidence_basis - float(neutral)) * damping
        return max(0.05, min(0.95, confidence))

    def record(self, *, model_fired: bool, was_confirmed: bool) -> None:
        if model_fired and was_confirmed:
            self.true_positives += 1
        elif model_fired and not was_confirmed:
            self.false_positives += 1
        elif not model_fired and was_confirmed:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "model_fingerprint": self.model_fingerprint,
            "last_reset_reason": self.last_reset_reason,
            "evidence_count": self.evidence_count,
            "feedback_count": self.feedback_count,
            "smoothed_precision": round(self.smoothed_precision(), 4),
            "smoothed_recall": round(self.smoothed_recall(), 4),
            "smoothed_f1": round(self.smoothed_f1(), 4),
            "precision_weighted_score": round(self.smoothed_fbeta(beta=0.5), 4),
            "model_confidence": round(self.feedback_confidence(), 4),
            "updated_at": self.updated_at,
            "source": "feedback_runtime",
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfidenceState":
        return cls(
            true_positives=int(payload.get("true_positives", 0) or 0),
            false_positives=int(payload.get("false_positives", 0) or 0),
            false_negatives=int(payload.get("false_negatives", 0) or 0),
            true_negatives=int(payload.get("true_negatives", 0) or 0),
            updated_at=payload.get("updated_at"),
            model_fingerprint=payload.get("model_fingerprint"),
            last_reset_reason=payload.get("last_reset_reason"),
            version=int(payload.get("version", 2) or 2),
        )


@dataclass
class ScopedModelConfidence:
    """A set of ModelConfidenceStates keyed by context (plan 1.1).

    One store per model-signal file. `scopes[GLOBAL_SCOPE]` is the site-wide
    aggregate (updated on every outcome, and the fallback for thin/absent
    context evidence); other keys are per-context (regime|tool|material — the
    same key the scorer uses for weight profiles and priors).

    Backward compatibility: a legacy flat file (just a ModelConfidenceState
    payload) loads as the global scope, and the global scope is mirrored to the
    top level on save so an old reader still sees a valid flat state.
    """

    scopes: Dict[str, ModelConfidenceState] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.scopes is None:
            self.scopes = {}
        self.scopes.setdefault(GLOBAL_SCOPE, ModelConfidenceState())

    @property
    def global_state(self) -> ModelConfidenceState:
        return self.scopes[GLOBAL_SCOPE]

    def confidence(
        self,
        context_key: Optional[str] = None,
        *,
        neutral: float = 0.5,
        damping_k: float = 5.0,
    ) -> float:
        """Context-scoped confidence with empirical-Bayes fallback to global.

        No context (or an unseen context) → the global confidence. A context
        with its own evidence blends toward its context-specific confidence in
        proportion to how much evidence it has accumulated, so a false-alarm-
        prone regime quiets selectively while a break-prone regime stays loud.
        """
        global_conf = self.global_state.feedback_confidence(neutral=neutral, damping_k=damping_k)
        if not context_key or context_key == GLOBAL_SCOPE:
            return global_conf
        ctx = self.scopes.get(context_key)
        if ctx is None or ctx.evidence_count <= 0:
            return global_conf
        ctx_conf = ctx.feedback_confidence(neutral=neutral, damping_k=damping_k)
        w = ctx.evidence_count / (ctx.evidence_count + _SCOPE_SHRINK_K)
        return w * ctx_conf + (1.0 - w) * global_conf

    def record(self, *, model_fired: bool, was_confirmed: bool,
               context_key: Optional[str] = None) -> None:
        """Record an outcome into the global scope and (if given) the context."""
        self.global_state.record(model_fired=model_fired, was_confirmed=was_confirmed)
        if context_key and context_key != GLOBAL_SCOPE:
            ctx = self.scopes.get(context_key)
            if ctx is None:
                ctx = ModelConfidenceState()
                self.scopes[context_key] = ctx
            ctx.record(model_fired=model_fired, was_confirmed=was_confirmed)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.global_state.to_dict())  # flat mirror for old readers
        payload["version"] = 3
        payload["scopes"] = {k: v.to_dict() for k, v in self.scopes.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScopedModelConfidence":
        raw_scopes = payload.get("scopes")
        if isinstance(raw_scopes, dict) and raw_scopes:
            scopes = {
                str(k): ModelConfidenceState.from_dict(v)
                for k, v in raw_scopes.items()
                if isinstance(v, dict)
            }
        else:
            # Legacy flat file: the whole payload is the global scope.
            scopes = {GLOBAL_SCOPE: ModelConfidenceState.from_dict(payload)}
        return cls(scopes=scopes)


def resolve_model_confidence_path(path: Optional[str | Path] = None) -> Path:
    return Path(path) if path is not None else Path(MODEL_CONFIDENCE_PATH)


def fingerprint_model_artifact(model_path: str | Path) -> Optional[str]:
    resolved = Path(model_path)
    if not resolved.exists() or not resolved.is_file():
        return None

    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except Exception as exc:
        logger.warning("Failed to fingerprint model artifact %s: %s", resolved, exc)
        return None
    return f"sha256:{digest.hexdigest()}"


def load_model_confidence_state(path: Optional[str | Path] = None) -> ModelConfidenceState:
    resolved = resolve_model_confidence_path(path)
    if not resolved.exists():
        return ModelConfidenceState()

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return ModelConfidenceState()
        return ModelConfidenceState.from_dict(payload)
    except Exception as exc:
        logger.warning("Failed to load model confidence from %s: %s", resolved, exc)
        return ModelConfidenceState()


def save_model_confidence_state(
    state: ModelConfidenceState,
    path: Optional[str | Path] = None,
) -> None:
    resolved = resolve_model_confidence_path(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2)
        tmp_path.replace(resolved)
    except Exception as exc:
        logger.warning("Failed to save model confidence to %s: %s", resolved, exc)


def load_scoped_model_confidence(path: Optional[str | Path] = None) -> ScopedModelConfidence:
    resolved = resolve_model_confidence_path(path)
    if not resolved.exists():
        return ScopedModelConfidence()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return ScopedModelConfidence()
        return ScopedModelConfidence.from_dict(payload)
    except Exception as exc:
        logger.warning("Failed to load scoped model confidence from %s: %s", resolved, exc)
        return ScopedModelConfidence()


def save_scoped_model_confidence(
    store: ScopedModelConfidence,
    path: Optional[str | Path] = None,
) -> None:
    resolved = resolve_model_confidence_path(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(store.to_dict(), handle, indent=2)
        tmp_path.replace(resolved)
    except Exception as exc:
        logger.warning("Failed to save scoped model confidence to %s: %s", resolved, exc)


def current_model_confidence(
    path: Optional[str | Path] = None,
    *,
    context_key: Optional[str] = None,
    neutral: float = 0.5,
    damping_k: float = 5.0,
) -> float:
    """Feedback-driven model confidence. With ``context_key`` (plan 1.1) the
    value is scoped to that context with empirical-Bayes fallback to the global
    aggregate; without it, the global (site-wide) confidence — identical to the
    pre-scoping behaviour."""
    store = load_scoped_model_confidence(path)
    return store.confidence(context_key, neutral=neutral, damping_k=damping_k)


def get_model_confidence_diagnostics(path: Optional[str | Path] = None) -> Dict[str, Any]:
    resolved = resolve_model_confidence_path(path)
    store = load_scoped_model_confidence(resolved)
    payload = store.global_state.to_dict()  # flat global view (back-compat)
    payload["path"] = str(resolved)
    payload["exists"] = resolved.exists()
    # Per-context scopes (plan 1.1) so the UI can show which contexts the model
    # is trusted in. Ordered least-trusted first — the ones feedback has quieted.
    scope_rows = []
    for key, st in store.scopes.items():
        if key == GLOBAL_SCOPE:
            continue
        scope_rows.append({
            "context": key,
            "model_confidence": round(store.confidence(key), 4),
            "confirmed": st.true_positives,
            "dismissed": st.false_positives,
            "evidence_count": st.evidence_count,
        })
    scope_rows.sort(key=lambda r: r["model_confidence"])
    payload["scopes"] = scope_rows
    payload["scope_count"] = len(scope_rows)
    return payload


def reset_model_confidence_state(
    path: Optional[str | Path] = None,
    *,
    model_fingerprint: Optional[str] = None,
    reason: str = "model_retrained",
) -> ModelConfidenceState:
    resolved = resolve_model_confidence_path(path)
    with _STATE_LOCK:
        state = ModelConfidenceState(
            updated_at=datetime.now(timezone.utc).isoformat(),
            model_fingerprint=model_fingerprint,
            last_reset_reason=reason,
        )
        save_model_confidence_state(state, resolved)
    return state


def record_model_feedback_outcome(
    *,
    model_fired: bool,
    was_confirmed: bool,
    path: Optional[str | Path] = None,
    context_key: Optional[str] = None,
    neutral: float = 0.5,
    damping_k: float = 5.0,
) -> ModelConfidenceState:
    """Record an outcome. Always updates the global aggregate; with
    ``context_key`` (plan 1.1) also updates that context's scope. Returns the
    global state (back-compat with the flat return contract)."""
    resolved = resolve_model_confidence_path(path)
    with _STATE_LOCK:
        store = load_scoped_model_confidence(resolved)
        store.record(model_fired=model_fired, was_confirmed=was_confirmed,
                     context_key=context_key)
        save_scoped_model_confidence(store, resolved)
        return ModelConfidenceState.from_dict(store.global_state.to_dict())
"""Runtime helpers for selecting and loading harmonic scorer implementations."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Maps HARMONIC_RUNTIME_DATASET values → the preset factory that points at the
# matching trained checkpoint under data/models/. Without this, the live
# orchestrator built a default config whose checkpoint path
# (data/models/harmonic_context.pt) does not exist, so no harmonic score was
# ever produced despite trained per-dataset weights being present.
_RUNTIME_PRESETS = {
    "casedata": "casedata_stoppage_preset",
    "casedata_peaks": "casedata_peak_context_preset",
    "site_a_line2": "site_a_line2_breakage_preset",
    "stoppage_1hz": "stoppage_1hz_preset",
}


def resolve_runtime_harmonic_config(env_value: str | None = None) -> Any | None:
    """Resolve a harmonic config for the live orchestrator from env.

    Reads ``HARMONIC_RUNTIME_DATASET`` (or the passed ``env_value``) and returns
    the matching dataset preset, or ``None`` when unset/unknown so the caller
    keeps its existing (default) behaviour.
    """
    raw = env_value if env_value is not None else os.environ.get("HARMONIC_RUNTIME_DATASET", "")
    name = str(raw or "").strip().lower()
    if not name:
        return None
    preset_name = _RUNTIME_PRESETS.get(name)
    if preset_name is None:
        logger.warning(
            "HARMONIC_RUNTIME_DATASET=%r not recognised (known: %s)",
            raw,
            ", ".join(sorted(_RUNTIME_PRESETS)),
        )
        return None
    try:
        from . import harmonic_config as _hc

        return getattr(_hc, preset_name)()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to build harmonic preset %s: %s", preset_name, exc)
        return None


def _scorer_kind(config: Any | None) -> str:
    kind = getattr(config, "scorer_kind", "context") if config is not None else "context"
    kind = str(kind or "context").strip().lower()
    return kind if kind in {"context", "pair"} else "context"


def harmonic_torch_available(config: Any | None = None) -> bool:
    """Return torch availability for the scorer family implied by config."""
    if _scorer_kind(config) == "pair":
        from .harmonic_pair_model import TORCH_AVAILABLE as PAIR_TORCH_AVAILABLE

        return bool(PAIR_TORCH_AVAILABLE)

    from .harmonic_model import TORCH_AVAILABLE as CONTEXT_TORCH_AVAILABLE

    return bool(CONTEXT_TORCH_AVAILABLE)


def build_harmonic_scorer(config: Any | None = None) -> Any:
    """Instantiate the scorer implementation implied by config."""
    if _scorer_kind(config) == "pair":
        from .harmonic_pair_model import HarmonicPairScorer

        return HarmonicPairScorer(config=config)

    from .harmonic_model import HarmonicContextScorer

    return HarmonicContextScorer(config=config)


def ensure_harmonic_scorer(config: Any | None = None) -> Any | None:
    """Instantiate and load the scorer, returning None if unavailable."""
    scorer = build_harmonic_scorer(config=config)
    if scorer._ensure_model():
        return scorer
    return None


def harmonic_feature_labels(config: Any) -> list[str]:
    """Return UI labels for the scorer family implied by config."""
    return build_harmonic_scorer(config=config).get_feature_labels()
"""Knowledge-pack builder + upstream sinks — Agent H (2026-04-24).

Bundles the learned state of one LFL site into a transportable JSON
artefact that can (a) bootstrap a new site with non-cold priors and
(b) be pushed to an upstream platform for fleet-level aggregation.

This module is the **data layer** only: pure dict-in/dict-out,
filesystem reads are tolerant (missing files → empty sections), and
no HTTP or MQTT dependencies at import time. Real I/O is behind the
sink protocol in :mod:`.sinks`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

KNOWLEDGE_PACK_VERSION = "1.0.0"
_REQUIRED_CONTEXT_KEYS = ("machine_type", "tool_type", "material", "regime")

# Files we pull from by default. Missing files are skipped silently.
_DEFAULT_SOURCES = {
    "pattern_priors": "pattern_priors.json",
    "discovered_patterns": "discovered_patterns.json",
    "pattern_index": "pattern_index.json",
    "pattern_aliases": "pattern_aliases.json",
    "rl_agent": "rl_agent.json",
    "rule_agreement_pairs": "rule_agreement_pairs.json",
}


@dataclass
class ContextKeys:
    """Similarity gate keys. Import must pass a minimum overlap."""

    machine_type: Optional[str] = None
    tool_type: Optional[str] = None
    material: Optional[str] = None
    regime: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)

    def missing_required_keys(self) -> List[str]:
        missing: List[str] = []
        for key in _REQUIRED_CONTEXT_KEYS:
            value = getattr(self, key, None)
            if value is None or not str(value).strip():
                missing.append(key)
        return missing

    def is_complete(self) -> bool:
        return not self.missing_required_keys()


@dataclass
class KnowledgePack:
    """Serialisable bundle of one site's learned state.

    Kept flat on purpose so operators can diff two packs with plain
    ``jq``/``diff``. Every section is optional; missing data becomes an
    empty dict rather than a missing key, so consumers never have to
    guard with ``.get(..., {})``.
    """

    version: str = KNOWLEDGE_PACK_VERSION
    site: str = "unknown"
    built_at: str = ""
    tenant_id: Optional[str] = None
    signer: Optional[str] = None
    signed_at: Optional[str] = None
    license: str = "internal-only"
    pii_scrub_level: str = "symbolic_only"
    expires_at: Optional[str] = None
    context: Dict[str, Optional[str]] = field(default_factory=dict)

    pattern_priors: Dict[str, Any] = field(default_factory=dict)
    discovered_patterns: Dict[str, Any] = field(default_factory=dict)
    pattern_index: Dict[str, Any] = field(default_factory=dict)
    pattern_aliases: Dict[str, Any] = field(default_factory=dict)
    rl_agent: Dict[str, Any] = field(default_factory=dict)
    rule_agreement_pairs: Dict[str, Any] = field(default_factory=dict)

    # Optional extras the caller may inject (computed at runtime).
    weight_profiles: Dict[str, Any] = field(default_factory=dict)
    adaptive_thresholds: Dict[str, Any] = field(default_factory=dict)
    rule_performance: Dict[str, Any] = field(default_factory=dict)
    seed_model_meta: Dict[str, Any] = field(default_factory=dict)

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def summary(self) -> Dict[str, int]:
        """Cheap inspectable counts — useful for tests and UI badges."""
        return {
            "priors": len(self.pattern_priors),
            "discovered_patterns": _count_patterns(self.discovered_patterns),
            "aliases": len(self.pattern_aliases),
            "weight_profiles": len(self.weight_profiles),
            "rule_performance": len(self.rule_performance),
            "notes": len(self.notes),
        }


@dataclass
class FleetPackApplication:
    """Result of similarity-gated fleet-pack application at a site."""

    allowed: bool
    score: float
    source_site: Optional[str] = None
    source_tenant_id: Optional[str] = None
    pattern_priors: Dict[str, float] = field(default_factory=dict)
    discovered_patterns: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _count_patterns(discovered: Mapping[str, Any]) -> int:
    # discovered_patterns.json wraps the dict under "patterns"; accept
    # either shape.
    if isinstance(discovered, Mapping) and "patterns" in discovered and isinstance(discovered["patterns"], Mapping):
        return len(discovered["patterns"])
    return len(discovered)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _read_json_tolerant(path: Path) -> Dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
        # Non-dict JSON (list/scalar) → wrap under "items" to keep the
        # pack shape predictable.
        return {"items": payload}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("knowledge_pack: failed to read %s: %s", path, exc)
        return {}


def _normalize_context_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_context_key(raw_context_key: Any) -> Dict[str, str]:
    if not isinstance(raw_context_key, str):
        return {}
    parsed: Dict[str, str] = {}
    for part in raw_context_key.split("|"):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        normalized_key = _normalize_context_value(key)
        normalized_value = _normalize_context_value(value)
        if normalized_key and normalized_value:
            parsed[normalized_key] = normalized_value
    return parsed


def _expected_export_context(context: ContextKeys) -> Dict[str, str]:
    return {
        "machine_type": _normalize_context_value(context.machine_type),
        "tool_type": _normalize_context_value(context.tool_type),
        "workpiece_material": _normalize_context_value(context.material),
        "operating_regime": _normalize_context_value(context.regime),
    }


def _context_key_matches(raw_context_key: Any, *, expected: Mapping[str, str]) -> bool:
    parsed = _parse_context_key(raw_context_key)
    return bool(parsed) and all(parsed.get(key) == value for key, value in expected.items())


def _summarize_feedback_counts(raw_counts: Any, *, patterns: Optional[Iterable[str]] = None) -> Dict[str, int]:
    if not isinstance(raw_counts, Mapping):
        return {}

    allowed_patterns = {str(pattern) for pattern in patterns} if patterns is not None else None
    summarized: Dict[str, int] = {}
    for pattern_key, counts in raw_counts.items():
        normalized_key = str(pattern_key)
        if allowed_patterns is not None and normalized_key not in allowed_patterns:
            continue

        total = 0
        if isinstance(counts, Mapping):
            total = sum(
                int(value)
                for value in counts.values()
                if isinstance(value, (int, float)) and value > 0
            )
        elif isinstance(counts, (int, float)) and counts > 0:
            total = int(counts)

        if total > 0:
            summarized[normalized_key] = total
    return summarized


def _prior_export_payload(raw_payload: Mapping[str, Any], *, context: ContextKeys) -> Dict[str, Any]:
    if not raw_payload:
        return {}

    raw_context_priors = raw_payload.get("pattern_priors_by_context")
    raw_context_counts = raw_payload.get("feedback_counts_by_context")
    if isinstance(raw_context_priors, Mapping):
        expected = _expected_export_context(context)
        for raw_context_key, priors in raw_context_priors.items():
            if not isinstance(priors, Mapping):
                continue
            if not _context_key_matches(raw_context_key, expected=expected):
                continue
            selected_priors = {str(pattern): value for pattern, value in priors.items()}
            payload = {
                "pattern_priors": selected_priors,
                "pattern_priors_by_context": {str(raw_context_key): selected_priors},
            }
            selected_counts = _summarize_feedback_counts(
                raw_context_counts.get(raw_context_key) if isinstance(raw_context_counts, Mapping) else {},
                patterns=selected_priors,
            )
            if selected_counts:
                payload["pattern_evidence_counts"] = selected_counts
            return payload
        return {
            "pattern_priors": {},
            "pattern_priors_by_context": {},
        }

    raw_priors = raw_payload.get("pattern_priors")
    if isinstance(raw_priors, Mapping):
        selected_priors = {str(pattern): value for pattern, value in raw_priors.items()}
        payload = {"pattern_priors": selected_priors}
        selected_counts = _summarize_feedback_counts(
            raw_payload.get("feedback_counts", {}),
            patterns=selected_priors,
        )
        if selected_counts:
            payload["pattern_evidence_counts"] = selected_counts
        return payload

    return {}


def _discovery_export_patterns(raw_payload: Mapping[str, Any], *, context: ContextKeys) -> Dict[str, Any]:
    if not raw_payload:
        return {}

    raw_patterns = raw_payload.get("patterns")
    if not isinstance(raw_patterns, Mapping):
        raw_patterns = raw_payload

    expected_context = _expected_export_context(context)

    filtered_patterns: Dict[str, Any] = {}
    for pattern_key, raw_pattern in raw_patterns.items():
        if not isinstance(raw_pattern, Mapping):
            continue
        if not bool(raw_pattern.get("promoted")):
            continue

        if not _context_key_matches(raw_pattern.get("context_key"), expected=expected_context):
            continue

        filtered_patterns[str(pattern_key)] = {
            str(field): value
            for field, value in raw_pattern.items()
            if field != "source_events"
        }

    return {"patterns": filtered_patterns}


def build_knowledge_pack(
    data_dir: str | Path,
    *,
    site: str,
    context: Optional[ContextKeys] = None,
    require_complete_context: bool = False,
    tenant_id: Optional[str] = None,
    signer: Optional[str] = None,
    license: str = "internal-only",
    pii_scrub_level: str = "symbolic_only",
    expires_at: Optional[str] = None,
    weight_profiles: Optional[Mapping[str, Any]] = None,
    adaptive_thresholds: Optional[Mapping[str, Any]] = None,
    rule_performance: Optional[Mapping[str, Any]] = None,
    seed_model_meta: Optional[Mapping[str, Any]] = None,
    extra_sources: Optional[Mapping[str, str]] = None,
    notes: Optional[Iterable[str]] = None,
) -> KnowledgePack:
    """Assemble a :class:`KnowledgePack` by reading JSON data files.

    Any missing file produces an empty section rather than a crash.
    Runtime-only data (weight profiles, adaptive thresholds, rule
    performance, seed-model metadata) is injected by the caller — this
    function only knows about filesystem artefacts.
    """
    root = Path(data_dir)
    ctx = context or ContextKeys()
    if require_complete_context:
        missing = ctx.missing_required_keys()
        if missing:
            raise ValueError(
                "knowledge pack export requires context keys: " + ", ".join(missing)
            )

    sources = dict(_DEFAULT_SOURCES)
    if extra_sources:
        sources.update(extra_sources)

    payload = {name: _read_json_tolerant(root / fname) for name, fname in sources.items()}
    built_at = _now_iso()
    signer_value = (signer or "").strip() or None
    pattern_priors = payload.get("pattern_priors", {})
    discovered_patterns = payload.get("discovered_patterns", {})
    if require_complete_context:
        pattern_priors = _prior_export_payload(pattern_priors, context=ctx)
        discovered_patterns = _discovery_export_patterns(discovered_patterns, context=ctx)

    pack = KnowledgePack(
        site=site,
        built_at=built_at,
        tenant_id=(tenant_id or "").strip() or None,
        signer=signer_value,
        signed_at=built_at if signer_value else None,
        license=(license or "internal-only").strip() or "internal-only",
        pii_scrub_level=(pii_scrub_level or "symbolic_only").strip() or "symbolic_only",
        expires_at=(expires_at or "").strip() or None,
        context=ctx.to_dict(),
        pattern_priors=pattern_priors,
        discovered_patterns=discovered_patterns,
        pattern_index=payload.get("pattern_index", {}),
        pattern_aliases=payload.get("pattern_aliases", {}),
        rl_agent=payload.get("rl_agent", {}),
        rule_agreement_pairs=payload.get("rule_agreement_pairs", {}),
        weight_profiles=dict(weight_profiles or {}),
        adaptive_thresholds=dict(adaptive_thresholds or {}),
        rule_performance=dict(rule_performance or {}),
        seed_model_meta=dict(seed_model_meta or {}),
        notes=list(notes or []),
    )
    return pack


def save_pack(pack: KnowledgePack, path: str | Path) -> Path:
    """Write a pack to disk atomically (``<path>.tmp`` → ``os.replace``)."""
    import os
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(pack.to_json(), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_pack(path: str | Path) -> KnowledgePack:
    """Read a pack from disk. Unknown keys are ignored."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f for f in KnowledgePack.__dataclass_fields__}
    return KnowledgePack(**{k: v for k, v in data.items() if k in known})


# ── Similarity gate for safe import ────────────────────────────────────

_DEFAULT_SIMILARITY_WEIGHTS = {
    "machine_type": 0.4,
    "tool_type": 0.3,
    "material": 0.2,
    "regime": 0.1,
}


def similarity_score(
    pack_context: Mapping[str, Optional[str]],
    target_context: Mapping[str, Optional[str]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Score how safe it is to apply *pack* onto a site with *target*.

    Returns a value in ``[0.0, 1.0]``. Each context key contributes its
    weight when both sides match (case-insensitive, trimmed). Missing
    or mismatched keys contribute zero. Keys outside the weight map
    are ignored.
    """
    w = dict(weights or _DEFAULT_SIMILARITY_WEIGHTS)
    total = sum(w.values()) or 1.0
    score = 0.0
    for key, weight in w.items():
        a = (pack_context.get(key) or "").strip().lower()
        b = (target_context.get(key) or "").strip().lower()
        if a and b and a == b:
            score += weight
    return round(score / total, 4)


def should_apply(
    pack: KnowledgePack,
    target_context: Mapping[str, Optional[str]],
    *,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Return ``(allowed, score)`` for an import decision."""
    s = similarity_score(pack.context, target_context)
    return (s >= threshold, s)


def _extract_pack_priors(payload: Mapping[str, Any]) -> Dict[str, float]:
    raw_priors: Any
    if isinstance(payload.get("pattern_priors"), Mapping):
        raw_priors = payload.get("pattern_priors")
    else:
        raw_priors = payload
    if not isinstance(raw_priors, Mapping):
        return {}
    extracted: Dict[str, float] = {}
    for pattern_key, prior in raw_priors.items():
        if isinstance(prior, (int, float)):
            extracted[str(pattern_key)] = float(prior)
    return extracted


def _extract_pack_evidence_counts(payload: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(payload, Mapping):
        return {}

    explicit_counts = payload.get("pattern_evidence_counts")
    if isinstance(explicit_counts, Mapping):
        return _summarize_feedback_counts(explicit_counts)

    return _summarize_feedback_counts(payload.get("feedback_counts", {}))


def _extract_promoted_discoveries(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw_patterns: Any
    if isinstance(payload.get("patterns"), Mapping):
        raw_patterns = payload.get("patterns")
    else:
        raw_patterns = payload
    if not isinstance(raw_patterns, Mapping):
        return {}
    promoted: Dict[str, Any] = {}
    for pattern_key, raw_pattern in raw_patterns.items():
        if not isinstance(raw_pattern, Mapping):
            continue
        if not bool(raw_pattern.get("promoted")):
            continue
        promoted[str(pattern_key)] = dict(raw_pattern)
    return {"patterns": promoted} if promoted else {}


def apply_fleet_pack(
    pack: KnowledgePack,
    target_context: Mapping[str, Optional[str]],
    *,
    threshold: float = 0.5,
) -> FleetPackApplication:
    """Similarity-gated site-side application of an upstream fleet pack.

    The current implementation is intentionally pure: it does not write to
    local files or scorer state directly. It returns the damped symbolic
    learnings that are safe to apply locally, leaving persistence/integration
    to the caller.
    """
    allowed, score = should_apply(pack, target_context, threshold=threshold)
    result = FleetPackApplication(
        allowed=allowed,
        score=score,
        source_site=(pack.site or None),
        source_tenant_id=(pack.tenant_id or None),
        notes=list(pack.notes or []),
    )
    if not allowed:
        return result

    priors = _extract_pack_priors(pack.pattern_priors if isinstance(pack.pattern_priors, Mapping) else {})
    result.pattern_priors = {
        pattern_key: round(0.5 + (prior - 0.5) * score, 4)
        for pattern_key, prior in priors.items()
    }
    result.discovered_patterns = _extract_promoted_discoveries(
        pack.discovered_patterns if isinstance(pack.discovered_patterns, Mapping) else {}
    )
    return result

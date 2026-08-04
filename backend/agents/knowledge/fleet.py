from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .pack import (
    KnowledgePack,
    _extract_pack_evidence_counts,
    _extract_pack_priors,
    _extract_promoted_discoveries,
)

logger = logging.getLogger(__name__)

FAMILY_SIGNATURE_OVERLAP = 0.80


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _normalize_context_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_context(context: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "machine_type": _normalize_context_value(context.get("machine_type")),
        "tool_type": _normalize_context_value(context.get("tool_type")),
        "material": _normalize_context_value(context.get("material")),
        "regime": _normalize_context_value(context.get("regime")),
    }


def _context_matches(pack_context: Mapping[str, Any], target_context: Mapping[str, Any]) -> bool:
    normalized_pack = _normalize_context(pack_context)
    normalized_target = _normalize_context(target_context)
    return all(normalized_pack.get(key) == normalized_target.get(key) for key in normalized_target)


def _pack_identity(pack: KnowledgePack) -> Optional[str]:
    for candidate in (pack.tenant_id, pack.site):
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _signature_tokens(payload: Mapping[str, Any]) -> set[str]:
    raw_features = payload.get("features")
    if not isinstance(raw_features, Mapping):
        return set()
    tokens: set[str] = set()
    for feature_name, direction in raw_features.items():
        name = str(feature_name or "").strip()
        trend = str(direction or "").strip().lower()
        if name and trend:
            tokens.add(f"{name}:{trend}")
    return tokens


def _signature_jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def _signature_containment(lhs: set[str], rhs: set[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / min(len(lhs), len(rhs))


def _family_similarity(lhs: set[str], rhs: set[str]) -> float:
    return max(_signature_jaccard(lhs, rhs), _signature_containment(lhs, rhs))


def _majority_signature(token_counts: Mapping[str, int], member_count: int) -> set[str]:
    threshold = max(1, (member_count // 2) + 1)
    return {token for token, count in token_counts.items() if count >= threshold}


def _tokens_to_features(tokens: set[str]) -> Dict[str, str]:
    features: Dict[str, str] = {}
    for token in sorted(tokens):
        feature_name, _, direction = token.rpartition(":")
        if feature_name and direction:
            features[feature_name] = direction
    return features


def _family_key(tokens: set[str]) -> str:
    if not tokens:
        return "family:discovered"
    return "family:" + "+".join(sorted(tokens))


def _context_key(context: Mapping[str, Any]) -> str:
    normalized = _normalize_context(context)
    return "|".join(
        f"{key}={normalized.get(key, '')}"
        for key in ("machine_type", "tool_type", "material", "regime")
    )


@dataclass
class FleetDiscoveryFamily:
    family_key: str
    canonical_name: str
    source: str = "discovered"
    status: str = "candidate"
    site_count: int = 0
    pattern_count: int = 0
    member_keys: List[str] = field(default_factory=list)
    representative_features: Dict[str, str] = field(default_factory=dict)
    prior: Optional[float] = None


@dataclass
class FleetFamilyReview:
    family_key: str
    context: Dict[str, Optional[str]]
    canonical_name: Optional[str] = None
    status: str = "candidate"
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    reviewed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["context"] = {
            key: (value or None)
            for key, value in _normalize_context(self.context).items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FleetFamilyReview":
        return cls(
            family_key=str(payload.get("family_key") or "").strip(),
            context={
                key: (value or None)
                for key, value in _normalize_context(payload.get("context") or {}).items()
            },
            canonical_name=(str(payload.get("canonical_name") or "").strip() or None),
            status=str(payload.get("status") or "candidate").strip() or "candidate",
            reviewer=(str(payload.get("reviewer") or "").strip() or None),
            reason=(str(payload.get("reason") or "").strip() or None),
            reviewed_at=str(payload.get("reviewed_at") or _now_iso()),
        )


@dataclass
class FleetKnowledgePack:
    built_at: str
    context: Dict[str, Optional[str]]
    pack_count: int = 0
    site_count: int = 0
    k_anonymity_threshold: int = 3
    k_anonymity_met: bool = False
    source_sites: List[str] = field(default_factory=list)
    pattern_priors: Dict[str, Any] = field(default_factory=dict)
    discovered_patterns: Dict[str, Any] = field(default_factory=dict)
    discovery_families: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_family_reviews(
    aggregate: FleetKnowledgePack,
    reviews: Iterable[FleetFamilyReview],
) -> FleetKnowledgePack:
    reviewed = FleetKnowledgePack(**aggregate.to_dict())
    review_index: Dict[tuple[str, str], FleetFamilyReview] = {}
    for review in reviews:
        normalized_review = FleetFamilyReview.from_dict(review.to_dict())
        if not normalized_review.family_key:
            continue
        review_index[(_context_key(normalized_review.context), normalized_review.family_key)] = normalized_review

    aggregate_context_key = _context_key(reviewed.context)
    applied_count = 0
    updated_families: List[Dict[str, Any]] = []
    for family in reviewed.discovery_families:
        family_copy = dict(family)
        review = review_index.get((aggregate_context_key, str(family_copy.get("family_key") or "")))
        if review is not None:
            if review.canonical_name:
                family_copy["canonical_name"] = review.canonical_name
            family_copy["status"] = review.status or family_copy.get("status", "candidate")
            family_copy["review"] = {
                "reviewer": review.reviewer,
                "reason": review.reason,
                "reviewed_at": review.reviewed_at,
            }
            applied_count += 1
        updated_families.append(family_copy)

    reviewed.discovery_families = updated_families
    if applied_count:
        reviewed.notes.append(f"applied {applied_count} family review(s)")
    return reviewed


def aggregate_fleet_packs(
    packs: Iterable[KnowledgePack],
    target_context: Mapping[str, Any],
    *,
    min_sites: int = 3,
) -> FleetKnowledgePack:
    threshold = max(1, int(min_sites))
    normalized_target = _normalize_context(target_context)
    matching_packs: List[KnowledgePack] = []
    identities: Dict[str, str] = {}
    priors_by_pattern: Dict[str, Dict[str, Dict[str, float]]] = {}
    discoveries_by_key: Dict[str, Dict[str, Dict[str, Any]]] = {}
    discovery_entries: List[Dict[str, Any]] = []

    for pack in packs:
        if not _context_matches(pack.context, normalized_target):
            continue
        identity = _pack_identity(pack)
        if not identity:
            continue
        identities[identity] = str(pack.site or identity)
        matching_packs.append(pack)

        priors = _extract_pack_priors(pack.pattern_priors if isinstance(pack.pattern_priors, Mapping) else {})
        evidence_counts = _extract_pack_evidence_counts(
            pack.pattern_priors if isinstance(pack.pattern_priors, Mapping) else {}
        )
        for pattern_key, prior in priors.items():
            priors_by_pattern.setdefault(str(pattern_key), {})[identity] = {
                "prior": float(prior),
                "weight": float(max(1, evidence_counts.get(str(pattern_key), 1))),
            }

        discoveries = _extract_promoted_discoveries(
            pack.discovered_patterns if isinstance(pack.discovered_patterns, Mapping) else {}
        )
        raw_patterns = discoveries.get("patterns") if isinstance(discoveries, Mapping) else {}
        if not isinstance(raw_patterns, Mapping):
            continue
        for pattern_key, payload in raw_patterns.items():
            if not isinstance(payload, Mapping):
                continue
            payload_copy = dict(payload)
            discoveries_by_key.setdefault(str(pattern_key), {})[identity] = payload_copy
            signature_tokens = _signature_tokens(payload_copy)
            if signature_tokens:
                discovery_entries.append(
                    {
                        "identity": identity,
                        "pattern_key": str(pattern_key),
                        "payload": payload_copy,
                        "signature_tokens": signature_tokens,
                    }
                )

    result = FleetKnowledgePack(
        built_at=_now_iso(),
        context={key: (value or None) for key, value in normalized_target.items()},
        pack_count=len(matching_packs),
        site_count=len(identities),
        k_anonymity_threshold=threshold,
        k_anonymity_met=len(identities) >= threshold,
        source_sites=sorted(identities.values()),
    )
    if not result.k_anonymity_met:
        result.notes.append(
            f"k-anonymity threshold not met: {result.site_count} site(s), requires >= {threshold}"
        )
        return result

    for pattern_key, values_by_identity in priors_by_pattern.items():
        site_count = len(values_by_identity)
        if site_count < threshold:
            continue
        evidence_count = int(sum(value["weight"] for value in values_by_identity.values()))
        weighted_prior = sum(value["prior"] * value["weight"] for value in values_by_identity.values())
        weighted_prior /= max(1.0, float(evidence_count))
        result.pattern_priors[pattern_key] = {
            "prior": round(weighted_prior, 4),
            "site_count": site_count,
            "evidence_count": evidence_count,
        }

    promoted_patterns: Dict[str, Any] = {}
    for pattern_key, payloads_by_identity in discoveries_by_key.items():
        site_count = len(payloads_by_identity)
        if site_count < threshold:
            continue
        first_payload = dict(next(iter(payloads_by_identity.values())))
        prior_values = [
            float(payload.get("prior"))
            for payload in payloads_by_identity.values()
            if isinstance(payload.get("prior"), (int, float))
        ]
        if prior_values:
            first_payload["prior"] = round(sum(prior_values) / len(prior_values), 4)
        first_payload["site_count"] = site_count
        promoted_patterns[pattern_key] = first_payload
    if promoted_patterns:
        result.discovered_patterns = {"patterns": promoted_patterns}

    family_accumulators: List[Dict[str, Any]] = []
    for entry in sorted(discovery_entries, key=lambda item: item["pattern_key"]):
        best_index: Optional[int] = None
        best_score = 0.0
        for index, family in enumerate(family_accumulators):
            score = _family_similarity(entry["signature_tokens"], family["family_signature"])
            if score >= FAMILY_SIGNATURE_OVERLAP and score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            family_accumulators.append(
                {
                    "seed_signature": set(entry["signature_tokens"]),
                    "family_signature": set(entry["signature_tokens"]),
                    "token_counts": {token: 1 for token in entry["signature_tokens"]},
                    "member_count": 1,
                    "identities": {entry["identity"]},
                    "member_keys": {entry["pattern_key"]},
                    "priors": [
                        float(entry["payload"].get("prior"))
                    ] if isinstance(entry["payload"].get("prior"), (int, float)) else [],
                }
            )
            continue
        family = family_accumulators[best_index]
        family["member_count"] += 1
        family["identities"].add(entry["identity"])
        family["member_keys"].add(entry["pattern_key"])
        for token in entry["signature_tokens"]:
            family["token_counts"][token] = family["token_counts"].get(token, 0) + 1
        majority_signature = _majority_signature(family["token_counts"], family["member_count"])
        family["family_signature"] = majority_signature or set(family["seed_signature"])
        if isinstance(entry["payload"].get("prior"), (int, float)):
            family["priors"].append(float(entry["payload"]["prior"]))

    for family in family_accumulators:
        site_count = len(family["identities"])
        if site_count < threshold:
            continue
        representative_tokens = family["family_signature"] or set(family["seed_signature"])
        prior_values = family["priors"]
        result.discovery_families.append(
            asdict(
                FleetDiscoveryFamily(
                    family_key=_family_key(representative_tokens),
                    canonical_name=_family_key(representative_tokens),
                    site_count=site_count,
                    pattern_count=len(family["member_keys"]),
                    member_keys=sorted(family["member_keys"]),
                    representative_features=_tokens_to_features(representative_tokens),
                    prior=round(sum(prior_values) / len(prior_values), 4) if prior_values else None,
                )
            )
        )
    result.discovery_families.sort(key=lambda family: (-family["site_count"], family["family_key"]))

    if not result.pattern_priors and not result.discovered_patterns and not result.discovery_families:
        result.notes.append(
            "no fleet artifacts met the per-pattern k-anonymity threshold"
        )
    return result


def fleet_store_path(path: str | Path | None = None) -> Path:
    return Path(path or "data/fleet_packs.jsonl")


def fleet_review_store_path(path: str | Path | None = None) -> Path:
    return Path(path or "data/fleet_family_reviews.jsonl")


def append_pack_to_store(pack: KnowledgePack, path: str | Path) -> int:
    target = fleet_store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pack.to_dict(), sort_keys=True, default=str) + "\n")
    return len(load_packs_from_store(target))


def append_family_review_to_store(review: FleetFamilyReview, path: str | Path) -> int:
    target = fleet_review_store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(review.to_dict(), sort_keys=True, default=str) + "\n")
    return len(load_family_reviews_from_store(target))


def load_packs_from_store(path: str | Path) -> List[KnowledgePack]:
    target = fleet_store_path(path)
    if not target.is_file():
        return []
    packs: List[KnowledgePack] = []
    known = {field_name for field_name in KnowledgePack.__dataclass_fields__}
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("fleet store: skipping corrupt line in %s", target)
                continue
            if not isinstance(payload, dict):
                continue
            packs.append(KnowledgePack(**{k: v for k, v in payload.items() if k in known}))
    return packs


def load_family_reviews_from_store(path: str | Path) -> List[FleetFamilyReview]:
    target = fleet_review_store_path(path)
    if not target.is_file():
        return []
    reviews: List[FleetFamilyReview] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("fleet review store: skipping corrupt line in %s", target)
                continue
            if not isinstance(payload, dict):
                continue
            reviews.append(FleetFamilyReview.from_dict(payload))
    return reviews
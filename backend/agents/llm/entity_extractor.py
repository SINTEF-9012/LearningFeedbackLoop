"""Closed-vocabulary entity extraction for the document semantic layer.

This module is intentionally self-contained so it can be validated before the
ingest pipeline is wired to call it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import httpx

from backend.agents.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL, GROQ_TIMEOUT

logger = logging.getLogger(__name__)

ALLOWED_ENTITY_TYPES: tuple[str, ...] = (
    "Machine",
    "Component",
    "Subsystem",
    "Parameter",
    "Procedure",
    "Alarm",
    "Symptom",
    "Material",
    "Operation",
    "Tool",
    "Document",
)

ALLOWED_REL_TYPES: tuple[str, ...] = (
    "PART_OF",
    "HAS_COMPONENT",
    "CONTROLS",
    "MEASURES",
    "CAUSES",
    "SYMPTOM_OF",
    "TROUBLESHOOTS",
    "REQUIRES",
    "APPLIES_TO",
    "DESCRIBES",
)


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    aliases: list[str]


@dataclass(frozen=True)
class ExtractedRelation:
    src_name: str
    src_type: str
    dst_name: str
    dst_type: str
    rel_type: str
    confidence: float = 0.5


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    warnings: list[str] = field(default_factory=list)
    raw_payload: Optional[dict[str, Any]] = None


class EntityExtractor:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> None:
        self.enabled = (
            bool(enabled)
            if enabled is not None
            else os.environ.get("LFL_ENTITY_EXTRACTOR_ENABLED", "false").strip().lower() == "true"
        )
        self.provider = str(provider or os.environ.get("LLM_PROVIDER", "groq")).strip().lower()
        self.groq_api_key = groq_api_key if groq_api_key is not None else GROQ_API_KEY
        self.groq_api_url = str(groq_api_url or GROQ_API_URL).rstrip("/")
        self.model = str(model or os.environ.get("LFL_ENTITY_EXTRACTOR_MODEL", GROQ_MODEL)).strip()
        self.timeout = float(timeout if timeout is not None else os.environ.get("LFL_ENTITY_EXTRACTOR_TIMEOUT", GROQ_TIMEOUT))
        self.max_completion_tokens = int(
            max_completion_tokens
            if max_completion_tokens is not None
            else os.environ.get("LFL_ENTITY_EXTRACTOR_MAX_TOKENS", 900)
        )
        # Reasoning models (e.g. gpt-oss) spend a large, highly variable share of
        # the completion budget on reasoning tokens; on dense chunks that can
        # starve the JSON output and trigger Groq `json_validate_failed` (400).
        # Capping reasoning effort keeps extraction reliable (and faster). Only
        # sent when set — non-reasoning models (llama) reject the param.
        self.reasoning_effort = str(
            os.environ.get("LFL_ENTITY_EXTRACTOR_REASONING_EFFORT", "")
        ).strip().lower()

    def is_enabled(self) -> bool:
        return self.enabled and self.provider == "groq" and bool(self.groq_api_key)

    def extract_from_chunk(
        self,
        chunk_text: str,
        *,
        usecase: str,
        machine_hint: Optional[str] = None,
        source_hint: Optional[str] = None,
    ) -> ExtractionResult:
        text = _clean_surface(chunk_text)
        if len(text) < 40:
            return ExtractionResult(entities=[], relations=[])
        if not self.is_enabled():
            return ExtractionResult(entities=[], relations=[])
        try:
            payload = self._request_payload(
                chunk_text=text,
                usecase=usecase,
                machine_hint=machine_hint,
                source_hint=source_hint,
            )
            return self._normalize_payload(payload)
        except Exception as exc:
            logger.warning("Entity extraction failed for usecase %s: %s", usecase, exc)
            return ExtractionResult(entities=[], relations=[], warnings=[f"extractor_error:{exc}"])

    def _request_payload(
        self,
        *,
        chunk_text: str,
        usecase: str,
        machine_hint: Optional[str],
        source_hint: Optional[str],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(
            chunk_text=chunk_text,
            usecase=usecase,
            machine_hint=machine_hint,
            source_hint=source_hint,
        )
        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            request_body["reasoning_effort"] = self.reasoning_effort
        response = httpx.post(
            f"{self.groq_api_url}/chat/completions",
            json=request_body,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=self.timeout,
                write=connect_timeout,
                pool=connect_timeout,
            ),
        )
        response.raise_for_status()
        message_text = _extract_message_text(response.json())
        if not message_text:
            raise ValueError("empty_response")
        payload = _parse_json_object(message_text)
        if not isinstance(payload, dict):
            raise ValueError("response_not_object")
        return payload

    def _build_prompt(
        self,
        *,
        chunk_text: str,
        usecase: str,
        machine_hint: Optional[str],
        source_hint: Optional[str],
    ) -> str:
        machine_line = f"Machine hint: {machine_hint}\n" if machine_hint else ""
        source_line = f"Source hint: {source_hint}\n" if source_hint else ""
        return (
            "You extract a grounded semantic layer from manufacturing documentation.\n"
            "Return only a JSON object with keys `entities` and `relations`.\n"
            "Every entity and relation must be directly supported by the chunk text.\n"
            "Do not infer facts that are not stated.\n"
            f"Usecase: {usecase}\n"
            f"{machine_line}"
            f"{source_line}"
            f"Allowed entity types: {', '.join(ALLOWED_ENTITY_TYPES)}\n"
            f"Allowed relation types: {', '.join(ALLOWED_REL_TYPES)}\n"
            "Entity schema: {\"name\": str, \"type\": str, \"aliases\": [str]}\n"
            "Relation schema: {\"src_name\": str, \"src_type\": str, \"dst_name\": str, \"dst_type\": str, \"rel_type\": str, \"confidence\": float}\n"
            "If nothing is grounded, return {\"entities\": [], \"relations\": []}.\n\n"
            f"Chunk:\n{chunk_text}"
        )

    @staticmethod
    def _normalize_payload(payload: Mapping[str, Any]) -> ExtractionResult:
        warnings: list[str] = []
        entities_by_key: dict[tuple[str, str], ExtractedEntity] = {}

        for raw_entity in payload.get("entities") or []:
            if not isinstance(raw_entity, Mapping):
                warnings.append("invalid_entity_record")
                continue
            name = _clean_surface(raw_entity.get("name"))
            entity_type = str(raw_entity.get("type") or "").strip()
            if entity_type not in ALLOWED_ENTITY_TYPES:
                warnings.append(f"invalid_entity_type:{entity_type or 'missing'}")
                continue
            name_norm = normalize_entity_name(name)
            if not name_norm:
                warnings.append("invalid_entity_name")
                continue
            aliases = _collect_aliases(raw_entity.get("aliases"), canonical_name=name)
            key = (entity_type, name_norm)
            existing = entities_by_key.get(key)
            if existing is None:
                entities_by_key[key] = ExtractedEntity(name=name, type=entity_type, aliases=aliases)
                continue
            merged_aliases = _merge_distinct(existing.aliases, aliases)
            entities_by_key[key] = ExtractedEntity(name=existing.name, type=existing.type, aliases=merged_aliases)

        relations: list[ExtractedRelation] = []
        relation_keys: set[tuple[tuple[str, str], tuple[str, str], str]] = set()
        for raw_relation in payload.get("relations") or []:
            if not isinstance(raw_relation, Mapping):
                warnings.append("invalid_relation_record")
                continue
            src_name = _clean_surface(raw_relation.get("src_name") or raw_relation.get("source") or raw_relation.get("src"))
            dst_name = _clean_surface(raw_relation.get("dst_name") or raw_relation.get("target") or raw_relation.get("dst"))
            src_type = str(raw_relation.get("src_type") or raw_relation.get("source_type") or "").strip()
            dst_type = str(raw_relation.get("dst_type") or raw_relation.get("target_type") or "").strip()
            rel_type = str(raw_relation.get("rel_type") or raw_relation.get("predicate") or raw_relation.get("type") or "").strip()
            if rel_type not in ALLOWED_REL_TYPES:
                warnings.append(f"invalid_relation_type:{rel_type or 'missing'}")
                continue
            src_key = (src_type, normalize_entity_name(src_name))
            dst_key = (dst_type, normalize_entity_name(dst_name))
            if src_key not in entities_by_key or dst_key not in entities_by_key:
                warnings.append(f"ungrounded_relation:{rel_type}")
                continue
            dedupe_key = (src_key, dst_key, rel_type)
            if dedupe_key in relation_keys:
                continue
            relation_keys.add(dedupe_key)
            relations.append(
                ExtractedRelation(
                    src_name=entities_by_key[src_key].name,
                    src_type=src_type,
                    dst_name=entities_by_key[dst_key].name,
                    dst_type=dst_type,
                    rel_type=rel_type,
                    confidence=_coerce_confidence(raw_relation.get("confidence")),
                )
            )

        entities = sorted(entities_by_key.values(), key=lambda item: (item.type, item.name.lower()))
        return ExtractionResult(
            entities=entities,
            relations=relations,
            warnings=warnings,
            raw_payload=dict(payload),
        )


def normalize_entity_name(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_surface(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _collect_aliases(raw_aliases: Any, *, canonical_name: str) -> list[str]:
    values = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases] if raw_aliases else []
    aliases: list[str] = []
    canonical_norm = normalize_entity_name(canonical_name)
    for alias in values:
        alias_text = _clean_surface(alias)
        if not alias_text:
            continue
        if normalize_entity_name(alias_text) == canonical_norm:
            continue
        aliases.append(alias_text)
    return _merge_distinct([], aliases)


def _merge_distinct(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *incoming]:
        key = normalize_entity_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(parsed, 1.0))


def _extract_message_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                return str(message.get("content") or "").strip()
    message = payload.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "").strip()
    return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("response_not_object")
    return parsed
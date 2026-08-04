"""Canonicalize extracted entities within a usecase before graph writes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Optional, Sequence

from backend.agents.llm.entity_extractor import ALLOWED_ENTITY_TYPES, normalize_entity_name


EntityCanonicalIdResolver = Callable[..., Optional[str]]


@dataclass(frozen=True)
class CanonicalEntity:
    id: str
    usecase: str
    type: str
    name: str
    name_norm: str
    canonical_id: Optional[str]
    aliases: list[str]


class EntityCanonicalizer:
    def __init__(
        self,
        *,
        usecase: str,
        similarity_threshold: float = 0.92,
        canonical_id_resolver: Optional[EntityCanonicalIdResolver] = None,
    ) -> None:
        self.usecase = str(usecase or "").strip().upper()
        self.similarity_threshold = float(similarity_threshold)
        self._canonical_id_resolver = canonical_id_resolver
        self._entities_by_id: dict[str, CanonicalEntity] = {}
        self._index: dict[tuple[str, str], str] = {}
        self._canonical_index: dict[tuple[str, str], str] = {}

    def seed(self, entities: Sequence[CanonicalEntity]) -> None:
        for entity in entities:
            if entity.usecase.upper() != self.usecase:
                raise ValueError("seed entity usecase mismatch")
            self._store(entity)

    def register(
        self,
        *,
        name: str,
        entity_type: str,
        aliases: Optional[Sequence[str]] = None,
        machine_hint: Optional[str] = None,
        machine_uri: Optional[str] = None,
    ) -> CanonicalEntity:
        normalized_type = str(entity_type or "").strip()
        if normalized_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {normalized_type}")
        surface_name = _clean_surface(name)
        name_norm = normalize_entity_name(surface_name)
        if not name_norm:
            raise ValueError("Entity name cannot be blank")

        surface_aliases = _merge_surface_aliases([], aliases or [], canonical_name=surface_name)
        canonical_id = self._resolve_canonical_id(
            name=surface_name,
            entity_type=normalized_type,
            aliases=surface_aliases,
            machine_hint=machine_hint,
            machine_uri=machine_uri,
        )
        existing = self._find_existing(
            normalized_type,
            name_norm,
            [surface_name, *surface_aliases],
            canonical_id=canonical_id,
        )
        if existing is None:
            entity = CanonicalEntity(
                id=_stable_entity_id(self.usecase, normalized_type, name_norm, canonical_id=canonical_id),
                usecase=self.usecase,
                type=normalized_type,
                name=surface_name,
                name_norm=name_norm,
                canonical_id=canonical_id,
                aliases=surface_aliases,
            )
            self._store(entity)
            return entity

        merged_aliases = _merge_surface_aliases(existing.aliases, [surface_name, *surface_aliases], canonical_name=existing.name)
        merged = CanonicalEntity(
            id=existing.id,
            usecase=existing.usecase,
            type=existing.type,
            name=existing.name,
            name_norm=existing.name_norm,
            canonical_id=existing.canonical_id or canonical_id,
            aliases=merged_aliases,
        )
        self._store(merged)
        return merged

    def list_entities(self) -> list[CanonicalEntity]:
        return sorted(self._entities_by_id.values(), key=lambda item: (item.type, item.name_norm, item.id))

    def _find_existing(
        self,
        entity_type: str,
        name_norm: str,
        surfaces: Sequence[str],
        *,
        canonical_id: Optional[str] = None,
    ) -> Optional[CanonicalEntity]:
        if canonical_id:
            existing_id = self._canonical_index.get((entity_type, canonical_id))
            if existing_id is not None:
                return self._entities_by_id[existing_id]

        search_keys = [name_norm, *(normalize_entity_name(surface) for surface in surfaces)]
        for key in search_keys:
            existing_id = self._index.get((entity_type, key))
            if existing_id is not None:
                return self._entities_by_id[existing_id]

        best_match: Optional[CanonicalEntity] = None
        best_score = 0.0
        for entity in self._entities_by_id.values():
            if entity.type != entity_type:
                continue
            candidates = [entity.name_norm, *(normalize_entity_name(alias) for alias in entity.aliases)]
            for candidate in candidates:
                score = SequenceMatcher(None, name_norm, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_match = entity
        if best_match is not None and best_score >= self.similarity_threshold:
            return best_match
        return None

    def _store(self, entity: CanonicalEntity) -> None:
        self._entities_by_id[entity.id] = entity
        keys = [(entity.type, entity.name_norm)]
        keys.extend((entity.type, normalize_entity_name(alias)) for alias in entity.aliases)
        for key in keys:
            if key[1]:
                self._index[key] = entity.id
        if entity.canonical_id:
            self._canonical_index[(entity.type, entity.canonical_id)] = entity.id

    def _resolve_canonical_id(
        self,
        *,
        name: str,
        entity_type: str,
        aliases: Sequence[str],
        machine_hint: Optional[str],
        machine_uri: Optional[str],
    ) -> Optional[str]:
        if self._canonical_id_resolver is None:
            return None
        value = self._canonical_id_resolver(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            usecase=self.usecase,
            machine_hint=machine_hint,
            machine_uri=machine_uri,
        )
        if value is None:
            return None
        canonical_id = str(value).strip()
        return canonical_id or None


def _stable_entity_id(usecase: str, entity_type: str, name_norm: str, *, canonical_id: Optional[str] = None) -> str:
    stable_key = canonical_id if canonical_id not in (None, "") else name_norm
    digest = hashlib.sha1(f"{usecase}:{entity_type}:{stable_key}".encode("utf-8")).hexdigest()[:16]
    return f"entity:{digest}"


def _clean_surface(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _merge_surface_aliases(
    existing: Sequence[str],
    incoming: Sequence[str],
    *,
    canonical_name: str,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = {normalize_entity_name(canonical_name)}
    for value in [*existing, *incoming]:
        surface = _clean_surface(value)
        key = normalize_entity_name(surface)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(surface)
    return merged
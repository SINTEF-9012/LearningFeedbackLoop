"""
Pluggable documentation-retrieval backend for RetrieverAgent.

This module provides a minimal interface and a default no-op implementation,
so PDF/Neo4j document retrieval can be added without changing retriever core
control flow.

The active Neo4j backend now combines chunk-level vector retrieval with a
grounded semantic layer over `:Entity` nodes when those entities are present
for the queried usecase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from backend.agents.config import (
    NEO4J_CONNECT_TIMEOUT_S,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from backend.agents.usecase import normalize_usecase, resolve_usecase, usecase_aliases

_DOC_SCHEMA_VERSION = 3
_DOC_VECTOR_INDEX = "doc_vector_index"
_ENTITY_VECTOR_INDEX = "entity_vector_index"
_RRF_K = 60
_GRAPH_SUPPORT_BONUS_STEP = 0.07
_GRAPH_SUPPORT_BONUS_CAP = 3
_RANKING_AGREEMENT_BONUS = 0.03
_SPREADSHEET_SCORE_MULTIPLIER = 0.6
_DOC_FEEDBACK_BONUS_STEP = 0.04
_DOC_FEEDBACK_BONUS_CAP = 3.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _adjust_match_score(document_type: Optional[str], score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    adjusted = float(score)
    if str(document_type or "").strip().lower() == "spreadsheet":
        adjusted *= _SPREADSHEET_SCORE_MULTIPLIER
    return adjusted


def _hybrid_match_score(
    *,
    document_type: Optional[str],
    vector_score: Optional[float],
    graph_support: Optional[int],
    ranking_count: Optional[int],
) -> Optional[float]:
    base_vector = max(0.0, float(vector_score or 0.0))
    bounded_graph_support = max(0, min(int(graph_support or 0), _GRAPH_SUPPORT_BONUS_CAP))
    graph_bonus = bounded_graph_support * _GRAPH_SUPPORT_BONUS_STEP
    agreement_bonus = max(0, int(ranking_count or 0) - 1) * _RANKING_AGREEMENT_BONUS
    if base_vector > 0.0:
        raw_score = min(1.0, base_vector + graph_bonus + agreement_bonus)
    else:
        raw_score = min(1.0, graph_bonus + agreement_bonus)
    return _adjust_match_score(document_type, raw_score)


def _feedback_rank_score(score: Optional[float], feedback_score: Optional[float]) -> float:
    base_score = max(0.0, float(score or 0.0))
    bounded_feedback = max(
        -_DOC_FEEDBACK_BONUS_CAP,
        min(float(feedback_score or 0.0), _DOC_FEEDBACK_BONUS_CAP),
    )
    return min(1.0, max(0.0, base_score + (bounded_feedback * _DOC_FEEDBACK_BONUS_STEP)))


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coverage_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _build_twin_health_summary(
    *,
    document_count: int,
    entity_count: int,
    semantic_coverage_ratio: float,
    semantic_gap_usecases: List[str],
    usecase_coverage: List[Dict[str, Any]],
    message: str,
) -> Dict[str, Any]:
    total_usecases = len(usecase_coverage)
    ready_usecases = sum(1 for item in usecase_coverage if bool(item.get("semantic_ready")))
    canonical_entity_count = sum(_coerce_int(item.get("canonical_entity_count")) for item in usecase_coverage)

    if document_count <= 0:
        status = "warning"
        headline = "—"
        summary = message
    elif semantic_gap_usecases:
        status = "warning"
        headline = f"{ready_usecases}/{total_usecases} ready" if total_usecases > 0 else f"{semantic_coverage_ratio * 100:.1f}% grounded"
        summary = (
            f"{ready_usecases}/{total_usecases} usecases semantically ready"
            if total_usecases > 0
            else message
        )
        if canonical_entity_count > 0:
            summary += f" · {canonical_entity_count} canonical entities"
    else:
        status = "ok"
        headline = f"{ready_usecases}/{total_usecases} ready" if total_usecases > 0 else f"{semantic_coverage_ratio * 100:.1f}% grounded"
        summary = (
            f"{ready_usecases}/{total_usecases} usecases semantically ready"
            if total_usecases > 0
            else message
        )
        if canonical_entity_count > 0 or entity_count > 0:
            summary += f" · {canonical_entity_count} canonical ids across {entity_count} entities"

    return {
        "status": status,
        "headline": headline,
        "summary": summary,
        "semantic_ready_usecases": ready_usecases,
        "total_usecases": total_usecases,
        "canonical_entity_count": canonical_entity_count,
        "semantic_coverage_ratio": semantic_coverage_ratio,
        "semantic_gap_usecases": list(semantic_gap_usecases),
    }


class DocsBackend(Protocol):
    """Interface for documentation retrieval providers."""

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        usecase: Optional[str] = None,
        source_filter: Optional[str] = None,
        machine: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    async def structured(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def status(self) -> Dict[str, Any]:
        ...


@dataclass
class NullDocsBackend:
    """Safe default backend used until real PDF retrieval is wired."""

    name: str = "docs_vector"

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        usecase: Optional[str] = None,
        source_filter: Optional[str] = None,
        machine: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "query": query,
            "top_k": top_k,
            "usecase": usecase,
            "source_filter": source_filter,
            "machine": machine,
            "document_type": document_type,
            "matches": [],
            "message": "No documentation backend configured yet.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def structured(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "backend": "docs_structured",
            "mode": args.get("mode", "template"),
            "records": [],
            "message": "No structured documentation backend configured yet.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def status(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "ready": False,
            "message": "Null backend active (placeholder).",
        }


class Neo4jDocsBackend:
    """Documentation backend that reads `:Document` nodes from Neo4j.

    This treats documentation as a separate graph domain inside the shared
    database. Queries stay partitioned by usecase using `d.usecase` when
    present, with fallback filters on `dataset_id`, `source`, `machine`, and
    `file_name` so partially migrated document graphs still stay scoped.
    """

    def __init__(
        self,
        *,
        uri: str = NEO4J_URI,
        username: str = NEO4J_USERNAME,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
        connect_timeout_s: float = NEO4J_CONNECT_TIMEOUT_S,
    ) -> None:
        self.name = "docs_neo4j"
        self._uri = uri
        self._username = username
        self._password = password
        self._connect_timeout_s = connect_timeout_s
        self._database = database
        self._driver = None
        self._driver_error: Optional[str] = None
        self._graph_database = None

        self._connect()

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        usecase: Optional[str] = None,
        source_filter: Optional[str] = None,
        machine: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_driver()
        resolved_usecase = resolve_usecase(
            usecase=usecase,
            machine=machine,
            source=source_filter,
        )
        params = self._build_filter_params(
            usecase=resolved_usecase,
            source_filter=source_filter,
            machine=machine,
            document_type=document_type,
        )
        params["top_k"] = max(1, int(top_k or 5))
        params["search_text"] = str(query or "")

        if self._driver is None:
            return {
                "backend": self.name,
                "query": query,
                "top_k": params["top_k"],
                "usecase": resolved_usecase,
                "matches": [],
                "ready": False,
                "message": self._driver_error or "Neo4j docs backend unavailable.",
                "timestamp": _now_iso(),
            }

        matches = self._hybrid_search(params) if self._entities_available(params) else self._vector_search(params)
        if not matches:
            matches = self._text_search(params)

        return {
            "backend": self.name,
            "query": query,
            "top_k": params["top_k"],
            "usecase": resolved_usecase,
            "source_filter": source_filter,
            "machine": machine,
            "document_type": document_type,
            "matches": matches,
            "message": "ok" if matches else "No matching documentation found.",
            "timestamp": _now_iso(),
        }

    async def structured(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_driver()
        params = self._build_filter_params(
            usecase=resolve_usecase(
                usecase=args.get("usecase"),
                machine=args.get("machine"),
                source=args.get("source_filter"),
            ),
            source_filter=args.get("source_filter"),
            machine=args.get("machine"),
            document_type=args.get("document_type"),
        )
        params["limit"] = max(1, min(int(args.get("limit", 25) or 25), 100))

        if self._driver is None:
            return {
                "backend": self.name,
                "mode": args.get("mode", "structured"),
                "records": [],
                "ready": False,
                "message": self._driver_error or "Neo4j docs backend unavailable.",
                "timestamp": _now_iso(),
            }

        where_clause = self._where_clause("d")
        rows = self._run(
            f"""
            MATCH (d:Document)
            WHERE {where_clause}
            WITH d, coalesce(d.file_name, d.relative_path, d.source, d.id) AS document_key
            ORDER BY document_key, coalesce(d.page, d.page_start, 0)
            WITH document_key, collect(d)[0] AS doc
            ORDER BY coalesce(doc.file_name, doc.relative_path, doc.source, ''), coalesce(doc.page, doc.page_start, 0)
            LIMIT $limit
            RETURN doc AS doc
            """,
            **params,
        )
        records = [self._format_match(row.get("doc"), None) for row in rows]
        return {
            "backend": self.name,
            "mode": args.get("mode", "structured"),
            "records": records,
            "ready": True,
            "timestamp": _now_iso(),
        }

    async def status(self) -> Dict[str, Any]:
        self._ensure_driver()
        if self._driver is None:
            return {
                "backend": self.name,
                "ready": False,
                "message": self._driver_error or "Neo4j docs backend unavailable.",
            }

        try:
            meta_rows = self._run(
                """
                MATCH (d:Document)
                RETURN collect(DISTINCT coalesce(d.usecase, d.dataset_id, d.source, 'unknown'))[..20] AS sources,
                       collect(DISTINCT coalesce(d.machine, d.machine_uri, 'unknown'))[..20] AS machines
                """
            )
            per_usecase_rows = self._run(
                """
                MATCH (d:Document)
                WITH coalesce(d.usecase, d.dataset_id, d.source, 'unknown') AS usecase,
                     count(d) AS document_count,
                     count(CASE WHEN EXISTS { MATCH (d)-[:MENTIONS]->(:Entity) } THEN 1 END) AS docs_with_mentions
                OPTIONAL MATCH (f:DocumentFile)
                WHERE coalesce(f.usecase, f.dataset_id, f.source, 'unknown') = usecase
                RETURN usecase,
                       document_count,
                       docs_with_mentions,
                       count(f) AS file_count
                ORDER BY usecase
                """
            )
            entity_rows = self._run(
                """
                MATCH (e:Entity)
                RETURN coalesce(e.usecase, 'unknown') AS usecase,
                       count(e) AS entity_count,
                       count(CASE WHEN e.canonical_id IS NOT NULL AND trim(toString(e.canonical_id)) <> '' THEN 1 END) AS canonical_entity_count
                ORDER BY usecase
                """
            )
            mention_rows = self._run(
                """
                MATCH (d:Document)-[m:MENTIONS]->(:Entity)
                RETURN coalesce(d.usecase, d.dataset_id, d.source, 'unknown') AS usecase,
                       count(m) AS mention_count
                ORDER BY usecase
                """
            )
            relation_rows = self._run(
                """
                MATCH (e:Entity)-[r:REL]->(:Entity)
                RETURN coalesce(e.usecase, 'unknown') AS usecase,
                       count(r) AS relation_count
                ORDER BY usecase
                """
            )

            meta_row = meta_rows[0] if meta_rows else {}
            coverage_by_usecase: Dict[str, Dict[str, Any]] = {}

            for row in per_usecase_rows:
                usecase = str(row.get("usecase") or "unknown")
                document_count = _coerce_int(row.get("document_count"))
                docs_with_mentions = _coerce_int(row.get("docs_with_mentions"))
                coverage_by_usecase[usecase] = {
                    "usecase": usecase,
                    "document_count": document_count,
                    "file_count": _coerce_int(row.get("file_count")),
                    "entity_count": 0,
                    "canonical_entity_count": 0,
                    "mention_count": 0,
                    "relation_count": 0,
                    "docs_with_mentions": docs_with_mentions,
                    "docs_without_mentions": max(0, document_count - docs_with_mentions),
                    "semantic_coverage_ratio": _coverage_ratio(docs_with_mentions, document_count),
                    "semantic_ready": False,
                }

            for row in entity_rows:
                usecase = str(row.get("usecase") or "unknown")
                entry = coverage_by_usecase.setdefault(
                    usecase,
                    {
                        "usecase": usecase,
                        "document_count": 0,
                        "file_count": 0,
                        "entity_count": 0,
                        "canonical_entity_count": 0,
                        "mention_count": 0,
                        "relation_count": 0,
                        "docs_with_mentions": 0,
                        "docs_without_mentions": 0,
                        "semantic_coverage_ratio": 0.0,
                        "semantic_ready": False,
                    },
                )
                entry["entity_count"] = _coerce_int(row.get("entity_count"))
                entry["canonical_entity_count"] = _coerce_int(row.get("canonical_entity_count"))
                entry["semantic_ready"] = entry["document_count"] > 0 and entry["entity_count"] > 0

            for row in mention_rows:
                usecase = str(row.get("usecase") or "unknown")
                entry = coverage_by_usecase.setdefault(
                    usecase,
                    {
                        "usecase": usecase,
                        "document_count": 0,
                        "file_count": 0,
                        "entity_count": 0,
                        "canonical_entity_count": 0,
                        "mention_count": 0,
                        "relation_count": 0,
                        "docs_with_mentions": 0,
                        "docs_without_mentions": 0,
                        "semantic_coverage_ratio": 0.0,
                        "semantic_ready": False,
                    },
                )
                entry["mention_count"] = _coerce_int(row.get("mention_count"))

            for row in relation_rows:
                usecase = str(row.get("usecase") or "unknown")
                entry = coverage_by_usecase.setdefault(
                    usecase,
                    {
                        "usecase": usecase,
                        "document_count": 0,
                        "file_count": 0,
                        "entity_count": 0,
                        "canonical_entity_count": 0,
                        "mention_count": 0,
                        "relation_count": 0,
                        "docs_with_mentions": 0,
                        "docs_without_mentions": 0,
                        "semantic_coverage_ratio": 0.0,
                        "semantic_ready": False,
                    },
                )
                entry["relation_count"] = _coerce_int(row.get("relation_count"))

            usecase_coverage = sorted(coverage_by_usecase.values(), key=lambda item: str(item.get("usecase") or ""))
            document_count = sum(_coerce_int(item.get("document_count")) for item in usecase_coverage)
            docs_with_mentions = sum(_coerce_int(item.get("docs_with_mentions")) for item in usecase_coverage)
            entity_count = sum(_coerce_int(item.get("entity_count")) for item in usecase_coverage)
            mention_count = sum(_coerce_int(item.get("mention_count")) for item in usecase_coverage)
            relation_count = sum(_coerce_int(item.get("relation_count")) for item in usecase_coverage)
            docs_without_mentions = max(0, document_count - docs_with_mentions)
            semantic_gap_usecases = [
                str(item.get("usecase") or "unknown")
                for item in usecase_coverage
                if _coerce_int(item.get("document_count")) > 0 and _coerce_int(item.get("entity_count")) <= 0
            ]

            if document_count <= 0:
                message = "No :Document nodes found in Neo4j."
            elif semantic_gap_usecases:
                message = f"Semantic coverage missing for: {', '.join(semantic_gap_usecases)}."
            elif docs_without_mentions > 0:
                message = (
                    f"Partial semantic coverage: {docs_with_mentions} of {document_count} document chunks "
                    "have grounded entity mentions."
                )
            else:
                message = "ok"

            semantic_coverage_ratio = _coverage_ratio(docs_with_mentions, document_count)
            twin_health = _build_twin_health_summary(
                document_count=document_count,
                entity_count=entity_count,
                semantic_coverage_ratio=semantic_coverage_ratio,
                semantic_gap_usecases=semantic_gap_usecases,
                usecase_coverage=usecase_coverage,
                message=message,
            )

            return {
                "backend": self.name,
                "ready": document_count > 0,
                "document_count": document_count,
                "entity_count": entity_count,
                "mention_count": mention_count,
                "relation_count": relation_count,
                "docs_with_mentions": docs_with_mentions,
                "docs_without_mentions": docs_without_mentions,
                "semantic_coverage_ratio": semantic_coverage_ratio,
                "semantic_gap_usecases": semantic_gap_usecases,
                "sources": meta_row.get("sources") or [],
                "machines": meta_row.get("machines") or [],
                "usecase_coverage": usecase_coverage,
                "twin_health": twin_health,
                "message": message,
            }
        except Exception as exc:
            return {
                "backend": self.name,
                "ready": False,
                "message": f"Docs status query failed: {exc}",
            }

    def _ensure_schema(self) -> None:
        if self._driver is None:
            return
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE INDEX doc_source_idx IF NOT EXISTS "
                "FOR (d:Document) ON (d.source)"
            )
            try:
                session.run(
                    "CREATE INDEX doc_usecase_idx IF NOT EXISTS "
                    "FOR (d:Document) ON (d.usecase)"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE INDEX entity_usecase_idx IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.usecase)"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE INDEX entity_type_idx IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.type)"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE INDEX entity_name_idx IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.name_norm)"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE INDEX entity_canonical_id_idx IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.canonical_id)"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE VECTOR INDEX doc_vector_index IF NOT EXISTS "
                    "FOR (d:Document) ON (d.embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: 384,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}"
                )
            except Exception:
                pass
            try:
                session.run(
                    f"CREATE VECTOR INDEX {_ENTITY_VECTOR_INDEX} IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: 384,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}"
                )
            except Exception:
                pass
            session.run(
                "MERGE (sv:SchemaVersion {domain: 'documents'}) "
                "SET sv.version = $version, sv.updated_at = $ts",
                version=_DOC_SCHEMA_VERSION,
                ts=_now_iso(),
            )

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]

            self._graph_database = GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
                connection_timeout=self._connect_timeout_s,
                connection_acquisition_timeout=self._connect_timeout_s,
            )
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1").consume()
            self._ensure_schema()
            self._driver_error = None
        except Exception as exc:
            if self._driver is not None:
                try:
                    self._driver.close()
                except Exception:
                    pass
            self._driver = None
            self._driver_error = str(exc)

    def _ensure_driver(self) -> None:
        if self._driver is not None:
            return
        self._connect()

    def _run(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        if self._driver is None:
            return []
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, **params).data()

    def _vector_search(
        self,
        params: Dict[str, Any],
        *,
        limit_to_top_k: bool = True,
        apply_feedback_ranking: bool = True,
    ) -> List[Dict[str, Any]]:
        embedding = _compute_embedding(params.get("search_text") or "")
        if embedding is None or self._driver is None:
            return []
        vector_params = dict(params)
        vector_params["embedding"] = embedding
        vector_params["candidate_limit"] = max(params["top_k"] * 6, params["top_k"])
        vector_params["result_limit"] = params["top_k"] if limit_to_top_k else vector_params["candidate_limit"]
        try:
            rows = self._run(
                f"""
                CALL db.index.vector.queryNodes('{_DOC_VECTOR_INDEX}', $candidate_limit, $embedding)
                YIELD node, score
                WHERE node:Document AND {self._where_clause('node')}
                RETURN node AS doc, score
                ORDER BY score DESC
                LIMIT $result_limit
                """,
                **vector_params,
            )
        except Exception:
            return []
        matches = [self._format_match(row.get("doc"), row.get("score")) for row in rows]
        if apply_feedback_ranking:
            return self._rank_matches_with_feedback(matches, top_k=params["top_k"] if limit_to_top_k else 0)
        return matches[: vector_params["result_limit"]]

    def _entity_vector_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        embedding = _compute_embedding(params.get("search_text") or "")
        if embedding is None or self._driver is None:
            return []
        vector_params = dict(params)
        vector_params["embedding"] = embedding
        vector_params["candidate_limit"] = max(params["top_k"] * 6, params["top_k"])
        vector_params["entity_top_k"] = max(params["top_k"] * 3, params["top_k"])
        try:
            rows = self._run(
                f"""
                CALL db.index.vector.queryNodes('{_ENTITY_VECTOR_INDEX}', $candidate_limit, $embedding)
                YIELD node, score
                WHERE node:Entity AND {self._entity_where_clause('node')}
                RETURN node AS entity, score
                ORDER BY score DESC
                LIMIT $entity_top_k
                """,
                **vector_params,
            )
        except Exception:
            return []
        return [self._format_entity_match(row.get("entity"), row.get("score")) for row in rows]

    def _entity_alias_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        terms = _query_terms(params.get("search_text") or "")
        if not terms:
            return []
        rows = self._run(
            f"""
            UNWIND $terms AS term
            MATCH (e:Entity)
            WHERE {self._entity_where_clause('e')}
              AND (
                toLower(coalesce(e.name_norm, '')) CONTAINS term OR
                any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS term)
              )
            RETURN e AS entity, count(DISTINCT term) AS score
            ORDER BY score DESC, coalesce(e.name, '')
            LIMIT $entity_top_k
            """,
            **params,
            terms=terms,
            entity_top_k=max(params["top_k"] * 3, params["top_k"]),
        )
        normalizer = float(len(terms) or 1)
        matches = []
        for row in rows:
            score = float(row.get("score") or 0.0) / normalizer
            matches.append(self._format_entity_match(row.get("entity"), score))
        return matches

    def _seed_entities_from_chunks(self, chunk_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        rows = self._run(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (d:Document {id: chunk_id})-[:MENTIONS]->(e:Entity)
            RETURN e AS entity, count(DISTINCT chunk_id) AS score
            ORDER BY score DESC, coalesce(e.name, '')
            LIMIT $entity_top_k
            """,
            chunk_ids=chunk_ids,
            entity_top_k=max(top_k * 3, top_k),
        )
        normalizer = float(len(chunk_ids) or 1)
        matches = []
        for row in rows:
            score = float(row.get("score") or 0.0) / normalizer
            matches.append(self._format_entity_match(row.get("entity"), score))
        return matches

    def _graph_expand_search(self, params: Dict[str, Any], seed_entity_ids: List[str]) -> List[Dict[str, Any]]:
        if not seed_entity_ids:
            return []
        rows = self._run(
            f"""
            UNWIND $seed_entity_ids AS seed_id
            MATCH (seed:Entity {{id: seed_id}})
            OPTIONAL MATCH (seed)-[:REL]-(neighbor:Entity)
            WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS raw_entities
            UNWIND raw_entities AS entity
            WITH DISTINCT entity
            WHERE entity IS NOT NULL
            MATCH (d:Document)-[:MENTIONS]->(entity)
            WHERE {self._where_clause('d')}
            RETURN d AS doc,
                   count(DISTINCT entity.id) AS entity_support,
                     collect(DISTINCT {{id: entity.id, canonical_id: entity.canonical_id, name: entity.name, type: entity.type}})[..5] AS evidence_entities
            ORDER BY entity_support DESC, coalesce(d.page, 0)
            LIMIT $candidate_limit
            """,
            **params,
            seed_entity_ids=seed_entity_ids,
            candidate_limit=max(params["top_k"] * 6, params["top_k"]),
        )
        matches: List[Dict[str, Any]] = []
        for row in rows:
            match = self._format_match(row.get("doc"), row.get("entity_support"))
            evidence_entities = row.get("evidence_entities") or []
            if evidence_entities:
                match["evidence_entities"] = evidence_entities
            match["graph_support"] = int(row.get("entity_support") or 0)
            matches.append(match)
        return matches

    def _hybrid_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        vector_matches = [
            dict(match, _score_source="vector")
            for match in self._vector_search(
                params,
                limit_to_top_k=False,
                apply_feedback_ranking=False,
            )
        ]
        seed_entity_matches = self._seed_entities_from_chunks(
            [match.get("id") for match in vector_matches if match.get("id")],
            params["top_k"],
        )
        entity_matches = _dedupe_entity_matches(
            [
                *self._entity_vector_search(params),
                *self._entity_alias_search(params),
                *seed_entity_matches,
            ]
        )
        graph_matches = [
            dict(match, _score_source="graph")
            for match in self._graph_expand_search(
                params,
                [match["id"] for match in entity_matches if match.get("id")],
            )
        ]
        rankings = [ranking for ranking in (vector_matches, graph_matches) if ranking]
        if not rankings:
            return []
        return self._rrf_merge(rankings, top_k=params["top_k"])

    def _rrf_merge(self, rankings: List[List[Dict[str, Any]]], *, top_k: int) -> List[Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}
        for ranking in rankings:
            for index, match in enumerate(ranking, start=1):
                match_id = str(match.get("id") or "").strip()
                if not match_id:
                    continue
                entry = combined.setdefault(match_id, dict(match))
                score_source = str(match.get("_score_source") or "vector")
                entry["_rrf_score"] = float(entry.get("_rrf_score") or 0.0) + 1.0 / float(_RRF_K + index)
                entry["_ranking_count"] = int(entry.get("_ranking_count") or 0) + 1
                if score_source == "vector":
                    entry["_best_vector_score"] = max(
                        float(entry.get("_best_vector_score") or 0.0),
                        float(match.get("score") or 0.0),
                    )
                if score_source == "graph":
                    entry["_best_graph_support"] = max(
                        int(entry.get("_best_graph_support") or 0),
                        int(match.get("graph_support") or 0),
                    )
                entry["evidence_entities"] = _merge_evidence_entities(
                    entry.get("evidence_entities") or [],
                    match.get("evidence_entities") or [],
                )
                entry["graph_support"] = max(
                    int(entry.get("graph_support") or 0),
                    int(match.get("graph_support") or 0),
                )
                if entry.get("text") in (None, "") and match.get("text"):
                    entry["text"] = match.get("text")

        merged_matches = []
        for entry in combined.values():
            final_score = _hybrid_match_score(
                document_type=entry.get("document_type"),
                vector_score=entry.get("_best_vector_score"),
                graph_support=entry.get("_best_graph_support") or entry.get("graph_support"),
                ranking_count=entry.get("_ranking_count"),
            )
            entry["score"] = round(float(final_score), 4) if final_score is not None else None
            if not entry.get("evidence_entities"):
                entry.pop("evidence_entities", None)
            entry.pop("_best_vector_score", None)
            entry.pop("_best_graph_support", None)
            entry.pop("_ranking_count", None)
            entry.pop("_score_source", None)
            merged_matches.append(entry)

        merged_matches = self._rank_matches_with_feedback(merged_matches, top_k=0)
        merged_matches.sort(
            key=lambda item: (
                float(item.get("ranking_score") or item.get("score") or 0.0),
                float(item.get("feedback_score") or 0.0),
                float(item.get("score") or 0.0),
                float(item.get("_rrf_score") or 0.0),
            ),
            reverse=True,
        )
        for entry in merged_matches:
            entry.pop("_rrf_score", None)
        return merged_matches[:top_k]

    def _text_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        where_clause = self._where_clause("d")
        rows = self._run(
            f"""
            MATCH (d:Document)
            WHERE {where_clause}
            RETURN d AS doc
            LIMIT 200
            """,
            **params,
        )
        query_terms = _query_terms(params.get("search_text") or "")
        scored: List[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            formatted = self._format_match(row.get("doc"), None)
            haystack = " ".join(
                str(formatted.get(key) or "")
                for key in ("text", "file_name", "machine", "source", "document_type")
            ).lower()
            if query_terms:
                hits = sum(1 for term in query_terms if term in haystack)
                if hits <= 0:
                    continue
                score = hits / float(len(query_terms))
            else:
                score = 0.0
            adjusted_score = _adjust_match_score(formatted.get("document_type"), score)
            formatted["score"] = round(float(adjusted_score or 0.0), 4)
            scored.append((score, formatted))
        ranked_matches = self._rank_matches_with_feedback([match for _, match in scored], top_k=0)
        ranked_matches.sort(
            key=lambda item: (
                float(item.get("ranking_score") or item.get("score") or 0.0),
                float(item.get("feedback_score") or 0.0),
                float(item.get("score") or 0.0),
            ),
            reverse=True,
        )
        return ranked_matches[: params["top_k"]]

    def _entities_available(self, params: Dict[str, Any]) -> bool:
        rows = self._run(
            f"""
            MATCH (e:Entity)
            WHERE {self._entity_where_clause('e')}
            RETURN count(e) AS entity_count
            """,
            **params,
        )
        row = rows[0] if rows else {}
        return int(row.get("entity_count") or 0) > 0

    def _build_filter_params(
        self,
        *,
        usecase: Optional[str],
        source_filter: Optional[str],
        machine: Optional[str],
        document_type: Optional[str],
    ) -> Dict[str, Any]:
        usecase_code = normalize_usecase(usecase)
        raw_source = str(source_filter or "").strip()
        source_usecase = normalize_usecase(raw_source)
        source_candidates = [raw_source.lower()] if raw_source else []
        if source_usecase:
            source_candidates.extend(usecase_aliases(source_usecase))
            source_candidates.append(source_usecase.lower())

        machine_text = str(machine or "").strip().lower()
        machine_tokens = _query_terms(machine_text)
        if machine_text and machine_text not in machine_tokens:
            machine_tokens.append(machine_text)

        usecase_alias_list = usecase_aliases(usecase_code)
        usecase_machine_tokens = list(usecase_alias_list)
        if usecase_code == "SITE_A":
            usecase_machine_tokens.extend(["machine_a1", "a1001"])
        elif usecase_code == "SITE_C":
            usecase_machine_tokens.extend(["c1001", "machine_c1"])
        elif usecase_code == "SITE_B":
            usecase_machine_tokens.extend(["b1001", "b1002"])

        return {
            "usecase": usecase_code,
            "usecase_aliases": sorted(set(usecase_alias_list)),
            "usecase_machine_tokens": sorted(set(usecase_machine_tokens)),
            "source_candidates": sorted(set(source_candidates)),
            "machine": machine_text,
            "machine_tokens": sorted(set(machine_tokens)),
            "document_type": str(document_type or "").strip().lower(),
        }

    @staticmethod
    def _where_clause(alias: str) -> str:
        return (
            "(" 
            "$usecase IS NULL OR "
            f"toUpper(coalesce({alias}.usecase, '')) = $usecase OR "
            f"toLower(coalesce({alias}.dataset_id, '')) IN $usecase_aliases OR "
            f"toLower(coalesce({alias}.source, '')) IN $usecase_aliases OR "
            f"any(token IN $usecase_machine_tokens WHERE toLower(coalesce({alias}.machine, '')) CONTAINS token) OR "
            f"any(token IN $usecase_machine_tokens WHERE toLower(coalesce({alias}.file_name, {alias}.relative_path, '')) CONTAINS token)"
            ") AND ("
            "$source_candidates = [] OR "
            f"toLower(coalesce({alias}.source, '')) IN $source_candidates OR "
            f"toUpper(coalesce({alias}.usecase, '')) IN [candidate IN $source_candidates | toUpper(candidate)]"
            ") AND ("
            "$machine = '' OR "
            f"toLower(coalesce({alias}.machine, '')) CONTAINS $machine OR "
            f"toLower(coalesce({alias}.machine_uri, '')) CONTAINS $machine OR "
            f"any(token IN $machine_tokens WHERE toLower(coalesce({alias}.file_name, {alias}.relative_path, '')) CONTAINS token)"
            ") AND ("
            "$document_type = '' OR "
            f"toLower(coalesce({alias}.document_type, '')) = $document_type"
            ")"
        )

    @staticmethod
    def _entity_where_clause(alias: str) -> str:
        return (
            "(" 
            "$usecase IS NULL OR "
            f"toUpper(coalesce({alias}.usecase, '')) = $usecase OR "
            f"toLower(coalesce({alias}.usecase, '')) IN $usecase_aliases"
            ")"
        )

    def _format_entity_match(self, node: Any, score: Optional[float]) -> Dict[str, Any]:
        props = dict(node or {})
        return {
            "id": props.get("id"),
            "canonical_id": props.get("canonical_id"),
            "name": props.get("name"),
            "type": props.get("type"),
            "aliases": props.get("aliases") or [],
            "score": round(float(score), 4) if score is not None else None,
        }

    def _format_match(self, node: Any, score: Optional[float]) -> Dict[str, Any]:
        props = dict(node or {})
        source = props.get("source")
        usecase = resolve_usecase(
            usecase=props.get("usecase"),
            dataset_id=props.get("dataset_id"),
            machine_id=props.get("machine_id"),
            machine_uri=props.get("machine_uri"),
            machine=props.get("machine"),
            source=source,
            metadata=props,
            fallback_generic=False,
        )
        file_name = props.get("file_name") or props.get("relative_path") or props.get("document_id")
        machine = props.get("machine") or props.get("machine_id") or props.get("machine_uri")
        page = props.get("page") or props.get("page_start")
        text = str(props.get("text") or "")
        document_type = props.get("document_type")
        if len(text) > 1200:
            text = text[:1197].rstrip() + "..."
        adjusted_score = _adjust_match_score(document_type, score)
        feedback_score = float(props.get("feedback_score") or 0.0)
        return {
            "id": props.get("id") or props.get("document_id"),
            "text": text,
            "source": source,
            "usecase": usecase,
            "file_name": file_name,
            "page": page,
            "machine": machine,
            "machine_uri": props.get("machine_uri"),
            "document_type": document_type,
            "language": props.get("language_code") or props.get("original_language"),
            "score": round(float(adjusted_score), 4) if adjusted_score is not None else None,
            "helpful_count": int(props.get("helpful_count") or 0),
            "not_helpful_count": int(props.get("not_helpful_count") or 0),
            "feedback_score": feedback_score,
            "ranking_score": round(_feedback_rank_score(adjusted_score, feedback_score), 4)
            if adjusted_score is not None
            else 0.0,
            "citation": _citation(usecase, file_name, page, machine),
        }

    def _doc_feedback_signals(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        unique_doc_ids = [doc_id for doc_id in dict.fromkeys(str(doc_id or "").strip() for doc_id in doc_ids) if doc_id]
        if not unique_doc_ids or getattr(self, "_driver", None) is None:
            return {}
        rows = self._run(
            """
            MATCH (m:Memory)
            WHERE any(doc_id IN $doc_ids WHERE doc_id IN coalesce(m.doc_link_ids, []))
            RETURN m.metadata_json AS metadata_json
            """,
            doc_ids=unique_doc_ids,
        )
        feedback_by_id: Dict[str, Dict[str, Any]] = {
            doc_id: {
                "helpful_count": 0,
                "not_helpful_count": 0,
                "feedback_score": 0.0,
            }
            for doc_id in unique_doc_ids
        }
        for row in rows:
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
            except (TypeError, ValueError):
                continue
            if not isinstance(metadata, dict):
                continue
            for link in metadata.get("doc_links") or []:
                if not isinstance(link, dict):
                    continue
                doc_id = str(link.get("id") or "").strip()
                if doc_id not in feedback_by_id:
                    continue
                helpful_count = int(link.get("helpful_count") or 0)
                not_helpful_count = int(link.get("not_helpful_count") or 0)
                feedback_by_id[doc_id]["helpful_count"] += helpful_count
                feedback_by_id[doc_id]["not_helpful_count"] += not_helpful_count
                feedback_by_id[doc_id]["feedback_score"] = float(
                    feedback_by_id[doc_id]["helpful_count"] - feedback_by_id[doc_id]["not_helpful_count"]
                )
        return feedback_by_id

    def _rank_matches_with_feedback(
        self,
        matches: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not matches:
            return []
        feedback_loader = getattr(self, "_doc_feedback_signals", None)
        if callable(feedback_loader):
            feedback_by_id = feedback_loader([str(match.get("id") or "") for match in matches]) or {}
        else:
            feedback_by_id = {}
        ranked: List[Dict[str, Any]] = []
        for match in matches:
            enriched = dict(match)
            doc_id = str(enriched.get("id") or "").strip()
            feedback = feedback_by_id.get(doc_id, {})
            helpful_count = int(feedback.get("helpful_count", enriched.get("helpful_count") or 0) or 0)
            not_helpful_count = int(feedback.get("not_helpful_count", enriched.get("not_helpful_count") or 0) or 0)
            feedback_score = float(feedback.get("feedback_score", enriched.get("feedback_score") or 0.0) or 0.0)
            enriched["helpful_count"] = helpful_count
            enriched["not_helpful_count"] = not_helpful_count
            enriched["feedback_score"] = feedback_score
            enriched["ranking_score"] = round(
                _feedback_rank_score(enriched.get("score"), feedback_score),
                4,
            )
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                float(item.get("ranking_score") or item.get("score") or 0.0),
                float(item.get("feedback_score") or 0.0),
                float(item.get("score") or 0.0),
            ),
            reverse=True,
        )
        if top_k > 0:
            return ranked[:top_k]
        return ranked


_DOCS_BACKEND: Optional[DocsBackend] = None


def get_docs_backend() -> DocsBackend:
    """Factory for docs backend.

    Keep this indirection so swapping to a real PDF/Neo4j backend is a
    one-line change.
    """

    global _DOCS_BACKEND
    if _DOCS_BACKEND is None:
        _DOCS_BACKEND = Neo4jDocsBackend()
    return _DOCS_BACKEND


def _compute_embedding(text: str) -> Optional[List[float]]:
    try:
        from backend.agents.memory.retriever import _get_embedding_model

        model = _get_embedding_model()
        if model is None:
            return None
        return [float(value) for value in model.encode(text).tolist()]
    except Exception:
        return None


def _query_terms(text: str) -> List[str]:
    terms: List[str] = []
    for raw in str(text or "").lower().split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_"}).strip("-_")
        if not token:
            continue
        if token.isdigit() or len(token) >= 3:
            terms.append(token)
    return list(dict.fromkeys(terms))


def _citation(usecase: Optional[str], file_name: Any, page: Any, machine: Any) -> str:
    parts: List[str] = []
    if usecase:
        parts.append(str(usecase))
    if file_name:
        parts.append(str(file_name))
    if page not in (None, ""):
        parts.append(f"p.{page}")
    if machine:
        parts.append(f"machine={machine}")
    return " / ".join(parts)


def _dedupe_entity_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for match in matches:
        match_id = _entity_match_key(match)
        if not match_id:
            continue
        current = deduped.get(match_id)
        if current is None or float(match.get("score") or 0.0) > float(current.get("score") or 0.0):
            deduped[match_id] = dict(match)
    return sorted(deduped.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)


def _merge_evidence_entities(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entity in [*existing, *incoming]:
        if not isinstance(entity, dict):
            continue
        entity_id = _entity_match_key(entity)
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        merged.append(entity)
    return merged


def _entity_match_key(match: Dict[str, Any]) -> str:
    canonical_id = str(match.get("canonical_id") or "").strip()
    if canonical_id:
        return canonical_id
    return str(match.get("id") or "").strip()

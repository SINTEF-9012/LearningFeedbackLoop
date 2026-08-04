"""
Retriever Agent — dual-source knowledge retrieval (SINDIT + Neo4j).

Provides document / entity / memory retrieval for the LFL backend by
querying two complementary sources:

1. **SINDIT** (knowledge graph) — asset metadata, RDF entities, docs ingested
   into the digital-twin knowledge graph (when ``SINDIT_ENABLED``).
2. **Neo4j memory store** — vector-similarity search over past event memories
   (when ``STORAGE_BACKEND == "neo4j"``).

Falls back gracefully when either source is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.agents.config import (
    SINDIT_API_URL,
    SINDIT_ENABLED,
    SINDIT_TIMEOUT_S,
    STORAGE_BACKEND,
)
from backend.agents.llm.retriever_models import DocsQueryArgs, QueryArgs

logger = logging.getLogger(__name__)


def _describe_memory_store(store: Any) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "configured_backend": STORAGE_BACKEND,
        "resolved_backend": None,
        "store_class": None,
    }
    if store is None:
        return details

    details["store_class"] = type(store).__name__
    if hasattr(store, "_driver"):
        details["resolved_backend"] = "neo4j"
        details["database"] = getattr(store, "_database", None)
    elif hasattr(store, "db_path"):
        details["resolved_backend"] = "sqlite"
        details["db_path"] = str(getattr(store, "db_path", ""))
    else:
        details["resolved_backend"] = STORAGE_BACKEND

    try:
        details["count"] = store.count()
    except Exception:
        details["count"] = "error"

    if hasattr(store, "subgraph_integrity"):
        try:
            details["subgraph_integrity"] = store.subgraph_integrity()
        except Exception as exc:
            details["subgraph_integrity"] = {"error": str(exc)}

    return details


def _describe_knowledge_graph(docs_status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status = dict(docs_status or {})
    return {
        "backend": status.get("backend"),
        "ready": bool(status.get("ready", False)),
        "document_count": status.get("document_count"),
        "entity_count": status.get("entity_count"),
        "mention_count": status.get("mention_count"),
        "relation_count": status.get("relation_count"),
        "docs_with_mentions": status.get("docs_with_mentions"),
        "docs_without_mentions": status.get("docs_without_mentions"),
        "semantic_coverage_ratio": status.get("semantic_coverage_ratio"),
        "semantic_gap_usecases": list(status.get("semantic_gap_usecases") or []),
        "sources": list(status.get("sources") or []),
        "machines": list(status.get("machines") or []),
        "usecase_coverage": list(status.get("usecase_coverage") or []),
        "twin_health": dict(status.get("twin_health") or {}),
        "message": status.get("message"),
    }


class RetrieverAgent:
    """Agent that answers document / knowledge / memory-search queries.

    Handles the ``retriever`` agent name in the router with actions:

    - ``search``  — free-text search across both sources
    - ``assets``  — list / search SINDIT assets
    - ``memory``  — vector-similarity search over stored memories
    - ``status``  — return connectivity status of both sources
    """

    def __init__(self) -> None:
        self._sindit_client: Any = None
        self._neo4j_store: Any = None
        self._docs_backend: Any = None
        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {
            "search": self._search,
            "assets": self._search_assets,
            "memory": self._memory_search,
            "docs.search": self._docs_search,
            "docs.structured": self._docs_structured,
            "docs.status": self._docs_status_args,
            "status": self._status_args,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Warm up connections (called by router on first dispatch)."""
        if SINDIT_ENABLED:
            try:
                from backend.agents.sindit.client import SinditClient

                self._sindit_client = SinditClient(
                    base_url=SINDIT_API_URL,
                    timeout=SINDIT_TIMEOUT_S,
                )
                await self._sindit_client.__aenter__()
            except Exception as exc:
                logger.warning("RetrieverAgent: SINDIT client init failed: %s", exc)

        if STORAGE_BACKEND == "neo4j":
            try:
                from backend.agents.memory.init import get_store

                self._neo4j_store = get_store()
            except Exception as exc:
                logger.debug("RetrieverAgent: Neo4j store unavailable: %s", exc)

        # Pluggable documentation backend (PDF/Neo4j chunks, etc.)
        try:
            from backend.agents.llm.docs_backend import get_docs_backend

            self._docs_backend = get_docs_backend()
        except Exception as exc:
            logger.debug("RetrieverAgent: docs backend unavailable: %s", exc)
            self._docs_backend = None

    # ------------------------------------------------------------------
    # Router interface
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        session_id: str,
        action: Optional[str],
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = action or "search"
        handler = self._action_handlers.get(action)
        if handler is None:
            return self._envelope(action=action, payload={}, ok=False, error=f"Unknown retriever action: {action}")

        try:
            payload = await handler(args or {})
            return self._envelope(action=action, payload=payload, ok=True)
        except Exception as exc:
            logger.exception("Retriever action failed: %s", action)
            return self._envelope(action=action, payload={}, ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Unified search across SINDIT assets + Neo4j memories."""
        parsed = QueryArgs.from_dict(args)
        query = parsed.query
        include_docs = bool(args.get("include_docs", True))
        results: Dict[str, Any] = {
            "query": query,
            "sources": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # SINDIT asset search
        if self._sindit_client:
            try:
                assets = await self._sindit_client.search_assets(query)
                if assets:
                    results["sindit_assets"] = self._normalize_sindit_assets(assets[:10])
                    results["sources"].append("sindit")
            except Exception as exc:
                logger.debug("SINDIT search failed: %s", exc)

        # Neo4j vector search
        memory_results = await self._memory_search({"query": query, "top_k": parsed.top_k})
        if memory_results.get("matches"):
            results["memories"] = memory_results["matches"]
            results["sources"].append("neo4j_memory")

        # Documentation search (pluggable backend)
        if include_docs:
            try:
                doc_results = await self._docs_search(args)
                if doc_results.get("matches"):
                    results["documents"] = doc_results["matches"]
                    results["sources"].append(doc_results.get("backend", "docs_vector"))
            except Exception as exc:
                logger.debug("Docs search failed: %s", exc)

        linked_assets = self._link_documents_and_assets(
            results.get("documents") if isinstance(results.get("documents"), list) else [],
            results.get("sindit_assets") if isinstance(results.get("sindit_assets"), list) else [],
        )
        if linked_assets:
            results["linked_assets"] = linked_assets

        if not results["sources"]:
            results["message"] = "No results found from any source."

        return results

    async def _search_assets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List or search SINDIT assets."""
        if not self._sindit_client:
            return {"error": "SINDIT is not enabled or unavailable."}

        query = QueryArgs.from_dict(args).query
        try:
            if query:
                assets = await self._sindit_client.search_assets(query)
            else:
                assets = await self._sindit_client.get_assets()
            return {"assets": assets, "count": len(assets)}
        except Exception as exc:
            return {"error": f"Asset search failed: {exc}"}

    async def _memory_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Vector-similarity search over stored memories."""
        parsed = QueryArgs.from_dict(args)
        query = parsed.query
        top_k = parsed.top_k

        if self._neo4j_store is None:
            # Fallback: use the text-based search on whatever store is active
            try:
                from backend.agents.memory.init import get_store
                store = get_store()
                memories = store.search(text_query=query)
                return {
                    "matches": [
                        {
                            "id": m.id,
                            "annotation": m.annotation_text,
                            "session_id": m.session_id,
                            "label": m.label,
                            "score": None,
                        }
                        for m in memories[:top_k]
                    ]
                }
            except Exception:
                return {"matches": []}

        # Neo4j store — try vector search if embedding available
        try:
            from backend.agents.storage.neo4j_store import Neo4jMemoryStore

            if isinstance(self._neo4j_store, Neo4jMemoryStore) and query:
                embedding = self._compute_embedding(query)
                if embedding is not None:
                    results = self._neo4j_store.vector_search(embedding, top_k=top_k)
                    return {
                        "matches": [
                            {
                                "id": mem.id,
                                "annotation": mem.annotation_text,
                                "session_id": mem.session_id,
                                "label": mem.label,
                                "score": round(score, 4),
                            }
                            for mem, score in results
                        ]
                    }
        except Exception as exc:
            logger.debug("Vector search failed: %s", exc)

        # Fallback to text search
        memories = self._neo4j_store.search(text_query=query)
        return {
            "matches": [
                {
                    "id": m.id,
                    "annotation": m.annotation_text,
                    "session_id": m.session_id,
                    "label": m.label,
                    "score": None,
                }
                for m in memories[:top_k]
            ]
        }

    async def _status(self) -> Dict[str, Any]:
        """Return connectivity status of both sources."""
        memory_graph = _describe_memory_store(self._neo4j_store)
        docs_status: Optional[Dict[str, Any]] = None
        status: Dict[str, Any] = {
            "sindit_enabled": SINDIT_ENABLED,
            "configured_storage_backend": STORAGE_BACKEND,
            "storage_backend": memory_graph.get("resolved_backend") or STORAGE_BACKEND,
            "memory_graph": memory_graph,
        }
        if self._sindit_client:
            status["sindit_reachable"] = await self._sindit_client.health()
        status["neo4j_memory_count"] = (
            memory_graph.get("count")
            if memory_graph.get("resolved_backend") == "neo4j"
            else None
        )
        if self._docs_backend:
            try:
                docs_status = await self._docs_backend.status()
                status["docs"] = docs_status
            except Exception:
                docs_status = {"ready": False, "error": "docs backend status failed"}
                status["docs"] = docs_status
        status["knowledge_graph"] = _describe_knowledge_graph(docs_status)
        return status

    async def _status_args(self, _: Dict[str, Any]) -> Dict[str, Any]:
        return await self._status()

    async def _docs_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search PDF/document chunks via pluggable docs backend."""
        parsed = DocsQueryArgs.from_dict(args)
        query = parsed.query
        top_k = parsed.top_k
        usecase = parsed.usecase
        source_filter = parsed.source_filter
        machine = parsed.machine
        document_type = parsed.document_type

        if not self._docs_backend:
            return {
                "backend": "docs_vector",
                "query": query,
                "matches": [],
                "message": "Documentation backend unavailable.",
            }

        return await self._docs_backend.search(
            query,
            top_k=top_k,
            usecase=usecase,
            source_filter=source_filter,
            machine=machine,
            document_type=document_type,
        )

    async def _docs_structured(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Structured machine-document retrieval via pluggable backend."""
        if not self._docs_backend:
            return {
                "backend": "docs_structured",
                "records": [],
                "message": "Documentation backend unavailable.",
            }
        return await self._docs_backend.structured(args)

    async def _docs_status(self) -> Dict[str, Any]:
        """Return status for docs backend."""
        if not self._docs_backend:
            return {
                "backend": "docs_vector",
                "ready": False,
                "message": "Documentation backend unavailable.",
            }
        return await self._docs_backend.status()

    async def _docs_status_args(self, _: Dict[str, Any]) -> Dict[str, Any]:
        return await self._docs_status()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_embedding(text: str) -> Optional[List[float]]:
        """Compute a 384-dim embedding for the query text.

        Uses the module-level cached model from
        :func:`backend.agents.memory.retriever._get_embedding_model` so
        the ~80 MB model is loaded at most once across the process.
        """
        try:
            from backend.agents.memory.retriever import _get_embedding_model

            model = _get_embedding_model()
            if model is None:
                return None
            vec = model.encode(text).tolist()
            return vec
        except Exception:
            return None

    @staticmethod
    def _normalize_sindit_assets(assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for asset in assets or []:
            if not isinstance(asset, dict):
                continue
            entry = dict(asset)
            canonical_id = str(entry.get("canonical_id") or entry.get("iri") or "").strip()
            if canonical_id:
                if canonical_id in seen:
                    continue
                seen.add(canonical_id)
                entry["canonical_id"] = canonical_id
            normalized.append(entry)
        return normalized

    @classmethod
    def _link_documents_and_assets(
        cls,
        documents: List[Dict[str, Any]],
        sindit_assets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        linked_by_canonical_id: Dict[str, Dict[str, Any]] = {}

        for asset in sindit_assets:
            if not isinstance(asset, dict):
                continue
            canonical_id = str(asset.get("canonical_id") or asset.get("iri") or "").strip()
            if not canonical_id:
                continue
            asset["canonical_id"] = canonical_id
            sources = asset.setdefault("sources", [])
            if "sindit" not in sources:
                sources.append("sindit")
            asset.setdefault("related_documents", [])
            linked_by_canonical_id[canonical_id] = asset

        for document in documents:
            if not isinstance(document, dict):
                continue
            related_assets: List[Dict[str, Any]] = []
            seen_related: set[str] = set()
            for reference in cls._document_asset_references(document):
                canonical_id = str(reference.get("canonical_id") or "").strip()
                if not canonical_id or canonical_id in seen_related:
                    continue
                seen_related.add(canonical_id)

                asset = linked_by_canonical_id.get(canonical_id)
                if asset is None:
                    asset = {
                        "canonical_id": canonical_id,
                        "iri": reference.get("iri") or canonical_id,
                        "label": reference.get("label"),
                        "type": reference.get("type"),
                        "sources": ["docs"],
                        "related_documents": [],
                    }
                    linked_by_canonical_id[canonical_id] = asset
                else:
                    sources = asset.setdefault("sources", [])
                    if "docs" not in sources:
                        sources.append("docs")
                    if not asset.get("label") and reference.get("label"):
                        asset["label"] = reference.get("label")
                    if not asset.get("type") and reference.get("type"):
                        asset["type"] = reference.get("type")

                related_doc = {
                    "id": document.get("id"),
                    "citation": document.get("citation"),
                    "file_name": document.get("file_name"),
                }
                if not any(existing.get("id") == related_doc["id"] for existing in asset["related_documents"]):
                    asset["related_documents"].append(related_doc)

                related_assets.append(
                    {
                        "canonical_id": canonical_id,
                        "iri": asset.get("iri") or canonical_id,
                        "label": asset.get("label") or reference.get("label"),
                        "type": asset.get("type") or reference.get("type"),
                    }
                )

            if related_assets:
                document["related_assets"] = related_assets

        return sorted(
            linked_by_canonical_id.values(),
            key=lambda item: (
                1 if "sindit" in (item.get("sources") or []) else 0,
                len(item.get("related_documents") or []),
                str(item.get("label") or item.get("iri") or ""),
            ),
            reverse=True,
        )

    @staticmethod
    def _document_asset_references(document: Dict[str, Any]) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for entity in document.get("evidence_entities") or []:
            if not isinstance(entity, dict):
                continue
            canonical_id = str(entity.get("canonical_id") or "").strip()
            if not canonical_id or canonical_id in seen:
                continue
            seen.add(canonical_id)
            references.append(
                {
                    "canonical_id": canonical_id,
                    "iri": canonical_id,
                    "label": entity.get("name"),
                    "type": entity.get("type"),
                }
            )

        machine_uri = str(document.get("machine_uri") or "").strip()
        if machine_uri and machine_uri not in seen:
            seen.add(machine_uri)
            references.append(
                {
                    "canonical_id": machine_uri,
                    "iri": machine_uri,
                    "label": document.get("machine"),
                    "type": "Machine",
                }
            )
        return references

    @staticmethod
    def _envelope(
        *,
        action: str,
        payload: Dict[str, Any],
        ok: bool,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified action response envelope.

        Keeps payload fields top-level for backward compatibility while adding
        stable metadata (`ok`, `action`, `timestamp`, `error`).
        """
        out: Dict[str, Any] = {
            "ok": ok,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out.update(payload or {})
        if error:
            out["error"] = error
        return out

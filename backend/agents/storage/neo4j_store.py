"""
Neo4j Memory Store — Graph-based persistent storage for LFL memories.

Implements ``MemoryStoreProtocol`` using Neo4j as the backend.  The graph
schema separates concerns via node labels:

  Memory domain:
    (:Memory), (:Pattern), (:Feedback), (:Trace), (:Session),
    (:DiscoveredPattern), (:SchemaVersion), (:Experiment),
    (:Machine), (:Tool), (:Snapshot)

  Relationships:
    [:HAS_PATTERN]     Memory  → Pattern
    [:IN_SESSION]      Memory  → Session
    [:IN_EXPERIMENT]   Session → Experiment
    [:HAS_SESSION]     Experiment → Session
    [:TESTED_PATTERN]  Experiment → Pattern
    [:ON_MACHINE]      Memory  → Machine
    [:USED_TOOL]       Memory  → Tool
    [:NEXT]            Memory  → Memory    (temporal, within session)
    [:ABOUT]           Feedback → Memory
    [:ON_PATTERN]      Feedback → Pattern
    [:SIMILAR_TO]      Memory  ↔ Memory   (cosine threshold)
    [:CO_OCCURS_WITH]  Pattern ↔ Pattern  (weight, window, updated_at)
    [:DISCOVERED_FROM] DiscoveredPattern → Memory  (provenance)
    [:EVOLVED_FROM]    Pattern → DiscoveredPattern (promotion link)

  Indices / constraints:
    UNIQUE  Memory.id, Pattern.key, Session.id, Experiment.run_id
    VECTOR  memory_embedding_index (384-dim, cosine)

Requirements:
    neo4j>=5.0  (pip install neo4j)
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..core.schemas import (
    Memory,
    MemoryProvenance,
    NumericMetrics,
    PatternKey,
    TimeRange,
)
from ..patterns.signatures import infer_pattern_kind
from ..usecase import resolve_usecase
from .graph_boundary import (
    collect_memory_graph_cleanup_preview,
    collect_subgraph_integrity,
    legacy_memory_candidate_predicate,
    normalize_doc_link_intent,
)
from .graph_write_outbox import GraphWriteIntent, GraphWriteOutbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph schema version — bump when Cypher migrations are needed.
# ---------------------------------------------------------------------------
NEO4J_SCHEMA_VERSION = 7

# Default vector dimensions (all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2)
_EMBEDDING_DIM = 384

# Module-level guard: _ensure_schema runs 8+ CREATE statements; don't redo per
# instance for the same (uri, database) target within one process.
# Agent Q performance pass: prior behaviour repeated all CREATEs on every
# Neo4jMemoryStore() construction.
_SCHEMA_INITIALIZED: set = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_num(d: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    """Extract a numeric value from a dict, returning None if missing."""
    if d is None:
        return None
    v = d.get(key)
    return float(v) if v is not None else None


def _pattern_kind(pk: PatternKey) -> str:
    pattern_type = pk.pattern_type.value if getattr(pk, "pattern_type", None) else None
    return infer_pattern_kind(pk.key, pattern_type)


def _serialize_pattern_key(pk: PatternKey) -> Dict[str, Any]:
    data = pk.model_dump()
    additional = dict(data.get("additional") or {})
    additional.setdefault("kind", _pattern_kind(pk))
    data["additional"] = additional
    return data


def _operation_node_id(
    operation_id: Optional[str],
    dataset_id: Optional[str],
    case_dir: Optional[str],
) -> Optional[str]:
    op = str(operation_id or "").strip()
    if not op:
        return None
    dataset = str(dataset_id or "").strip() or "unknown-dataset"
    case = str(case_dir or "").strip() or "unknown-case"
    return f"{dataset}::{case}::{op}"


def _deserialize_doc_link_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_doc_link(row)


def _load_metadata_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _normalize_doc_link(link: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(link or {})
    evidence_entities = item.get("evidence_entities")
    if not isinstance(evidence_entities, list):
        evidence_entities = []
    if not evidence_entities and item.get("evidence_entities_json"):
        try:
            decoded = json.loads(item.get("evidence_entities_json") or "[]")
            evidence_entities = list(decoded) if isinstance(decoded, list) else []
        except (TypeError, ValueError):
            evidence_entities = []
    item["query_used"] = str(item.get("query_used") or "")
    item["pattern_key"] = str(item.get("pattern_key") or "")
    item["doc_feedback"] = str(item.get("doc_feedback") or "").strip() or None
    item["helpful_count"] = int(item.get("helpful_count") or 0)
    item["not_helpful_count"] = int(item.get("not_helpful_count") or 0)
    item["feedback_score"] = float(item.get("feedback_score") or 0.0)
    item["evidence_entities"] = evidence_entities
    item.pop("evidence_entities_json", None)
    return item


def _doc_link_rank(link: Dict[str, Any]) -> Tuple[float, float]:
    return (
        float(link.get("feedback_score") or 0.0),
        float(link.get("score") or 0.0),
    )


def _doc_link_key(link: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(link.get("id") or "").strip(),
        str(link.get("page") or "").strip(),
    )


def _sort_doc_links(doc_links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [_normalize_doc_link(link) for link in doc_links if isinstance(link, dict)],
        key=_doc_link_rank,
        reverse=True,
    )


def _merge_doc_links(*doc_link_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_links in doc_link_groups:
        for raw_link in doc_links or []:
            link = _normalize_doc_link(raw_link)
            doc_id, _page = _doc_link_key(link)
            if not doc_id:
                continue
            key = _doc_link_key(link)
            current = best_by_key.get(key)
            if current is None or _doc_link_rank(link) > _doc_link_rank(current):
                best_by_key[key] = link
    return _sort_doc_links(list(best_by_key.values()))


def _doc_link_ids(doc_links: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for link in doc_links or []:
        doc_id = str((link or {}).get("id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)
    return ids


class Neo4jMemoryStore:
    """Neo4j-backed implementation of ``MemoryStoreProtocol``.

    All public methods are **synchronous** (the neo4j Python driver provides
    both sync and async sessions; we use sync here to stay compatible with
    the existing ``MemoryStore`` surface which the orchestrator, scorer and
    retriever call synchronously).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        connect_timeout_s: Optional[float] = None,
        max_pool_size: Optional[int] = None,
        max_transaction_retry_s: Optional[float] = None,
        graph_outbox_path: Optional[str] = None,
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'neo4j' package is required for Neo4jMemoryStore. "
                "Install with: pip install neo4j>=5.0"
            ) from exc

        driver_kwargs: Dict[str, Any] = {}
        if connect_timeout_s is not None:
            driver_kwargs["connection_timeout"] = connect_timeout_s
            driver_kwargs["connection_acquisition_timeout"] = connect_timeout_s
        if max_pool_size is not None and int(max_pool_size) > 0:
            driver_kwargs["max_connection_pool_size"] = int(max_pool_size)
        if max_transaction_retry_s is not None and float(max_transaction_retry_s) >= 0.0:
            driver_kwargs["max_transaction_retry_time"] = float(max_transaction_retry_s)

        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            **driver_kwargs,
        )
        self._database = database
        self._doc_link_feedback_lock = threading.Lock()
        self._graph_write_outbox = GraphWriteOutbox(graph_outbox_path) if graph_outbox_path else None

        # Warm-up: verify connectivity and ensure schema.
        # Schema bootstrap is idempotent but each CREATE still round-trips to
        # Neo4j; skip if we've already initialized this target in-process.
        _key = (uri, database)
        if _key not in _SCHEMA_INITIALIZED:
            self._ensure_schema()
            _SCHEMA_INITIALIZED.add(_key)
        else:
            # Still verify the driver can connect (cheap) so constructor errors
            # surface in the same place as before.
            try:
                with self._driver.session(database=self._database) as _s:
                    _s.run("RETURN 1").consume()
            except Exception:
                # Connection broke since initialization; re-run schema to be safe.
                _SCHEMA_INITIALIZED.discard(_key)
                self._ensure_schema()
                _SCHEMA_INITIALIZED.add(_key)

        self._flush_graph_write_outbox()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create constraints, indices and schema version node."""
        with self._driver.session(database=self._database) as session:
            # Unique constraints
            session.run(
                "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS "
                "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT pattern_key_unique IF NOT EXISTS "
                "FOR (p:Pattern) REQUIRE p.key IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT session_id_unique IF NOT EXISTS "
                "FOR (s:Session) REQUIRE s.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT discovered_pattern_key_unique IF NOT EXISTS "
                "FOR (dp:DiscoveredPattern) REQUIRE dp.key IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT experiment_run_id_unique IF NOT EXISTS "
                "FOR (e:Experiment) REQUIRE e.run_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT snapshot_id_unique IF NOT EXISTS "
                "FOR (sn:Snapshot) REQUIRE sn.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT operation_id_unique IF NOT EXISTS "
                "FOR (o:Operation) REQUIRE o.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT co_occurrence_update_id_unique IF NOT EXISTS "
                "FOR (cu:CoOccurrenceUpdate) REQUIRE cu.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS "
                "FOR (d:Dataset) REQUIRE d.id IS UNIQUE"
            )
            try:
                session.run(
                    "CREATE INDEX memory_usecase_idx IF NOT EXISTS "
                    "FOR (m:Memory) ON (m.usecase)"
                )
            except Exception:
                logger.debug("Memory usecase index creation skipped")

            try:
                session.run(
                    "CREATE INDEX pattern_prior_idx IF NOT EXISTS "
                    "FOR (p:Pattern) ON (p.prior)"
                )
            except Exception:
                logger.debug("Pattern prior index creation skipped")

            try:
                session.run(
                    "CREATE INDEX co_occurs_weight_idx IF NOT EXISTS "
                    "FOR ()-[r:CO_OCCURS_WITH]-() ON (r.weight)"
                )
            except Exception:
                logger.debug("CO_OCCURS_WITH weight index creation skipped")

            # Text index for annotation search
            try:
                session.run(
                    "CREATE TEXT INDEX memory_annotation_text IF NOT EXISTS "
                    "FOR (m:Memory) ON (m.annotation_text)"
                )
            except Exception:
                # Older Neo4j versions may not support text indices
                logger.debug("Text index creation skipped (not supported)")

            # Vector index for embeddings (Neo4j 5.11+)
            try:
                session.run(
                    "CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS "
                    "FOR (m:Memory) ON (m.embedding) "
                    "OPTIONS {indexConfig: {"
                    "  `vector.dimensions`: $dim,"
                    "  `vector.similarity_function`: 'cosine'"
                    "}}",
                    dim=_EMBEDDING_DIM,
                )
            except Exception:
                logger.warning(
                    "Vector index creation failed — vector search will be unavailable. "
                    "Requires Neo4j 5.11+ with vector index support."
                )

            # Schema version marker
            session.run(
                "MERGE (sv:SchemaVersion {domain: 'memory'}) "
                "SET sv.version = $v, sv.updated_at = $ts",
                v=NEO4J_SCHEMA_VERSION,
                ts=_now_iso(),
            )
            try:
                self._migrate_legacy_doc_links(session)
            except Exception:
                logger.warning("Legacy doc-link migration skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, cypher: str, **params: Any) -> Any:
        """Run a single Cypher statement and return the Result."""
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, **params).data()

    def _run_single(self, cypher: str, **params: Any) -> Optional[Dict[str, Any]]:
        rows = self._run(cypher, **params)
        return rows[0] if rows else None

    def _enqueue_graph_write(self, kind: str, payload: Dict[str, Any]) -> GraphWriteIntent:
        intent = GraphWriteIntent(kind=kind, payload=dict(payload or {}))
        outbox = getattr(self, "_graph_write_outbox", None)
        if not outbox:
            self._apply_graph_write_intent(intent)
            return intent
        try:
            outbox.append(intent)
        except OSError:
            logger.exception("Neo4j graph outbox append failed for %s; applying directly", kind)
            self._apply_graph_write_intent(intent)
            return intent
        self._flush_graph_write_outbox()
        return intent

    def _flush_graph_write_outbox(self, *, limit: int = 0) -> int:
        outbox = getattr(self, "_graph_write_outbox", None)
        if not outbox:
            return 0
        processed = 0
        for intent in outbox.iter_pending():
            try:
                self._apply_graph_write_intent(intent)
            except Exception:
                logger.warning(
                    "Neo4j graph outbox replay failed for kind=%s sequence=%s",
                    intent.kind,
                    intent.sequence,
                    exc_info=True,
                )
                break
            outbox.mark_acked(intent.sequence)
            processed += 1
            if limit > 0 and processed >= limit:
                break
        # Keep the append-only durability log from growing without bound over
        # long-running sessions; only rewrites once it has actually grown large.
        try:
            outbox.maybe_compact()
        except Exception:
            logger.debug("Graph outbox compaction skipped", exc_info=True)
        return processed

    def _apply_graph_write_intent(self, intent: GraphWriteIntent) -> None:
        if intent.kind == "trace":
            self._apply_trace_intent(intent.payload)
            return
        if intent.kind == "feedback_event":
            self._apply_feedback_event_intent(intent.payload)
            return
        if intent.kind == "co_occurrence":
            self._apply_co_occurrence_intent(intent)
            return
        if intent.kind == "doc_links":
            self._apply_doc_links_intent(intent.payload)
            return
        if intent.kind == "discovered_pattern":
            self._apply_discovered_pattern_intent(intent.payload)
            return
        if intent.kind == "experiment":
            self._apply_experiment_intent(intent.payload)
            return
        raise ValueError(f"Unsupported graph write intent kind: {intent.kind}")

    def _apply_trace_intent(self, payload: Dict[str, Any]) -> None:
        self._run(
            "MERGE (t:Trace {id: $tid}) "
            "SET t.session_id = $sid, t.memory_id = $mid, "
            "    t.trace_type = $tt, t.created_at = $ts, t.payload_json = $pj",
            tid=str(payload.get("trace_id") or ""),
            sid=payload.get("session_id"),
            mid=payload.get("memory_id"),
            tt=str(payload.get("trace_type") or ""),
            ts=str(payload.get("created_at") or _now_iso()),
            pj=json.dumps(payload.get("payload") or {}, sort_keys=True),
        )

    def _apply_feedback_event_intent(self, payload: Dict[str, Any]) -> None:
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Feedback graph write intent missing event_id")

        memory_id = payload.get("memory_id")
        action = str(payload.get("action") or "")
        user_id = str(payload.get("user_id") or "")
        timestamp = str(payload.get("timestamp") or _now_iso())
        context_key = payload.get("context_key")
        context = dict(payload.get("context") or {})
        data = dict(payload.get("data") or {})
        pattern_keys = [
            str(pattern_key).strip()
            for pattern_key in (payload.get("pattern_keys") or [])
            if str(pattern_key).strip()
        ]
        try:
            feedback_weight = max(0.0, float(payload.get("weight", 1.0)))
        except (TypeError, ValueError):
            feedback_weight = 1.0

        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> None:
                tx.run(
                    "MERGE (f:Feedback {id: $eid}) "
                    "SET f.memory_id = $mid, f.action = $action, "
                    "    f.user_id = $uid, f.timestamp = $ts, "
                    "    f.weight = $weight, f.context_key = $ck, "
                    "    f.context_json = $cj, f.data_json = $dj",
                    eid=event_id,
                    mid=memory_id,
                    action=action,
                    uid=user_id,
                    ts=timestamp,
                    weight=feedback_weight,
                    ck=context_key,
                    cj=json.dumps(context),
                    dj=json.dumps(data),
                )

                if memory_id:
                    tx.run(
                        "MATCH (f:Feedback {id: $eid}), (m:Memory {id: $mid}) "
                        "MERGE (f)-[:ABOUT]->(m)",
                        eid=event_id,
                        mid=memory_id,
                    )

                for pattern_key in pattern_keys:
                    tx.run(
                        "MATCH (f:Feedback {id: $eid}) "
                        "MERGE (p:Pattern {key: $pk}) "
                        "MERGE (f)-[:ON_PATTERN]->(p)",
                        eid=event_id,
                        pk=pattern_key,
                    )

            session.execute_write(_tx)

    def _apply_co_occurrence_intent(self, intent: GraphWriteIntent) -> None:
        payload = dict(intent.payload or {})
        intent_id = int(intent.sequence or 0)
        if intent_id <= 0:
            raise ValueError("Co-occurrence graph write intent missing outbox sequence")

        a = str(payload.get("pattern_key_a") or "").strip()
        b = str(payload.get("pattern_key_b") or "").strip()
        session_id = str(payload.get("session_id") or "")
        created_at = str(payload.get("created_at") or _now_iso())
        if not a or not b or a == b:
            return

        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> None:
                rows = tx.run(
                    "MERGE (cu:CoOccurrenceUpdate {id: $intent_id}) "
                    "ON CREATE SET cu.pattern_key_a = $a, cu.pattern_key_b = $b, "
                    "              cu.session_id = $sid, cu.created_at = $ts "
                    "RETURN cu.applied_at AS applied_at",
                    intent_id=intent_id,
                    a=a,
                    b=b,
                    sid=session_id,
                    ts=created_at,
                ).data()
                if rows and rows[0].get("applied_at"):
                    return

                applied_rows = tx.run(
                    "MATCH (pa:Pattern {key: $a}), (pb:Pattern {key: $b}) "
                    "MERGE (pa)-[r:CO_OCCURS_WITH]-(pb) "
                    "ON CREATE SET r.weight = 1, r.first_session = $sid, "
                    "             r.created_at = $ts "
                    "ON MATCH SET r.weight = r.weight + 1 "
                    "SET r.last_session = $sid, r.updated_at = $ts "
                    "RETURN 1 AS applied",
                    a=a,
                    b=b,
                    sid=session_id,
                    ts=created_at,
                ).data()
                if not applied_rows:
                    raise RuntimeError(
                        f"Co-occurrence replay could not match patterns: {a}, {b}"
                    )

                tx.run(
                    "MATCH (cu:CoOccurrenceUpdate {id: $intent_id}) "
                    "SET cu.applied_at = $ts",
                    intent_id=intent_id,
                    ts=created_at,
                )

            session.execute_write(_tx)

    def _apply_discovered_pattern_intent(self, payload: Dict[str, Any]) -> None:
        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> None:
                tx.run(
                    "MERGE (dp:DiscoveredPattern {key: $key}) "
                    "SET dp.features_json = $fj, "
                    "    dp.confirmation_count = $cc, "
                    "    dp.promoted = $prom, "
                    "    dp.prior = $prior, "
                    "    dp.first_seen = $fs, "
                    "    dp.last_seen = $ls, "
                    "    dp.updated_at = $now",
                    key=str(payload.get("key") or ""),
                    fj=json.dumps(payload.get("features") or {}),
                    cc=int(payload.get("confirmation_count") or 0),
                    prom=bool(payload.get("promoted")),
                    prior=float(payload.get("prior") or 0.0),
                    fs=str(payload.get("first_seen") or ""),
                    ls=str(payload.get("last_seen") or ""),
                    now=_now_iso(),
                )

                for memory_id in (payload.get("source_memory_ids") or []):
                    if not memory_id:
                        continue
                    tx.run(
                        "MATCH (dp:DiscoveredPattern {key: $key}), "
                        "      (m:Memory {id: $mid}) "
                        "MERGE (dp)-[:DISCOVERED_FROM]->(m)",
                        key=str(payload.get("key") or ""),
                        mid=str(memory_id),
                    )

                if bool(payload.get("promoted")):
                    tx.run(
                        "MERGE (p:Pattern {key: $key}) "
                        "ON CREATE SET p.pattern_type = 'cluster', "
                        "              p.fault_type = 'unknown' "
                        "WITH p "
                        "MATCH (dp:DiscoveredPattern {key: $key}) "
                        "MERGE (p)-[:EVOLVED_FROM]->(dp)",
                        key=str(payload.get("key") or ""),
                    )

            session.execute_write(_tx)

    def _apply_experiment_intent(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or "")
        experiment_type = str(payload.get("experiment_type") or "")
        session_ids = [
            str(session_id).strip()
            for session_id in (payload.get("session_ids") or [])
            if str(session_id).strip()
        ]

        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> None:
                tx.run(
                    "MERGE (e:Experiment {run_id: $rid}) "
                    "SET e.experiment_type = $etype, "
                    "    e.config_json = $cfg, "
                    "    e.test_f1 = $tf1, "
                    "    e.test_precision = $tp, "
                    "    e.test_recall = $tr, "
                    "    e.eval_f1 = $ef1, "
                    "    e.eval_precision = $ep, "
                    "    e.eval_recall = $er, "
                    "    e.delta_f1 = $df1, "
                    "    e.pct_improvement = $pct, "
                    "    e.created_at = $ts",
                    rid=run_id,
                    etype=experiment_type,
                    cfg=json.dumps(payload.get("config") or {}),
                    tf1=_safe_num(payload.get("test_metrics"), "f1"),
                    tp=_safe_num(payload.get("test_metrics"), "precision"),
                    tr=_safe_num(payload.get("test_metrics"), "recall"),
                    ef1=_safe_num(payload.get("eval_metrics"), "f1"),
                    ep=_safe_num(payload.get("eval_metrics"), "precision"),
                    er=_safe_num(payload.get("eval_metrics"), "recall"),
                    df1=_safe_num(payload.get("comparison"), "delta_f1"),
                    pct=_safe_num(payload.get("comparison"), "pct_f1_improvement"),
                    ts=str(payload.get("created_at") or _now_iso()),
                )

                for session_id in session_ids:
                    tx.run(
                        "MATCH (e:Experiment {run_id: $rid}) "
                        "MERGE (s:Session {id: $sid}) "
                        "MERGE (e)-[:HAS_SESSION]->(s) "
                        "MERGE (s)-[:IN_EXPERIMENT]->(e)",
                        rid=run_id,
                        sid=session_id,
                    )

                if session_ids:
                    tx.run(
                        "MATCH (e:Experiment {run_id: $rid}) "
                        "MATCH (s:Session)<-[:IN_SESSION]-(m:Memory)-[:HAS_PATTERN]->(p:Pattern) "
                        "WHERE s.id IN $sids "
                        "MERGE (e)-[:TESTED_PATTERN]->(p)",
                        rid=run_id,
                        sids=session_ids,
                    )

            session.execute_write(_tx)

    def _apply_doc_links_intent(self, payload: Dict[str, Any]) -> int:
        with self._driver.session(database=self._database) as session:
            return session.execute_write(lambda tx: self._apply_doc_links_metadata_intent(tx, payload))

    def _apply_doc_links_metadata_intent(self, tx: Any, payload: Dict[str, Any]) -> int:
        memory_id = str(payload.get("memory_id") or "").strip()
        if not memory_id:
            return 0

        valid_links = _sort_doc_links(
            [
                dict(link)
                for link in (payload.get("doc_links") or [])
                if str((link or {}).get("id") or "").strip()
            ]
        )
        if not valid_links:
            return 0

        rows = tx.run(
            "MATCH (m:Memory {id: $memory_id}) RETURN m.metadata_json AS metadata_json",
            memory_id=memory_id,
        ).data()
        if not rows:
            return 0

        metadata = _load_metadata_json(rows[0].get("metadata_json"))
        metadata["doc_links"] = valid_links
        tx.run(
            "MATCH (m:Memory {id: $memory_id}) "
            "SET m.metadata_json = $metadata_json, "
            "    m.doc_link_ids = $doc_link_ids, "
            "    m.updated_at = $updated_at "
            "RETURN 1 AS updated",
            memory_id=memory_id,
            metadata_json=json.dumps(metadata),
            doc_link_ids=_doc_link_ids(valid_links),
            updated_at=_now_iso(),
        ).consume()
        return len(valid_links)

    def _migrate_legacy_doc_links(self, session: Any) -> None:
        rows = session.run(
            "MATCH (m:Memory)-[r:CITES]->(d:Document) "
            "RETURN m.id AS memory_id, "
            "       m.metadata_json AS metadata_json, "
            "       d.id AS id, "
            "       r.citation AS citation, "
            "       r.score AS score, "
            "       coalesce(r.page, d.page) AS page, "
            "       d.file_name AS file_name, "
            "       d.source AS source, "
            "       d.usecase AS usecase, "
            "       d.machine AS machine, "
            "       d.text AS text, "
            "       d.document_type AS document_type, "
            "       coalesce(d.language_code, d.original_language, properties(d)['language']) AS language, "
            "       r.query AS query_used, "
            "       r.pattern_key AS pattern_key, "
            "       r.doc_feedback AS doc_feedback, "
            "       coalesce(r.helpful_count, 0) AS helpful_count, "
            "       coalesce(r.not_helpful_count, 0) AS not_helpful_count, "
            "       coalesce(r.feedback_score, 0.0) AS feedback_score, "
            "       r.evidence_entities_json AS evidence_entities_json"
        ).data()
        if not rows:
            return

        by_memory: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            memory_id = str(row.get("memory_id") or "").strip()
            if not memory_id:
                continue
            bucket = by_memory.setdefault(
                memory_id,
                {
                    "metadata": _load_metadata_json(row.get("metadata_json")),
                    "doc_links": [],
                },
            )
            bucket["doc_links"].append(_deserialize_doc_link_row(row))

        migrated_links = 0
        for memory_id, payload in by_memory.items():
            metadata = dict(payload.get("metadata") or {})
            merged_links = _merge_doc_links(
                metadata.get("doc_links") if isinstance(metadata.get("doc_links"), list) else [],
                payload.get("doc_links") or [],
            )
            metadata["doc_links"] = merged_links
            session.run(
                "MATCH (m:Memory {id: $memory_id}) "
                "SET m.metadata_json = $metadata_json, "
                "    m.doc_link_ids = $doc_link_ids, "
                "    m.updated_at = $updated_at",
                memory_id=memory_id,
                metadata_json=json.dumps(metadata),
                doc_link_ids=_doc_link_ids(merged_links),
                updated_at=_now_iso(),
            ).consume()
            migrated_links += len(payload.get("doc_links") or [])

        session.run("MATCH ()-[r:CITES]->(:Document) DELETE r").consume()
        session.run("MATCH ()-[r:DOCUMENTED_BY]->(:Document) DELETE r").consume()
        logger.info(
            "Migrated %s legacy doc links into memory metadata across %s memories",
            migrated_links,
            len(by_memory),
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_memory(memory: Memory) -> Dict[str, Any]:
        """Flatten a Memory to a dict of Neo4j-safe properties."""
        metadata = dict(memory.metadata or {})
        if "doc_links" in metadata:
            raw_doc_links = metadata.get("doc_links")
            metadata["doc_links"] = _sort_doc_links(raw_doc_links if isinstance(raw_doc_links, list) else [])
        doc_link_ids = _doc_link_ids(metadata.get("doc_links") if isinstance(metadata.get("doc_links"), list) else [])
        if memory.numeric_vector is not None:
            metadata["_numeric_vector"] = memory.numeric_vector
        # NOTE: text_embedding is stored as the native 'embedding' property
        # on the node (used by the vector index).  Do NOT duplicate it into
        # metadata_json — that would add ~3 KB of redundant JSON per memory.

        # Agent B (2026-04-24): expose SINDIT machine URI as a top-level
        # property on Memory nodes so history queries can be scoped per
        # asset without parsing metadata_json. Sourced from the cutting
        # context's machine_uri / asset_iri fields if present.
        cutting_ctx = metadata.get("cutting_context") if isinstance(metadata.get("cutting_context"), dict) else metadata
        cutting_extra = cutting_ctx.get("extra") if isinstance(cutting_ctx.get("extra"), dict) else {}
        casedata = metadata.get("casedata") if isinstance(metadata.get("casedata"), dict) else {}
        machine_uri = None
        if isinstance(cutting_ctx, dict):
            machine_uri = (
                cutting_ctx.get("machine_uri")
                or cutting_ctx.get("asset_iri")
                or cutting_ctx.get("machine_iri")
            )
        if not machine_uri and memory.machine_uri:
            machine_uri = memory.machine_uri
        usecase = resolve_usecase(
            metadata=metadata,
            machine_uri=machine_uri or memory.machine_uri,
            fallback_generic=False,
        )
        operation_id = casedata.get("operation_id")
        dataset_id = metadata.get("dataset_id") or casedata.get("dataset_id")
        source_dataset_id = metadata.get("source_dataset_id")
        case_dir = casedata.get("case_dir")
        machine_family = metadata.get("machine_family") or cutting_extra.get("machine_family")
        machine_iri = metadata.get("machine_iri") or cutting_ctx.get("machine_iri") or cutting_ctx.get("asset_iri")
        sindit_asset_iri = metadata.get("sindit_asset_iri") or machine_uri or memory.machine_uri
        sindit_tool_iri = cutting_extra.get("sindit_tool_iri") or metadata.get("sindit_tool_iri")
        operation_node_id = _operation_node_id(operation_id, dataset_id, case_dir)

        return {
            "id": memory.id or str(uuid.uuid4()),
            "session_id": memory.session_id,
            "annotation_text": memory.annotation_text,
            "pattern_keys_json": json.dumps(
                [_serialize_pattern_key(pk) for pk in memory.pattern_keys]
            ) if memory.pattern_keys else "[]",
            "metrics_json": memory.metrics.model_dump_json() if memory.metrics else None,
            "time_range_json": _serialize_time_range(memory.time_range),
            "channels_json": json.dumps(memory.channels) if memory.channels else "[]",
            "tags_json": json.dumps(memory.tags) if memory.tags else "[]",
            "label": memory.label,
            "provenance_json": memory.provenance.model_dump_json() if memory.provenance else None,
            "metadata_json": json.dumps(metadata) if metadata else "{}",
            "doc_link_ids": doc_link_ids,
            "machine_uri": machine_uri,
            "usecase": usecase,
            "operation_id": str(operation_id) if operation_id not in (None, "") else None,
            "operation_node_id": operation_node_id,
            "dataset_id": str(dataset_id) if dataset_id not in (None, "") else None,
            "source_dataset_id": str(source_dataset_id) if source_dataset_id not in (None, "") else None,
            "case_dir": str(case_dir) if case_dir not in (None, "") else None,
            "machine_family": str(machine_family) if machine_family not in (None, "") else None,
            "machine_iri": str(machine_iri) if machine_iri not in (None, "") else None,
            "sindit_asset_iri": str(sindit_asset_iri) if sindit_asset_iri not in (None, "") else None,
            "sindit_tool_iri": str(sindit_tool_iri) if sindit_tool_iri not in (None, "") else None,
            "visibility": memory.visibility or "active",
            "created_at": memory.created_at.isoformat() if memory.created_at else _now_iso(),
            "updated_at": _now_iso(),
            "created_by": memory.created_by or "operator",
            "embedding": memory.text_embedding,
        }

    @staticmethod
    def _deserialize_memory(props: Dict[str, Any]) -> Memory:
        """Reconstruct a Memory from Neo4j node properties."""
        pattern_keys_raw = props.get("pattern_keys_json", "[]")
        pattern_keys = [PatternKey(**pk) for pk in json.loads(pattern_keys_raw)]
        for pk in pattern_keys:
            additional = dict(pk.additional or {})
            additional.setdefault(
                "kind",
                infer_pattern_kind(pk.key, pk.pattern_type.value if pk.pattern_type else None),
            )
            pk.additional = additional

        provenance = MemoryProvenance()
        prov_raw = props.get("provenance_json")
        if prov_raw:
            provenance = MemoryProvenance(**json.loads(prov_raw))

        metadata = json.loads(props.get("metadata_json") or "{}")
        numeric_vector = metadata.get("_numeric_vector")
        # text_embedding is stored as the native node property 'embedding'
        # (used by the vector index), not in metadata_json.
        text_embedding = props.get("embedding") or metadata.get("_text_embedding")

        time_range = _deserialize_time_range(props.get("time_range_json"))

        created_at_str = props.get("created_at", _now_iso())
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        return Memory(
            id=props["id"],
            session_id=props.get("session_id", ""),
            annotation_text=props.get("annotation_text", ""),
            pattern_keys=pattern_keys,
            metrics=(
                NumericMetrics(**json.loads(props["metrics_json"]))
                if props.get("metrics_json")
                else NumericMetrics()
            ),
            time_range=time_range,
            channels=json.loads(props.get("channels_json") or "[]"),
            tags=json.loads(props.get("tags_json") or "[]"),
            label=props.get("label"),
            provenance=provenance,
            metadata=metadata,
            numeric_vector=numeric_vector,
            text_embedding=text_embedding,
            visibility=props.get("visibility", "active"),
            created_at=created_at,
            created_by=props.get("created_by", "operator"),
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def store(self, memory: Memory) -> str:
        if not memory.id:
            memory.id = str(uuid.uuid4())
        if not memory.created_at:
            memory.created_at = datetime.now(timezone.utc)

        props = self._serialize_memory(memory)

        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> str:
                # Upsert Memory node
                tx.run(
                    "MERGE (m:Memory {id: $id}) "
                    "SET m += $props",
                    id=props["id"],
                    props={k: v for k, v in props.items() if k != "embedding"},
                )

                # Set embedding separately (list property)
                if props.get("embedding"):
                    tx.run(
                        "MATCH (m:Memory {id: $id}) SET m.embedding = $emb",
                        id=props["id"],
                        emb=props["embedding"],
                    )

                # Ensure Session node and link
                tx.run(
                    "MERGE (s:Session {id: $sid}) "
                    "WITH s "
                    "MATCH (m:Memory {id: $mid}) "
                    "MERGE (m)-[:IN_SESSION]->(s)",
                    sid=memory.session_id,
                    mid=props["id"],
                )

                # Pattern nodes and links
                tx.run(
                    "MATCH (m:Memory {id: $mid})-[r:HAS_PATTERN]->() DELETE r",
                    mid=props["id"],
                )
                for pk in memory.pattern_keys:
                    kind = _pattern_kind(pk)
                    tx.run(
                        "MERGE (p:Pattern {key: $key}) "
                        "SET p.pattern_type = $pt, p.fault_type = $ft, p.kind = $kind "
                        "WITH p "
                        "MATCH (m:Memory {id: $mid}) "
                        "MERGE (m)-[r:HAS_PATTERN]->(p) "
                        "SET r.strength = $strength, r.source_metric = $source_metric",
                        key=pk.key,
                        pt=pk.pattern_type.value if pk.pattern_type else "custom",
                        ft=pk.fault_type,
                        kind=kind,
                        mid=props["id"],
                        strength=float(getattr(pk, "confidence", 1.0) or 1.0),
                        source_metric=getattr(pk, "source_metric", None),
                    )

                # --- Machine / Tool nodes (physical context) --------
                cutting_ctx = json.loads(props.get("metadata_json") or "{}")
                if not isinstance(cutting_ctx, dict):
                    cutting_ctx = {}
                # metadata may carry an explicit "cutting_context": null (set by the
                # orchestrator when no cutting context is available, e.g. SINDIT
                # disabled/unreachable) — fall back to {} rather than crashing on None.
                nested = cutting_ctx.get("cutting_context", cutting_ctx)
                cutting_ctx = nested if isinstance(nested, dict) else {}
                machine_id = cutting_ctx.get("machine_id") or cutting_ctx.get("machine_type")
                tool_id = cutting_ctx.get("tool_id") or cutting_ctx.get("tool_type")

                if machine_id:
                    tx.run(
                        "MERGE (ma:Machine {id: $mid}) "
                        "ON CREATE SET ma.machine_type = $mt "
                        "WITH ma "
                        "MATCH (m:Memory {id: $memid}) "
                        "MERGE (m)-[:ON_MACHINE]->(ma)",
                        mid=str(machine_id),
                        mt=cutting_ctx.get("machine_type") or str(machine_id),
                        memid=props["id"],
                    )

                if tool_id:
                    tx.run(
                        "MERGE (t:Tool {id: $tid}) "
                        "ON CREATE SET t.tool_type = $tt, t.num_teeth = $nt, "
                        "    t.tool_diameter = $td "
                        "WITH t "
                        "MATCH (m:Memory {id: $memid}) "
                        "MERGE (m)-[:USED_TOOL]->(t)",
                        tid=str(tool_id),
                        tt=cutting_ctx.get("tool_type"),
                        nt=cutting_ctx.get("num_teeth"),
                        td=cutting_ctx.get("tool_diameter") or (cutting_ctx.get("extra") or {}).get("d"),
                        memid=props["id"],
                    )

                dataset_id = props.get("dataset_id")
                source_dataset_id = props.get("source_dataset_id")
                operation_id = props.get("operation_id")
                operation_node_id = props.get("operation_node_id")
                case_dir = props.get("case_dir")

                if dataset_id:
                    tx.run(
                        "MERGE (ds:Dataset {id: $dataset_id}) "
                        "SET ds.usecase = coalesce($usecase, ds.usecase), "
                        "    ds.source_dataset_id = coalesce($source_dataset_id, ds.source_dataset_id)",
                        dataset_id=str(dataset_id),
                        usecase=props.get("usecase"),
                        source_dataset_id=str(source_dataset_id) if source_dataset_id else None,
                    )

                if operation_id and operation_node_id:
                    tx.run(
                        "MERGE (op:Operation {id: $operation_node_id}) "
                        "SET op.operation_id = $operation_id, "
                        "    op.case_dir = $case_dir, "
                        "    op.dataset_id = $dataset_id "
                        "WITH op "
                        "MATCH (m:Memory {id: $memory_id}) "
                        "MERGE (m)-[:DURING]->(op)",
                        operation_node_id=str(operation_node_id),
                        operation_id=str(operation_id),
                        case_dir=str(case_dir) if case_dir else None,
                        dataset_id=str(dataset_id) if dataset_id else None,
                        memory_id=props["id"],
                    )
                    if dataset_id:
                        tx.run(
                            "MATCH (op:Operation {id: $operation_node_id}) "
                            "MATCH (ds:Dataset {id: $dataset_id}) "
                            "MERGE (op)-[:OF_DATASET]->(ds)",
                            operation_node_id=str(operation_node_id),
                            dataset_id=str(dataset_id),
                        )

                # --- Temporal edge: [:NEXT] -------------------------
                # Link this memory to the previous memory in the same
                # session so event sequences are graph-traversable.
                tx.run(
                    "MATCH (prev:Memory)-[:IN_SESSION]->(:Session {id: $sid}) "
                    "WHERE prev.id <> $mid "
                    "WITH prev ORDER BY prev.created_at DESC LIMIT 1 "
                    "MATCH (cur:Memory {id: $mid}) "
                    "MERGE (prev)-[:NEXT]->(cur)",
                    sid=memory.session_id,
                    mid=props["id"],
                )

                return props["id"]

            return session.execute_write(_tx)

    def persist_doc_links(
        self,
        *,
        memory_id: str,
        pattern_keys: List[str],
        doc_links: List[Dict[str, Any]],
    ) -> int:
        payload = normalize_doc_link_intent(
            memory_id=memory_id,
            pattern_keys=pattern_keys,
            doc_links=doc_links,
        )
        valid_links = list(payload["doc_links"])
        if not valid_links:
            return 0
        try:
            linked = self._apply_doc_links_intent(payload)
            self._flush_graph_write_outbox()
            return linked
        except Exception:
            outbox = getattr(self, "_graph_write_outbox", None)
            if not outbox:
                raise
            logger.warning(
                "Neo4j doc-link persistence failed for memory=%s; queued for replay",
                memory_id,
                exc_info=True,
            )
            outbox.append(GraphWriteIntent(kind="doc_links", payload=payload))
            return len(valid_links)

    def get_doc_links(
        self,
        memory_id: str,
        *,
        score_floor: float = 0.0,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        row = self._run_single(
            "MATCH (m:Memory {id: $memory_id}) RETURN m.metadata_json AS metadata_json",
            memory_id=memory_id,
        )
        if not row:
            return []
        metadata = _load_metadata_json(row.get("metadata_json"))
        doc_links = _sort_doc_links(
            [
                dict(link)
                for link in (metadata.get("doc_links") or [])
                if isinstance(link, dict)
            ]
        )
        filtered: List[Dict[str, Any]] = []
        for link in doc_links:
            try:
                score = float(link.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score >= float(score_floor):
                filtered.append(link)
        if limit > 0:
            filtered = filtered[: int(limit)]
        return filtered

    def set_doc_link_feedback(
        self,
        *,
        memory_id: str,
        doc_id: str,
        feedback: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        feedback_value = str(feedback or "").strip().lower()
        if feedback_value not in {"helpful", "not_helpful"}:
            raise ValueError("feedback must be 'helpful' or 'not_helpful'")

        helpful_delta = 1 if feedback_value == "helpful" else 0
        not_helpful_delta = 1 if feedback_value == "not_helpful" else 0

        def _write_feedback() -> Optional[Dict[str, Any]]:
            with self._driver.session(database=self._database) as session:

                def _tx(tx: Any) -> Optional[Dict[str, Any]]:
                    rows = tx.run(
                        "MATCH (m:Memory {id: $memory_id}) RETURN m.metadata_json AS metadata_json",
                        memory_id=memory_id,
                    ).data()
                    if not rows:
                        return None

                    metadata = _load_metadata_json(rows[0].get("metadata_json"))
                    doc_links = [
                        dict(link)
                        for link in (metadata.get("doc_links") or [])
                        if isinstance(link, dict)
                    ]
                    updated_link: Optional[Dict[str, Any]] = None
                    for index, link in enumerate(doc_links):
                        if str(link.get("id") or "") != str(doc_id):
                            continue
                        current = _normalize_doc_link(link)
                        if helpful_delta:
                            current["helpful_count"] = int(current.get("helpful_count") or 0) + helpful_delta
                        if not_helpful_delta:
                            current["not_helpful_count"] = int(current.get("not_helpful_count") or 0) + not_helpful_delta
                        current["feedback_score"] = float(current.get("helpful_count") or 0) - float(current.get("not_helpful_count") or 0)
                        current["doc_feedback"] = feedback_value
                        current["doc_feedback_user_id"] = str(user_id or "")
                        current["doc_feedback_reason"] = reason
                        current["doc_feedback_updated_at"] = _now_iso()
                        doc_links[index] = current
                        updated_link = current
                        break
                    if updated_link is None:
                        return None

                    metadata["doc_links"] = _sort_doc_links(doc_links)
                    tx.run(
                        "MATCH (m:Memory {id: $memory_id}) "
                        "SET m.metadata_json = $metadata_json, "
                        "    m.doc_link_ids = $doc_link_ids, "
                        "    m.updated_at = $updated_at",
                        memory_id=memory_id,
                        metadata_json=json.dumps(metadata),
                        doc_link_ids=_doc_link_ids(metadata["doc_links"]),
                        updated_at=_now_iso(),
                    ).consume()
                    return dict(updated_link)

                return session.execute_write(_tx)

        lock = getattr(self, "_doc_link_feedback_lock", None)
        if lock is None:
            return _write_feedback()
        with lock:
            return _write_feedback()

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._run_single(
            "MATCH (m:Memory {id: $id}) RETURN properties(m) AS p",
            id=memory_id,
        )
        if not row:
            return None
        return self._deserialize_memory(row["p"])

    def update(self, memory_id: str, updates: Optional[Dict[str, Any]] = None, **fields: Any) -> bool:
        """Update a memory by ID using a dict (preferred) or **kwargs."""
        merged: Dict[str, Any] = dict(updates) if updates else {}
        merged.update(fields)
        if not merged:
            return False

        memory = self.get(memory_id)
        if memory is None:
            return False

        for key, value in merged.items():
            setattr(memory, key, value)

        self.store(memory)
        return True

    def delete(self, memory_id: str) -> bool:
        result = self._run(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m RETURN count(m) AS c",
            id=memory_id,
        )
        return bool(result and result[0].get("c", 0) > 0)

    # ------------------------------------------------------------------
    # Queries / listing
    # ------------------------------------------------------------------

    def list_all(self, limit: int = 1000, visibility: str = "active") -> List[Memory]:
        if visibility:
            rows = self._run(
                "MATCH (m:Memory) WHERE m.visibility = $vis "
                "RETURN properties(m) AS p ORDER BY m.created_at DESC LIMIT $lim",
                vis=visibility,
                lim=limit,
            )
        else:
            rows = self._run(
                "MATCH (m:Memory) RETURN properties(m) AS p "
                "ORDER BY m.created_at DESC LIMIT $lim",
                lim=limit,
            )
        return [self._deserialize_memory(r["p"]) for r in rows]

    def list_by_session(self, session_id: str, limit: int = 100) -> List[Memory]:
        rows = self._run(
            "MATCH (m:Memory)-[:IN_SESSION]->(s:Session {id: $sid}) "
            "RETURN properties(m) AS p ORDER BY m.created_at DESC LIMIT $lim",
            sid=session_id,
            lim=limit,
        )
        return [self._deserialize_memory(r["p"]) for r in rows]

    def search(
        self,
        text_query: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        if session_id:
            conditions.append("m.session_id = $sid")
            params["sid"] = session_id

        if text_query:
            conditions.append("m.annotation_text CONTAINS $q")
            params["q"] = text_query

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        cypher = f"MATCH (m:Memory){where} RETURN properties(m) AS p ORDER BY m.created_at DESC"
        rows = self._run(cypher, **params)
        memories = [self._deserialize_memory(r["p"]) for r in rows]

        if time_range is None:
            return memories

        q0, q1 = float(time_range[0]), float(time_range[1])
        filtered: List[Memory] = []
        for mem in memories:
            tr = mem.time_range
            if isinstance(tr, tuple) and len(tr) == 2:
                m0, m1 = float(tr[0]), float(tr[1])
            elif isinstance(tr, TimeRange):
                m0, m1 = float(tr.t0), float(tr.t1)
            else:
                continue
            if m0 <= q1 and m1 >= q0:
                filtered.append(mem)
        return filtered

    # ------------------------------------------------------------------
    # Resolution history — "did we see this before and how was it resolved?"
    # Agent E (2026-04-24): feeds ExplanationContext.similar_memories and the
    # /memory/{id}/resolution_chain endpoint.
    # ------------------------------------------------------------------

    def get_similar_with_resolution(
        self,
        memory_id: str,
        k: int = 5,
        *,
        include_dismissed: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` similar memories with their feedback resolution.

        "Similar" means one of:
          1. Connected via a ``:SIMILAR_TO`` edge (cosine-threshold at write time).
          2. Shares at least one ``:Pattern`` with the query memory.

        For each similar memory the result includes:
          - id, session_id, created_at, annotation_text, label
          - shared_pattern_keys (List[str])
                    - shared_pattern_details (List[dict]) with query/candidate strength
                        and source-metric metadata when available
          - feedback: {confirm_count, dismiss_count, last_action, last_comment,
                       last_action_ts}

        Results are ordered by a cheap heuristic: (confirm_count desc,
        shared_pattern_count desc, created_at desc). This is good enough for
        "show operator the last N similar cases" UX; a proper ranker is a
        future improvement.
        """
        cypher = """
        MATCH (q:Memory {id: $mid})
        OPTIONAL MATCH (q)-[:SIMILAR_TO]-(s1:Memory)
        WITH q, collect(DISTINCT s1) AS via_similar
        OPTIONAL MATCH (q)-[:HAS_PATTERN]->(p:Pattern)<-[:HAS_PATTERN]-(s2:Memory)
          WHERE s2.id <> q.id
                WITH q, via_similar, collect(DISTINCT s2) AS via_pattern
                WITH q, [m IN via_similar + via_pattern WHERE m IS NOT NULL] AS cands
        UNWIND cands AS c
                WITH DISTINCT c AS cand, q
                OPTIONAL MATCH (q)-[qr:HAS_PATTERN]->(shared_p:Pattern)<-[cr:HAS_PATTERN]-(cand)
                WITH cand,
                         collect(DISTINCT shared_p.key) AS shared_patterns,
                         collect(DISTINCT CASE
                             WHEN shared_p IS NULL THEN NULL
                             ELSE {
                                 key: shared_p.key,
                                 query_strength: coalesce(qr.strength, 1.0),
                                 query_source_metric: qr.source_metric,
                                 candidate_strength: coalesce(cr.strength, 1.0),
                                 candidate_source_metric: cr.source_metric
                             }
                         END) AS shared_pattern_details
        OPTIONAL MATCH (f:Feedback)-[:ABOUT]->(cand)
                WITH cand, shared_patterns, shared_pattern_details,
             collect(f) AS feedbacks
        RETURN
          properties(cand) AS mem_props,
                    [k IN shared_patterns WHERE k IS NOT NULL] AS shared_patterns,
                    [d IN shared_pattern_details WHERE d IS NOT NULL] AS shared_pattern_details,
          feedbacks
        LIMIT $k_outer
        """
        rows = self._run(cypher, mid=memory_id, k_outer=max(int(k) * 4, 20))

        enriched: List[Dict[str, Any]] = []
        for row in rows:
            mem_props = row.get("mem_props") or {}
            shared = list(row.get("shared_patterns") or [])
            shared_pattern_details = [
                detail
                for detail in list(row.get("shared_pattern_details") or [])
                if isinstance(detail, dict) and detail.get("key")
            ]
            feedbacks = row.get("feedbacks") or []

            confirm_count = 0
            dismiss_count = 0
            last_action: Optional[str] = None
            last_comment: Optional[str] = None
            last_action_ts: Optional[str] = None

            # Sort feedbacks by timestamp if available.
            def _fb_ts(fb: Any) -> str:
                props = dict(fb) if not isinstance(fb, dict) else fb
                return str(props.get("timestamp") or props.get("created_at") or "")

            sorted_fbs = sorted(feedbacks, key=_fb_ts)
            for fb in sorted_fbs:
                props = dict(fb) if not isinstance(fb, dict) else fb
                action = props.get("action") or props.get("type")
                if action == "confirm":
                    confirm_count += 1
                elif action == "dismiss":
                    dismiss_count += 1
                last_action = action or last_action
                cmt = props.get("comment")
                if cmt:
                    last_comment = cmt
                ts = _fb_ts(fb)
                if ts:
                    last_action_ts = ts

            if not include_dismissed and last_action == "dismiss" and confirm_count == 0:
                continue

            enriched.append({
                "id": mem_props.get("id"),
                "session_id": mem_props.get("session_id"),
                "created_at": mem_props.get("created_at"),
                "annotation_text": mem_props.get("annotation_text"),
                "label": mem_props.get("label"),
                "machine_uri": mem_props.get("machine_uri"),
                "shared_pattern_keys": shared,
                "shared_pattern_details": shared_pattern_details,
                "feedback": {
                    "confirm_count": confirm_count,
                    "dismiss_count": dismiss_count,
                    "last_action": last_action,
                    "last_comment": last_comment,
                    "last_action_ts": last_action_ts,
                },
            })

        # Heuristic ranking: confirmed cases first, more shared patterns next,
        # then recency (ISO-8601 strings sort lexicographically as dates).
        enriched.sort(
            key=lambda e: (
                int(e["feedback"]["confirm_count"]),
                len(e["shared_pattern_keys"]),
                e["created_at"] or "",
            ),
            reverse=True,
        )

        return enriched[: int(k)]

    def count(self) -> int:
        row = self._run_single("MATCH (m:Memory) RETURN count(m) AS c")
        return int(row["c"]) if row else 0

    def stats(self) -> Dict[str, Any]:
        total = self.count()
        session_row = self._run_single(
            "MATCH (m:Memory) RETURN count(DISTINCT m.session_id) AS c"
        )
        sessions = int(session_row["c"]) if session_row else 0
        return {
            "total_memories": total,
            "unique_sessions": sessions,
            "backend": "neo4j",
        }

    def clear(self) -> None:
        """Delete all Memory-domain nodes (label-scoped, safe for shared DB)."""
        self.clear_memory_graph()

    def preview_memory_graph_cleanup(self) -> Dict[str, Any]:
        return collect_memory_graph_cleanup_preview(self._run)

    def clear_memory_graph(self) -> Dict[str, int]:
        labels = [
            "Memory", "Pattern", "Session", "Feedback", "Trace",
            "DiscoveredPattern", "Experiment", "Machine", "Tool", "Snapshot",
            "CoOccurrenceUpdate",
        ]
        counts: Dict[str, int] = {}
        for label in labels:
            rows = self._run(
                f"MATCH (n:{label}) WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c"
            )
            counts[label] = int(rows[0]["c"]) if rows else 0
        return counts

    def clear_legacy_candidate_memories(self) -> Dict[str, int]:
        predicate = legacy_memory_candidate_predicate("m")
        session_rows = self._run(
            "MATCH (m:Memory) "
            f"WHERE {predicate} "
            "RETURN DISTINCT m.session_id AS sid"
        )
        candidate_session_ids = [
            str(row.get("sid") or "").strip()
            for row in session_rows
            if str(row.get("sid") or "").strip()
        ]
        pattern_rows = self._run(
            "MATCH (m:Memory)-[:HAS_PATTERN]->(p:Pattern) "
            f"WHERE {predicate} "
            "RETURN DISTINCT p.key AS pattern_key"
        )
        candidate_pattern_keys = [
            str(row.get("pattern_key") or "").strip()
            for row in pattern_rows
            if str(row.get("pattern_key") or "").strip()
        ]
        machine_rows = self._run(
            "MATCH (m:Memory)-[:ON_MACHINE]->(ma:Machine) "
            f"WHERE {predicate} "
            "RETURN DISTINCT ma.id AS machine_id"
        )
        candidate_machine_ids = [
            str(row.get("machine_id") or "").strip()
            for row in machine_rows
            if str(row.get("machine_id") or "").strip()
        ]
        tool_rows = self._run(
            "MATCH (m:Memory)-[:USED_TOOL]->(t:Tool) "
            f"WHERE {predicate} "
            "RETURN DISTINCT t.id AS tool_id"
        )
        candidate_tool_ids = [
            str(row.get("tool_id") or "").strip()
            for row in tool_rows
            if str(row.get("tool_id") or "").strip()
        ]
        experiment_rows = self._run(
            "MATCH (m:Memory)-[:IN_SESSION]->(:Session)-[:IN_EXPERIMENT]->(e:Experiment) "
            f"WHERE {predicate} "
            "RETURN DISTINCT e.run_id AS run_id"
        )
        candidate_run_ids = [
            str(row.get("run_id") or "").strip()
            for row in experiment_rows
            if str(row.get("run_id") or "").strip()
        ]

        counts: Dict[str, int] = {}
        rows = self._run(
            "MATCH (m:Memory) "
            f"WHERE {predicate} "
            "MATCH (f:Feedback)-[:ABOUT]->(m) "
            "WITH DISTINCT f LIMIT 50000 DETACH DELETE f RETURN count(f) AS c"
        )
        counts["Feedback"] = int(rows[0]["c"]) if rows else 0

        rows = self._run(
            "MATCH (m:Memory) "
            f"WHERE {predicate} "
            "MATCH (t:Trace) WHERE t.memory_id = m.id "
            "WITH DISTINCT t LIMIT 50000 DETACH DELETE t RETURN count(t) AS c"
        )
        counts["Trace"] = int(rows[0]["c"]) if rows else 0

        rows = self._run(
            "MATCH (m:Memory) "
            f"WHERE {predicate} "
            "WITH m LIMIT 50000 DETACH DELETE m RETURN count(m) AS c"
        )
        counts["Memory"] = int(rows[0]["c"]) if rows else 0

        if candidate_session_ids:
            rows = self._run(
                "MATCH (cu:CoOccurrenceUpdate) "
                "WHERE cu.session_id IN $sids "
                "  AND NOT EXISTS { MATCH (:Memory)-[:IN_SESSION]->(:Session {id: cu.session_id}) } "
                "WITH cu LIMIT 50000 DETACH DELETE cu RETURN count(cu) AS c",
                sids=candidate_session_ids,
            )
            counts["CoOccurrenceUpdate"] = int(rows[0]["c"]) if rows else 0

            rows = self._run(
                "MATCH (s:Session) WHERE s.id IN $sids "
                "  AND NOT EXISTS { MATCH (:Memory)-[:IN_SESSION]->(s) } "
                "WITH s LIMIT 50000 DETACH DELETE s RETURN count(s) AS c",
                sids=candidate_session_ids,
            )
            counts["Session"] = int(rows[0]["c"]) if rows else 0
        else:
            counts["CoOccurrenceUpdate"] = 0
            counts["Session"] = 0

        if candidate_run_ids:
            rows = self._run(
                "MATCH (e:Experiment) WHERE e.run_id IN $run_ids "
                "  AND NOT EXISTS { MATCH (e)-[:HAS_SESSION]->(:Session) } "
                "WITH e LIMIT 50000 DETACH DELETE e RETURN count(e) AS c",
                run_ids=candidate_run_ids,
            )
            counts["Experiment"] = int(rows[0]["c"]) if rows else 0
        else:
            counts["Experiment"] = 0

        if candidate_machine_ids:
            rows = self._run(
                "MATCH (ma:Machine) WHERE ma.id IN $machine_ids "
                "  AND NOT EXISTS { MATCH (:Memory)-[:ON_MACHINE]->(ma) } "
                "WITH ma LIMIT 50000 DETACH DELETE ma RETURN count(ma) AS c",
                machine_ids=candidate_machine_ids,
            )
            counts["Machine"] = int(rows[0]["c"]) if rows else 0
        else:
            counts["Machine"] = 0

        if candidate_tool_ids:
            rows = self._run(
                "MATCH (t:Tool) WHERE t.id IN $tool_ids "
                "  AND NOT EXISTS { MATCH (:Memory)-[:USED_TOOL]->(t) } "
                "WITH t LIMIT 50000 DETACH DELETE t RETURN count(t) AS c",
                tool_ids=candidate_tool_ids,
            )
            counts["Tool"] = int(rows[0]["c"]) if rows else 0
        else:
            counts["Tool"] = 0

        if candidate_pattern_keys:
            rows = self._run(
                "MATCH (p:Pattern) WHERE p.key IN $pattern_keys "
                "  AND NOT EXISTS { MATCH (:Memory)-[:HAS_PATTERN]->(p) } "
                "  AND NOT EXISTS { MATCH (:Feedback)-[:ON_PATTERN]->(p) } "
                "  AND NOT EXISTS { MATCH (:Experiment)-[:TESTED_PATTERN]->(p) } "
                "WITH p LIMIT 50000 DETACH DELETE p RETURN count(p) AS c",
                pattern_keys=candidate_pattern_keys,
            )
            counts["Pattern"] = int(rows[0]["c"]) if rows else 0
        else:
            counts["Pattern"] = 0
        return counts

    def subgraph_integrity(self) -> Dict[str, Any]:
        return collect_subgraph_integrity(self._run)

    def clear_all(self) -> Dict[str, int]:
        """Delete ALL graph nodes including Experiment, Machine, Tool, Snapshot.

        Returns a dict of label → deleted count.
        """
        labels = [
            "Memory", "Pattern", "Session", "Feedback", "Trace",
            "DiscoveredPattern", "Experiment", "Machine", "Tool", "Snapshot",
            "CoOccurrenceUpdate",
            "Operation", "Dataset",
        ]
        counts: Dict[str, int] = {}
        for label in labels:
            rows = self._run(
                f"MATCH (n:{label}) WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c"
            )
            counts[label] = int(rows[0]["c"]) if rows else 0
        return counts

    # ------------------------------------------------------------------
    # Per-experiment deletion
    # ------------------------------------------------------------------

    def delete_experiment(self, run_id: str) -> Dict[str, int]:
        """Delete an experiment and all its owned data.

        Removes the :Experiment node, its :Session nodes, any :Memory
        nodes linked to those sessions, and related :Feedback / :Trace
        nodes.  Shared :Pattern and :CO_OCCURS_WITH edges are left
        intact (they belong to the global learning state).

        Returns counts of deleted entities per label.
        """
        counts: Dict[str, int] = {}

        # Find session IDs belonging to this experiment
        session_rows = self._run(
            "MATCH (e:Experiment {run_id: $rid})-[:HAS_SESSION]->(s:Session) "
            "RETURN s.id AS sid",
            rid=run_id,
        )
        session_ids = [r["sid"] for r in session_rows]

        if session_ids:
            # Delete Feedback nodes linked to memories in these sessions
            rows = self._run(
                "MATCH (s:Session)<-[:IN_SESSION]-(m:Memory) "
                "WHERE s.id IN $sids "
                "MATCH (f:Feedback)-[:ABOUT]->(m) "
                "DETACH DELETE f RETURN count(f) AS c",
                sids=session_ids,
            )
            counts["Feedback"] = int(rows[0]["c"]) if rows else 0

            # Delete Trace nodes linked to memories in these sessions
            rows = self._run(
                "MATCH (s:Session)<-[:IN_SESSION]-(m:Memory) "
                "WHERE s.id IN $sids "
                "MATCH (t:Trace) WHERE t.memory_id = m.id "
                "DETACH DELETE t RETURN count(t) AS c",
                sids=session_ids,
            )
            counts["Trace"] = int(rows[0]["c"]) if rows else 0

            rows = self._run(
                "MATCH (cu:CoOccurrenceUpdate) "
                "WHERE cu.session_id IN $sids "
                "DETACH DELETE cu RETURN count(cu) AS c",
                sids=session_ids,
            )
            counts["CoOccurrenceUpdate"] = int(rows[0]["c"]) if rows else 0

            # Delete Memory nodes in these sessions
            rows = self._run(
                "MATCH (s:Session)<-[:IN_SESSION]-(m:Memory) "
                "WHERE s.id IN $sids "
                "DETACH DELETE m RETURN count(m) AS c",
                sids=session_ids,
            )
            counts["Memory"] = int(rows[0]["c"]) if rows else 0

            # Delete Session nodes
            rows = self._run(
                "MATCH (s:Session) WHERE s.id IN $sids "
                "DETACH DELETE s RETURN count(s) AS c",
                sids=session_ids,
            )
            counts["Session"] = int(rows[0]["c"]) if rows else 0

        # Delete the Experiment node itself
        rows = self._run(
            "MATCH (e:Experiment {run_id: $rid}) "
            "DETACH DELETE e RETURN count(e) AS c",
            rid=run_id,
        )
        counts["Experiment"] = int(rows[0]["c"]) if rows else 0

        logger.info("Deleted experiment %s: %s", run_id, counts)
        return counts

    # ------------------------------------------------------------------
    # Snapshots — capture & restore graph state
    # ------------------------------------------------------------------

    def capture_snapshot(
        self,
        *,
        run_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Capture the current pattern-prior and co-occurrence state.

        Stores a :Snapshot node containing serialised priors and
        co-occurrence edges.  Use ``restore_snapshot()`` to roll back.

        Returns the snapshot id.
        """
        snap_id = str(uuid.uuid4())
        ts = _now_iso()

        # 1. Capture all Pattern priors
        prior_rows = self._run(
            "MATCH (p:Pattern) RETURN p.key AS k, coalesce(p.prior, 0.5) AS pr"
        )
        priors = {r["k"]: float(r["pr"]) for r in prior_rows}

        # 2. Capture all CO_OCCURS_WITH edges
        cooc_rows = self._run(
            "MATCH (a:Pattern)-[r:CO_OCCURS_WITH]-(b:Pattern) "
            "WHERE a.key < b.key "
            "RETURN a.key AS a, b.key AS b, r.weight AS w"
        )
        co_occurrence = [
            {"source": r["a"], "target": r["b"], "weight": int(r["w"])}
            for r in cooc_rows
        ]

        # 3. Node counts per label
        count_labels = [
            "Memory", "Pattern", "Session", "Feedback", "Trace",
            "DiscoveredPattern", "Experiment", "CoOccurrenceUpdate",
        ]
        node_counts: Dict[str, int] = {}
        for lbl in count_labels:
            rows = self._run(f"MATCH (n:{lbl}) RETURN count(n) AS c")
            node_counts[lbl] = int(rows[0]["c"]) if rows else 0

        # 4. Store the Snapshot node
        self._run(
            "CREATE (sn:Snapshot {"
            "  id: $id, run_id: $rid, label: $lbl,"
            "  created_at: $ts,"
            "  pattern_priors_json: $pj,"
            "  co_occurrence_json: $cj,"
            "  node_counts_json: $nj"
            "})",
            id=snap_id,
            rid=run_id,
            lbl=label or f"snapshot_{ts[:19]}",
            ts=ts,
            pj=json.dumps(priors),
            cj=json.dumps(co_occurrence),
            nj=json.dumps(node_counts),
        )

        logger.info(
            "Captured snapshot %s (label=%s, priors=%d, cooc_edges=%d)",
            snap_id, label, len(priors), len(co_occurrence),
        )
        return snap_id

    def list_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return all Snapshot nodes, newest first."""
        rows = self._run(
            "MATCH (sn:Snapshot) "
            "RETURN properties(sn) AS props "
            "ORDER BY sn.created_at DESC LIMIT $lim",
            lim=limit,
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            props = dict(row["props"])
            # Parse counts for the summary without sending bulky JSON
            try:
                priors = json.loads(props.get("pattern_priors_json") or "{}")
                cooc = json.loads(props.get("co_occurrence_json") or "[]")
                props["n_priors"] = len(priors)
                props["n_co_occurrence_edges"] = len(cooc)
            except Exception:
                props["n_priors"] = 0
                props["n_co_occurrence_edges"] = 0
            # Don't send bulky JSON to listing
            props.pop("pattern_priors_json", None)
            props.pop("co_occurrence_json", None)
            results.append(props)
        return results

    def restore_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore pattern priors and co-occurrence edges from a snapshot.

        This overwrites all ``Pattern.prior`` values and replaces all
        ``[:CO_OCCURS_WITH]`` edges.  Memory / Feedback / Session nodes
        are NOT affected — they are historical records.

        Returns summary of what was restored.
        """
        row = self._run_single(
            "MATCH (sn:Snapshot {id: $id}) RETURN properties(sn) AS props",
            id=snapshot_id,
        )
        if not row:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        props = row["props"]
        priors = json.loads(props.get("pattern_priors_json") or "{}")
        co_occurrence = json.loads(props.get("co_occurrence_json") or "[]")

        with self._driver.session(database=self._database) as session:

            def _tx(tx: Any) -> None:
                # 1. Reset all pattern priors from snapshot
                for key, prior in priors.items():
                    tx.run(
                        "MATCH (p:Pattern {key: $k}) SET p.prior = $pr",
                        k=key,
                        pr=float(prior),
                    )

                # 2. Delete all current CO_OCCURS_WITH edges
                tx.run("MATCH ()-[r:CO_OCCURS_WITH]-() DELETE r")

                # 3. Recreate from snapshot
                for edge in co_occurrence:
                    tx.run(
                        "MATCH (a:Pattern {key: $ak}), (b:Pattern {key: $bk}) "
                        "CREATE (a)-[:CO_OCCURS_WITH {weight: $w, "
                        "  restored_from: $sid, updated_at: $ts}]->(b)",
                        ak=edge["source"],
                        bk=edge["target"],
                        w=int(edge["weight"]),
                        sid=snapshot_id,
                        ts=_now_iso(),
                    )

            session.execute_write(_tx)

        logger.info(
            "Restored snapshot %s: %d priors, %d co-occurrence edges",
            snapshot_id, len(priors), len(co_occurrence),
        )
        return {
            "snapshot_id": snapshot_id,
            "label": props.get("label"),
            "restored_priors": len(priors),
            "restored_co_occurrence_edges": len(co_occurrence),
        }

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a single Snapshot node."""
        rows = self._run(
            "MATCH (sn:Snapshot {id: $id}) DETACH DELETE sn RETURN count(sn) AS c",
            id=snapshot_id,
        )
        deleted = bool(rows and rows[0].get("c", 0) > 0)
        if deleted:
            logger.info("Deleted snapshot %s", snapshot_id)
        return deleted

    def graph_stats(self) -> Dict[str, Any]:
        """Return node and relationship counts for graph management UI."""
        labels = [
            "Memory", "Pattern", "Session", "Feedback", "Trace",
            "DiscoveredPattern", "Experiment", "Machine", "Tool", "Snapshot",
            "CoOccurrenceUpdate",
        ]
        node_counts: Dict[str, int] = {}
        for lbl in labels:
            rows = self._run(f"MATCH (n:{lbl}) RETURN count(n) AS c")
            node_counts[lbl] = int(rows[0]["c"]) if rows else 0

        rel_types = [
            "HAS_PATTERN", "IN_SESSION", "IN_EXPERIMENT", "HAS_SESSION",
            "TESTED_PATTERN", "ON_MACHINE", "USED_TOOL", "NEXT",
            "ABOUT", "ON_PATTERN", "SIMILAR_TO", "CO_OCCURS_WITH",
            "DISCOVERED_FROM", "EVOLVED_FROM", "DURING", "OF_DATASET",
            "CITES", "DOCUMENTED_BY",
        ]
        rel_counts: Dict[str, int] = {}
        for rt in rel_types:
            rows = self._run(f"MATCH ()-[r:{rt}]-() RETURN count(r) AS c")
            rel_counts[rt] = int(rows[0]["c"]) if rows else 0

        total_nodes = sum(node_counts.values())
        total_rels = sum(rel_counts.values())
        subgraph_integrity = self.subgraph_integrity()

        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "subgraph_integrity": subgraph_integrity,
        }

    # ------------------------------------------------------------------
    # Feedback events
    # ------------------------------------------------------------------

    def add_feedback_event(
        self,
        *,
        memory_id: Optional[str],
        action: str,
        user_id: str,
        timestamp: Optional[str] = None,
        pattern_keys: Optional[List[str]] = None,
        context_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        event_id = str(uuid.uuid4())
        ts = timestamp or _now_iso()
        pats = [str(p).strip() for p in (pattern_keys or []) if str(p).strip()]
        try:
            feedback_weight = max(0.0, float(weight))
        except (TypeError, ValueError):
            feedback_weight = 1.0

        payload = {
            "event_id": event_id,
            "memory_id": memory_id,
            "action": action,
            "user_id": user_id,
            "timestamp": ts,
            "pattern_keys": pats,
            "context_key": context_key,
            "context": dict(context or {}),
            "data": dict(data or {}),
            "weight": feedback_weight,
        }
        try:
            self._apply_feedback_event_intent(payload)
            self._flush_graph_write_outbox()
        except Exception:
            outbox = getattr(self, "_graph_write_outbox", None)
            if not outbox:
                raise
            try:
                outbox.append(GraphWriteIntent(kind="feedback_event", payload=payload))
            except OSError:
                logger.exception(
                    "Neo4j feedback-event outbox append failed for event_id=%s",
                    event_id,
                )
            else:
                logger.warning(
                    "Neo4j feedback event persistence failed for event_id=%s; queued for replay",
                    event_id,
                    exc_info=True,
                )
            raise

        return event_id

    def list_feedback_events(self, memory_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._run(
            "MATCH (f:Feedback) WHERE f.memory_id = $mid "
            "OPTIONAL MATCH (f)-[:ON_PATTERN]->(p:Pattern) "
            "RETURN f, collect(p.key) AS pks "
            "ORDER BY f.timestamp ASC LIMIT $lim",
            mid=memory_id,
            lim=limit,
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            f = row["f"] if isinstance(row["f"], dict) else dict(row["f"])
            f["pattern_keys"] = row.get("pks", [])
            f["context"] = json.loads(f.pop("context_json", "{}") or "{}")
            f["data"] = json.loads(f.pop("data_json", "{}") or "{}")
            results.append(f)
        return results

    def list_pattern_keys_with_feedback(self, user_id: Optional[str] = None) -> List[str]:
        if user_id is None:
            rows = self._run(
                "MATCH (f:Feedback)-[:ON_PATTERN]->(p:Pattern) "
                "RETURN DISTINCT p.key AS pk"
            )
        else:
            rows = self._run(
                "MATCH (f:Feedback {user_id: $uid})-[:ON_PATTERN]->(p:Pattern) "
                "RETURN DISTINCT p.key AS pk",
                uid=str(user_id),
            )
        return [r["pk"] for r in rows]

    def get_feedback_counts(
        self,
        *,
        pattern_key: str,
        context_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        pattern_key = str(pattern_key).strip()
        if context_key is None and user_id is None:
            rows = self._run(
                "MATCH (f:Feedback)-[:ON_PATTERN]->(p:Pattern {key: $pk}) "
                "RETURN f.action AS action, sum(coalesce(f.weight, 1.0)) AS cnt",
                pk=pattern_key,
            )
        elif context_key is None:
            rows = self._run(
                "MATCH (f:Feedback {user_id: $uid})-[:ON_PATTERN]->(p:Pattern {key: $pk}) "
                "RETURN f.action AS action, sum(coalesce(f.weight, 1.0)) AS cnt",
                pk=pattern_key,
                uid=str(user_id),
            )
        elif user_id is None:
            rows = self._run(
                "MATCH (f:Feedback)-[:ON_PATTERN]->(p:Pattern {key: $pk}) "
                "WHERE f.context_key = $ck "
                "RETURN f.action AS action, sum(coalesce(f.weight, 1.0)) AS cnt",
                pk=pattern_key,
                ck=str(context_key),
            )
        else:
            rows = self._run(
                "MATCH (f:Feedback {user_id: $uid})-[:ON_PATTERN]->(p:Pattern {key: $pk}) "
                "WHERE f.context_key = $ck "
                "RETURN f.action AS action, sum(coalesce(f.weight, 1.0)) AS cnt",
                pk=pattern_key,
                ck=str(context_key),
                uid=str(user_id),
            )
        counts = {str(r["action"]).lower(): float(r["cnt"]) for r in rows}
        confirm = counts.get("confirm", 0) + counts.get("missed", 0)
        return (confirm, counts.get("dismiss", 0))

    def global_feedback_totals(self) -> Tuple[int, int]:
        """All-time (confirmed, dismissed) across every Feedback node — a single
        aggregate query, so it's correct regardless of any memory list cap."""
        rows = self._run("MATCH (f:Feedback) RETURN f.action AS action, count(*) AS cnt")
        counts = {str(r["action"]).lower(): int(r["cnt"]) for r in rows}
        return (counts.get("confirm", 0) + counts.get("missed", 0), counts.get("dismiss", 0))

    def count_memories(self, visibility: str = "active") -> int:
        """Total Memory node count (uncapped)."""
        row = self._run_single("MATCH (m:Memory) RETURN count(m) AS c")
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def add_trace(
        self,
        *,
        session_id: Optional[str],
        memory_id: Optional[str],
        trace_type: str,
        payload: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> str:
        trace_id = str(uuid.uuid4())
        ts = created_at or _now_iso()
        self._enqueue_graph_write(
            "trace",
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "memory_id": memory_id,
                "trace_type": trace_type,
                "created_at": ts,
                "payload": dict(payload or {}),
            },
        )
        return trace_id

    def list_traces(
        self,
        *,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_type: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        conditions: List[str] = []
        params: Dict[str, Any] = {"lim": limit}
        if memory_id is not None:
            conditions.append("t.memory_id = $mid")
            params["mid"] = memory_id
        if session_id is not None:
            conditions.append("t.session_id = $sid")
            params["sid"] = session_id
        if trace_type is not None:
            conditions.append("t.trace_type = $tt")
            params["tt"] = trace_type

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._run(
            f"MATCH (t:Trace){where} "
            "RETURN properties(t) AS p ORDER BY t.created_at ASC LIMIT $lim",
            **params,
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            p = dict(row["p"])
            p["payload"] = json.loads(p.pop("payload_json", "{}") or "{}")
            results.append(p)
        return results

    def get_runtime_identity_snapshot(
        self,
        *,
        operation_node_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the persisted graph identity for a live runtime context."""
        snapshot: Dict[str, Any] = {
            "operation_node": None,
            "dataset_node": None,
        }
        if operation_node_id:
            row = self._run_single(
                "MATCH (op:Operation {id: $id}) RETURN properties(op) AS props",
                id=str(operation_node_id),
            )
            if row:
                snapshot["operation_node"] = dict(row["props"])
        if dataset_id:
            row = self._run_single(
                "MATCH (ds:Dataset {id: $id}) RETURN properties(ds) AS props",
                id=str(dataset_id),
            )
            if row:
                snapshot["dataset_node"] = dict(row["props"])
        return snapshot

    # ------------------------------------------------------------------
    # Graph-specific: vector search
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """Find similar memories using the Neo4j vector index.

        Returns list of (Memory, similarity_score) tuples.
        """
        rows = self._run(
            "CALL db.index.vector.queryNodes("
            "  'memory_embedding_index', $k, $emb"
            ") YIELD node, score "
            "RETURN properties(node) AS p, score",
            k=top_k,
            emb=query_embedding,
        )
        return [
            (self._deserialize_memory(r["p"]), float(r["score"]))
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Graph-specific: Experiment subgraph
    # ------------------------------------------------------------------

    def store_experiment(
        self,
        *,
        run_id: str,
        experiment_type: str,
        config: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
        comparison: Optional[Dict[str, Any]] = None,
        session_ids: Optional[List[str]] = None,
    ) -> None:
        """Create an :Experiment node and link it to its :Session and :Pattern nodes.

        Graph additions::

            (:Experiment {run_id, experiment_type, config_json, ...})
            (:Experiment)-[:HAS_SESSION]->(:Session)
            (:Session)-[:IN_EXPERIMENT]->(:Experiment)
            (:Experiment)-[:TESTED_PATTERN]->(:Pattern)
        """
        payload = {
            "run_id": run_id,
            "experiment_type": experiment_type,
            "config": dict(config or {}),
            "test_metrics": dict(test_metrics or {}),
            "eval_metrics": dict(eval_metrics or {}),
            "comparison": dict(comparison or {}),
            "session_ids": list(session_ids or []),
            "created_at": _now_iso(),
        }
        try:
            self._apply_experiment_intent(payload)
            self._flush_graph_write_outbox()
        except Exception:
            outbox = getattr(self, "_graph_write_outbox", None)
            if not outbox:
                raise
            logger.warning(
                "Neo4j experiment persistence failed for run_id=%s; queued for replay",
                run_id,
                exc_info=True,
            )
            outbox.append(GraphWriteIntent(kind="experiment", payload=payload))
        logger.info("Stored experiment %s in Neo4j (type=%s)", run_id, experiment_type)

    def get_experiment_graph(self, run_id: str) -> Dict[str, Any]:
        """Return co-occurrence data scoped to a single experiment.

        Returns {nodes, edges} where edges are CO_OCCURS_WITH links
        between patterns that appeared in sessions of this experiment.
        """
        rows = self._run(
            "MATCH (e:Experiment {run_id: $rid})-[:HAS_SESSION]->(s:Session) "
            "<-[:IN_SESSION]-(m:Memory)-[:HAS_PATTERN]->(p:Pattern) "
            "WITH collect(DISTINCT p.key) AS pkeys "
            "UNWIND pkeys AS pk "
            "MATCH (pa:Pattern {key: pk})-[r:CO_OCCURS_WITH]-(pb:Pattern) "
            "WHERE pb.key IN pkeys AND pa.key < pb.key "
            "RETURN pa.key AS a, pb.key AS b, r.weight AS w "
            "ORDER BY r.weight DESC LIMIT 100",
            rid=run_id,
        )
        node_set: Dict[str, int] = {}
        edges = []
        for row in rows:
            a_key, b_key, w = str(row["a"]), str(row["b"]), int(row.get("w", 1))
            node_set[a_key] = node_set.get(a_key, 0) + w
            node_set[b_key] = node_set.get(b_key, 0) + w
            edges.append({"source": a_key, "target": b_key, "weight": w})

        # Get priors for these patterns
        prior_rows = self._run(
            "MATCH (p:Pattern) WHERE p.key IN $keys "
            "RETURN p.key AS k, coalesce(p.prior, 0.5) AS pr",
            keys=list(node_set.keys()),
        ) if node_set else []
        priors = {r["k"]: float(r["pr"]) for r in prior_rows}

        nodes = [
            {"id": k, "weight": v, "prior": priors.get(k, 0.5)}
            for k, v in sorted(node_set.items())
        ]
        return {"nodes": nodes, "edges": edges}

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return all Experiment nodes, newest first."""
        rows = self._run(
            "MATCH (e:Experiment) "
            "OPTIONAL MATCH (e)-[:HAS_SESSION]->(s:Session) "
            "OPTIONAL MATCH (s)<-[:IN_SESSION]-(m:Memory) "
            "WITH e, count(DISTINCT s) AS n_sessions, count(DISTINCT m) AS n_memories "
            "RETURN properties(e) AS props, n_sessions, n_memories "
            "ORDER BY e.created_at DESC LIMIT $lim",
            lim=limit,
        )
        results = []
        for row in rows:
            props = row["props"]
            props["n_sessions"] = row["n_sessions"]
            props["n_memories"] = row["n_memories"]
            results.append(props)
        return results

    # ------------------------------------------------------------------
    # Graph-specific: co-occurrence
    # ------------------------------------------------------------------

    def upsert_co_occurrence(
        self,
        pattern_key_a: str,
        pattern_key_b: str,
        session_id: str,
    ) -> None:
        """Increment the co-occurrence weight between two patterns.

        Also stamps ``updated_at`` so the time-decay method can age out
        stale edges.
        """
        if pattern_key_a == pattern_key_b:
            return
        # Canonical ordering so we don't create duplicate edges
        a, b = sorted([pattern_key_a, pattern_key_b])
        created_at = _now_iso()
        try:
            self._run(
                "MATCH (pa:Pattern {key: $a}), (pb:Pattern {key: $b}) "
                "MERGE (pa)-[r:CO_OCCURS_WITH]-(pb) "
                "ON CREATE SET r.weight = 1, r.first_session = $sid, "
                "             r.created_at = $ts "
                "ON MATCH SET r.weight = r.weight + 1 "
                "SET r.last_session = $sid, r.updated_at = $ts",
                a=a,
                b=b,
                sid=session_id,
                ts=created_at,
            )
            self._flush_graph_write_outbox()
        except Exception:
            outbox = getattr(self, "_graph_write_outbox", None)
            if not outbox:
                raise
            logger.warning(
                "Neo4j co-occurrence persistence failed for pair=(%s, %s) session=%s; queued for replay",
                a,
                b,
                session_id,
                exc_info=True,
            )
            outbox.append(
                GraphWriteIntent(
                    kind="co_occurrence",
                    payload={
                        "pattern_key_a": a,
                        "pattern_key_b": b,
                        "session_id": session_id,
                        "created_at": created_at,
                    },
                )
            )

    def decay_old_co_occurrence(
        self,
        *,
        max_age_days: int = 30,
        decay_factor: float = 0.5,
        prune_below: int = 1,
    ) -> int:
        """Apply time-based decay to CO_OCCURS_WITH edges.

        Edges older than *max_age_days* have their weight halved.
        Edges whose weight drops to *prune_below* or less are deleted.

        Call periodically (e.g. weekly or after each experiment) to keep
        the co-occurrence graph focused on the current operating regime.

        Returns the number of edges pruned.
        """
        cutoff = (
            datetime.now(timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
        )
        from datetime import timedelta
        cutoff_iso = (cutoff - timedelta(days=max_age_days)).isoformat()

        # Decay old edges
        self._run(
            "MATCH ()-[r:CO_OCCURS_WITH]-() "
            "WHERE r.updated_at < $cutoff "
            "SET r.weight = toInteger(r.weight * $factor)",
            cutoff=cutoff_iso,
            factor=decay_factor,
        )
        # Prune negligible edges
        rows = self._run(
            "MATCH ()-[r:CO_OCCURS_WITH]-() "
            "WHERE r.weight <= $min "
            "DELETE r "
            "RETURN count(r) AS pruned",
            min=prune_below,
        )
        pruned = rows[0]["pruned"] if rows else 0
        logger.info(
            "Co-occurrence decay: cutoff=%s, factor=%.2f, pruned=%d edges",
            cutoff_iso, decay_factor, pruned,
        )
        return int(pruned)

    def get_co_occurring_patterns(
        self,
        pattern_key: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-k co-occurring patterns for *pattern_key*.

        Queries ``[:CO_OCCURS_WITH]`` edges in Neo4j.
        Each result dict has keys: ``source``, ``target``, ``weight``.
        """
        rows = self._run(
            "MATCH (p:Pattern {key: $pk})-[r:CO_OCCURS_WITH]-(n:Pattern) "
            "RETURN p.key AS source, n.key AS target, r.weight AS weight "
            "ORDER BY r.weight DESC LIMIT $k",
            pk=str(pattern_key).strip(),
            k=int(top_k),
        )
        return [
            {"source": r["source"], "target": r["target"], "weight": int(r["weight"])}
            for r in rows
        ]

    def sync_pattern_prior(self, pattern_key: str, prior: float) -> None:
        """Persist the scorer-derived prior for a single pattern node."""
        key = str(pattern_key).strip()
        if not key:
            return
        self._run(
            "MERGE (p:Pattern {key: $key}) "
            "SET p.prior = $prior",
            key=key,
            prior=float(prior),
        )

    def propagate_prior_update(
        self,
        pattern_key: str,
        delta: float,
        decay: float = 0.3,
        max_hops: int = 1,
    ) -> List[Tuple[str, float]]:
        """Propagate a prior-update along CO_OCCURS_WITH edges.

        Computes a decayed delta for each co-occurring pattern and
        **writes it directly** to the ``Pattern.prior`` property in
        Neo4j.  Returns list of (neighbor_key, applied_delta) for
        logging / audit.
        """
        rows = self._run(
            "MATCH (p:Pattern {key: $pk})-[r:CO_OCCURS_WITH]-(n:Pattern) "
            "RETURN n.key AS nk, r.weight AS w "
            "ORDER BY r.weight DESC LIMIT 20",
            pk=pattern_key,
        )
        updates: List[Tuple[str, float]] = []
        for row in rows:
            nk = row["nk"]
            weight = float(row["w"])
            applied = delta * decay * min(weight / 10.0, 1.0)
            if abs(applied) < 1e-6:
                continue
            # Persist the prior change directly in Neo4j
            self._run(
                "MATCH (n:Pattern {key: $nk}) "
                "SET n.prior = coalesce(n.prior, 0.5) + $delta",
                nk=nk,
                delta=applied,
            )
            updates.append((nk, applied))
        return updates

    # ------------------------------------------------------------------
    # Graph-specific: discovered-pattern persistence
    # ------------------------------------------------------------------

    def store_discovered_pattern(
        self,
        *,
        key: str,
        features: Dict[str, str],
        confirmation_count: int,
        promoted: bool,
        prior: float,
        first_seen: str,
        last_seen: str,
        source_memory_ids: Optional[List[str]] = None,
    ) -> None:
        """Upsert a :DiscoveredPattern node and link to source :Memory nodes.

        Graph schema additions::

            (:DiscoveredPattern {key, features_json, confirmation_count,
                                 promoted, prior, first_seen, last_seen})
            (:DiscoveredPattern)-[:DISCOVERED_FROM]->(:Memory)
            (:Pattern)-[:EVOLVED_FROM]->(:DiscoveredPattern)   // when promoted
        """
        self._enqueue_graph_write(
            "discovered_pattern",
            {
                "key": key,
                "features": dict(features or {}),
                "confirmation_count": confirmation_count,
                "promoted": promoted,
                "prior": prior,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "source_memory_ids": list(source_memory_ids or []),
            },
        )

    def list_discovered_patterns(self, promoted_only: bool = False) -> List[Dict[str, Any]]:
        """Return all DiscoveredPattern nodes with their source memory IDs."""
        where = "WHERE dp.promoted = true " if promoted_only else ""
        rows = self._run(
            f"MATCH (dp:DiscoveredPattern) {where}"
            "OPTIONAL MATCH (dp)-[:DISCOVERED_FROM]->(m:Memory) "
            "RETURN properties(dp) AS dp, collect(m.id) AS source_ids "
            "ORDER BY dp.last_seen DESC"
        )
        results: List[Dict[str, Any]] = []
        for row in rows:
            dp = dict(row["dp"])
            dp["features"] = json.loads(dp.pop("features_json", "{}"))
            dp["source_memory_ids"] = row.get("source_ids", [])
            results.append(dp)
        return results

    # ------------------------------------------------------------------
    # Graph outbox status
    # ------------------------------------------------------------------

    def graph_outbox_enabled(self) -> bool:
        return bool(getattr(self, "_graph_write_outbox", None))

    def graph_outbox_pending_count(self) -> int:
        outbox = getattr(self, "_graph_write_outbox", None)
        if not outbox:
            return 0
        return outbox.pending_count()

    def list_pending_graph_writes(self, limit: int = 0) -> List[Dict[str, Any]]:
        outbox = getattr(self, "_graph_write_outbox", None)
        if not outbox:
            return []
        pending: List[Dict[str, Any]] = []
        for index, intent in enumerate(outbox.iter_pending()):
            if limit > 0 and index >= int(limit):
                break
            pending.append(intent.to_dict())
        return pending

    def replay_pending_graph_writes(self, limit: int = 0) -> int:
        return self._flush_graph_write_outbox(limit=max(0, int(limit or 0)))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level helpers (shared with SQLite store serialisation)
# ---------------------------------------------------------------------------

def _serialize_time_range(time_range: Any) -> Optional[str]:
    if time_range is None:
        return None
    if isinstance(time_range, (tuple, list)) and len(time_range) == 2:
        return json.dumps([float(time_range[0]), float(time_range[1])])
    if isinstance(time_range, TimeRange):
        return time_range.model_dump_json()
    if isinstance(time_range, dict):
        return json.dumps(time_range)
    return None


def _deserialize_time_range(raw: Optional[str]) -> Any:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, list) and len(parsed) == 2:
        return (float(parsed[0]), float(parsed[1]))
    if isinstance(parsed, dict):
        if {"i0", "i1", "t0", "t1", "fs"}.issubset(parsed.keys()):
            return TimeRange(**parsed)
        if "t0" in parsed and "t1" in parsed:
            return (float(parsed["t0"]), float(parsed["t1"]))
    return None

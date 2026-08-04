from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


MEMORY_GRAPH_LABELS = (
    "Memory",
    "Pattern",
    "Session",
    "Feedback",
    "Trace",
    "DiscoveredPattern",
    "Experiment",
    "Machine",
    "Tool",
    "Snapshot",
    "CoOccurrenceUpdate",
)

KNOWLEDGE_GRAPH_LABELS = (
    "Document",
    "Entity",
)

ALLOWED_CROSS_GRAPH_RELATIONSHIPS = ()

LEGACY_MEMORY_CANDIDATE_HEURISTIC = (
    "dataset_id|source_dataset_id|case_dir|operation_id|created_by!=operator|linked_experiment"
)


def legacy_memory_candidate_predicate(alias: str = "m") -> str:
    return (
        f"trim(coalesce(toString({alias}.dataset_id), '')) <> '' OR "
        f"trim(coalesce(toString({alias}.source_dataset_id), '')) <> '' OR "
        f"trim(coalesce(toString({alias}.case_dir), '')) <> '' OR "
        f"trim(coalesce(toString({alias}.operation_id), '')) <> '' OR "
        f"coalesce({alias}.created_by, 'operator') <> 'operator' OR "
        f"EXISTS {{ MATCH ({alias})-[:IN_SESSION]->(:Session)-[:IN_EXPERIMENT]->(:Experiment) }}"
    )


def normalize_doc_link_intent(
    *,
    memory_id: str,
    pattern_keys: List[str],
    doc_links: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "memory_id": str(memory_id or "").strip(),
        "pattern_keys": sorted(
            {
                str(key).strip()
                for key in (pattern_keys or [])
                if str(key).strip()
            }
        ),
        "doc_links": [
            dict(link)
            for link in (doc_links or [])
            if str((link or {}).get("id") or "").strip()
        ],
    }


def apply_doc_link_intent(tx: Any, payload: Dict[str, Any]) -> int:
    memory_id = str(payload.get("memory_id") or "").strip()
    if not memory_id:
        return 0

    valid_pattern_keys = {
        str(key).strip()
        for key in (payload.get("pattern_keys") or [])
        if str(key).strip()
    }
    valid_links = [
        dict(link)
        for link in (payload.get("doc_links") or [])
        if str((link or {}).get("id") or "").strip()
    ]
    if not valid_links:
        return 0

    tx.run(
        "MATCH (m:Memory {id: $memory_id})-[r:CITES]->(:Document) DELETE r",
        memory_id=memory_id,
    )

    linked = 0
    for link in valid_links:
        score = link.get("score")
        pattern_key = str(link.get("pattern_key") or "").strip()
        evidence_entities = link.get("evidence_entities") or []
        rows = tx.run(
            "MATCH (m:Memory {id: $memory_id}) "
            "MATCH (d:Document {id: $doc_id}) "
            "MERGE (m)-[r:CITES]->(d) "
            "SET r.score = $score, r.page = $page, r.citation = $citation, r.query = $query_used, "
            "    r.pattern_key = $pattern_key, r.evidence_entities_json = $evidence_entities_json "
            "RETURN 1 AS linked",
            memory_id=memory_id,
            doc_id=str(link.get("id")),
            score=float(score) if isinstance(score, (int, float)) else None,
            page=link.get("page"),
            citation=link.get("citation"),
            query_used=link.get("query_used"),
            pattern_key=pattern_key or None,
            evidence_entities_json=json.dumps(evidence_entities),
        ).data()
        if not rows:
            continue
        linked += 1
        if pattern_key and pattern_key in valid_pattern_keys:
            tx.run(
                "MATCH (p:Pattern {key: $pattern_key}) "
                "MATCH (d:Document {id: $doc_id}) "
                "MERGE (p)-[r:DOCUMENTED_BY]->(d) "
                "SET r.score = $score, r.page = $page, r.citation = $citation, r.query = $query_used",
                pattern_key=pattern_key,
                doc_id=str(link.get("id")),
                score=float(score) if isinstance(score, (int, float)) else None,
                page=link.get("page"),
                citation=link.get("citation"),
                query_used=link.get("query_used"),
            )
    return linked


def apply_doc_link_feedback(
    tx: Any,
    *,
    memory_id: str,
    doc_id: str,
    feedback: str,
    user_id: str,
    reason: Optional[str],
    updated_at: str,
    helpful_delta: int,
    not_helpful_delta: int,
) -> Optional[Dict[str, Any]]:
    rows = tx.run(
        "MATCH (m:Memory {id: $memory_id})-[r:CITES]->(d:Document {id: $doc_id}) "
        "SET r.doc_feedback = $feedback, "
        "    r.doc_feedback_user_id = $user_id, "
        "    r.doc_feedback_reason = $reason, "
        "    r.doc_feedback_updated_at = $updated_at, "
        "    r.helpful_count = coalesce(r.helpful_count, 0) + $helpful_delta, "
        "    r.not_helpful_count = coalesce(r.not_helpful_count, 0) + $not_helpful_delta, "
        "    r.feedback_score = (coalesce(r.helpful_count, 0) + $helpful_delta) - (coalesce(r.not_helpful_count, 0) + $not_helpful_delta) "
        "RETURN d.id AS id, "
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
        "       r.evidence_entities_json AS evidence_entities_json",
        memory_id=memory_id,
        doc_id=doc_id,
        feedback=feedback,
        user_id=user_id,
        reason=reason,
        updated_at=updated_at,
        helpful_delta=helpful_delta,
        not_helpful_delta=not_helpful_delta,
    ).data()
    if not rows:
        return None
    return rows[0]


def collect_subgraph_integrity(run_query: Callable[..., List[Dict[str, Any]]]) -> Dict[str, Any]:
    params = {
        "memory_labels": list(MEMORY_GRAPH_LABELS),
        "knowledge_labels": list(KNOWLEDGE_GRAPH_LABELS),
        "allowed_relationships": list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    }
    mixed_rows = run_query(
        "MATCH (n) "
        "WHERE any(label IN labels(n) WHERE label IN $memory_labels) "
        "  AND any(label IN labels(n) WHERE label IN $knowledge_labels) "
        "RETURN count(n) AS mixed_label_nodes",
        **params,
    )
    edge_rows = run_query(
        "MATCH (a)-[r]->(b) "
        "WHERE (("
        "        any(label IN labels(a) WHERE label IN $memory_labels) "
        "    AND any(label IN labels(b) WHERE label IN $knowledge_labels)"
        "       ) OR ("
        "        any(label IN labels(a) WHERE label IN $knowledge_labels) "
        "    AND any(label IN labels(b) WHERE label IN $memory_labels)"
        "       )) "
        "  AND NOT type(r) IN $allowed_relationships "
        "RETURN count(r) AS disallowed_cross_graph_edges, "
        "       collect(DISTINCT type(r))[..20] AS disallowed_relationship_types",
        **params,
    )
    mixed = int((mixed_rows[0] if mixed_rows else {}).get("mixed_label_nodes") or 0)
    edge_row = edge_rows[0] if edge_rows else {}
    disallowed_edges = int(edge_row.get("disallowed_cross_graph_edges") or 0)
    disallowed_types = [
        str(rel_type)
        for rel_type in (edge_row.get("disallowed_relationship_types") or [])
        if str(rel_type).strip()
    ]
    return {
        "healthy": mixed == 0 and disallowed_edges == 0,
        "mixed_label_nodes": mixed,
        "disallowed_cross_graph_edges": disallowed_edges,
        "disallowed_relationship_types": disallowed_types,
        "memory_labels": list(MEMORY_GRAPH_LABELS),
        "knowledge_labels": list(KNOWLEDGE_GRAPH_LABELS),
        "allowed_cross_relationships": list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    }


def collect_memory_graph_cleanup_preview(run_query: Callable[..., List[Dict[str, Any]]]) -> Dict[str, Any]:
    node_counts: Dict[str, int] = {}
    for label in MEMORY_GRAPH_LABELS:
        rows = run_query(f"MATCH (n:{label}) RETURN count(n) AS c")
        node_counts[label] = int((rows[0] if rows else {}).get("c") or 0)

    relationship_rows = run_query(
        "MATCH (a)-[r]->(b) "
        "WHERE any(label IN labels(a) WHERE label IN $memory_labels) "
        "   OR any(label IN labels(b) WHERE label IN $memory_labels) "
        "RETURN count(r) AS total_relationships_to_delete",
        memory_labels=list(MEMORY_GRAPH_LABELS),
    )
    bridge_rows = run_query(
        "MATCH (a)-[r]->(b) "
        "WHERE type(r) IN $allowed_relationships "
        "  AND (("
        "        any(label IN labels(a) WHERE label IN $memory_labels) "
        "    AND any(label IN labels(b) WHERE label IN $knowledge_labels)"
        "       ) OR ("
        "        any(label IN labels(a) WHERE label IN $knowledge_labels) "
        "    AND any(label IN labels(b) WHERE label IN $memory_labels)"
        "       )) "
        "RETURN type(r) AS relationship_type, count(r) AS c",
        memory_labels=list(MEMORY_GRAPH_LABELS),
        knowledge_labels=list(KNOWLEDGE_GRAPH_LABELS),
        allowed_relationships=list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    )
    bridge_relationship_counts = {
        str(row.get("relationship_type")): int(row.get("c") or 0)
        for row in (bridge_rows or [])
        if str(row.get("relationship_type") or "").strip()
    }
    legacy_candidate_predicate = legacy_memory_candidate_predicate("m")
    legacy_summary_rows = run_query(
        "MATCH (m:Memory) "
        f"RETURN count(m) AS total_memories, "
        f"count(CASE WHEN {legacy_candidate_predicate} THEN 1 END) AS candidate_memories, "
        f"count(DISTINCT CASE WHEN {legacy_candidate_predicate} THEN m.session_id END) AS candidate_sessions, "
        "min(m.created_at) AS oldest_memory_at, "
        "max(m.created_at) AS newest_memory_at, "
        f"min(CASE WHEN {legacy_candidate_predicate} THEN m.created_at END) AS oldest_candidate_created_at, "
        f"max(CASE WHEN {legacy_candidate_predicate} THEN m.created_at END) AS newest_candidate_created_at"
    )
    legacy_created_by_rows = run_query(
        "MATCH (m:Memory) "
        f"WHERE {legacy_candidate_predicate} "
        "RETURN coalesce(m.created_by, 'operator') AS created_by, count(m) AS c "
        "ORDER BY c DESC, created_by ASC LIMIT 5"
    )
    legacy_usecase_rows = run_query(
        "MATCH (m:Memory) "
        f"WHERE {legacy_candidate_predicate} "
        "RETURN coalesce(m.usecase, 'unknown') AS usecase, count(m) AS c "
        "ORDER BY c DESC, usecase ASC LIMIT 5"
    )
    legacy_session_rows = run_query(
        "MATCH (m:Memory) "
        f"WHERE {legacy_candidate_predicate} "
        "RETURN m.session_id AS session_id, count(m) AS memory_count, "
        "       min(m.created_at) AS oldest_created_at, max(m.created_at) AS newest_created_at "
        "ORDER BY memory_count DESC, newest_created_at DESC LIMIT 5"
    )
    total_nodes_to_delete = sum(node_counts.values())
    total_relationships_to_delete = int(
        (relationship_rows[0] if relationship_rows else {}).get("total_relationships_to_delete") or 0
    )
    legacy_summary = (legacy_summary_rows[0] if legacy_summary_rows else {}) or {}
    return {
        "scope": "memory_graph",
        "total_nodes_to_delete": total_nodes_to_delete,
        "total_relationships_to_delete": total_relationships_to_delete,
        "node_counts": node_counts,
        "bridge_relationship_counts": bridge_relationship_counts,
        "legacy_candidate_summary": {
            "heuristic": LEGACY_MEMORY_CANDIDATE_HEURISTIC,
            "total_memories": int(legacy_summary.get("total_memories") or 0),
            "candidate_memories": int(legacy_summary.get("candidate_memories") or 0),
            "candidate_sessions": int(legacy_summary.get("candidate_sessions") or 0),
            "oldest_memory_at": legacy_summary.get("oldest_memory_at"),
            "newest_memory_at": legacy_summary.get("newest_memory_at"),
            "oldest_candidate_created_at": legacy_summary.get("oldest_candidate_created_at"),
            "newest_candidate_created_at": legacy_summary.get("newest_candidate_created_at"),
            "created_by_counts": {
                str(row.get("created_by") or "operator"): int(row.get("c") or 0)
                for row in (legacy_created_by_rows or [])
                if str(row.get("created_by") or "operator").strip()
            },
            "usecase_counts": {
                str(row.get("usecase") or "unknown"): int(row.get("c") or 0)
                for row in (legacy_usecase_rows or [])
                if str(row.get("usecase") or "unknown").strip()
            },
            "top_sessions": [
                {
                    "session_id": str(row.get("session_id") or "").strip(),
                    "memory_count": int(row.get("memory_count") or 0),
                    "oldest_created_at": row.get("oldest_created_at"),
                    "newest_created_at": row.get("newest_created_at"),
                }
                for row in (legacy_session_rows or [])
                if str(row.get("session_id") or "").strip()
            ],
        },
        "memory_labels": list(MEMORY_GRAPH_LABELS),
        "knowledge_labels_preserved": list(KNOWLEDGE_GRAPH_LABELS),
        "allowed_cross_relationships": list(ALLOWED_CROSS_GRAPH_RELATIONSHIPS),
    }
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.agents.domain_config import FaultTypeConfig, get_active_domain
from backend.agents.patterns.signatures import infer_pattern_kind, normalize_signature_key, signature_key_for_fault_name

DEFAULT_DOC_LINK_TOP_K = 3
DEFAULT_DOC_LINK_SCORE_FLOOR = 0.6
DEFAULT_DOC_LINK_LIMIT = 5


def _humanize_key(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    token = raw.split(":", 1)[-1] if ":" in raw else raw
    return " ".join(part for part in token.replace("_", " ").replace("-", " ").split() if part).strip()


def _context_terms(cutting_context: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(cutting_context, dict):
        return []
    values = [
        cutting_context.get("machine_type"),
        cutting_context.get("tool_type"),
        cutting_context.get("workpiece_material"),
    ]
    terms: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in terms:
            terms.append(text)
    return terms


def _fault_config_index(channel_names: Optional[Sequence[str]]) -> Dict[str, FaultTypeConfig]:
    domain = get_active_domain(channel_names=list(channel_names or []), force=bool(channel_names))
    index: Dict[str, FaultTypeConfig] = {}
    for fault in domain.fault_types:
        canonical = normalize_signature_key(fault.pattern_key)
        index[canonical] = fault
        index[fault.pattern_key] = fault
        index[fault.name] = fault
    return index


def _build_fault_query(fault: FaultTypeConfig, context_terms: Sequence[str]) -> str:
    parts: List[str] = []
    description = str(fault.description or "").strip().rstrip(".")
    if description:
        parts.append(description)
    indicator_labels = [
        str(indicator.display_name or indicator.feature_name or "").strip()
        for indicator in fault.indicators[:3]
        if str(indicator.display_name or indicator.feature_name or "").strip()
    ]
    if indicator_labels:
        parts.append(" ".join(indicator_labels))
    if context_terms:
        parts.append(" ".join(context_terms))
    return " ".join(part for part in parts if part).strip()


def build_alert_doc_queries(
    pattern_keys: Sequence[str],
    *,
    fault_name: Optional[str] = None,
    cutting_context: Optional[Dict[str, Any]] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    context_terms = _context_terms(cutting_context)
    fault_index = _fault_config_index(channel_names)
    candidates: List[str] = [str(key).strip() for key in pattern_keys or [] if str(key).strip()]
    if fault_name:
        candidates.append(f"fault:{str(fault_name).strip()}")

    queries: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    for raw_key in candidates:
        canonical = normalize_signature_key(raw_key)
        kind = infer_pattern_kind(raw_key)
        query = ""

        if kind == "domain_rule" and raw_key == raw_key.upper():
            query = raw_key
        else:
            fault = fault_index.get(canonical) or fault_index.get(raw_key) or fault_index.get(str(fault_name or "").strip())
            if fault is not None:
                query = _build_fault_query(fault, context_terms)
            else:
                base = _humanize_key(canonical if canonical.startswith("signature:") else raw_key)
                query = " ".join(part for part in [base, *context_terms] if part).strip()

        if not query:
            continue
        dedupe_key = (raw_key, query)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        queries.append({"pattern_key": raw_key, "query": query})

    return queries


def _doc_link_key(match: Dict[str, Any]) -> Tuple[str, str]:
    doc_id = str(match.get("id") or match.get("file_name") or match.get("citation") or "")
    page = str(match.get("page") or "")
    return doc_id, page


def _doc_link_rank(match: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(match.get("ranking_score") or match.get("score") or 0.0),
        float(match.get("feedback_score") or 0.0),
        float(match.get("score") or 0.0),
    )


async def propose_alert_doc_links(
    docs_backend: Any,
    *,
    pattern_keys: Sequence[str],
    usecase: Optional[str],
    machine: Optional[str] = None,
    fault_name: Optional[str] = None,
    cutting_context: Optional[Dict[str, Any]] = None,
    channel_names: Optional[Sequence[str]] = None,
    top_k: int = DEFAULT_DOC_LINK_TOP_K,
    score_floor: float = DEFAULT_DOC_LINK_SCORE_FLOOR,
    limit: int = DEFAULT_DOC_LINK_LIMIT,
) -> Dict[str, Any]:
    query_candidates = build_alert_doc_queries(
        pattern_keys,
        fault_name=fault_name,
        cutting_context=cutting_context,
        channel_names=channel_names,
    )
    best_by_doc: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for candidate in query_candidates:
        result = await docs_backend.search(
            candidate["query"],
            top_k=top_k,
            usecase=usecase,
            machine=machine,
        )
        for match in list(result.get("matches") or []):
            score = match.get("score")
            if score is None or float(score) < float(score_floor):
                continue
            enriched = dict(match)
            enriched["query_used"] = candidate["query"]
            enriched["pattern_key"] = candidate["pattern_key"]
            doc_key = _doc_link_key(enriched)
            existing = best_by_doc.get(doc_key)
            if existing is None or _doc_link_rank(enriched) > _doc_link_rank(existing):
                best_by_doc[doc_key] = enriched

    doc_links = sorted(
        best_by_doc.values(),
        key=_doc_link_rank,
        reverse=True,
    )
    if limit > 0:
        doc_links = doc_links[:limit]

    return {
        "query_candidates": query_candidates,
        "doc_links": doc_links,
    }
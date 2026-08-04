"""
Analytics Agent — memory-based trend analysis and pattern statistics.

Provides historical analysis over the Neo4j / SQLite memory store:
pattern frequency, session history, trend detection, and summary stats.
Unlike the alternate/ AnalyticsAgent that was tightly coupled to SINDIT
timeseries + factory component config, this agent works purely off the
LFL memory pipeline data.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Keywords for router auto-routing
ANALYTICS_KEYWORDS = frozenset([
    "trend", "chart", "history", "stats", "statistics", "pattern",
    "frequency", "count", "summary", "cycle", "session", "analyse",
    "analyze", "analytics", "report",
])


class AnalyticsAgent:
    """Agent that provides trend analysis & statistics over memory data.

    Registered as ``"analytics"`` in the agent router with actions:

    - ``pattern_stats``    — frequency / confirmation counts per pattern
    - ``session_history``  — list sessions with memory counts
    - ``trend``            — pattern occurrence trend over sessions
    - ``summary``          — aggregate system summary
    - ``query``            — free-text dispatch to the best sub-action
    """

    def __init__(self) -> None:
        self._store: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Acquire reference to the active memory store."""
        try:
            from backend.agents.memory.init import get_store
            self._store = get_store()
        except Exception as exc:
            logger.warning("AnalyticsAgent: memory store unavailable: %s", exc)

    def _ensure_store(self) -> Any:
        if self._store is not None:
            return self._store
        try:
            from backend.agents.memory.init import get_store
            self._store = get_store()
        except Exception:
            pass
        return self._store

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
        action = action or "query"

        if action == "pattern_stats":
            return self._pattern_stats(args)
        if action == "session_history":
            return self._session_history(args)
        if action == "trend":
            return self._trend(args)
        if action == "summary":
            return self._summary()
        if action == "query":
            return self._free_query(args)

        return {"error": f"Unknown analytics action: {action}"}

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pattern_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Frequency and feedback counts per pattern key."""
        store = self._ensure_store()
        if store is None:
            return {"error": "Memory store unavailable."}

        limit = int(args.get("limit", 1000))
        memories = store.list_all(limit=limit)

        # Count pattern occurrences
        pattern_counter: Counter[str] = Counter()
        for mem in memories:
            for pk in (mem.pattern_keys or []):
                pattern_counter[pk.key] += 1

        # Enrich with feedback counts
        stats: List[Dict[str, Any]] = []
        for key, count in pattern_counter.most_common():
            confirm, dismiss = 0, 0
            try:
                confirm, dismiss = store.get_feedback_counts(pattern_key=key)
            except Exception:
                pass
            stats.append({
                "pattern_key": key,
                "occurrences": count,
                "confirms": confirm,
                "dismissals": dismiss,
                "net_feedback": confirm - dismiss,
            })

        return {"patterns": stats, "total_memories": len(memories)}

    def _session_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List sessions with memory counts."""
        store = self._ensure_store()
        if store is None:
            return {"error": "Memory store unavailable."}

        limit = int(args.get("limit", 2000))
        memories = store.list_all(limit=limit)

        session_map: Dict[str, Dict[str, Any]] = {}
        for mem in memories:
            sid = mem.session_id or "unknown"
            if sid not in session_map:
                session_map[sid] = {
                    "session_id": sid,
                    "memory_count": 0,
                    "patterns": set(),
                    "labels": set(),
                    "earliest": None,
                    "latest": None,
                }
            entry = session_map[sid]
            entry["memory_count"] += 1
            for pk in (mem.pattern_keys or []):
                entry["patterns"].add(pk.key)
            if mem.label:
                entry["labels"].add(mem.label)
            ts = mem.created_at
            if ts:
                if entry["earliest"] is None or ts < entry["earliest"]:
                    entry["earliest"] = ts
                if entry["latest"] is None or ts > entry["latest"]:
                    entry["latest"] = ts

        # Serialize
        sessions: List[Dict[str, Any]] = []
        for entry in sorted(
            session_map.values(),
            key=lambda e: e.get("latest") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        ):
            sessions.append({
                "session_id": entry["session_id"],
                "memory_count": entry["memory_count"],
                "unique_patterns": len(entry["patterns"]),
                "top_patterns": list(entry["patterns"])[:5],
                "labels": list(entry["labels"]),
                "earliest": entry["earliest"].isoformat() if entry["earliest"] else None,
                "latest": entry["latest"].isoformat() if entry["latest"] else None,
            })

        return {"sessions": sessions, "total_sessions": len(sessions)}

    def _trend(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern occurrence trend bucketed by session."""
        store = self._ensure_store()
        if store is None:
            return {"error": "Memory store unavailable."}

        pattern_filter = args.get("pattern_key")
        limit = int(args.get("limit", 2000))
        memories = store.list_all(limit=limit)

        # Group by session
        session_order: List[str] = []
        session_seen: set = set()
        buckets: Dict[str, Counter] = {}

        for mem in memories:
            sid = mem.session_id or "unknown"
            if sid not in session_seen:
                session_order.append(sid)
                session_seen.add(sid)
                buckets[sid] = Counter()
            for pk in (mem.pattern_keys or []):
                if pattern_filter is None or pk.key == pattern_filter:
                    buckets[sid][pk.key] += 1

        # Build trend series
        trend: List[Dict[str, Any]] = []
        for sid in session_order:
            trend.append({
                "session_id": sid,
                "counts": dict(buckets[sid]),
                "total": sum(buckets[sid].values()),
            })

        return {"trend": trend, "sessions_analysed": len(trend)}

    def _summary(self) -> Dict[str, Any]:
        """Aggregate system summary."""
        store = self._ensure_store()
        if store is None:
            return {"error": "Memory store unavailable."}

        store_stats = store.stats()
        total = store_stats.get("total_memories", store.count())

        # Pattern and feedback overview
        pattern_data = self._pattern_stats({"limit": 5000})
        patterns = pattern_data.get("patterns", [])

        top_confirmed = sorted(patterns, key=lambda p: p["confirms"], reverse=True)[:5]
        top_dismissed = sorted(patterns, key=lambda p: p["dismissals"], reverse=True)[:5]
        most_common = patterns[:5]  # already sorted by occurrence

        return {
            "total_memories": total,
            "backend": store_stats.get("backend", "unknown"),
            "unique_sessions": store_stats.get("unique_sessions", 0),
            "unique_patterns": len(patterns),
            "most_common_patterns": most_common,
            "top_confirmed": top_confirmed,
            "top_dismissed": top_dismissed,
        }

    def _free_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route a free-text query to the best sub-action."""
        query = (args.get("query") or args.get("q", "")).lower()
        if not query:
            return self._summary()

        if any(w in query for w in ("trend", "over time", "progression")):
            return self._trend(args)
        if any(w in query for w in ("session", "history", "list")):
            return self._session_history(args)
        if any(w in query for w in ("pattern", "frequency", "count")):
            return self._pattern_stats(args)
        return self._summary()

"""HTTP client for routing experiment events through the live backend API.

When ``ExperimentConfig.api_mode`` is True, the evaluator uses this module
instead of a local in-memory MemoryStore.  Every scored sample is POSTed
to the backend's ``/agent/memory/events`` endpoint, and feedback goes
through ``PATCH /agent/memory/{id}/feedback``.

This means that co-occurrence tracking, prior propagation, SINDIT context
enrichment, and Neo4j persistence all happen server-side — exactly as they
would in a real production deployment.

Usage
-----
>>> client = ExperimentAPIClient("http://localhost:8000")
>>> result = client.submit_event(session_id, pattern_keys, metadata, cutting_ctx)
>>> client.submit_feedback(memory_id, "CONFIRM", pattern_keys)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _format_http_error(exc: httpx.HTTPError) -> str:
    """Build a compact diagnostic string from an httpx HTTPError.

    Includes the response status code and a truncated body when the error
    carries a response (HTTPStatusError); falls back to the bare exception
    string for transport-level errors (timeouts, connect failures).
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.text or ""
        except Exception:
            body = ""
        body = body.strip().replace("\n", " ")
        if len(body) > 500:
            body = body[:500] + "…"
        return f"HTTP {resp.status_code} {resp.reason_phrase or ''}: {body}".strip()
    return f"{type(exc).__name__}: {exc}"


class ExperimentAPIClient:
    """Synchronous HTTP client for the memory API used in experiment mode.

    The experiment evaluator runs synchronously (iterating pandas rows),
    so this uses ``httpx.Client`` rather than async.  All calls have a
    generous timeout since the backend may trigger ML inference.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._healthy: Optional[bool] = None
        # Most recent failure detail (status + truncated body) for strict-mode reporting.
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Verify the backend API is reachable."""
        try:
            r = self._client.get("/health")
            self._healthy = r.status_code == 200
        except httpx.HTTPError:
            self._healthy = False
        return self._healthy

    # ------------------------------------------------------------------
    # Submit event (replaces local MemoryStore.store)
    # ------------------------------------------------------------------

    def submit_event(
        self,
        session_id: str,
        pattern_keys: List[str],
        metadata: Dict[str, Any],
        significance_score: float,
        annotation: str = "",
        cutting_context: Optional[Dict[str, Any]] = None,
        label: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        derive_patterns: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """POST an event to ``/agent/memory/events`` and return the full API response.

        Mirrors the ``MemoryEvent`` schema expected by the backend router.
        The backend orchestrator handles:
        - Significance scoring (the API's own scorer — authoritative in api_mode)
        - Pattern co-occurrence tracking
        - SINDIT context enrichment (when enabled)
        - Neo4j / SQLite storage

        Returns a dict with ``memory_id``, ``significance_score``, ``action``,
        ``significant`` on success, or None on failure.
        """
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "pattern_keys": pattern_keys,
            "derive_patterns": bool(derive_patterns),
            "metadata": {
                **(metadata or {}),
                "experiment_score": significance_score,
                "label": label,
            },
            "annotation_text": annotation,
            "tags": tags or [],
        }
        if cutting_context:
            payload["cutting_context"] = cutting_context
        if metrics:
            payload["metrics"] = metrics

        try:
            r = self._client.post("/agent/memory/events", json=payload)
            r.raise_for_status()
            data = r.json()
            mem_id = data.get("memory_id")
            api_score = data.get("significance_score", 0.0)
            api_action = data.get("action", "IGNORE")
            logger.info(
                "API → POST /events  session=%s  patterns=%s  "
                "local_score=%.3f  api_score=%.3f  action=%s  label=%s  → %s",
                session_id, pattern_keys, significance_score,
                api_score, api_action, label, mem_id,
            )
            return {
                "memory_id": mem_id,
                "significance_score": api_score,
                "action": api_action,
                "significant": data.get("significant", False),
                "pattern_keys": data.get("pattern_keys_used") or [],
                "model_breakdown": data.get("model_breakdown") or {},
                "explanation": data.get("explanation"),
                "explanation_source": data.get("explanation_source"),
                "alert_line": data.get("alert_line"),
                "alert_line_source": data.get("alert_line_source"),
                "prior_boost": float(data.get("prior_boost") or 0.0),
                "pattern_rule_score": float(data.get("pattern_rule_score") or 0.0),
                "triggered_rules": list(data.get("triggered_rules") or []),
            }
        except httpx.HTTPError as exc:
            self.last_error = _format_http_error(exc)
            logger.warning("API event submission failed: %s", self.last_error)
            return None

    # ------------------------------------------------------------------
    # Batch submit events (eliminates per-event HTTP round-trips)
    # ------------------------------------------------------------------

    def submit_events_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """POST a batch of events to ``/agent/memory/events/batch``.

        Each element in *events* must be a dict with the same keys as
        ``submit_event`` (session_id, pattern_keys, metadata, …).
        Returns a list of result dicts (same format as ``submit_event``),
        with ``None`` for any individual failures.
        """
        payloads = []
        for ev in events:
            p: Dict[str, Any] = {
                "session_id": ev["session_id"],
                "pattern_keys": ev.get("pattern_keys", []),
                "derive_patterns": bool(ev.get("derive_patterns", False)),
                "metadata": ev.get("metadata"),
                "annotation_text": ev.get("annotation_text", ""),
                "tags": ev.get("tags", []),
            }
            if ev.get("cutting_context"):
                p["cutting_context"] = ev["cutting_context"]
            if ev.get("metrics"):
                p["metrics"] = ev["metrics"]
            payloads.append(p)

        try:
            r = self._client.post(
                "/agent/memory/events/batch",
                json={"events": payloads},
            )
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("results", []):
                if not item.get("processed"):
                    results.append(None)
                    continue
                results.append({
                    "memory_id": item.get("memory_id"),
                    "significance_score": item.get("significance_score", 0.0),
                    "action": item.get("action", "IGNORE"),
                    "significant": item.get("significant", False),
                    "pattern_keys": item.get("pattern_keys_used") or [],
                    "model_breakdown": item.get("model_breakdown") or {},
                    "explanation": item.get("explanation"),
                    "explanation_source": item.get("explanation_source"),
                    "alert_line": item.get("alert_line"),
                    "alert_line_source": item.get("alert_line_source"),
                    "prior_boost": float(item.get("prior_boost") or 0.0),
                    "pattern_rule_score": float(item.get("pattern_rule_score") or 0.0),
                    "triggered_rules": list(item.get("triggered_rules") or []),
                })
            logger.info(
                "API → POST /events/batch  count=%d  ok=%d",
                len(payloads), sum(1 for r in results if r),
            )
            return results
        except httpx.HTTPError as exc:
            self.last_error = _format_http_error(exc)
            logger.warning("API batch event submission failed: %s", self.last_error)
            return [None] * len(events)

    # ------------------------------------------------------------------
    # Submit feedback (replaces local MemoryStore.add_feedback_event)
    # ------------------------------------------------------------------

    def submit_feedback(
        self,
        memory_id: str,
        action: str,
        user_id: str = "experiment",
        pattern_keys: Optional[List[str]] = None,
    ) -> bool:
        """PATCH feedback to ``/agent/memory/{id}/feedback``.

        The backend orchestrator updates the pattern prior (a Beta(1,1)-smoothed
        estimate over *saturated* effective counts — see
        ``backend/agents/memory/prior_math.py``; not a plain Beta-Binomial),
        propagates co-occurrence, and persists to Neo4j.
        """
        normalized_action = str(action).strip().lower()
        payload: Dict[str, Any] = {
            "action": normalized_action,
            "user_id": user_id,
        }
        if pattern_keys:
            payload["pattern_keys"] = pattern_keys

        try:
            r = self._client.patch(f"/agent/memory/{memory_id}/feedback", json=payload)
            r.raise_for_status()
            logger.info(
                "API → PATCH /feedback  memory=%s  action=%s  patterns=%s",
                memory_id, normalized_action, pattern_keys or [],
            )
            return True
        except httpx.HTTPError as exc:
            self.last_error = _format_http_error(exc)
            logger.warning("API feedback submission failed: %s", self.last_error)
            return False

    def submit_missed_event(
        self,
        *,
        session_id: str,
        user_id: str,
        pattern_keys: Optional[List[str]] = None,
        raw_metrics: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        timestamp: Optional[str] = None,
        derive_patterns: bool = False,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "pattern_keys": pattern_keys or [],
            "derive_patterns": bool(derive_patterns),
            "user_id": user_id,
        }
        if raw_metrics:
            payload["raw_metrics"] = raw_metrics
        if reason:
            payload["reason"] = reason
        if timestamp:
            payload["timestamp"] = timestamp

        try:
            r = self._client.post("/agent/memory/feedback/missed-event", json=payload)
            r.raise_for_status()
            data = r.json()
            logger.info(
                "API → POST /feedback/missed-event  session=%s  patterns=%s  user=%s",
                session_id,
                data.get("patterns_boosted") or pattern_keys or [],
                user_id,
            )
            return data
        except httpx.HTTPError as exc:
            self.last_error = _format_http_error(exc)
            logger.warning("API missed-event submission failed: %s", self.last_error)
            return None

    # ------------------------------------------------------------------
    # Query current priors (read back server-side state)
    # ------------------------------------------------------------------

    def get_priors(self) -> Dict[str, float]:
        """GET current pattern priors from the backend scorer.

        Returns a dict of pattern_key → prior_value.
        Falls back to empty dict on failure.
        """
        try:
            r = self._client.get("/agent/memory/scorer/priors")
            r.raise_for_status()
            data = r.json()
            return data.get("pattern_priors", data.get("priors", {}))
        except httpx.HTTPError as exc:
            logger.debug("API priors query failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Query co-occurrence graph
    # ------------------------------------------------------------------

    def get_co_occurrence(self) -> Dict[str, int]:
        """GET the co-occurrence graph from the backend.

        Returns a dict of "patternA|patternB" → weight.
        """
        try:
            r = self._client.get("/agent/memory/graph/co-occurrence")
            r.raise_for_status()
            data = r.json()
            edges = data.get("edges", [])
            return {
                f"{e['source']}|{e['target']}": e.get("weight", 1)
                for e in edges
            }
        except httpx.HTTPError as exc:
            logger.debug("API co-occurrence query failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Verification: count memories in the store
    # ------------------------------------------------------------------

    def get_memory_count(self, session_id: Optional[str] = None) -> int:
        """Count how many memories are stored server-side."""
        try:
            if session_id:
                r = self._client.get(f"/agent/memory/session/{session_id}")
            else:
                r = self._client.get("/agent/memory/session/all")
            r.raise_for_status()
            data = r.json()
            memories = data.get("memories", [])
            return len(memories)
        except httpx.HTTPError:
            return 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

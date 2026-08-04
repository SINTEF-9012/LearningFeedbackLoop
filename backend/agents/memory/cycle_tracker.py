from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_nested(mapping: Dict[str, Any], *path: str) -> Optional[Any]:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


@dataclass
class CycleEnded:
    session_id: str
    part_id: Optional[str]
    operation_id: Optional[str]
    started_at: float
    ended_at: float


@dataclass
class _ActiveCycle:
    part_id: Optional[str]
    operation_id: Optional[str]
    started_at: float
    last_seen_at: float


class CycleTracker:
    def __init__(self) -> None:
        self._active: Dict[str, _ActiveCycle] = {}

    @staticmethod
    def extract_identifiers(metadata: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
        if not isinstance(metadata, dict):
            return (None, None)

        part_id = (
            _clean_text(metadata.get("part_id"))
            or _clean_text(metadata.get("part"))
            or _clean_text(_get_nested(metadata, "casedata", "part_id"))
            or _clean_text(_get_nested(metadata, "casedata", "part"))
        )
        operation_id = (
            _clean_text(metadata.get("operation_id"))
            or _clean_text(metadata.get("operation"))
            or _clean_text(metadata.get("of_id"))
            or _clean_text(_get_nested(metadata, "casedata", "operation_id"))
            or _clean_text(_get_nested(metadata, "casedata", "operation"))
            or _clean_text(_get_nested(metadata, "casedata", "of_id"))
        )
        return (part_id, operation_id)

    def observe(self, session_id: str, metadata: Optional[Dict[str, Any]], ts: float) -> Optional[CycleEnded]:
        part_id, operation_id = self.extract_identifiers(metadata)
        active = self._active.get(session_id)
        observed_at = float(ts)

        if part_id is None and operation_id is None:
            if active is None:
                return None
            self._active.pop(session_id, None)
            return CycleEnded(
                session_id=session_id,
                part_id=active.part_id,
                operation_id=active.operation_id,
                started_at=active.started_at,
                ended_at=observed_at,
            )

        if active is None:
            self._active[session_id] = _ActiveCycle(
                part_id=part_id,
                operation_id=operation_id,
                started_at=observed_at,
                last_seen_at=observed_at,
            )
            return None

        if active.part_id == part_id and active.operation_id == operation_id:
            active.last_seen_at = observed_at
            return None

        ended = CycleEnded(
            session_id=session_id,
            part_id=active.part_id,
            operation_id=active.operation_id,
            started_at=active.started_at,
            ended_at=observed_at,
        )

        if part_id is None and operation_id == active.operation_id and active.part_id is not None:
            self._active.pop(session_id, None)
            return ended

        self._active[session_id] = _ActiveCycle(
            part_id=part_id,
            operation_id=operation_id,
            started_at=observed_at,
            last_seen_at=observed_at,
        )
        return ended

    def clear_session(self, session_id: str) -> None:
        self._active.pop(session_id, None)

    def flush_session(self, session_id: str, ended_at: Optional[float] = None) -> Optional[CycleEnded]:
        active = self._active.pop(session_id, None)
        if active is None:
            return None

        final_timestamp = active.last_seen_at
        if ended_at is not None:
            try:
                final_timestamp = max(final_timestamp, float(ended_at))
            except (TypeError, ValueError):
                final_timestamp = active.last_seen_at

        return CycleEnded(
            session_id=session_id,
            part_id=active.part_id,
            operation_id=active.operation_id,
            started_at=active.started_at,
            ended_at=final_timestamp,
        )


_cycle_tracker: Optional[CycleTracker] = None


def get_cycle_tracker() -> CycleTracker:
    global _cycle_tracker
    if _cycle_tracker is None:
        _cycle_tracker = CycleTracker()
    return _cycle_tracker
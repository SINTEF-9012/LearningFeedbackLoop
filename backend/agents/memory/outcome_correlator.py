from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from ..core.context import CuttingContext
from ..core.schemas import Memory
from .cycle_tracker import CycleEnded
from .scorer import FEEDBACK_WEIGHTS

logger = logging.getLogger(__name__)


def _time_bounds(memory: Memory) -> tuple[Optional[float], Optional[float]]:
    time_range = getattr(memory, "time_range", None)
    if time_range is None:
        return (None, None)
    if isinstance(time_range, (tuple, list)) and len(time_range) >= 2:
        try:
            return (float(time_range[0]), float(time_range[1]))
        except (TypeError, ValueError):
            return (None, None)
    start = getattr(time_range, "t0", None)
    end = getattr(time_range, "t1", None)
    try:
        start_value = float(start) if start is not None else None
    except (TypeError, ValueError):
        start_value = None
    try:
        end_value = float(end) if end is not None else None
    except (TypeError, ValueError):
        end_value = None
    return (start_value, end_value)


def _overlaps_cycle(memory: Memory, cycle: CycleEnded) -> bool:
    start, end = _time_bounds(memory)
    if start is None or end is None:
        return False
    return end >= float(cycle.started_at) and start <= float(cycle.ended_at)


def _pattern_keys(memory: Memory) -> List[str]:
    keys: List[str] = []
    for pattern in getattr(memory, "pattern_keys", None) or []:
        key = getattr(pattern, "key", None)
        if key:
            keys.append(str(key))
    return keys


def _has_feedback(memory_id: Optional[str], store: Any) -> bool:
    if not memory_id or not hasattr(store, "list_feedback_events"):
        return False
    try:
        events = store.list_feedback_events(memory_id, limit=200)
    except Exception:
        return False
    return bool(events)


def _cutting_context(memory: Memory) -> Optional[CuttingContext]:
    metadata = getattr(memory, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    raw_context = metadata.get("cutting_context")
    if not isinstance(raw_context, dict) or not raw_context:
        return None
    try:
        return CuttingContext.model_validate(raw_context)
    except Exception:
        logger.debug("Failed to rebuild CuttingContext for passive outcome", exc_info=True)
        return None


def attach_passive_outcome(
    *,
    cycle: CycleEnded,
    memories: Iterable[Memory],
    store: Any,
    scorer: Any,
) -> int:
    weight = FEEDBACK_WEIGHTS["passive_cycle_completed_without_intervention"]
    affected = 0

    for memory in memories:
        if getattr(memory, "session_id", None) != cycle.session_id:
            continue
        if not _overlaps_cycle(memory, cycle):
            continue
        if _has_feedback(getattr(memory, "id", None), store):
            continue

        memory_id = getattr(memory, "id", None)
        pattern_keys = _pattern_keys(memory)
        if hasattr(store, "add_feedback_event"):
            try:
                store.add_feedback_event(
                    memory_id=memory_id,
                    action="dismiss",
                    user_id="system:cycle_tracker",
                    pattern_keys=pattern_keys,
                    data={
                        "reason": "cycle completed without intervention",
                        "source": "passive_cycle_completed_without_intervention",
                        "emitted_by": "passive_cycle_tracker",
                        "cycle": {
                            "part_id": cycle.part_id,
                            "operation_id": cycle.operation_id,
                            "started_at": cycle.started_at,
                            "ended_at": cycle.ended_at,
                        },
                        "weight": weight,
                    },
                    weight=weight,
                )
            except Exception:
                logger.warning(
                    "Failed to persist passive feedback event for memory %s",
                    memory_id,
                    exc_info=True,
                )

        if scorer is not None and hasattr(scorer, "update_pattern_prior"):
            cutting_context = _cutting_context(memory)
            for pattern_key in pattern_keys:
                scorer.update_pattern_prior(
                    pattern_key,
                    was_significant=False,
                    context=cutting_context,
                    weight=weight,
                    source="passive_cycle_completed_without_intervention",
                )

        affected += 1

    return affected
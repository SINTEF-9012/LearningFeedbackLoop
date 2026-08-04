"""SINDIT runtime current-state tracker — Agent G (2026-04-24).

Tracks **active-only** runtime entities (experiments, phases, operations,
currently-detected patterns) and emits diff-based SINDIT upserts so that
the twin reflects *what is happening now*, not a growing event log.

Design invariants:

1. **Active set is authoritative.** The tracker stores a mapping of
   ``(kind, id) → asset payload``. Calling :py:meth:`set_active` with a
   new set replaces it. Anything that fell off is marked inactive (or
   removed, depending on policy).
2. **Bounded growth.** The tracker itself never accumulates history —
   that stays in Neo4j. The only persistent effect on SINDIT is the
   up-to-date active set.
3. **Pure + testable.** :class:`RuntimeStateTracker` takes a client
   callable protocol; tests stub it. The live bridge wires a real
   :class:`SinditClient`.
4. **Fail-open.** Every sync call is best-effort — exceptions are
   logged and counted, never re-raised up the call chain.

Entity URN formats:

- ``urn:lfl:experiment:{id}``
- ``urn:lfl:phase:{id}``
- ``urn:lfl:operation:{id}``
- ``urn:lfl:pattern:{key}``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set, Tuple

from .asset_catalog import (
    LFL_ASSET_KIND_KEY,
    SAMM_ASSET_TYPE,
    _now_iso,
    _slug,
)

logger = logging.getLogger(__name__)


# ── URN formats ────────────────────────────────────────────────────────

_URN_FORMATS = {
    "experiment": "urn:lfl:experiment:{id}",
    "phase": "urn:lfl:phase:{id}",
    "operation": "urn:lfl:operation:{id}",
    "pattern": "urn:lfl:pattern:{id}",
}

_KIND_LABEL = {
    "experiment": "Experiment",
    "phase": "Phase",
    "operation": "Operation",
    "pattern": "Pattern",
}

_VALID_KINDS = frozenset(_URN_FORMATS.keys())


# ── Client protocol + record ───────────────────────────────────────────


class _SinditWriteClient(Protocol):
    """Minimal client surface used by the tracker."""

    async def post_asset(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]: ...

    async def update_node(
        self, node_uri: str, fields: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]: ...


@dataclass(frozen=True)
class ActiveEntity:
    kind: str                            # "experiment" / "phase" / "operation" / "pattern"
    entity_id: str                       # user-supplied id (gets slugified for the URN)
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def payload(self) -> Dict[str, Any]:
        return build_runtime_asset(self.kind, self.entity_id, dict(self.metadata))


def build_runtime_asset(
    kind: str,
    entity_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    active: bool = True,
) -> Dict[str, Any]:
    """Build an upsert payload for a runtime entity.

    The payload mirrors the shape produced by
    :mod:`backend.agents.sindit.asset_catalog` so downstream consumers
    don't need a second parser.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown runtime kind={kind!r}; allowed={sorted(_VALID_KINDS)}")
    slug = _slug(entity_id)
    uri = _URN_FORMATS[kind].format(id=slug)
    meta: Dict[str, Any] = dict(metadata or {})
    meta.setdefault("active", bool(active))
    meta.setdefault("lastSeenAt", _now_iso())
    return {
        "uri": uri,
        "label": meta.pop("label", None) or f"{_KIND_LABEL[kind]} {entity_id}",
        "assetType": SAMM_ASSET_TYPE,
        "assetDescription": meta.pop("description", f"Runtime {_KIND_LABEL[kind].lower()}."),
        LFL_ASSET_KIND_KEY: _KIND_LABEL[kind],
        "metadata": meta,
    }


# ── Tracker ────────────────────────────────────────────────────────────


@dataclass
class _TrackerStats:
    upserts_ok: int = 0
    upserts_fail: int = 0
    deactivations_ok: int = 0
    deactivations_fail: int = 0
    last_sync_at: Optional[str] = None
    active_by_kind: Dict[str, int] = field(default_factory=dict)


class RuntimeStateTracker:
    """Tracks the live set of runtime entities and diffs them to SINDIT.

    Typical call sequence (per session tick / experiment-runner cycle)::

        tracker = RuntimeStateTracker()
        await tracker.sync_active(
            client,
            [
                ActiveEntity("experiment", "expt_2026-04-24_01"),
                ActiveEntity("phase", "roughing"),
                ActiveEntity("pattern", "chatter:high_ratio"),
            ],
        )

    On the next call the tracker upserts new entries, refreshes
    ``lastSeenAt`` on entries still present, and deactivates (writes
    ``active=False`` + ``endedAt``) anything that fell out of the list.

    Set ``purge_on_deactivate=True`` to instead issue a hard update that
    clears the entity's presence — useful when you want the twin to
    drop the IRI entirely once the run ends. SINDIT proper doesn't have
    a DELETE verb in our client, so we still write ``active=False`` and
    rely on the operator/downstream to archive.
    """

    def __init__(self, *, purge_on_deactivate: bool = False):
        self._active: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._purge_on_deactivate = bool(purge_on_deactivate)
        self._stats = _TrackerStats()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    @property
    def stats(self) -> _TrackerStats:
        # Refresh per-kind counters on every read.
        by_kind: Dict[str, int] = {}
        for (kind, _), _payload in self._active.items():
            by_kind[kind] = by_kind.get(kind, 0) + 1
        self._stats.active_by_kind = by_kind
        return self._stats

    def active_keys(self) -> List[Tuple[str, str]]:
        return sorted(self._active.keys())

    def active_uris(self) -> List[str]:
        return sorted(p.get("uri", "") for p in self._active.values())

    # ------------------------------------------------------------------
    # Pure diff (exposed for tests)
    # ------------------------------------------------------------------

    def plan_sync(
        self, active: Iterable[ActiveEntity]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compute (to_upsert, to_deactivate) without touching the client.

        Also mutates the tracker's in-memory active set so that a
        following ``sync_active`` call would see the new state; however,
        tests can call ``plan_sync`` alone to validate the decision
        logic without needing a client stub.
        """
        new_active: Dict[Tuple[str, str], Dict[str, Any]] = {}
        to_upsert: List[Dict[str, Any]] = []
        for ent in active:
            key = (ent.kind, _slug(ent.entity_id))
            payload = ent.payload()
            new_active[key] = payload
            to_upsert.append(payload)

        dropped_keys = [k for k in self._active if k not in new_active]
        to_deactivate: List[Dict[str, Any]] = []
        for key in dropped_keys:
            previous = self._active[key]
            to_deactivate.append(previous)

        self._active = new_active
        return to_upsert, to_deactivate

    # ------------------------------------------------------------------
    # Async sync (hits the client)
    # ------------------------------------------------------------------

    async def sync_active(
        self,
        client: _SinditWriteClient,
        active: Iterable[ActiveEntity],
    ) -> Dict[str, Any]:
        """Apply the diff: upsert new/current entries, deactivate dropped.

        Returns a summary dict; never raises.
        """
        to_upsert, to_deactivate = self.plan_sync(active)

        for payload in to_upsert:
            ok = False
            try:
                ok = bool(await client.post_asset(payload))
            except Exception:
                logger.exception("RuntimeStateTracker: post_asset failed for %s", payload.get("uri"))
            if ok:
                self._stats.upserts_ok += 1
            else:
                self._stats.upserts_fail += 1

        for previous in to_deactivate:
            ok = False
            try:
                ok = bool(
                    await client.update_node(
                        node_uri=previous.get("uri", ""),
                        fields={
                            "active": False,
                            "endedAt": _now_iso(),
                        },
                    )
                )
            except Exception:
                logger.exception(
                    "RuntimeStateTracker: update_node(deactivate) failed for %s",
                    previous.get("uri"),
                )
            if ok:
                self._stats.deactivations_ok += 1
            else:
                self._stats.deactivations_fail += 1

        self._stats.last_sync_at = _now_iso()
        return {
            "upserts": len(to_upsert),
            "deactivations": len(to_deactivate),
            "stats": {
                "upserts_ok": self._stats.upserts_ok,
                "upserts_fail": self._stats.upserts_fail,
                "deactivations_ok": self._stats.deactivations_ok,
                "deactivations_fail": self._stats.deactivations_fail,
                "active_by_kind": self.stats.active_by_kind,
                "last_sync_at": self._stats.last_sync_at,
            },
        }

    async def clear(self, client: _SinditWriteClient) -> Dict[str, Any]:
        """Deactivate every tracked entity — call on session end."""
        # Re-use sync_active with empty set.
        return await self.sync_active(client, [])


# ── Convenience builders ──────────────────────────────────────────────


def experiment(entity_id: str, **metadata: Any) -> ActiveEntity:
    return _entity("experiment", entity_id, metadata)


def phase(entity_id: str, **metadata: Any) -> ActiveEntity:
    return _entity("phase", entity_id, metadata)


def operation(entity_id: str, **metadata: Any) -> ActiveEntity:
    return _entity("operation", entity_id, metadata)


def pattern(pattern_key: str, **metadata: Any) -> ActiveEntity:
    return _entity("pattern", pattern_key, metadata)


def _entity(kind: str, entity_id: str, metadata: Dict[str, Any]) -> ActiveEntity:
    # Dataclass is frozen so metadata needs to be hashable — tuple of items.
    meta_items: List[Tuple[str, Any]] = []
    for k, v in sorted(metadata.items()):
        meta_items.append((k, _freeze(v)))
    return ActiveEntity(kind=kind, entity_id=entity_id, metadata=tuple(meta_items))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value

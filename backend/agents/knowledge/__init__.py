"""Knowledge subsystem — Agent H (2026-04-24).

Public API:

- :class:`KnowledgePack`, :func:`build_knowledge_pack`, :func:`save_pack`,
    :func:`load_pack`, :class:`ContextKeys`, :func:`similarity_score`,
    :func:`should_apply`, :class:`FleetPackApplication`,
    :func:`apply_fleet_pack` from :mod:`.pack`.
- :class:`FleetKnowledgePack`, :class:`FleetDiscoveryFamily`,
    :class:`FleetFamilyReview`, :func:`aggregate_fleet_packs`,
    :func:`apply_family_reviews` from
    :mod:`.fleet`.
- :class:`UpstreamSink`, :class:`FileSink`, :class:`MqttSink`,
    :class:`HttpSink`,
  :func:`push_to_sinks` from :mod:`.sinks`.
"""

from .pack import (  # noqa: F401
    KNOWLEDGE_PACK_VERSION,
    ContextKeys,
    FleetPackApplication,
    KnowledgePack,
    apply_fleet_pack,
    build_knowledge_pack,
    load_pack,
    save_pack,
    should_apply,
    similarity_score,
)
from .fleet import (  # noqa: F401
    FleetDiscoveryFamily,
    FleetFamilyReview,
    FleetKnowledgePack,
    aggregate_fleet_packs,
    apply_family_reviews,
)
from .sinks import (  # noqa: F401
    FileSink,
    HttpSink,
    MqttSink,
    UpstreamSink,
    push_to_sinks,
)

__all__ = [
    "KNOWLEDGE_PACK_VERSION",
    "ContextKeys",
    "FleetDiscoveryFamily",
    "FleetFamilyReview",
    "FleetPackApplication",
    "FleetKnowledgePack",
    "KnowledgePack",
    "aggregate_fleet_packs",
    "apply_family_reviews",
    "apply_fleet_pack",
    "build_knowledge_pack",
    "load_pack",
    "save_pack",
    "should_apply",
    "similarity_score",
    "FileSink",
    "HttpSink",
    "MqttSink",
    "UpstreamSink",
    "push_to_sinks",
]

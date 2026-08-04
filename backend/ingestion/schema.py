"""Typed envelopes for inbound frames and outbound learnings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


FRAME_SCHEMA_VERSION = 1
LEARNING_SCHEMA_VERSION = 1


@dataclass
class FrameEnvelope:
    """Typed payload for inbound stream frames."""

    kind: str
    session_id: str
    ts_unix: float
    position: int
    fs: float
    schema_version: int = FRAME_SCHEMA_VERSION
    source: str = "unknown"
    window_seconds: Optional[float] = None
    signals: Dict[str, float] = field(default_factory=dict)
    frame: Optional[Dict[str, List[float]]] = None
    spectrum: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    patterns: Optional[List[str]] = None
    external_signals: Optional[Dict[str, Any]] = None
    cutting_context: Optional[Dict[str, Any]] = None
    batch: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEnvelope:
    """Typed payload for outbound learnings and feedback."""

    kind: str
    ts_unix: float
    session_id: str
    payload: Dict[str, Any]
    batch: Optional[Dict[str, Any]] = None
    schema_version: int = LEARNING_SCHEMA_VERSION
    source: str = "feedback_loop"
    machine_uri: Optional[str] = None
    tenant_id: Optional[str] = None
    site_id: Optional[str] = None
    pii_scrub_level: Optional[str] = None


def envelope_to_dict(envelope: FrameEnvelope | LearningEnvelope) -> Dict[str, Any]:
    """Serialize a typed envelope while omitting None fields."""

    data = asdict(envelope)
    return {key: value for key, value in data.items() if value is not None}
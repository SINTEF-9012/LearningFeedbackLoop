"""Configuration endpoints — Agent K (2026-04-24).

Surfaces runtime URLs so the UI stops hardcoding ``localhost:9017`` /
``localhost:7200``. All values are env-overridable with sensible
defaults. The endpoint never returns secrets — bearer tokens or
credentials stay server-side.

Also exposes the ``POST /knowledge/push`` endpoint deferred from
Agent H: builds a knowledge pack and fans it out to configured sinks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.agents.knowledge import (
    ContextKeys,
    FileSink,
    HttpSink,
    MqttSink,
    build_knowledge_pack,
    push_to_sinks,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v or default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ConfigUrlsResponse(BaseModel):
    sindit: str
    graphdb: str
    influxdb: str
    neo4j: str
    upstream_knowledge: Optional[str] = None
    ui: Optional[str] = None


class LearningsConfigResponse(BaseModel):
    websocket_global: str
    websocket_session_template: str
    mqtt_enabled: bool
    mqtt_configured: bool
    mqtt_transport_available: bool
    mqtt_forwarding_active: bool
    mqtt_state: str
    mqtt_topic: Optional[str] = None
    mqtt_broker_host: str
    mqtt_broker_port: int
    mqtt_qos: int
    mqtt_retain: bool = False
    published_count: int = 0
    connected_at: Optional[float] = None
    last_published_at: Optional[float] = None
    last_error: Optional[str] = None


@router.get("/config/urls", response_model=ConfigUrlsResponse)
async def get_config_urls() -> ConfigUrlsResponse:
    """Return runtime URLs for external services.

    Never returns secrets — only endpoint base URLs.
    """
    return ConfigUrlsResponse(
        sindit=_env("SINDIT_URL", "http://localhost:9017"),
        graphdb=_env("GRAPHDB_URL", "http://localhost:7200"),
        influxdb=_env("INFLUXDB_URL", "http://localhost:8086"),
        neo4j=_env("NEO4J_URL", "bolt://localhost:7687"),
        upstream_knowledge=(os.environ.get("UPSTREAM_KNOWLEDGE_URL") or None),
        ui=(os.environ.get("UI_URL") or None),
    )


@router.get("/config/learnings", response_model=LearningsConfigResponse)
async def get_learnings_config(request: Request) -> LearningsConfigResponse:
    mqtt_enabled = _env_flag("MQTT_LEARNINGS_ENABLED")
    mqtt_topic = os.environ.get("MQTT_LEARNINGS_TOPIC", "").strip() or None
    mqtt_transport_available = True
    try:
        from backend.mqtt_transport import ensure_mqtt_transport_available

        ensure_mqtt_transport_available()
    except Exception:
        mqtt_transport_available = False

    runtime_status: Dict[str, Any] = {}
    publisher = getattr(request.app.state, "mqtt_learning_publisher", None)
    if publisher is not None and hasattr(publisher, "status"):
        try:
            runtime_status = dict(publisher.status())
        except Exception:
            logger.debug("Could not read mqtt learning publisher status", exc_info=True)

    mqtt_configured = bool(mqtt_topic)
    mqtt_state = str(runtime_status.get("state") or (
        "disabled"
        if not mqtt_enabled
        else "transport_unavailable"
        if not mqtt_transport_available
        else "misconfigured"
        if not mqtt_configured
        else "configured"
    ))

    return LearningsConfigResponse(
        websocket_global="/learnings/ws",
        websocket_session_template="/learnings/ws/{session_id}",
        mqtt_enabled=mqtt_enabled,
        mqtt_configured=mqtt_configured,
        mqtt_transport_available=mqtt_transport_available,
        mqtt_forwarding_active=bool(runtime_status.get("task_active")) and mqtt_state in {"starting", "connecting", "connected"},
        mqtt_state=mqtt_state,
        mqtt_topic=mqtt_topic,
        mqtt_broker_host=_env("MQTT_BROKER_HOST", "localhost"),
        mqtt_broker_port=int(_env("MQTT_BROKER_PORT", "1883")),
        mqtt_qos=int(_env("MQTT_LEARNINGS_QOS", "0")),
        mqtt_retain=_env_flag("MQTT_LEARNINGS_RETAIN"),
        published_count=int(runtime_status.get("published_count") or 0),
        connected_at=runtime_status.get("connected_at"),
        last_published_at=runtime_status.get("last_published_at"),
        last_error=runtime_status.get("last_error"),
    )


class KnowledgePushRequest(BaseModel):
    site: str
    data_dir: str = "data"
    context: Dict[str, Optional[str]] = Field(default_factory=dict)
    tenant_id: Optional[str] = None
    signer: Optional[str] = None
    license: str = "internal-only"
    pii_scrub_level: str = "symbolic_only"
    expires_at: Optional[str] = None
    sinks: List[str] = Field(
        default_factory=lambda: ["file"],
        description="Sink names to target; allowed: 'file', 'mqtt'.",
    )
    notes: List[str] = Field(default_factory=list)


class KnowledgePushResponse(BaseModel):
    site: str
    summary: Dict[str, int]
    sinks: Dict[str, bool]


def _validated_context(raw_context: Dict[str, Optional[str]]) -> ContextKeys:
    ctx = ContextKeys(
        machine_type=raw_context.get("machine_type"),
        tool_type=raw_context.get("tool_type"),
        material=raw_context.get("material"),
        regime=raw_context.get("regime"),
    )
    missing = ctx.missing_required_keys()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "knowledge/push requires a complete context for reusable fleet exports",
                "missing_context_keys": missing,
            },
        )
    return ctx


def _build_sinks(names: List[str]) -> List[Any]:
    """Instantiate the requested sinks from env-configured defaults."""
    out: List[Any] = []
    for raw_name in names:
        name = (raw_name or "").strip().lower()
        if name == "file":
            out.append(
                FileSink(
                    directory=_env("KNOWLEDGE_PACK_DIR", "data/knowledge_packs"),
                    prefix=_env("KNOWLEDGE_PACK_PREFIX", "knowledge_pack"),
                    name="file",
                )
            )
        elif name == "mqtt":
            enabled = os.environ.get("MQTT_ENABLED", "").lower() in {"1", "true", "yes"}
            out.append(
                MqttSink(
                    broker_url=_env("MQTT_BROKER_URL", ""),
                    topic=_env("MQTT_TOPIC", ""),
                    mode=_env("MQTT_MODE", "delta"),
                    enabled=enabled,
                    name="mqtt",
                )
            )
        elif name in {"http", "upstream"}:
            out.append(
                HttpSink(
                    url=_env("UPSTREAM_KNOWLEDGE_URL", ""),
                    timeout_seconds=float(_env("UPSTREAM_KNOWLEDGE_TIMEOUT_SECONDS", "10")),
                    name="http",
                )
            )
        else:
            logger.warning("knowledge/push: unknown sink name=%r ignored", raw_name)
    return out


@router.post("/knowledge/push", response_model=KnowledgePushResponse)
async def push_knowledge_pack(request: KnowledgePushRequest) -> KnowledgePushResponse:
    """Build a knowledge pack for the given site and push to configured sinks.

    Never raises — sink failures are reflected in the response body
    (``sinks[name] = False``) and ``NotImplementedError`` from a
    misconfigured-but-enabled MQTT sink is caught and reported as
    ``False`` with a note so the caller learns without a 500.
    """
    ctx = _validated_context(request.context)
    pack = build_knowledge_pack(
        request.data_dir,
        site=request.site,
        context=ctx,
        require_complete_context=True,
        tenant_id=(request.tenant_id or os.environ.get("KNOWLEDGE_TENANT_ID") or request.site),
        signer=(request.signer or os.environ.get("KNOWLEDGE_PACK_SIGNER") or "knowledge_push"),
        license=request.license,
        pii_scrub_level=request.pii_scrub_level,
        expires_at=request.expires_at,
        notes=request.notes,
    )

    sinks = _build_sinks(request.sinks)
    try:
        results = await push_to_sinks(sinks, pack.to_dict())
    except NotImplementedError as exc:
        logger.warning("knowledge/push: sink not implemented: %s", exc)
        results = {s.name: False for s in sinks}

    return KnowledgePushResponse(
        site=pack.site,
        summary=pack.summary(),
        sinks=results,
    )

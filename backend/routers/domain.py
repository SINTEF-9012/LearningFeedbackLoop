"""Domain endpoints — Agent J (2026-04-24).

Exposes the currently-active domain config + the list of registered
packs so the UI can render a domain badge and let operators switch.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.agents.domain_config import (
    DomainConfig,
    get_active_domain,
    list_domains,
    reset_active_domain,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Domain packs are loaded lazily from YAML by domain_config on first access
# (get_active_domain / list_domains), so no import-time registration is needed here.


class DomainFaultSummary(BaseModel):
    name: str
    pattern_key: str
    severity: float
    indicator_count: int


class DomainActiveResponse(BaseModel):
    name: str
    display_name: str
    channel_roles: Dict[str, str]
    signature_channels: List[str]
    pattern_keys: List[str]
    thresholds: Dict[str, float] = Field(default_factory=dict)
    fault_types: List[DomainFaultSummary]
    source: str = Field(
        default="python",
        description="Where the active domain came from: 'yaml' or 'python'.",
    )


class DomainListResponse(BaseModel):
    active: str
    domains: List[str]


def _summarise(config: DomainConfig) -> DomainActiveResponse:
    thresholds = getattr(config, "thresholds", {}) or {}
    return DomainActiveResponse(
        name=config.name,
        display_name=config.display_name,
        channel_roles=dict(config.channel_roles),
        signature_channels=sorted(config.signature_channels),
        pattern_keys=list(config.pattern_keys),
        thresholds={k: float(v) for k, v in thresholds.items()},
        fault_types=[
            DomainFaultSummary(
                name=ft.name,
                pattern_key=ft.pattern_key,
                severity=ft.severity,
                indicator_count=len(ft.indicators),
            )
            for ft in config.fault_types
        ],
        source="yaml" if getattr(config, "loaded_from_yaml", False) else "python",
    )


@router.get("/domain/active", response_model=DomainActiveResponse)
async def get_active() -> DomainActiveResponse:
    """Return the currently active domain config."""
    config = get_active_domain()
    return _summarise(config)


@router.get("/domain/list", response_model=DomainListResponse)
async def list_available() -> DomainListResponse:
    """Return every registered domain + the active one."""
    active = get_active_domain()
    return DomainListResponse(active=active.name, domains=sorted(list_domains()))


class DomainSetRequest(BaseModel):
    name: str


@router.post("/domain/active", response_model=DomainActiveResponse)
async def set_active(request: DomainSetRequest) -> DomainActiveResponse:
    """Switch the active domain by name.

    Uses the ``DOMAIN_PROFILE`` env var to pin the choice; downstream
    callers of :func:`get_active_domain` observe the new value after
    the cache is cleared.
    """
    if request.name not in list_domains():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown domain {request.name!r}; available={list_domains()}",
        )
    os.environ["DOMAIN_PROFILE"] = request.name
    reset_active_domain()
    return _summarise(get_active_domain())

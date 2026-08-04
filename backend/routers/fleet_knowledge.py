from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.knowledge import ContextKeys, KnowledgePack
from backend.agents.knowledge.fleet import (
    aggregate_fleet_packs,
    append_family_review_to_store,
    apply_family_reviews,
    append_pack_to_store,
    fleet_review_store_path,
    fleet_store_path,
    load_family_reviews_from_store,
    load_packs_from_store,
    FleetFamilyReview,
)

router = APIRouter(prefix="/fleet", tags=["fleet-knowledge"])


def _store_path() -> Path:
    raw = (os.environ.get("FLEET_KNOWLEDGE_STORE_PATH") or "").strip() or "data/fleet_packs.jsonl"
    return fleet_store_path(raw)


def _review_store_path() -> Path:
    raw = (os.environ.get("FLEET_KNOWLEDGE_REVIEWS_PATH") or "").strip() or "data/fleet_family_reviews.jsonl"
    return fleet_review_store_path(raw)


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
                "message": "fleet pack ingest requires a complete context",
                "missing_context_keys": missing,
            },
        )
    return ctx


def _coerce_pack(payload: Dict[str, Any]) -> KnowledgePack:
    known = {field_name for field_name in KnowledgePack.__dataclass_fields__}
    pack = KnowledgePack(**{k: v for k, v in payload.items() if k in known})
    ctx = _validated_context(dict(pack.context or {}))
    pack.context = ctx.to_dict()
    return pack


class FleetPackIngestResponse(BaseModel):
    site: str
    summary: Dict[str, int]
    stored_count: int


class FleetPackResponse(BaseModel):
    built_at: str
    context: Dict[str, Optional[str]]
    pack_count: int
    site_count: int
    k_anonymity_threshold: int
    k_anonymity_met: bool
    source_sites: list[str] = Field(default_factory=list)
    pattern_priors: Dict[str, Any] = Field(default_factory=dict)
    discovered_patterns: Dict[str, Any] = Field(default_factory=dict)
    discovery_families: list[Dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FleetFamilyReviewRequest(BaseModel):
    family_key: str
    context: Dict[str, Optional[str]]
    canonical_name: Optional[str] = None
    status: str = "candidate"
    reviewer: Optional[str] = None
    reason: Optional[str] = None


class FleetFamilyReviewResponse(BaseModel):
    family_key: str
    context: Dict[str, Optional[str]]
    canonical_name: Optional[str] = None
    status: str
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    reviewed_at: str
    stored_count: int


@router.post("/pack", response_model=FleetPackIngestResponse)
async def ingest_fleet_pack(payload: Dict[str, Any]) -> FleetPackIngestResponse:
    pack = _coerce_pack(payload)
    stored_count = append_pack_to_store(pack, _store_path())
    return FleetPackIngestResponse(
        site=pack.site,
        summary=pack.summary(),
        stored_count=stored_count,
    )


@router.post("/family/review", response_model=FleetFamilyReviewResponse)
async def review_fleet_family(request: FleetFamilyReviewRequest) -> FleetFamilyReviewResponse:
    ctx = _validated_context(request.context)
    review = FleetFamilyReview(
        family_key=request.family_key,
        context=ctx.to_dict(),
        canonical_name=(request.canonical_name or "").strip() or None,
        status=(request.status or "candidate").strip() or "candidate",
        reviewer=(request.reviewer or "").strip() or None,
        reason=(request.reason or "").strip() or None,
    )
    stored_count = append_family_review_to_store(review, _review_store_path())
    return FleetFamilyReviewResponse(stored_count=stored_count, **review.to_dict())


@router.get("/pack", response_model=FleetPackResponse)
async def get_fleet_pack(
    machine_type: str = Query(...),
    tool_type: str = Query(...),
    material: str = Query(...),
    regime: str = Query(...),
    min_sites: int = Query(default=3, ge=1),
) -> FleetPackResponse:
    ctx = ContextKeys(
        machine_type=machine_type,
        tool_type=tool_type,
        material=material,
        regime=regime,
    )
    aggregate = aggregate_fleet_packs(
        load_packs_from_store(_store_path()),
        ctx.to_dict(),
        min_sites=min_sites,
    )
    aggregate = apply_family_reviews(
        aggregate,
        load_family_reviews_from_store(_review_store_path()),
    )
    return FleetPackResponse(**aggregate.to_dict())
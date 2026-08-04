"""
Memory Feedback Handler - Process user feedback on memories.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module handles user feedback on memory records.
# Simple implementation - does not trigger re-analysis (as per requirements).
# ===========================================================================

Responsibilities:
1. Store user comments/annotations
2. Track confirmations and dismissals
3. Update pattern priors based on feedback
4. Maintain feedback audit trail
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uuid

import numpy as np
from pydantic import BaseModel, Field

from ..core.context import CuttingContext
from .scorer import FEEDBACK_WEIGHTS, normalize_pattern_key

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# [PROTOTYPE_LLM_MEMORY_V1] - Feedback types
class FeedbackAction(str, Enum):
    """Types of feedback actions."""
    CONFIRM = "confirm"  # User confirms event is significant
    DISMISS = "dismiss"  # User dismisses as not significant
    COMMENT = "comment"  # User adds comment
    LABEL = "label"  # User assigns/changes label
    TAG = "tag"  # User adds tags
    LINK = "link"  # User links to another memory


# [PROTOTYPE_LLM_MEMORY_V1] - Feedback record
class FeedbackRecord(BaseModel):
    """Individual feedback action record."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    action: FeedbackAction
    user_id: str = "operator"  # Default user
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Pydantic v2 compatible config
    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()}
    }


# [PROTOTYPE_LLM_MEMORY_V1] - Feedback request schema
class MemoryFeedbackRequest(BaseModel):
    """Request to add feedback to a memory."""
    action: FeedbackAction
    user_id: str = "operator"
    
    # For COMMENT action
    comment: Optional[str] = None
    
    # For LABEL action
    label: Optional[str] = None
    
    # For TAG action
    tags: Optional[List[str]] = None
    
    # For LINK action
    linked_memory_ids: Optional[List[str]] = None
    
    # For CONFIRM/DISMISS - optional reason
    reason: Optional[str] = None
    severity_target: Optional[Literal["info", "warning", "critical"]] = None

    # Two-tier recommendation model (2026-07-12): which facet of the alert the
    # operator is rating. "explanation" = the why/what diagnosis; "recommendation"
    # = the immediate breakage-avoidance action. Absent (None) → whole-alert
    # feedback, identical to prior behaviour. Recorded on the feedback record so
    # explanation vs recommendation quality can be tracked independently.
    aspect: Optional[Literal["explanation", "recommendation"]] = None

    # Episode-level alerting (plan 1.4). When the operator adjudicates an alert
    # that represents an episode (a run of windows sharing a fault signature),
    # the UI passes the alert's episode_id. The handler then applies the learning
    # update (prior EMA + model-trust counts) ONCE per episode instead of once
    # per window, while still recording each memory's ground-truth label. Absent
    # (None) → no dedup, identical to per-memory behaviour.
    episode_id: Optional[str] = None


# [PROTOTYPE_LLM_MEMORY_V1] - Feedback response
class MemoryFeedbackResponse(BaseModel):
    """Response after processing feedback."""
    success: bool
    feedback_id: str
    memory_id: str
    action: FeedbackAction
    message: str
    updated_fields: List[str] = Field(default_factory=list)


# [PROTOTYPE_LLM_MEMORY_V1] - Pattern knowledge update
@dataclass
class PatternFeedbackUpdate:
    """Update to pattern knowledge from feedback."""
    pattern_key: str
    was_significant: bool
    user_id: str
    memory_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# [PROTOTYPE_LLM_MEMORY_V1] - Callback types
from typing import Callable, Awaitable
FeedbackCallback = Callable[[str, FeedbackAction, Dict[str, Any]], Awaitable[None]]


# [PROTOTYPE_LLM_MEMORY_V1] - Main handler class
class MemoryFeedbackHandler:
    """
    Handles user feedback on memories.
    
    [INTEGRATION_POINT] Requires MemoryStore for persistence.
    [INTEGRATION_POINT] Should notify SignificanceScorer of feedback.
    """
    
    def __init__(
        self,
        memory_store: Any = None,  # MemoryStore instance
        significance_scorer: Any = None,  # SignificanceScorer instance
        dpp_registry: Any = None,  # optional DPPRegistry for CO2/cost impact-weighting
    ):
        self.store = memory_store
        self.scorer = significance_scorer

        # [PROTOTYPE_LLM_MEMORY_V1] - In-memory feedback storage
        # Production should use database
        self._feedback_records: Dict[str, List[FeedbackRecord]] = {}  # memory_id -> records
        self._pattern_updates: List[PatternFeedbackUpdate] = []

        # Episode-level learning dedup (plan 1.4): episode_ids whose learning
        # update (prior EMA + model trust) has already been applied, so later
        # windows of the same episode only record their label, not another
        # learning nudge. Empty/unused when the UI does not pass episode_id.
        self._episodes_learned: set = set()

        # Optional DPP/CO2 impact-weighting of the learning weight (flag-gated, default
        # off so behaviour is unchanged). When on, a confirm/dismiss on a high-carbon
        # part moves the prior faster — the loop prioritises protecting high-impact parts.
        self._impact_weighting = _env_flag("FEEDBACK_IMPACT_WEIGHTING", False)
        self._impact_ref_pcf_kg = float(os.getenv("FEEDBACK_IMPACT_REF_PCF_KG", "900") or 900)
        self._impact_max = float(os.getenv("FEEDBACK_IMPACT_MAX", "3.0") or 3.0)
        self._dpp_registry = dpp_registry
        self._dpp_loaded = dpp_registry is not None

        # Optional collaborators wired by the orchestrator at runtime.
        self.pattern_discovery: Any = None
        self.retrainer: Any = None
        self.harmonic_retrainer: Any = None
        
        # Callbacks for external notification
        self._callbacks: List[FeedbackCallback] = []
    
    def register_callback(self, callback: FeedbackCallback):
        """Register callback to be notified of feedback."""
        self._callbacks.append(callback)
    
    async def process_feedback(
        self,
        memory_id: str,
        request: MemoryFeedbackRequest,
    ) -> MemoryFeedbackResponse:
        """
        Process feedback for a memory.
        
        Args:
            memory_id: ID of the memory
            request: Feedback request
        
        Returns:
            MemoryFeedbackResponse with result
        """
        # Get memory (if store available)
        memory = None
        if self.store:
            memory = await self._get_memory(memory_id)
            if memory is None:
                return MemoryFeedbackResponse(
                    success=False,
                    feedback_id="",
                    memory_id=memory_id,
                    action=request.action,
                    message=f"Memory {memory_id} not found",
                )
        
        # Create feedback record
        record = FeedbackRecord(
            memory_id=memory_id,
            action=request.action,
            user_id=request.user_id,
            data=self._extract_feedback_data(request),
        )
        
        # Store feedback
        if memory_id not in self._feedback_records:
            self._feedback_records[memory_id] = []
        self._feedback_records[memory_id].append(record)
        
        # Process based on action type
        updated_fields: List[str] = []
        
        if request.action == FeedbackAction.CONFIRM:
            await self._handle_confirm(memory_id, memory, request)
            updated_fields.append("significance_confirmed")
            
        elif request.action == FeedbackAction.DISMISS:
            await self._handle_dismiss(memory_id, memory, request)
            updated_fields.append("severity_corrected" if request.severity_target else "significance_dismissed")
            
        elif request.action == FeedbackAction.COMMENT:
            await self._handle_comment(memory_id, memory, request)
            updated_fields.append("annotation_text")
            
        elif request.action == FeedbackAction.LABEL:
            await self._handle_label(memory_id, memory, request)
            updated_fields.append("label")
            
        elif request.action == FeedbackAction.TAG:
            await self._handle_tag(memory_id, memory, request)
            updated_fields.append("tags")
            
        elif request.action == FeedbackAction.LINK:
            await self._handle_link(memory_id, memory, request)
            updated_fields.append("related_memory_ids")
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(memory_id, request.action, record.data)
            except Exception as e:
                logger.warning(f"Feedback callback failed: {e}")
        
        logger.info(f"Processed {request.action.value} feedback for memory {memory_id}")
        
        return MemoryFeedbackResponse(
            success=True,
            feedback_id=record.id,
            memory_id=memory_id,
            action=request.action,
            message=f"Feedback recorded: {request.action.value}",
            updated_fields=updated_fields,
        )
    
    def get_feedback_history(self, memory_id: str) -> List[FeedbackRecord]:
        """Get feedback history for a memory."""
        return self._feedback_records.get(memory_id, [])

    @staticmethod
    def _feedback_action_name(item: Any) -> str:
        action = getattr(item, "action", None)
        if action is None and isinstance(item, dict):
            action = item.get("action")
        if isinstance(action, FeedbackAction):
            return action.value
        return str(action or "")

    @staticmethod
    def _feedback_data(item: Any) -> Dict[str, Any]:
        data = getattr(item, "data", None)
        if data is None and isinstance(item, dict):
            data = item.get("data")
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _feedback_weight(item: Any) -> float:
        weight = getattr(item, "weight", None)
        if weight is None and isinstance(item, dict):
            weight = item.get("weight")
        if weight is None:
            data = MemoryFeedbackHandler._feedback_data(item)
            weight = data.get("weight", 0.0)
        try:
            return max(0.0, float(weight or 0.0))
        except (TypeError, ValueError):
            return 0.0
    
    def get_feedback_stats(self, memory_id: str) -> Dict[str, Any]:
        """Get feedback statistics for a memory."""
        items: List[Any] = []
        if self.store and hasattr(self.store, "list_feedback_events"):
            try:
                items = list(self.store.list_feedback_events(memory_id, limit=200))
            except Exception:
                items = []

        if not items:
            items = list(self._feedback_records.get(memory_id, []))

        confirms = 0
        dismisses = 0
        severity_corrections = 0
        passive_outcomes = 0
        comments = 0
        effective_weight_total = 0.0
        passive_outcome_weight_total = 0.0

        for item in items:
            action_name = self._feedback_action_name(item)
            data = self._feedback_data(item)
            weight = self._feedback_weight(item)
            source = str(data.get("source") or "").strip().lower()
            is_passive = source in {
                "passive_cycle_tracker",
                "passive_cycle_completed_without_intervention",
            }
            is_severity_correction = action_name == "severity_correction" or bool(data.get("severity_target"))

            if action_name == FeedbackAction.CONFIRM.value:
                confirms += 1
                effective_weight_total += weight
                continue
            if action_name == FeedbackAction.COMMENT.value:
                comments += 1
                continue
            if is_severity_correction:
                severity_corrections += 1
                effective_weight_total += weight
                continue
            if action_name == FeedbackAction.DISMISS.value and is_passive:
                passive_outcomes += 1
                passive_outcome_weight_total += weight
                effective_weight_total += weight
                continue
            if action_name == FeedbackAction.DISMISS.value:
                dismisses += 1
                effective_weight_total += weight

        return {
            "total_feedback": len(items),
            "confirms": confirms,
            "dismisses": dismisses,
            "confirm_count": confirms,
            "dismiss_count": dismisses,
            "severity_corrections": severity_corrections,
            "passive_outcomes": passive_outcomes,
            "passive_outcome_count": passive_outcomes,
            "passive_outcome_weight_total": passive_outcome_weight_total,
            "effective_weight_total": effective_weight_total,
            "comments": comments,
            "net_significance": confirms - dismisses,
        }
    
    async def _get_memory(self, memory_id: str) -> Optional[Any]:
        """Get memory from store."""
        if not self.store:
            return None
        
        try:
            if hasattr(self.store, 'get'):
                return self.store.get(memory_id)
            elif hasattr(self.store, 'get_memory'):
                return await self.store.get_memory(memory_id)
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
        return None
    
    async def _update_memory(self, memory_id: str, updates: Dict[str, Any]):
        """Update memory in store."""
        if not self.store:
            return

        try:
            memory = await self._get_memory(memory_id)
            if memory is None:
                return

            # Normalize metadata updates: support callers passing either
            # (a) full metadata dict under "metadata" or
            # (b) legacy dotted keys like "metadata.user_confirmed".
            metadata_patch: Dict[str, Any] = {}
            field_updates: Dict[str, Any] = {}
            for key, value in (updates or {}).items():
                if key == "metadata" and isinstance(value, dict):
                    metadata_patch.update(value)
                elif key.startswith("metadata."):
                    metadata_patch[key[len("metadata."):]] = value
                else:
                    field_updates[key] = value

            if metadata_patch:
                existing_meta = dict(getattr(memory, "metadata", None) or {})
                existing_meta.update(metadata_patch)
                field_updates["metadata"] = existing_meta

            if not field_updates:
                return

            # Preferred API: MemoryStore.update(memory_id, **fields)
            if hasattr(self.store, "update") and callable(getattr(self.store, "update")):
                try:
                    self.store.update(memory_id, **field_updates)
                    return
                except TypeError:
                    # Back-compat for older store adapters: update(id, updates_dict)
                    self.store.update(memory_id, field_updates)
                    return

            # Async adapter surface
            if hasattr(self.store, "update_memory") and callable(getattr(self.store, "update_memory")):
                await self.store.update_memory(memory_id, field_updates)
                return

            # Last resort: mutate the object and re-store if supported.
            if hasattr(self.store, "store") and callable(getattr(self.store, "store")):
                for key, value in field_updates.items():
                    setattr(memory, key, value)
                self.store.store(memory)
                return

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}", exc_info=True)
    
    def _extract_feedback_data(self, request: MemoryFeedbackRequest) -> Dict[str, Any]:
        """Extract feedback data from request."""
        data: Dict[str, Any] = {}
        if request.comment:
            data["comment"] = request.comment
        if request.label:
            data["label"] = request.label
        if request.tags:
            data["tags"] = request.tags
        if request.linked_memory_ids:
            data["linked_memory_ids"] = request.linked_memory_ids
        if request.reason:
            data["reason"] = request.reason
        if request.severity_target:
            data["severity_target"] = request.severity_target
        if request.aspect:
            data["aspect"] = request.aspect
        if request.action in {FeedbackAction.CONFIRM, FeedbackAction.DISMISS}:
            data["source"] = self._resolve_feedback_source(request)
            data["weight"] = self._resolve_feedback_weight(request)
        return data

    def _resolve_feedback_source(self, request: MemoryFeedbackRequest) -> str:
        if request.action == FeedbackAction.CONFIRM:
            return "confirm_explicit"
        if request.action == FeedbackAction.DISMISS:
            if request.severity_target:
                return "severity_correction"
            if request.reason and str(request.reason).strip():
                return "dismiss_with_reason"
            return "dismiss_oneclick"
        return str(request.action.value)

    def _resolve_feedback_weight(self, request: MemoryFeedbackRequest) -> float:
        """Return the learning weight for confirm/dismiss feedback."""
        if request.action == FeedbackAction.CONFIRM:
            return FEEDBACK_WEIGHTS["confirm_explicit"]
        if request.action == FeedbackAction.DISMISS:
            if request.severity_target:
                return FEEDBACK_WEIGHTS["severity_correction"]
            if request.reason and str(request.reason).strip():
                return FEEDBACK_WEIGHTS["dismiss_with_reason"]
            return FEEDBACK_WEIGHTS["dismiss_oneclick"]
        return 1.0

    def _get_dpp_registry(self) -> Any:
        """Lazily load a DPPRegistry from the configured directory (cached)."""
        if not self._dpp_loaded:
            self._dpp_loaded = True
            try:
                from ..maas import DPPRegistry
                self._dpp_registry = DPPRegistry.from_dir(
                    os.getenv("DPP_DIR", "data/supplementary_data")
                )
            except Exception as exc:  # defensive: never let impact-weighting break feedback
                logger.warning("impact-weighting: DPP load failed (%s); disabling", exc)
                self._dpp_registry = None
        return self._dpp_registry

    def _impact_multiplier(self, memory: Any) -> float:
        """Bounded CO2/cost multiplier for the learning weight (1.0 when off/unknown).

        Scales the prior/threshold delta by the part's embodied carbon so confirmed
        catches on high-PCF parts are learned faster. Applied symmetrically to confirm
        and dismiss so it changes learning *rate*, not direction (no upward bias).
        """
        if not self._impact_weighting:
            return 1.0
        registry = self._get_dpp_registry()
        if registry is None:
            return 1.0
        event_id = self._resolve_part_event_id(memory)
        impact = registry.resolve(event_id)
        if impact is None:
            return 1.0
        pcf = impact.pcf_processing_kg or impact.pcf_total_kg
        if not pcf or self._impact_ref_pcf_kg <= 0:
            return 1.0
        # floor at 1.0 (never down-weight cheap parts), cap at _impact_max
        return round(max(1.0, min(self._impact_max, pcf / self._impact_ref_pcf_kg)), 3)

    def _resolve_part_event_id(self, memory: Any) -> Optional[str]:
        """Find a part/event identifier on the memory to key the DPP lookup."""
        metadata = self._memory_metadata(memory) if memory is not None else {}
        for key in ("event_id", "part_id", "part", "work_order", "of"):
            value = metadata.get(key) if isinstance(metadata, dict) else None
            if value:
                return str(value)
        return None
    
    def _should_apply_episode_learning(self, request: MemoryFeedbackRequest) -> bool:
        """Episode-level dedup (plan 1.4).

        Returns True when the learning update (prior EMA + model-trust counts)
        should be applied for this feedback. With no ``episode_id`` (the default)
        this is always True — behaviour identical to per-memory feedback. When an
        ``episode_id`` is given, only the first adjudication of that episode
        applies learning; later windows of the same episode record their label
        but do not nudge the priors again.
        """
        eid = getattr(request, "episode_id", None)
        if not eid:
            return True
        if eid in self._episodes_learned:
            return False
        self._episodes_learned.add(eid)
        return True

    async def _handle_confirm(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle confirmation feedback."""
        feedback_weight = self._resolve_feedback_weight(request) * self._impact_multiplier(memory)
        cutting_context = self._memory_cutting_context(self._memory_metadata(memory)) if memory is not None else None
        apply_learning = self._should_apply_episode_learning(request)

        # Update memory metadata — the per-window ground-truth label, always applied.
        await self._update_memory(memory_id, {
            "metadata": {
                "user_confirmed": True,
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
                "confirmed_by": request.user_id,
            }
        })

        if apply_learning:
            # Update pattern priors
            if memory and hasattr(memory, 'pattern_keys'):
                for pattern in memory.pattern_keys:
                    self._update_pattern_prior(pattern.key, True, request.user_id, memory_id)

            # Persist feedback event to the durable store so scorer's
            # get_pattern_prior() observes updated confirm/dismiss counts.
            self._persist_feedback_event(memory_id, memory, "confirm", request)

            # Notify scorer
            if self.scorer and memory:
                for pattern in memory.pattern_keys:
                    self.scorer.update_pattern_prior(
                        pattern.key,
                        was_significant=True,
                        context=cutting_context,
                        weight=feedback_weight,
                        source=self._resolve_feedback_source(request),
                    )

            self._apply_learning_feedback(
                memory_id=memory_id,
                memory=memory,
                request=request,
                was_significant=True,
                feedback_weight=feedback_weight,
            )
        else:
            logger.debug(
                "Episode %s already adjudicated — recording confirm label only (no prior update)",
                request.episode_id,
            )
    
    async def _handle_dismiss(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle dismissal feedback."""
        feedback_weight = self._resolve_feedback_weight(request) * self._impact_multiplier(memory)
        is_severity_correction = bool(request.severity_target)
        metadata = self._memory_metadata(memory) if memory is not None else {}
        cutting_context = self._memory_cutting_context(metadata)
        current_score = self._memory_significance_score(metadata)
        current_action = self._memory_significance_action(metadata)
        # Severity corrections re-tune a specific memory and are not episode-
        # deduped; a plain dismiss is deduped per episode (plan 1.4).
        apply_learning = is_severity_correction or self._should_apply_episode_learning(request)

        # Update memory metadata — per-window ground-truth label, always applied.
        await self._update_memory(memory_id, {
            "metadata": {
                "user_dismissed": True,
                "dismissed_at": datetime.now(timezone.utc).isoformat(),
                "dismissed_by": request.user_id,
                "dismiss_reason": request.reason,
                "severity_target": request.severity_target,
            }
        })

        if not apply_learning:
            logger.debug(
                "Episode %s already adjudicated — recording dismiss label only (no prior update)",
                request.episode_id,
            )
            return

        # Update pattern priors
        if not is_severity_correction and memory and hasattr(memory, 'pattern_keys'):
            for pattern in memory.pattern_keys:
                self._update_pattern_prior(pattern.key, False, request.user_id, memory_id)

        # Persist feedback event to the durable store so scorer's
        # get_pattern_prior() observes updated confirm/dismiss counts.
        self._persist_feedback_event(
            memory_id,
            memory,
            "severity_correction" if is_severity_correction else "dismiss",
            request,
        )

        # Notify scorer
        if self.scorer and memory:
            for pattern in memory.pattern_keys:
                if is_severity_correction and hasattr(self.scorer, "record_severity_correction"):
                    self.scorer.record_severity_correction(
                        pattern.key,
                        target_severity=str(request.severity_target),
                        current_score=current_score,
                        current_severity=current_action,
                        weight=feedback_weight,
                    )
                else:
                    self.scorer.update_pattern_prior(
                        pattern.key,
                        was_significant=False,
                        context=cutting_context,
                        weight=feedback_weight,
                        source=self._resolve_feedback_source(request),
                    )

        if not is_severity_correction:
            self._apply_learning_feedback(
                memory_id=memory_id,
                memory=memory,
                request=request,
                was_significant=False,
                feedback_weight=feedback_weight,
            )
    
    async def _handle_comment(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle comment feedback.

        Agent I (2026-04-24): also run the LLM comment classifier to extract
        structured metadata (root_cause, action_taken, tool_change) and merge
        it into ``memory.metadata_json.comment_classification``. Best-effort:
        classification failures never abort feedback processing.
        """
        if not request.comment:
            return

        # Append to existing annotation or create new
        if memory and hasattr(memory, 'annotation_text'):
            new_annotation = f"{memory.annotation_text}\n\n[{request.user_id}]: {request.comment}"
        else:
            new_annotation = f"[{request.user_id}]: {request.comment}"

        updates: Dict[str, Any] = {"annotation_text": new_annotation}

        try:
            from ..llm.comment_classifier import classify_comment
            classification = await classify_comment(
                request.comment,
                llm_agent=getattr(self, "_llm_agent", None),
            )
            # Merge into metadata under a dedicated key; _update_memory's
            # metadata-normalisation path handles dotted keys.
            updates["metadata.comment_classification"] = classification
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("Comment classifier failed (non-fatal): %s", exc)

        await self._update_memory(memory_id, updates)
        self._persist_feedback_event(memory_id, memory, FeedbackAction.COMMENT.value, request)
    
    async def _handle_label(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle label feedback."""
        if not request.label:
            return
        
        await self._update_memory(memory_id, {
            "label": request.label,
        })
        self._persist_feedback_event(memory_id, memory, FeedbackAction.LABEL.value, request)
    
    async def _handle_tag(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle tag feedback."""
        if not request.tags:
            return
        
        # Merge with existing tags
        existing_tags = []
        if memory and hasattr(memory, 'tags'):
            existing_tags = memory.tags or []
        
        new_tags = list(set(existing_tags + request.tags))
        
        await self._update_memory(memory_id, {
            "tags": new_tags,
        })
        self._persist_feedback_event(memory_id, memory, FeedbackAction.TAG.value, request)
    
    async def _handle_link(self, memory_id: str, memory: Any, request: MemoryFeedbackRequest):
        """Handle link feedback."""
        if not request.linked_memory_ids:
            return
        
        # Merge with existing links
        existing_links = []
        if memory and hasattr(memory, 'related_memory_ids'):
            existing_links = memory.related_memory_ids or []
        
        new_links = list(set(existing_links + request.linked_memory_ids))
        
        await self._update_memory(memory_id, {
            "related_memory_ids": new_links,
        })
    
    def _update_pattern_prior(
        self,
        pattern_key: str,
        was_significant: bool,
        user_id: str,
        memory_id: str,
    ):
        """Record pattern feedback update."""
        update = PatternFeedbackUpdate(
            pattern_key=pattern_key,
            was_significant=was_significant,
            user_id=user_id,
            memory_id=memory_id,
        )
        self._pattern_updates.append(update)
        logger.debug(f"Pattern prior update: {pattern_key} -> {'significant' if was_significant else 'not significant'}")

    def _persist_feedback_event(
        self,
        memory_id: str,
        memory: Any,
        action: str,
        request: MemoryFeedbackRequest,
    ) -> None:
        """Append a feedback event to the durable store (best-effort).

        The ``SignificanceScorer.get_pattern_prior`` implementation reads
        confirm/dismiss counts from the feedback store when available. If
        we never persist the event, the scorer always returns the neutral
        prior (0.5) regardless of how many times ``update_pattern_prior``
        was called. Writing the event here closes that gap for all store
        backends that implement ``add_feedback_event`` (sqlite, Neo4j,
        and the in-memory orchestrator adapter).
        """
        if not self.store or not hasattr(self.store, "add_feedback_event"):
            return
        pattern_keys: List[str] = []
        if memory is not None and hasattr(memory, "pattern_keys"):
            pattern_keys = [
                normalize_pattern_key(str(getattr(p, "key", "")).strip())
                for p in memory.pattern_keys
                if getattr(p, "key", None)
            ]
            pattern_keys = [p for p in pattern_keys if p]
        event_data = self._extract_feedback_data(request)
        feedback_weight = float(event_data.get("weight", 1.0) or 1.0)
        try:
            self.store.add_feedback_event(
                memory_id=memory_id,
                action=str(action),
                user_id=str(request.user_id or ""),
                pattern_keys=pattern_keys,
                data=event_data or None,
                weight=feedback_weight,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Failed to persist feedback event for memory %s: %s",
                memory_id, exc,
            )

    def _apply_learning_feedback(
        self,
        *,
        memory_id: str,
        memory: Any,
        request: MemoryFeedbackRequest,
        was_significant: bool,
        feedback_weight: float,
    ) -> None:
        """Feed confirm/dismiss outcomes into the live learning hooks.

        Normal operator feedback should shape the same adaptive machinery that
        experiment review uses: discovery, suppression, retraining buffers, and
        the scorer's rule and weight adaptation.
        """
        if memory is None:
            return

        metadata = self._memory_metadata(memory)
        pattern_keys = self._memory_pattern_keys(memory)
        raw_metrics = self._memory_raw_metrics(metadata)
        cutting_context = self._memory_cutting_context(metadata)
        external_signals = self._memory_external_signals(metadata)
        triggered_rules = self._memory_triggered_rules(metadata)
        significance_score = self._memory_significance_score(metadata)
        significance_action = self._memory_significance_action(metadata)
        harmonic_context = self._memory_harmonic_context(metadata)
        harmonic_runtime = self._memory_harmonic_runtime(metadata)

        self._apply_scorer_feedback(
            was_significant=was_significant,
            cutting_context=cutting_context,
            triggered_rules=triggered_rules,
            external_signals=external_signals,
            significance_score=significance_score,
            significance_action=significance_action,
            feedback_weight=feedback_weight,
        )
        self._apply_pattern_discovery_feedback(
            was_significant=was_significant,
            raw_metrics=raw_metrics,
            cutting_context=cutting_context,
            pattern_keys=pattern_keys,
            memory_id=memory_id,
            session_id=self._memory_session_id(memory),
        )
        self._apply_retrainer_feedback(
            was_significant=was_significant,
            raw_metrics=raw_metrics,
            pattern_keys=pattern_keys,
            memory_id=memory_id,
        )
        self._apply_harmonic_retrainer_feedback(
            was_significant=was_significant,
            raw_metrics=raw_metrics,
            harmonic_context=harmonic_context,
            harmonic_runtime=harmonic_runtime,
            cutting_context=cutting_context,
            external_signals=external_signals,
            metadata=metadata,
            memory_id=memory_id,
            session_id=self._memory_session_id(memory),
        )

    def _apply_scorer_feedback(
        self,
        *,
        was_significant: bool,
        cutting_context: Optional[CuttingContext],
        triggered_rules: List[str],
        external_signals: Dict[str, Any],
        significance_score: Optional[float],
        significance_action: Optional[str],
        feedback_weight: float,
    ) -> None:
        if not self.scorer:
            return

        try:
            if hasattr(self.scorer, "record_rule_feedback"):
                self.scorer.record_rule_feedback(triggered_rules, was_significant)
        except Exception:
            logger.debug("record_rule_feedback failed", exc_info=True)

        try:
            if hasattr(self.scorer, "update_weight_profile_from_feedback"):
                self.scorer.update_weight_profile_from_feedback(
                    cutting_context,
                    triggered_rules,
                    was_significant,
                )
        except Exception:
            logger.debug("update_weight_profile_from_feedback failed", exc_info=True)

        try:
            if hasattr(self.scorer, "record_model_feedback"):
                self.scorer.record_model_feedback(
                    triggered_rules=triggered_rules,
                    was_confirmed=was_significant,
                    external_signals=external_signals,
                    cutting_context=cutting_context,
                )
        except Exception:
            logger.debug("record_model_feedback failed", exc_info=True)

        try:
            if hasattr(self.scorer, "record_rl_feedback"):
                self.scorer.record_rl_feedback(
                    feedback_action="confirm" if was_significant else "dismiss",
                    was_alerted=(significance_action in {"alert", "critical"}),
                    external_signals=external_signals,
                    context=cutting_context,
                )
        except Exception:
            logger.debug("record_rl_feedback failed", exc_info=True)

        try:
            if (
                significance_score is not None
                and significance_action is not None
                and hasattr(self.scorer, "record_feedback_for_adaptive_thresholds")
            ):
                self.scorer.record_feedback_for_adaptive_thresholds(
                    float(significance_score),
                    str(significance_action),
                    was_significant,
                    weight=feedback_weight,
                )
        except Exception:
            logger.debug("record_feedback_for_adaptive_thresholds failed", exc_info=True)

    def _apply_pattern_discovery_feedback(
        self,
        *,
        was_significant: bool,
        raw_metrics: Dict[str, float],
        cutting_context: Optional[CuttingContext],
        pattern_keys: List[str],
        memory_id: str,
        session_id: str,
    ) -> None:
        if not raw_metrics or self.pattern_discovery is None:
            return

        try:
            if was_significant and hasattr(self.pattern_discovery, "analyse_confirmed_event"):
                self.pattern_discovery.analyse_confirmed_event(
                    raw_metrics,
                    existing_pattern_keys=pattern_keys,
                    scorer=self.scorer,
                    memory_id=memory_id,
                    session_id=session_id,
                    cutting_context=cutting_context,
                )
            elif not was_significant and hasattr(self.pattern_discovery, "analyse_dismissed_event"):
                self.pattern_discovery.analyse_dismissed_event(
                    raw_metrics,
                    existing_pattern_keys=pattern_keys,
                    memory_id=memory_id,
                    session_id=session_id,
                    cutting_context=cutting_context,
                )
        except Exception:
            logger.debug("pattern discovery feedback failed", exc_info=True)

    def _apply_retrainer_feedback(
        self,
        *,
        was_significant: bool,
        raw_metrics: Dict[str, float],
        pattern_keys: List[str],
        memory_id: str,
    ) -> None:
        if not raw_metrics:
            return

        retrainer = self.retrainer
        if retrainer is None:
            try:
                from .retrainer import get_retrainer

                retrainer = get_retrainer(
                    model_confidence_path=getattr(self.scorer, "_model_confidence_path", None),
                )
                self.retrainer = retrainer
            except Exception:
                logger.debug("retrainer unavailable for feedback buffering", exc_info=True)
                return

        try:
            from ..processing.classical_models import features_from_dict

            feature_vec = np.asarray(features_from_dict(raw_metrics), dtype=np.float64)
        except Exception:
            logger.debug("feature extraction for retrainer failed", exc_info=True)
            return

        try:
            retrainer.record_feedback(
                features=feature_vec,
                is_significant=was_significant,
                pattern_keys=pattern_keys,
                memory_id=memory_id,
            )
        except Exception:
            logger.debug("retrainer.record_feedback failed", exc_info=True)

    def _apply_harmonic_retrainer_feedback(
        self,
        *,
        was_significant: bool,
        raw_metrics: Dict[str, float],
        harmonic_context: Dict[str, Any],
        harmonic_runtime: Dict[str, Any],
        cutting_context: Optional[CuttingContext],
        external_signals: Dict[str, Any],
        metadata: Dict[str, Any],
        memory_id: str,
        session_id: str,
    ) -> None:
        if not raw_metrics and not harmonic_context:
            return
        if not external_signals.get("harmonic_context_score") and not harmonic_context:
            return

        retrainer = self.harmonic_retrainer
        if retrainer is None:
            try:
                from ..processing.harmonic_feedback_retrainer import get_harmonic_feedback_retrainer

                retrainer = get_harmonic_feedback_retrainer(
                    model_confidence_path=getattr(self.scorer, "_model_confidence_path", None),
                )
                self.harmonic_retrainer = retrainer
            except Exception:
                logger.debug("harmonic retrainer unavailable for feedback buffering", exc_info=True)
                return

        try:
            retrainer.record_feedback(
                was_significant=was_significant,
                raw_metrics=raw_metrics,
                harmonic_context=harmonic_context,
                harmonic_runtime=harmonic_runtime,
                cutting_context=cutting_context.model_dump() if cutting_context else None,
                source=str(metadata.get("source") or ""),
                casedata=metadata.get("casedata") if isinstance(metadata.get("casedata"), dict) else None,
                memory_id=memory_id,
                session_id=session_id,
            )
        except Exception:
            logger.debug("harmonic_retrainer.record_feedback failed", exc_info=True)

    @staticmethod
    def _memory_metadata(memory: Any) -> Dict[str, Any]:
        metadata = getattr(memory, "metadata", None)
        return dict(metadata or {}) if isinstance(metadata, dict) else {}

    @staticmethod
    def _memory_pattern_keys(memory: Any) -> List[str]:
        keys: List[str] = []
        for pattern in getattr(memory, "pattern_keys", None) or []:
            key = getattr(pattern, "key", None)
            if key:
                keys.append(str(key))
        return keys

    @staticmethod
    def _memory_raw_metrics(metadata: Dict[str, Any]) -> Dict[str, float]:
        raw = metadata.get("raw_metrics") or {}
        if not isinstance(raw, dict):
            return {}
        cleaned: Dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                cleaned[str(key)] = float(value)
        return cleaned

    @staticmethod
    def _memory_cutting_context(metadata: Dict[str, Any]) -> Optional[CuttingContext]:
        raw_context = metadata.get("cutting_context")
        if not isinstance(raw_context, dict) or not raw_context:
            return None
        try:
            return CuttingContext.model_validate(raw_context)
        except Exception:
            logger.debug("failed to rebuild CuttingContext from memory metadata", exc_info=True)
            return None

    @staticmethod
    def _memory_external_signals(metadata: Dict[str, Any]) -> Dict[str, Any]:
        raw = metadata.get("external_signals") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _memory_harmonic_context(metadata: Dict[str, Any]) -> Dict[str, Any]:
        raw = metadata.get("harmonic_context") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _memory_harmonic_runtime(metadata: Dict[str, Any]) -> Dict[str, Any]:
        raw = metadata.get("harmonic_runtime") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _memory_triggered_rules(metadata: Dict[str, Any]) -> List[str]:
        raw = metadata.get("triggered_rules")
        if isinstance(raw, list):
            return [str(item) for item in raw if item]
        significance = metadata.get("significance") or {}
        if isinstance(significance, dict):
            rules = significance.get("triggered_rules")
            if isinstance(rules, list):
                return [str(item) for item in rules if item]
        return []

    @staticmethod
    def _memory_significance_score(metadata: Dict[str, Any]) -> Optional[float]:
        value = metadata.get("significance_score")
        if isinstance(value, (int, float)):
            return float(value)
        significance = metadata.get("significance") or {}
        if isinstance(significance, dict):
            nested = significance.get("score")
            if isinstance(nested, (int, float)):
                return float(nested)
        return None

    @staticmethod
    def _memory_significance_action(metadata: Dict[str, Any]) -> Optional[str]:
        value = metadata.get("significance_action")
        if isinstance(value, str) and value:
            return value
        significance = metadata.get("significance") or {}
        if isinstance(significance, dict):
            nested = significance.get("action")
            if isinstance(nested, str) and nested:
                return nested
        return None

    @staticmethod
    def _memory_session_id(memory: Any) -> str:
        value = getattr(memory, "session_id", None)
        return str(value or "")


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function
def create_feedback_handler(
    memory_store: Any = None,
    significance_scorer: Any = None,
) -> MemoryFeedbackHandler:
    """Create a MemoryFeedbackHandler instance."""
    return MemoryFeedbackHandler(memory_store, significance_scorer)

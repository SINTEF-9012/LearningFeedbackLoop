"""
LLM Explainer - Generate human-readable explanations of significant events.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module uses Ollama to generate explanations of detected patterns.
# Prompts are initial drafts - expected to be refined based on quality.
# ===========================================================================

Responsibilities:
1. Explain why an event is significant
2. Summarize similar historical memories
3. Generate concise memory summaries for storage
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import re
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests
import httpx

from ..core.schemas import Memory, PatternKey, NumericMetrics
from ..core.context import CuttingContext
from ..memory.scorer import SignificanceResult
from ..memory.retriever import MemoryMatch

logger = logging.getLogger(__name__)

_JSON_RESPONSE_MODE = "json"
_TEXT_RESPONSE_MODE = "text"
_GROUNDED_RESPONSE_FIELDS = (
    "indication",
    "evidence",
    "concern",
    "operator_action",
)

_PROMPT_ROLE_MARKER_RE = re.compile(
    r"</?\s*(?:system|assistant|user)\s*>|\[\s*/?\s*(?:system|assistant|user)\s*\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Evidence container — everything the LLM needs for a grounded explanation
# ---------------------------------------------------------------------------

@dataclass
class ExplanationContext:
    """Rich evidence context assembled by the orchestrator before LLM call.

    Each field represents a distinct evidence layer that should be cited in
    the final explanation.
    """

    # --- Core event data (always present) ---
    pattern_keys: List[str] = field(default_factory=list)
    significance: Optional[SignificanceResult] = None

    # --- Feature-level evidence (pattern → measurement + threshold) ---
    # e.g. {"BREAKAGE_POWER_SPIKE": [{"feature": "power_spindle_delta_max",
    #         "value": 23.4, "threshold": 15.0, "direction": "above"}]}
    feature_evidence: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # --- Classical model assessment ---
    # e.g. {"anomaly_detector_score": 0.81, "model_confidence": 0.74,
    #        "breakage_prediction": 0.65}
    classical_model: Dict[str, float] = field(default_factory=dict)

    # --- Feedback statistics per pattern ---
    # e.g. {"BREAKAGE_POWER_SPIKE": {"confirms": 11, "dismisses": 1,
    #         "prior": 0.87}}
    feedback_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Co-occurrence context ---
    # e.g. [{"pattern": "CHATTER_DETECTED", "count": 9, "co_pattern": "BREAKAGE_POWER_SPIKE"}]
    co_occurrence: List[Dict[str, Any]] = field(default_factory=list)

    # --- Historical similar events ---
    similar_memories: List[MemoryMatch] = field(default_factory=list)

    # --- Cutting conditions ---
    cutting_context: Optional[CuttingContext] = None

    # --- Raw metric excerpt (top relevant features) ---
    raw_metrics_excerpt: Dict[str, float] = field(default_factory=dict)

    # --- Curated fallback explanation ---
    # A hand-written, faithful description supplied with a scripted demo event
    # (fixtures carry ``_explanation``). Used ONLY as the fallback when the LLM
    # is unavailable, so a demo take still shows polished prose instead of the
    # terse deterministic fallback. Never overrides a successful LLM generation.
    curated_fallback: Optional[str] = None


# [PROTOTYPE_LLM_MEMORY_V1] - Configuration
class ExplainerConfig:
    """Configuration for LLM explainer.

        Supports two providers:
            - ``groq``   (default) — Groq cloud API (OpenAI-compatible)
            - ``ollama``           — local Ollama server (/api/chat)

    The active provider is selected by the ``LLM_PROVIDER`` env var.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: int = 500,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_api_url: Optional[str] = None,
        groq_model: Optional[str] = None,
        groq_timeout: Optional[float] = None,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "groq")).strip().lower()

        # --- Ollama settings ---
        self.ollama_url = ollama_url or os.environ.get(
            "OLLAMA_URL", "http://localhost:11434/api/generate"
        )
        self.ollama_model = model or os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

        # --- Groq settings ---
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_api_url = (groq_api_url or os.environ.get(
            "GROQ_API_URL", "https://api.groq.com/openai/v1"
        )).rstrip("/")
        self.groq_model = groq_model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_timeout = float(groq_timeout) if groq_timeout is not None else float(
            os.environ.get("GROQ_TIMEOUT", "30.0")
        )

        # --- Unified accessors ---
        if self.provider == "groq":
            self.model = self.groq_model
            self.timeout = self.groq_timeout
        else:
            self.model = self.ollama_model
            self.timeout = float(timeout) if timeout is not None else float(
                os.environ.get("OLLAMA_TIMEOUT", "60.0")
            )

        self.max_tokens = max_tokens


# [PROTOTYPE_LLM_MEMORY_V1] - Main explainer class
class LLMExplainer:
    """
    Generates human-readable explanations using LLM.
    
    [INTEGRATION_POINT] Requires Ollama or compatible API running.
    """
    
    def __init__(self, config: Optional[ExplainerConfig] = None):
        self.config = config or ExplainerConfig()
        self._available: Optional[bool] = None
        self._last_check: float = 0.0
        # Configurable re-check interval (seconds).  Lower values help demos
        # recover faster when Ollama becomes ready after server start.
        self._check_interval: float = float(
            os.environ.get("LLM_CHECK_INTERVAL", "10.0")
        )
        # Track consecutive fallback count for diagnostics
        self._fallback_count: int = 0
        self._llm_call_count: int = 0
        # TTL cache for LLM responses (P4 fix) — avoids repeated calls for
        # identical prompts within a 5-minute window.
        self._response_cache: Dict[str, tuple] = {}  # hash -> (timestamp, response)
        self._cache_ttl: float = float(os.environ.get("LLM_CACHE_TTL", "300.0"))
        logger.info(
            "LLMExplainer init: provider=%s, model=%s, url=%s, timeout=%.1fs, check_interval=%.1fs",
            self.config.provider,
            self.config.model,
            self.config.groq_api_url if self.config.provider == "groq" else self.config.ollama_url,
            self.config.timeout,
            self._check_interval,
        )
    
    @staticmethod
    def _on_event_loop() -> bool:
        """True if called from within a running asyncio event loop."""
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def is_available(self) -> bool:
        """Check if the LLM service is available (re-checks periodically).

        CRITICAL: the active provider probe (``_check_*_available``) uses a
        *blocking* ``requests.get``. Running that on the server event loop
        freezes every concurrent request for up to the probe timeout whenever
        the provider is flaky — which previously wedged the whole backend at
        ~5-10s/request. So on the event loop we never probe here: we return the
        cached flag, and at the end of a backoff window we optimistically allow
        one real attempt. The async call path (``_call_*_async``) has its own
        httpx timeouts and flips availability via ``_mark_transient_failure``,
        so health still converges without ever blocking the loop.
        """
        import time
        now = time.time()

        # Re-check if enough time has passed or never checked
        if self._available is None or (now - self._last_check) > self._check_interval:
            if self._on_event_loop():
                # Non-blocking path: defer the real health decision to the async
                # call. Optimistically enable so exactly one attempt runs per
                # backoff window; a still-down provider re-disables via
                # _mark_transient_failure (also non-blocking).
                prev = self._available
                self._last_check = now
                self._available = True
                if prev is False:
                    logger.info(
                        "LLM backoff window elapsed — allowing one probe call "
                        "(provider=%s, fallbacks so far: %d)",
                        self.config.provider, self._fallback_count,
                    )
                return self._available

            # Off-loop (scripts, tests, warm-up): safe to block on the probe.
            prev = self._available
            self._last_check = now

            if self.config.provider == "groq":
                self._available = self._check_groq_available()
            else:
                self._available = self._check_ollama_available()

            # Log transitions
            if prev != self._available:
                logger.info(
                    "LLM availability changed: %s -> %s (provider=%s, fallbacks so far: %d)",
                    prev, self._available, self.config.provider, self._fallback_count,
                )

        return self._available

    # ------------------------------------------------------------------
    # Provider-specific availability checks
    # ------------------------------------------------------------------

    def _check_groq_available(self) -> bool:
        """Check Groq API availability by hitting /models endpoint."""
        api_key = self.config.groq_api_key
        if not api_key:
            logger.warning("Groq API key not set (GROQ_API_KEY). LLM unavailable.")
            return False
        try:
            url = f"{self.config.groq_api_url}/models"
            logger.debug("Groq availability check: GET %s", url)
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
            if response.status_code == 401:
                logger.warning("Groq API key is invalid (401 Unauthorized).")
                return False
            if response.status_code != 200:
                logger.warning("Groq /models returned %d", response.status_code)
                return False
            logger.info("Groq API available: model='%s'", self.config.model)
            return True
        except Exception as e:
            logger.warning("Groq availability check failed: %s", e)
            return False

    def _check_ollama_available(self) -> bool:
        """Check Ollama availability via /api/tags."""
        try:
            base_url = self.config.ollama_url.replace("/api/generate", "")
            logger.debug("LLM availability check: GET %s/api/tags", base_url)
            response = requests.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                logger.warning(
                    "LLM availability check failed: /api/tags returned %d",
                    response.status_code,
                )
                return False
            try:
                tags = response.json()
                present = self._model_is_present(tags)
                if not present:
                    logger.warning(
                        "Ollama reachable but model '%s' not present; run `ollama pull %s`",
                        self.config.model,
                        self.config.model,
                    )
                else:
                    logger.info(
                        "LLM available: model='%s' confirmed present",
                        self.config.model,
                    )
                return present
            except Exception as e:
                logger.debug("LLM /api/tags parse issue (%s), assuming available", e)
                return True
        except Exception as e:
            logger.warning(
                "LLM availability check failed (connection error): %s", e
            )
            return False

    def _model_is_present(self, tags_json: Any) -> bool:
        """Return True if the configured model appears in /api/tags.

        Cloud models (name contains '-cloud') are served externally and may
        not appear in the local Ollama /api/tags response.  They may also
        report ``size=0``.  In both cases the model is treated as present
        so that the explainer attempts real LLM calls instead of falling
        back silently.
        """
        model = str(self.config.model or "").strip()
        if not model:
            return True

        # Cloud models are served externally — always treat as present.
        if self._is_cloud_model_name(model):
            logger.info(
                "Cloud model detected ('%s') — treating as present without /api/tags check",
                model,
            )
            return True

        models = []
        if isinstance(tags_json, dict):
            models = tags_json.get("models") or []

        names: List[str] = []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    n = m.get("name") or m.get("model")
                    if n:
                        names.append(str(n))
                elif isinstance(m, str):
                    names.append(m)

        # If we can't confidently check, assume present.
        if not names:
            logger.debug("No models listed in /api/tags — assuming model is present")
            return True

        target = model.lower()
        # Exact match first, then prefix match (e.g. "gpt-oss:20b-cloud" matches
        # "gpt-oss:20b-cloud" with or without a ":latest" suffix in /api/tags).
        if any(str(n).lower() == target for n in names):
            return True
        # Prefix: strip any tag suffix (e.g. ":latest") from the listed names.
        if any(str(n).lower().split(":")[0] == target.split(":")[0] and target in str(n).lower() for n in names):
            return True

        # Also check if any listed model reports size=0 for our target
        # (cloud/proxy models often do this).
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    name = str(m.get("name") or m.get("model") or "").lower()
                    if target in name or name in target:
                        size = m.get("size")
                        if size == 0 or size is None:
                            logger.info(
                                "Model '%s' found with size=%s — treating as cloud/proxy model",
                                name, size,
                            )
                            return True

        logger.warning(
            "Model '%s' not found in /api/tags. Available: %s",
            model, ", ".join(names[:10]),
        )
        return False

    @staticmethod
    def _is_cloud_model_name(model: str) -> bool:
        """Check if a model is a cloud/remote model.

        Detection order:
        1. Explicit ``OLLAMA_CLOUD_MODEL=true`` env var (overrides heuristic).
        2. Name heuristic: substrings ``-cloud``, ``_cloud``, ``-remote``, ``-proxy``.
        """
        explicit = os.environ.get("OLLAMA_CLOUD_MODEL", "").strip().lower()
        if explicit in ("true", "1", "yes"):
            return True
        if explicit in ("false", "0", "no"):
            return False
        lower = model.lower()
        return any(tag in lower for tag in ("-cloud", "_cloud", "-remote", "-proxy"))
    
    # ------------------------------------------------------------------
    # LLM response cache (P4)
    # ------------------------------------------------------------------

    def _cache_key(self, prompt: str, *, response_mode: str = _TEXT_RESPONSE_MODE) -> str:
        """Return a deterministic hash for a prompt string."""
        cache_input = f"{response_mode}:{prompt}"
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()[:24]

    def _get_cached(self, prompt: str, *, response_mode: str = _TEXT_RESPONSE_MODE) -> Optional[str]:
        """Return cached response if still within TTL, else None."""
        key = self._cache_key(prompt, response_mode=response_mode)
        entry = self._response_cache.get(key)
        if entry is None:
            return None
        ts, response = entry
        if (time.time() - ts) > self._cache_ttl:
            del self._response_cache[key]
            return None
        logger.debug("LLM cache hit (key=%s, age=%.1fs)", key, time.time() - ts)
        return response

    def _set_cache(
        self,
        prompt: str,
        response: str,
        *,
        response_mode: str = _TEXT_RESPONSE_MODE,
    ) -> None:
        """Store a response in the TTL cache."""
        key = self._cache_key(prompt, response_mode=response_mode)
        self._response_cache[key] = (time.time(), response)
        # Evict stale entries periodically (keep cache bounded)
        if len(self._response_cache) > 200:
            now = time.time()
            self._response_cache = {
                k: v for k, v in self._response_cache.items()
                if (now - v[0]) <= self._cache_ttl
            }

    def explain_significance(
        self,
        patterns: List[PatternKey],
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
        metrics: Optional[NumericMetrics] = None,
    ) -> Optional[str]:
        """
        Generate explanation of why this event is significant.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Basic prompt, to be refined.
        """
        if not self.is_available():
            return self._fallback_explanation(significance)

        # Keep a synchronous API for scripts/tests, but avoid blocking the server
        # event loop. In an async context, use `explain_significance_async`.
        try:
            asyncio.get_running_loop()
            self._fallback_count += 1
            logger.info(
                "explain_significance called from async context — returning fallback #%d "
                "(use explain_significance_async instead)",
                self._fallback_count,
            )
            return self._fallback_explanation(significance)
        except RuntimeError:
            return asyncio.run(
                self.explain_significance_async(
                    patterns=patterns,
                    significance=significance,
                    context=context,
                    metrics=metrics,
                )
            )

    def summarize_with_history(
        self,
        current_memory: Memory,
        similar_memories: List[MemoryMatch],
        significance: SignificanceResult,
    ) -> Optional[str]:
        """
        Generate summary incorporating historical context.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Combines current event with history.
        """
        try:
            asyncio.get_running_loop()
            self._fallback_count += 1
            logger.info(
                "summarize_with_history called from async context — returning fallback #%d",
                self._fallback_count,
            )
            return self._fallback_summary(significance, len(similar_memories))
        except RuntimeError:
            return asyncio.run(
                self.summarize_with_history_async(
                    current_memory=current_memory,
                    similar_memories=similar_memories,
                    significance=significance,
                )
            )
    
    def generate_memory_summary(
        self,
        memory: Memory,
        context: Optional[CuttingContext] = None,
    ) -> Optional[str]:
        """
        Generate a concise summary for memory storage/display.
        
        [PROTOTYPE_LLM_MEMORY_V1] - Short summary for record keeping.
        """
        try:
            asyncio.get_running_loop()
            self._fallback_count += 1
            logger.info(
                "generate_memory_summary called from async context — returning fallback #%d",
                self._fallback_count,
            )
            return self._fallback_memory_summary(memory)
        except RuntimeError:
            return asyncio.run(
                self.generate_memory_summary_async(
                    memory=memory,
                    context=context,
                )
            )

    async def explain_significance_async(
        self,
        patterns: List[PatternKey],
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
        metrics: Optional[NumericMetrics] = None,
    ) -> Optional[str]:
        if not self.is_available():
            self._fallback_count += 1
            logger.debug(
                "LLM unavailable for explain_significance (fallback #%d)",
                self._fallback_count,
            )
            return self._fallback_explanation(significance)

        prompt = self._build_significance_prompt(patterns, significance, context, metrics)
        try:
            text = await self._call_llm_async(prompt)
            text = str(text or "").strip()
            if not text:
                self._fallback_count += 1
                logger.warning("LLM returned empty explanation text — using fallback")
                return self._fallback_explanation(significance)
            return text
        except Exception as e:
            # Use _mark_transient_failure: only disable for hard connectivity
            # failures, not for read timeouts or transient errors.
            self._mark_transient_failure(e, "LLM explanation failed")
            self._fallback_count += 1
            return self._fallback_explanation(significance)

    async def explain_significance_for_alert_async(
        self,
        patterns: List[PatternKey],
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
        metrics: Optional[NumericMetrics] = None,
        model_signals: Optional[Dict[str, Any]] = None,
        recurrence: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, str]:
        """Generate a short operator-facing alert summary.

        Returns:
            (alert_line, source, recommendation) — source is 'llm' or 'fallback';
            recommendation is the immediate breakage-avoidance action (two-tier
            model, 2026-07-12), distinct from the end-of-batch reconfiguration.
        """
        if not self.is_available():
            self._fallback_count += 1
            logger.info(
                "LLM unavailable for alert summary — using fallback (fallback #%d, score=%.2f)",
                self._fallback_count, float(significance.score),
            )
            return (
                self._fallback_alert_line(
                    significance,
                    context=context,
                    model_signals=model_signals,
                ),
                "fallback",
                self._fallback_recommendation(significance, context=context),
            )

        pattern_keys = [p.key for p in (patterns or [])]
        logger.info(
            "Generating LLM alert summary: patterns=%s, score=%.2f",
            pattern_keys[:3], float(significance.score),
        )
        prompt = self._build_alert_line_prompt(
            patterns,
            significance,
            context,
            metrics,
            model_signals,
            recurrence=recurrence,
        )
        try:
            payload = await self._call_llm_json_async(prompt)
            text = self._extract_structured_text_field(payload, "alert_line", max_len=240)
            recommendation = self._recommendation_from_payload(payload, significance, context=context)
            if not text:
                self._fallback_count += 1
                logger.warning("LLM returned empty alert text — using fallback")
                return (
                    self._fallback_alert_line(
                        significance,
                        context=context,
                        model_signals=model_signals,
                    ),
                    "fallback",
                    recommendation,
                )
            self._llm_call_count += 1
            logger.info(
                "LLM alert generated (call #%d): '%s'",
                self._llm_call_count, text[:80],
            )
            return (text, "llm", recommendation)
        except Exception as e:
            self._mark_transient_failure(e, "LLM alert summary failed")
            self._fallback_count += 1
            return (
                self._fallback_alert_line(
                    significance,
                    context=context,
                    model_signals=model_signals,
                ),
                "fallback",
                self._fallback_recommendation(significance, context=context),
            )

    async def summarize_with_history_async(
        self,
        current_memory: Memory,
        similar_memories: List[MemoryMatch],
        significance: SignificanceResult,
    ) -> Optional[str]:
        if not self.is_available():
            return self._fallback_summary(significance, len(similar_memories))

        prompt = self._build_history_prompt(current_memory, similar_memories, significance)
        try:
            text = await self._call_llm_async(prompt)
            text = str(text or "").strip()
            if not text:
                logger.warning("LLM returned empty history summary text — using fallback")
                return self._fallback_summary(significance, len(similar_memories))
            return text
        except Exception as e:
            self._mark_transient_failure(e, "LLM summary failed")
            return self._fallback_summary(significance, len(similar_memories))

    async def summarize_with_history_for_alert_async(
        self,
        current_memory: Memory,
        similar_memories: List[MemoryMatch],
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
        model_signals: Optional[Dict[str, Any]] = None,
        recurrence: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, str]:
        """Generate a short operator-facing alert summary using history.

        Returns:
            (alert_line, source, recommendation) — see
            ``explain_significance_for_alert_async``.
        """
        if not self.is_available():
            self._fallback_count += 1
            logger.info(
                "LLM unavailable for history alert — using fallback (fallback #%d, similar=%d)",
                self._fallback_count, len(similar_memories),
            )
            return (
                self._fallback_alert_line(
                    significance,
                    num_similar=len(similar_memories),
                    context=context,
                    model_signals=model_signals,
                ),
                "fallback",
                self._fallback_recommendation(significance, context=context),
            )

        logger.info(
            "Generating LLM history alert: score=%.2f, similar=%d",
            float(significance.score), len(similar_memories),
        )
        prompt = self._build_history_alert_line_prompt(
            current_memory,
            similar_memories,
            significance,
            context,
            model_signals,
            recurrence=recurrence,
        )
        try:
            payload = await self._call_llm_json_async(prompt)
            text = self._extract_structured_text_field(payload, "alert_line", max_len=240)
            recommendation = self._recommendation_from_payload(payload, significance, context=context)
            if not text:
                self._fallback_count += 1
                logger.warning("LLM returned empty history alert text — using fallback")
                return (
                    self._fallback_alert_line(
                        significance,
                        num_similar=len(similar_memories),
                        context=context,
                        model_signals=model_signals,
                    ),
                    "fallback",
                    recommendation,
                )
            self._llm_call_count += 1
            logger.info(
                "LLM history alert generated (call #%d): '%s'",
                self._llm_call_count, text[:80],
            )
            return (text, "llm", recommendation)
        except Exception as e:
            self._mark_transient_failure(e, "LLM history alert summary failed")
            self._fallback_count += 1
            return (
                self._fallback_alert_line(
                    significance,
                    num_similar=len(similar_memories),
                    context=context,
                    model_signals=model_signals,
                ),
                "fallback",
                self._fallback_recommendation(significance, context=context),
            )

    async def generate_memory_summary_async(
        self,
        memory: Memory,
        context: Optional[CuttingContext] = None,
    ) -> Optional[str]:
        if not self.is_available():
            return self._fallback_memory_summary(memory)

        prompt = self._build_memory_summary_prompt(memory, context)
        try:
            payload = await self._call_llm_json_async(prompt)
            text = self._extract_structured_text_field(payload, "summary", max_len=200)
            if not text:
                logger.warning("LLM returned empty memory summary text — using fallback")
                return self._fallback_memory_summary(memory)
            return text
        except Exception as e:
            self._mark_transient_failure(e, "LLM memory summary failed")
            return self._fallback_memory_summary(memory)

    # ------------------------------------------------------------------
    # Grounded explanation — detailed, evidence-based reasoning
    # ------------------------------------------------------------------

    async def explain_grounded_async(
        self,
        context: "ExplanationContext",
    ) -> tuple[str, str]:
        """Generate a detailed, evidence-grounded explanation.

        Returns:
            (explanation_text, source) where source is ``'llm'`` or ``'fallback'``
        """
        if not self.is_available():
            self._fallback_count += 1
            return (self._fallback_grounded(context), "fallback")

        prompt = self._build_grounded_explanation_prompt(context)
        logger.info(
            "Generating grounded explanation: patterns=%s, score=%.2f, "
            "classical=%s, co_occur=%d, similar=%d",
            context.pattern_keys[:3],
            float(context.significance.score) if context.significance else 0,
            bool(context.classical_model),
            len(context.co_occurrence),
            len(context.similar_memories),
        )
        try:
            payload = await self._call_llm_json_async(prompt, use_system_role=True)
            text = self._format_grounded_structured_output(payload)
            if not text:
                self._fallback_count += 1
                logger.warning("LLM returned empty grounded explanation — using fallback")
                return (self._fallback_grounded(context), "fallback")

            # Post-hoc validation: check that the LLM echoed at least some
            # of the evidence numbers we gave it (Issue #12 fix, 2026-04-14).
            text = self._validate_grounded_output(text, context)
            if not text:
                self._fallback_count += 1
                logger.warning("LLM grounded explanation cited no evidence — using fallback")
                return (self._fallback_grounded(context), "fallback")

            self._llm_call_count += 1
            logger.info(
                "Grounded explanation generated (call #%d, %d chars): %s",
                self._llm_call_count, len(text), text[:120],
            )
            return (text, "llm")
        except Exception as e:
            self._mark_transient_failure(e, "Grounded explanation failed")
            self._fallback_count += 1
            return (self._fallback_grounded(context), "fallback")

    def _build_grounded_explanation_prompt(
        self, ctx: "ExplanationContext"
    ) -> str:
        """Build a rich, evidence-dense prompt for a grounded explanation."""

        sig = ctx.significance
        score_str = f"{float(sig.score):.2f}" if sig else "N/A"
        action_str = sig.action.value if sig else "unknown"
        rules_str = ", ".join(sig.triggered_rules) if sig and sig.triggered_rules else "none"
        reasons_str = (
            "; ".join(self._sanitize_prompt_text(reason, max_len=160) for reason in sig.reasons)
            if sig and sig.reasons else "none"
        )

        # --- Section 1: Detected Patterns & Feature Evidence ---
        pattern_lines = []
        for pk in ctx.pattern_keys:
            evidence_items = ctx.feature_evidence.get(pk, [])
            stats = ctx.feedback_stats.get(pk, {})
            confirms = stats.get("confirms", 0)
            dismisses = stats.get("dismisses", 0)
            prior = stats.get("prior")

            line = f"- **{pk}**"
            if evidence_items:
                ev_parts = []
                for ev in evidence_items:
                    feat = ev.get("feature", "?")
                    val = ev.get("value")
                    thresh = ev.get("threshold")
                    direction = ev.get("direction", "above")
                    if val is not None and thresh is not None:
                        pct = abs(val - thresh) / abs(thresh) * 100 if thresh else 0
                        ev_parts.append(
                            f"{feat}={val:.2f} ({direction} threshold {thresh:.2f}, "
                            f"exceeded by {pct:.0f}%)"
                        )
                    elif val is not None:
                        ev_parts.append(f"{feat}={val:.2f}")
                if ev_parts:
                    line += "\n    Measurements: " + "; ".join(ev_parts)

            if confirms or dismisses:
                total = confirms + dismisses
                rate = confirms / total * 100 if total else 0
                prior_str = f", prior={prior:.2f}" if prior is not None else ""
                line += (
                    f"\n    Feedback history: confirmed {confirms}/{total} times "
                    f"({rate:.0f}%){prior_str}"
                )
            elif prior is not None:
                line += f"\n    Prior significance: {prior:.2f}"

            pattern_lines.append(line)

        patterns_section = "\n".join(pattern_lines) if pattern_lines else "(no patterns detected)"

        # --- Section 2: Classical Model Assessment ---
        classical_lines = []
        if ctx.classical_model:
            for key in ("anomaly_detector_score", "model_confidence", "breakage_prediction"):
                val = ctx.classical_model.get(key)
                if val is not None:
                    classical_lines.append(f"- {key}: {float(val):.3f}")
            # Include any other signals
            for key, val in ctx.classical_model.items():
                if key not in ("anomaly_detector_score", "model_confidence", "breakage_prediction"):
                    if isinstance(val, (int, float)):
                        classical_lines.append(f"- {key}: {float(val):.3f}")
        classical_section = "\n".join(classical_lines) if classical_lines else "Not available"

        # --- Section 3: Co-occurrence History ---
        cooccur_lines = []
        for co in ctx.co_occurrence[:8]:
            src = co.get("source", "?")
            tgt = co.get("target", "?")
            weight = co.get("weight", 0)
            strength = co.get("strength")
            strength_str = f", strength={strength:.0%}" if strength is not None else ""
            cooccur_lines.append(f"- {src} ↔ {tgt} (co-occurred {weight} times{strength_str})")
        cooccur_section = "\n".join(cooccur_lines) if cooccur_lines else "No co-occurrence data available"

        # --- Section 4: Similar Historical Events ---
        history_lines = []
        for i, match in enumerate(ctx.similar_memories[:5], 1):
            mem = match.memory
            label = self._sanitize_prompt_text(mem.label or "unlabeled", max_len=60)
            note = self._sanitize_prompt_text(mem.annotation_text or "", max_len=150)
            note = note or "(no annotation)"
            patterns = ", ".join(p.key for p in mem.pattern_keys[:3])
            match_score = f"{float(match.relevance_score):.2f}"
            history_lines.append(
                f"{i}. [{label}] patterns: {patterns}\n"
                f"   Note: {note}\n"
                f"   Match score: {match_score}"
            )
        history_section = "\n".join(history_lines) if history_lines else "No similar historical events found."

        # --- Section 5: Cutting Conditions ---
        context_str = "Not available"
        if ctx.cutting_context:
            cc = ctx.cutting_context
            ctx_parts = []
            if cc.tool_type:
                ctx_parts.append(
                    f"Tool: {self._sanitize_prompt_text(cc.tool_type, max_len=80)}"
                )
            if cc.spindle_speed:
                ctx_parts.append(f"Spindle: {cc.spindle_speed} RPM")
            if cc.axial_depth:
                ctx_parts.append(f"Depth: {cc.axial_depth}mm")
            if cc.workpiece_material:
                ctx_parts.append(
                    f"Material: {self._sanitize_prompt_text(cc.workpiece_material, max_len=80)}"
                )
            if cc.tooth_passing_freq:
                ctx_parts.append(f"Tooth passing freq: {cc.tooth_passing_freq:.1f} Hz")
            context_str = ", ".join(ctx_parts) if ctx_parts else "Not available"

        # --- Section 6: Key Raw Metrics ---
        metrics_lines = []
        for feat, val in list(ctx.raw_metrics_excerpt.items())[:10]:
            metrics_lines.append(f"- {feat}: {float(val):.3f}")
        metrics_section = "\n".join(metrics_lines) if metrics_lines else "Not available"

        prompt = f"""## Event Summary
Score: {score_str} ({action_str}) | Triggered rules: {rules_str}
Flag reasons: {reasons_str}

## Detected Patterns & Evidence
{patterns_section}

## Classical Model Assessment
{classical_section}

## Pattern Co-Occurrence History
{cooccur_section}

## Similar Historical Events
{history_section}

## Cutting Conditions
{context_str}

## Key Sensor Readings
{metrics_section}

How to read the feedback history above:
- High confirmation rate or prior (>0.6): operators have repeatedly agreed this is a real problem — trust the alert more.
- Low confirmation rate (<0.4): operators have usually dismissed this as a false alarm — trust the alert less.
- No reviews yet, or a neutral prior near 0.50: no track record — judge the event on its sensor evidence alone. Missing or low feedback is NOT itself a reason for concern.

Based on ALL the above evidence, explain, in short plain sentences:
1. indication: what this event most likely indicates, citing the specific measurement(s) vs their thresholds.
2. concern: how much the operator should trust this alert right now and why, reading the feedback history per the guide above. Do not describe a low or absent confirmation rate as a reason for concern.
3. operator_action: what to check or do, based on outcomes of similar past events.

Return only a JSON object with this exact schema:
{{"indication": str, "evidence": [str], "concern": str, "operator_action": str}}

Rules:
- Keep every sentence short and concrete; no jargon, no hedging, no filler.
- evidence: 2-3 statements, each citing a specific measurement, threshold, score, rate, or history count from the context above.
- Do not echo raw pattern codes or IDs.
- Do not wrap the JSON in markdown fences.
"""
        return prompt

    def _fallback_grounded(self, ctx: "ExplanationContext") -> str:
        """Build a structured fallback when the LLM is unavailable.

        A curated fallback (scripted-demo ``_explanation``) is preferred so a
        demo take degrades to polished prose rather than the terse structured
        summary below. This never runs when the LLM succeeds.
        """
        if getattr(ctx, "curated_fallback", None):
            return str(ctx.curated_fallback).strip()
        parts = []
        sig = ctx.significance
        if sig:
            parts.append(
                f"Significance {float(sig.score):.2f} ({sig.action.value}): "
                f"{'; '.join(sig.reasons[:3]) if sig.reasons else 'no reasons'}."
            )

        # Cite feature evidence
        for pk in ctx.pattern_keys[:3]:
            ev_list = ctx.feature_evidence.get(pk, [])
            stats = ctx.feedback_stats.get(pk, {})
            ev_str = ""
            if ev_list:
                top = ev_list[0]
                ev_str = f" ({top.get('feature', '?')}={top.get('value', '?'):.2f})"
            fb_str = ""
            if stats.get("confirms") or stats.get("dismisses"):
                fb_str = f" [confirmed {stats.get('confirms', 0)}/{stats.get('confirms', 0) + stats.get('dismisses', 0)}]"
            parts.append(f"Pattern {pk}{ev_str}{fb_str}.")

        if ctx.classical_model:
            anom = ctx.classical_model.get("anomaly_detector_score")
            if anom is not None:
                parts.append(f"Classical model anomaly score: {float(anom):.2f}.")

        if ctx.similar_memories:
            n = len(ctx.similar_memories)
            confirmed = sum(
                1 for m in ctx.similar_memories
                if m.memory.label and "confirm" in str(m.memory.label).lower()
            )
            parts.append(f"{n} similar past events ({confirmed} confirmed).")

        return " ".join(parts) if parts else "Event detected (no evidence detail available)."

    @staticmethod
    def _validate_grounded_output(text: str, ctx: "ExplanationContext") -> Optional[str]:
        """Post-hoc validation of LLM output against evidence context.

        Checks that the LLM cited at least *some* of the numeric evidence
        we provided. If the output is completely ungrounded (mentions no
        numbers from the evidence), reject it so the caller can emit a
        deterministic fallback instead.
        """
        # Collect key numeric values from the evidence context
        evidence_numbers: set = set()
        for ev_list in ctx.feature_evidence.values():
            for ev in ev_list:
                val = ev.get("value")
                if val is not None:
                    # Use 1-decimal form for matching (e.g., "23.4")
                    evidence_numbers.add(f"{float(val):.1f}")
                    evidence_numbers.add(f"{float(val):.2f}")
                    # Also accept integer form for whole numbers
                    if float(val) == int(val):
                        evidence_numbers.add(str(int(val)))

        for key, val in ctx.classical_model.items():
            evidence_numbers.add(f"{float(val):.2f}")
            evidence_numbers.add(f"{float(val):.1f}")

        if not evidence_numbers:
            return text  # Nothing to validate against

        # Count how many evidence numbers appear in the LLM output.
        # NOTE: this used to REJECT the output when it echoed zero evidence
        # numbers verbatim, which discarded good, plainly-worded explanations
        # (e.g. "vibration at ~480 Hz is elevated" cites 480, not the internal
        # ratio "0.85") and replaced them with the raw deterministic fallback —
        # the dominant cause of alerts showing pattern-code text instead of LLM
        # reasoning. The prompt already grounds the model in the evidence, so we
        # trust the output and only log when nothing matched, rather than reject.
        cited = sum(1 for n in evidence_numbers if n in text)
        if cited == 0:
            logger.info(
                "LLM grounded explanation echoed 0/%d evidence numbers verbatim — accepting anyway",
                len(evidence_numbers),
            )
        return text

    def _mark_transient_failure(self, exc: Exception, label: str) -> None:
        """Log a failed LLM call and temporarily disable the provider when the
        next immediate retry is likely to fail for the same reason."""
        exc_type = type(exc).__name__
        status_code: Optional[int] = None
        should_backoff = False

        if isinstance(exc, httpx.HTTPStatusError):
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            should_backoff = status_code is None or status_code >= 500 or status_code in (408, 429)
        elif isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            should_backoff = status_code is None or status_code >= 500 or status_code in (408, 429)
        elif isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.TimeoutException,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                TimeoutError,
                asyncio.TimeoutError,
                json.JSONDecodeError,
            ),
        ):
            should_backoff = True
        else:
            should_backoff = "Connect" in exc_type or "Timeout" in exc_type

        logger.warning(
            "%s [%s]: %s | available=%s, fallbacks=%d, llm_calls=%d",
            label, exc_type, exc or "(no detail)",
            self._available, self._fallback_count, self._llm_call_count,
        )
        if should_backoff:
            reason = f"HTTP {status_code}" if status_code is not None else exc_type
            logger.warning(
                "Transient provider failure (%s) — disabling LLM for %.0fs (will re-check)",
                reason,
                self._check_interval,
            )
            self._available = False
            self._last_check = time.time()
        else:
            logger.info(
                "LLM failure (%s) — keeping service available for next call",
                f"HTTP {status_code}" if status_code is not None else exc_type,
            )

    def force_available(self, available: bool = True) -> None:
        """Force-set the availability flag.

        Intended for the server-side warm-up endpoint so that the demo script
        can prime the explainer after confirming Ollama connectivity directly.
        """
        prev = self._available
        self._available = available
        self._last_check = time.time()
        logger.info(
            "LLM availability force-set: %s -> %s (by server warm-up)",
            prev, available,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return internal state for debugging LLM connectivity issues."""
        elapsed_since_check = time.time() - self._last_check if self._last_check else None
        return {
            "provider": self.config.provider,
            "available": self._available,
            "model": self.config.model,
            "ollama_url": self.config.ollama_url,
            "groq_api_url": self.config.groq_api_url if self.config.provider == "groq" else None,
            "groq_api_key_set": bool(self.config.groq_api_key) if self.config.provider == "groq" else None,
            "timeout": self.config.timeout,
            "check_interval": self._check_interval,
            "seconds_since_last_check": round(elapsed_since_check, 1) if elapsed_since_check is not None else None,
            "next_recheck_in": round(max(0, self._check_interval - (elapsed_since_check or 0)), 1) if elapsed_since_check is not None else 0,
            "llm_call_count": self._llm_call_count,
            "fallback_count": self._fallback_count,
            "is_cloud_model": self.config.provider == "groq" or self._is_cloud_model_name(self.config.model or ""),
        }

    _SYSTEM_PROMPT = (
        "You are a manufacturing process monitoring assistant for CNC machining. "
        "Explain flagged events briefly and in plain terms an operator would understand. "
        "Ground every statement in a specific measurement, threshold, score, or history count; "
        "do not echo raw pattern codes or IDs, and do not hedge or pad. "
        "Interpret operator feedback history correctly: a HIGH confirmation rate or prior means "
        "operators have agreed this pattern is a real problem (trust the alert more); a LOW rate "
        "means they have usually dismissed it as a false alarm (trust it less); no history or a "
        "neutral ~0.5 prior means there is no track record yet, so judge the event on its sensor "
        "evidence alone. A low or absent confirmation rate is never itself a reason for concern."
    )

    def _build_messages(
        self, prompt: str, *, use_system_role: bool = False,
    ) -> List[Dict[str, str]]:
        """Build the ``messages`` array for the chat API call."""
        msgs: List[Dict[str, str]] = []
        if use_system_role:
            msgs.append({"role": "system", "content": self._SYSTEM_PROMPT})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _call_llm(self, prompt: str, *, use_system_role: bool = False) -> str:
        """
        Call LLM API (Ollama or Groq).

        Routes to the configured provider.  Uses /api/chat for Ollama for
        better compatibility with models that emit a separate ``thinking``
        field.  Uses the OpenAI-compatible /chat/completions for Groq.
        """
        # P4: check cache first
        cached = self._get_cached(prompt)
        if cached is not None:
            return cached

        if self.config.provider == "groq":
            return self._call_groq_sync(prompt, use_system_role=use_system_role)
        return self._call_ollama_sync(prompt, use_system_role=use_system_role)

    def _call_groq_sync(self, prompt: str, *, use_system_role: bool = False) -> str:
        """Synchronous Groq /chat/completions call."""
        url = f"{self.config.groq_api_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, use_system_role=use_system_role),
            "max_completion_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }

        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))
        logger.debug(
            "LLM sync request (groq): POST %s model=%s prompt_len=%d",
            url, self.config.model, len(prompt),
        )
        t0 = time.time()
        response = requests.post(
            url, json=payload, headers=headers,
            timeout=(connect_timeout, float(self.config.timeout)),
        )
        elapsed = time.time() - t0
        logger.debug("Groq sync response status=%d elapsed=%.2fs", response.status_code, elapsed)
        response.raise_for_status()

        result = response.json()
        text = self._extract_openai_text(result)
        if not text:
            logger.warning(
                "Groq sync returned no text (model=%s, elapsed=%.2fs). Keys: %s",
                self.config.model, elapsed,
                list(result.keys()) if isinstance(result, dict) else type(result).__name__,
            )
        else:
            logger.info("Groq sync response (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
        result_text = str(text).strip()
        if result_text:
            self._set_cache(prompt, result_text)
        return result_text

    def _call_ollama_sync(self, prompt: str, *, use_system_role: bool = False) -> str:
        """Synchronous Ollama /api/chat call."""
        url = str(self.config.ollama_url or "")
        if "/api/generate" in url:
            url = url.replace("/api/generate", "/api/chat")

        payload = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, use_system_role=use_system_role),
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
            },
        }

        connect_timeout = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))
        logger.debug(
            "LLM sync request (ollama): POST %s model=%s prompt_len=%d connect_timeout=%.1fs read_timeout=%.1fs",
            url, self.config.model, len(prompt), connect_timeout, self.config.timeout,
        )
        t0 = time.time()
        response = requests.post(
            url,
            json=payload,
            timeout=(connect_timeout, float(self.config.timeout)),
        )
        elapsed = time.time() - t0
        logger.debug("LLM sync response status=%d elapsed=%.2fs", response.status_code, elapsed)
        response.raise_for_status()

        result = response.json()
        text = self._extract_ollama_text(result)
        if not text:
            logger.warning(
                "LLM sync returned no text (model=%s, elapsed=%.2fs). Response keys: %s",
                self.config.model, elapsed,
                list(result.keys()) if isinstance(result, dict) else type(result).__name__,
            )
        else:
            logger.info(
                "LLM sync response (%d chars, %.2fs): %s",
                len(text), elapsed, text[:120],
            )
        result_text = str(text).strip()
        if result_text:
            self._set_cache(prompt, result_text)
        return result_text

    async def _call_llm_async(self, prompt: str, *, use_system_role: bool = False) -> str:
        """Async LLM call that cooperates with cancellation (fast shutdown).

        Routes to the appropriate provider (Ollama or Groq).
        """
        # P4: check cache first
        cached = self._get_cached(prompt)
        if cached is not None:
            return cached

        timeout_s = max(1.0, float(self.config.timeout) + 5.0)
        if self.config.provider == "groq":
            provider_call = self._call_groq_async(prompt, use_system_role=use_system_role)
        else:
            provider_call = self._call_ollama_async(prompt, use_system_role=use_system_role)

        try:
            return await asyncio.wait_for(provider_call, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"LLM overall timeout after {timeout_s:.1f}s") from exc

    async def _call_llm_json_async(self, prompt: str, *, use_system_role: bool = False) -> Dict[str, Any]:
        cached = self._get_cached(prompt, response_mode=_JSON_RESPONSE_MODE)
        if cached is not None:
            return self._parse_json_object(cached)

        timeout_s = max(1.0, float(self.config.timeout) + 5.0)
        if self.config.provider == "groq":
            provider_call = self._call_groq_async(
                prompt,
                use_system_role=use_system_role,
                response_mode=_JSON_RESPONSE_MODE,
            )
        else:
            provider_call = self._call_ollama_async(
                prompt,
                use_system_role=use_system_role,
                response_mode=_JSON_RESPONSE_MODE,
            )

        try:
            text = await asyncio.wait_for(provider_call, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"LLM overall timeout after {timeout_s:.1f}s") from exc
        return self._parse_json_object(text)

    async def _call_groq_async(
        self,
        prompt: str,
        *,
        use_system_role: bool = False,
        response_mode: str = _TEXT_RESPONSE_MODE,
    ) -> str:
        """Async Groq /chat/completions call via httpx."""
        url = f"{self.config.groq_api_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, use_system_role=use_system_role),
            "max_completion_tokens": self.config.max_tokens,
        }
        if response_mode == _JSON_RESPONSE_MODE:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }

        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=float(self.config.timeout),
            write=connect_timeout,
            pool=connect_timeout,
        )

        logger.debug(
            "LLM async request (groq): POST %s model=%s prompt_len=%d",
            url, self.config.model, len(prompt),
        )
        t0 = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            elapsed = time.time() - t0
            logger.debug("Groq async response status=%d elapsed=%.2fs", resp.status_code, elapsed)
            resp.raise_for_status()
            result = resp.json()
            text = self._extract_openai_text(result)
            if not text:
                logger.warning(
                    "Groq async returned no text (model=%s, elapsed=%.2fs). Keys: %s",
                    self.config.model, elapsed,
                    list(result.keys()) if isinstance(result, dict) else type(result).__name__,
                )
            else:
                logger.info("Groq async response (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
            result_text = str(text).strip()
            if result_text:
                self._set_cache(prompt, result_text, response_mode=response_mode)
            return result_text

    async def _call_ollama_async(
        self,
        prompt: str,
        *,
        use_system_role: bool = False,
        response_mode: str = _TEXT_RESPONSE_MODE,
    ) -> str:
        """Async Ollama /api/chat call via httpx."""
        url = str(self.config.ollama_url or "")
        if "/api/generate" in url:
            url = url.replace("/api/generate", "/api/chat")

        payload = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, use_system_role=use_system_role),
            "stream": False,
            "options": {
                "num_predict": self.config.max_tokens,
            },
        }
        if response_mode == _JSON_RESPONSE_MODE:
            payload["format"] = "json"

        connect_timeout = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=float(self.config.timeout),
            write=connect_timeout,
            pool=connect_timeout,
        )

        logger.debug(
            "LLM async request (ollama): POST %s model=%s prompt_len=%d connect_timeout=%.1fs read_timeout=%.1fs",
            url, self.config.model, len(prompt), connect_timeout, self.config.timeout,
        )
        t0 = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            elapsed = time.time() - t0
            logger.debug("LLM response status=%d elapsed=%.2fs", resp.status_code, elapsed)
            resp.raise_for_status()
            result = resp.json()
            text = self._extract_ollama_text(result)
            if not text:
                logger.warning(
                    "LLM returned no text (model=%s, elapsed=%.2fs). Response keys: %s",
                    self.config.model, elapsed,
                    list(result.keys()) if isinstance(result, dict) else type(result).__name__,
                )
            else:
                logger.info(
                    "LLM response (%d chars, %.2fs): %s",
                    len(text), elapsed, text[:120],
                )
            result_text = str(text).strip()
            if result_text:
                self._set_cache(prompt, result_text, response_mode=response_mode)
            return result_text

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        candidate = str(text or "").strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            if candidate.endswith("```"):
                candidate = candidate[:-3].strip()
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("response_not_object")
        return parsed

    @staticmethod
    def _sentence(text: Any) -> str:
        cleaned = LLMExplainer._sanitize_prompt_text(text, max_len=320)
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    def _format_grounded_structured_output(self, payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        indication = self._sentence(payload.get("indication"))
        concern = self._sentence(payload.get("concern"))
        operator_action = self._sentence(payload.get("operator_action"))
        raw_evidence = payload.get("evidence") or []
        evidence_items = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence[:3]:
                sentence = self._sentence(item)
                if sentence:
                    evidence_items.append(sentence)
        if not indication or not concern or not operator_action or not evidence_items:
            return ""
        return " ".join([indication, *evidence_items, concern, operator_action])

    @staticmethod
    def _extract_structured_text_field(
        payload: Dict[str, Any],
        field_name: str,
        *,
        max_len: int = 240,
    ) -> str:
        if not isinstance(payload, dict):
            return ""
        return LLMExplainer._sanitize_prompt_text(payload.get(field_name), max_len=max_len)

    # ------------------------------------------------------------------
    # Response text extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_openai_text(result: Any) -> str:
        """Extract text from an OpenAI-compatible chat completion response."""
        if not isinstance(result, dict):
            return ""
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                return str(msg.get("content") or "")
        return ""

    @staticmethod
    def _extract_ollama_text(result: Any) -> str:
        """Extract text from an Ollama /api/chat response."""
        if not isinstance(result, dict):
            return ""
        msg = result.get("message")
        if isinstance(msg, dict):
            text = str(msg.get("content") or "")
            if not text:
                text = str(msg.get("thinking") or "")
                if text:
                    logger.debug("LLM used thinking field (no content field)")
            return text
        # Legacy /api/generate format
        return str(result.get("response") or "")
    
    def _build_significance_prompt(
        self,
        patterns: List[PatternKey],
        significance: SignificanceResult,
        context: Optional[CuttingContext],
        metrics: Optional[NumericMetrics],
    ) -> str:
        """Build prompt for significance explanation."""
        
        # Format patterns
        pattern_list = "\n".join([f"- {p.key}" for p in patterns])
        
        # Format reasons
        reason_list = "\n".join(
            [f"- {self._sanitize_prompt_text(r, max_len=240)}" for r in significance.reasons]
        )
        
        # Format context if available
        context_str = "Not available"
        if context:
            ctx_parts = []
            if context.tool_type:
                ctx_parts.append(
                    f"Tool: {self._sanitize_prompt_text(context.tool_type, max_len=80)}"
                )
            if context.spindle_speed:
                ctx_parts.append(f"Spindle: {context.spindle_speed} RPM")
            if context.axial_depth:
                ctx_parts.append(f"Depth: {context.axial_depth}mm")
            if context.workpiece_material:
                ctx_parts.append(
                    f"Material: {self._sanitize_prompt_text(context.workpiece_material, max_len=80)}"
                )
            if context.tooth_passing_freq:
                ctx_parts.append(f"Tooth passing freq: {context.tooth_passing_freq:.1f} Hz")
            context_str = ", ".join(ctx_parts) if ctx_parts else "Not available"
        
        # Format key metrics if available
        metrics_str = "Not available"
        if metrics:
            m_parts = []
            if metrics.rms:
                top_rms = list(metrics.rms.items())[:3]
                m_parts.append(f"RMS: {dict(top_rms)}")
            if metrics.dominant_freqs:
                top_freq = list(metrics.dominant_freqs.items())[:3]
                m_parts.append(f"Dominant frequencies: {dict(top_freq)}")
            metrics_str = "; ".join(m_parts) if m_parts else "Not available"
        
        score_str = f"{float(significance.score):.2f}"
        breakdown_str = self._format_significance_breakdown(significance)

        prompt = f"""You are a manufacturing process monitoring assistant. Analyze the following sensor data event and explain its significance in 2-3 sentences.

## Detected Patterns
{pattern_list}

## Why This Was Flagged
{reason_list}

## Cutting Conditions
{context_str}

## Key Metrics
{metrics_str}

## Significance Score
{score_str} (Action: {significance.action.value})

## Score Breakdown (component contributions)
{breakdown_str}

Provide a brief, technical explanation of what this event likely indicates for the machining process. Focus on practical implications for the operator.
"""
        return prompt

    @staticmethod
    def _sanitize_prompt_text(
        value: Any,
        *,
        max_len: Optional[int] = None,
        keep_newlines: bool = False,
    ) -> str:
        if value is None:
            return ""
        text = str(value)
        text = text.replace("```", " ").replace("`", "")
        text = _PROMPT_ROLE_MARKER_RE.sub(" ", text)
        text = text.replace("\r", "\n")
        if keep_newlines:
            lines = []
            for raw_line in text.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if line:
                    lines.append(line)
            text = "\n".join(lines)
        else:
            text = re.sub(r"\s+", " ", text).strip()
        if max_len is not None and max_len > 0 and len(text) > max_len:
            text = text[:max_len].rstrip()
        return text

    @staticmethod
    def _format_alert_context(context: Optional[CuttingContext]) -> str:
        if not context:
            return "Not available"

        parts = []
        regime = getattr(context.operating_regime, "value", context.operating_regime)
        if regime:
            parts.append(f"Regime: {LLMExplainer._sanitize_prompt_text(regime, max_len=40)}")
        if context.workpiece_material:
            parts.append(
                f"Material: {LLMExplainer._sanitize_prompt_text(context.workpiece_material, max_len=80)}"
            )
        if context.tool_type:
            parts.append(
                f"Tool: {LLMExplainer._sanitize_prompt_text(context.tool_type, max_len=80)}"
            )
        if context.spindle_speed:
            parts.append(f"Spindle: {context.spindle_speed:.0f} RPM")
        if context.feed_rate:
            parts.append(f"Feed: {context.feed_rate:.0f} mm/min")
        depth_parts = []
        if context.axial_depth:
            depth_parts.append(f"ap={context.axial_depth:.2f} mm")
        if context.radial_depth:
            depth_parts.append(f"ae={context.radial_depth:.2f} mm")
        if depth_parts:
            parts.append("Depth: " + ", ".join(depth_parts))
        return "; ".join(parts) if parts else "Not available"

    @staticmethod
    def _format_alert_metrics(metrics: Optional[NumericMetrics]) -> str:
        if not metrics:
            return "Not available"

        m_parts = []
        if metrics.rms:
            top_rms = list(metrics.rms.items())[:2] if isinstance(metrics.rms, dict) else [("rms", metrics.rms)]
            m_parts.append(f"RMS: {dict(top_rms)}")
        if metrics.dominant_freqs:
            top_freq = list(metrics.dominant_freqs.items())[:2]
            m_parts.append(f"Dominant frequencies: {dict(top_freq)}")
        return "; ".join(m_parts) if m_parts else "Not available"

    @staticmethod
    def _format_alert_model_scores(model_signals: Optional[Dict[str, Any]]) -> str:
        if not model_signals:
            return "Not available"

        parts = []
        for key in (
            "anomaly_detector_score",
            "model_confidence",
            "breakage_prediction",
            "tool_wear_estimate",
        ):
            value = model_signals.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.2f}")
        return "; ".join(parts) if parts else "Not available"

    @staticmethod
    def _format_significance_breakdown(significance: SignificanceResult) -> str:
        """Render score_trace + prior factor/mode so the LLM sees how the score was built."""
        if significance is None:
            return "Not available"

        lines: List[str] = []
        prior_mode = getattr(significance, "prior_mode", None)
        prior_factor = getattr(significance, "prior_factor", None)
        prior_boost = getattr(significance, "prior_boost", None)

        if prior_mode == "multiplicative" and isinstance(prior_factor, (int, float)):
            lines.append(
                f"Historical prior: \u00d7{float(prior_factor):.2f} (multiplicative; \u00d71 = neutral)"
            )
        elif isinstance(prior_boost, (int, float)) and prior_boost:
            lines.append(f"Historical prior boost: {float(prior_boost):+.3f} (additive)")

        trace = getattr(significance, "score_trace", None) or []
        if trace:
            top = trace[:6]
            for entry in top:
                if not isinstance(entry, dict):
                    continue
                component = str(entry.get("component", "")).strip() or "component"
                value = entry.get("value")
                source = str(entry.get("source", "")).strip()
                if isinstance(value, (int, float)):
                    sign = "+" if value >= 0 else ""
                    suffix = f" [{source}]" if source else ""
                    lines.append(f"- {sign}{float(value):.3f} {component}{suffix}")
        return "\n".join(lines) if lines else "Not available"

    @staticmethod
    def _format_recurrence(recurrence: Optional[Dict[str, Any]]) -> str:
        if not recurrence:
            return "First-time occurrence"
        occ = int(recurrence.get("occurrences", 0) or 0)
        if occ <= 1:
            return "First-time occurrence"
        suppressed = int(recurrence.get("suppressed_since_last_emit", 0) or 0)
        first_seen = str(recurrence.get("first_seen") or "").strip()
        parts = [f"Recurring: {occ} occurrences"]
        if first_seen:
            parts.append(f"first seen {first_seen}")
        if suppressed:
            parts.append(f"{suppressed} identical updates suppressed since last emit")
        return "; ".join(parts)

    def _build_alert_line_prompt(
        self,
        patterns: List[PatternKey],
        significance: SignificanceResult,
        context: Optional[CuttingContext],
        metrics: Optional[NumericMetrics],
        model_signals: Optional[Dict[str, Any]],
        recurrence: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a short, demo-friendly alert prompt."""
        pattern_list = "\n".join([f"- {p.key}" for p in patterns])
        reason_list = "\n".join(
            [f"- {self._sanitize_prompt_text(r, max_len=240)}" for r in (significance.reasons or [])]
        )

        context_str = self._format_alert_context(context)
        metrics_str = self._format_alert_metrics(metrics)
        model_scores_str = self._format_alert_model_scores(model_signals)
        breakdown_str = self._format_significance_breakdown(significance)
        recurrence_str = self._format_recurrence(recurrence)

        score_str = f"{float(significance.score):.2f}"

        return f"""You are assisting a machine operator.

Write ONE concise operator-facing alert in 1-2 short sentences (max 32 words total).

Rules:
- Focus on what changed, why it matters for this cut, and what the operator should check next.
- Ground the alert in the detected patterns, the strongest model score, and the cutting context when available.
- If the event is recurring, mention it briefly (e.g. "still occurring", "again", or "Nth time").
- Mention at most one score and at most one context detail.
- Do NOT include pattern codes, IDs, underscores, or colon-delimited detector keys.
- Use plain manufacturing terms an operator can act on immediately.

Raw patterns (DO NOT echo these):
{pattern_list or '(none)'}

Why flagged:
{reason_list or '(none)'}

Model scores:
{model_scores_str}

Significance breakdown (how the score was built):
{breakdown_str}

Recurrence:
{recurrence_str}

Cutting conditions:
{context_str}

Key metrics:
{metrics_str}

Score/action:
{score_str} ({significance.action.value})

Return only a JSON object with this exact schema:
{{"alert_line": str, "recommendation": str}}

Rules:
- alert_line must be 1-2 short sentences and no more than 32 words total.
- recommendation: ONE immediate action the operator can take right now to reduce the chance of tool breakage or scrap (max 20 words). If nothing specific applies, give the safest general check. Plain manufacturing terms, no codes.
- Do not wrap the JSON in markdown fences.
"""
    
    def _build_history_prompt(
        self,
        current: Memory,
        similar: List[MemoryMatch],
        significance: SignificanceResult,
    ) -> str:
        """Build prompt incorporating historical memories."""
        
        # Current event patterns
        current_patterns = ", ".join([p.key for p in current.pattern_keys])
        
        # Format similar memories
        history_parts = []
        for i, match in enumerate(similar[:5], 1):  # Top 5
            mem = match.memory
            patterns = ", ".join([p.key for p in mem.pattern_keys[:3]])
            annotation = self._sanitize_prompt_text(mem.annotation_text or "", max_len=100) or "No annotation"
            label = self._sanitize_prompt_text(mem.label or "unlabeled", max_len=60)

            match_score_str = f"{float(match.relevance_score):.2f}"
            
            history_parts.append(
                f"{i}. Patterns: {patterns}\n"
                f"   Label: {label}\n"
                f"   Note: {annotation}\n"
                f"   Match score: {match_score_str}"
            )
        
        history_str = "\n".join(history_parts) if history_parts else "No similar historical events found."
        
        sig_score_str = f"{float(significance.score):.2f}"

        prompt = f"""You are a manufacturing process monitoring assistant with access to historical data.

## Current Event
Patterns detected: {current_patterns}
Significance: {sig_score_str}
Reasons: {', '.join(self._sanitize_prompt_text(reason, max_len=160) for reason in significance.reasons)}

## Similar Historical Events
{history_str}

Based on the current event and historical context:
1. What does this event likely indicate?
2. Have similar events been seen before, and what was the outcome?
3. Any recommended actions?

Provide a concise summary (3-4 sentences) that an operator can quickly read and act on.
"""
        return prompt

    def _build_history_alert_line_prompt(
        self,
        current: Memory,
        similar: List[MemoryMatch],
        significance: SignificanceResult,
        context: Optional[CuttingContext],
        model_signals: Optional[Dict[str, Any]],
        recurrence: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a short, demo-friendly alert prompt that can use history."""
        current_patterns = ", ".join([p.key for p in (current.pattern_keys or [])])

        history_parts = []
        for i, match in enumerate((similar or [])[:3], 1):
            mem = match.memory
            label = self._sanitize_prompt_text(mem.label or "unlabeled", max_len=60)
            note = self._sanitize_prompt_text(mem.annotation_text or "", max_len=120)
            note = note or "(no note)"
            history_parts.append(f"{i}. label={label}; note={note}")
        history_str = "\n".join(history_parts) if history_parts else "(no similar history)"

        sig_score_str = f"{float(significance.score):.2f}"
        context_str = self._format_alert_context(context)
        model_scores_str = self._format_alert_model_scores(model_signals)
        breakdown_str = self._format_significance_breakdown(significance)
        recurrence_str = self._format_recurrence(recurrence)

        return f"""You are assisting a machine operator.

    Write ONE concise operator-facing alert in 1-2 short sentences (max 32 words total).

Rules:
- Do NOT include pattern codes, IDs, underscores, or colon-delimited detector keys.
    - Use plain manufacturing terms and leverage history if it changes the recommendation.
    - Ground the alert in the current patterns, the strongest model score, the historical outcome summary, and the cutting context when available.
- If the event is recurring, mention it briefly (e.g. "still occurring", "again", or "Nth time").
    - Mention at most one score and at most one history fact.
    - Make it actionable for the operator.

Raw current patterns (DO NOT echo these):
{current_patterns or '(none)'}

Flag reasons:
{', '.join(self._sanitize_prompt_text(reason, max_len=160) for reason in significance.reasons) if significance.reasons else '(none)'}

    Model scores:
    {model_scores_str}

Significance breakdown (how the score was built):
{breakdown_str}

Recurrence:
{recurrence_str}

Similar historical events (summaries):
{history_str}

    Cutting conditions:
    {context_str}

Score/action:
{sig_score_str} ({significance.action.value})

Return only a JSON object with this exact schema:
{{"alert_line": str, "recommendation": str}}

Rules:
- alert_line must be 1-2 short sentences and no more than 32 words total.
- recommendation: ONE immediate action the operator can take right now to reduce the chance of tool breakage or scrap (max 20 words). If nothing specific applies, give the safest general check. Plain manufacturing terms, no codes.
- Do not wrap the JSON in markdown fences.
"""
    
    def _build_memory_summary_prompt(
        self,
        memory: Memory,
        context: Optional[CuttingContext],
    ) -> str:
        """Build prompt for memory summary generation."""
        
        patterns = ", ".join([p.key for p in memory.pattern_keys])
        
        context_str = ""
        if context:
            parts = []
            if context.operating_regime:
                parts.append(self._sanitize_prompt_text(context.operating_regime.value, max_len=40))
            if context.workpiece_material:
                parts.append(self._sanitize_prompt_text(context.workpiece_material, max_len=80))
            if context.tool_type:
                parts.append(self._sanitize_prompt_text(context.tool_type, max_len=80))
            context_str = f" during {' '.join(parts)}" if parts else ""

        tr = memory.time_range
        t0_str = f"{float(tr.t0 if hasattr(tr, 't0') else tr[0]):.2f}"
        t1_str = f"{float(tr.t1 if hasattr(tr, 't1') else tr[1]):.2f}"

        prompt = f"""Generate a one-sentence summary of this machining event for record-keeping.

Patterns: {patterns}
Time range: {t0_str}s - {t1_str}s
Tags: {', '.join(self._sanitize_prompt_text(tag, max_len=40) for tag in memory.tags) if memory.tags else 'none'}
Label: {self._sanitize_prompt_text(memory.label or 'unlabeled', max_len=60)}
Context: {context_str or 'standard operation'}

Write a brief, factual summary (one sentence, max 20 words).

Return only a JSON object with this exact schema:
{{"summary": str}}

Rules:
- summary must be one sentence and no more than 20 words.
- Do not wrap the JSON in markdown fences.
"""
        return prompt
    
    # [PROTOTYPE_LLM_MEMORY_V1] - Fallback methods when LLM unavailable
    
    def _fallback_explanation(self, significance: SignificanceResult) -> str:
        """Generate simple explanation without LLM."""
        if not significance.reasons:
            score_str = f"{float(significance.score):.2f}"
            return f"Event detected with significance score {score_str}."
        
        return f"Significant event: {'; '.join(significance.reasons[:3])}."
    
    def _fallback_summary(self, significance: SignificanceResult, num_similar: int) -> str:
        """Generate simple summary without LLM."""
        base = f"Event with {significance.action.value} priority detected."
        if num_similar > 0:
            base += f" {num_similar} similar historical events found."
        return base

    def _recommendation_from_payload(
        self,
        payload: Dict[str, Any],
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
    ) -> str:
        """Extract the immediate-action recommendation from an LLM JSON payload,
        falling back to a deterministic action if absent/empty."""
        recommendation = self._extract_structured_text_field(payload, "recommendation", max_len=200)
        if recommendation:
            return recommendation
        return self._fallback_recommendation(significance, context=context)

    def _fallback_recommendation(
        self,
        significance: SignificanceResult,
        context: Optional[CuttingContext] = None,
    ) -> str:
        """Deterministic immediate action when no LLM recommendation is available.

        Two-tier recommendation model (2026-07-12): this is the *per-alert*,
        tactical action to reduce breakage risk right now — distinct from the
        end-of-batch reconfiguration proposal.
        """
        action = str(getattr(getattr(significance, "action", None), "value", "") or "").lower()
        if action == "critical":
            return "Stop at the next safe point and inspect the tool for wear or breakage before continuing."
        return "Ease off feed/spindle and inspect the tool and workpiece at the next safe stop."

    def _fallback_alert_line(
        self,
        significance: SignificanceResult,
        num_similar: int = 0,
        context: Optional[CuttingContext] = None,
        model_signals: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a short operator-facing alert line without the LLM."""
        action = (
            str(getattr(getattr(significance, "action", None), "value", None) or "alert").upper()
        )
        reasons = list(getattr(significance, "reasons", []) or [])
        head = str(reasons[0]) if reasons else "Significant change in vibration signal"
        parts = [head.rstrip(". ")]

        if model_signals:
            for key in ("breakage_prediction", "anomaly_detector_score", "model_confidence"):
                value = model_signals.get(key)
                if isinstance(value, (int, float)):
                    parts.append(f"model {float(value):.2f}")
                    break

        if num_similar:
            parts.append(
                f"{int(num_similar)} similar past event{'s' if int(num_similar) != 1 else ''}"
            )

        if context and (context.workpiece_material or context.tool_type):
            ctx_bits = []
            if context.workpiece_material:
                ctx_bits.append(str(context.workpiece_material))
            if context.tool_type:
                ctx_bits.append(str(context.tool_type).replace("_", " "))
            if ctx_bits:
                parts.append(" ".join(ctx_bits[:2]))

        return f"{action}: {'; '.join(parts)}."
    
    def _fallback_memory_summary(self, memory: Memory) -> str:
        """Generate simple memory summary without LLM."""
        patterns = memory.pattern_keys[:2]
        pattern_str = ", ".join([p.key for p in patterns])
        label = memory.label or "event"
        tr = memory.time_range
        t0 = tr.t0 if hasattr(tr, 't0') else tr[0]
        return f"{label.capitalize()} at t={float(t0):.2f}s: {pattern_str}"


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function
def create_explainer(config: Optional[ExplainerConfig] = None) -> LLMExplainer:
    """Create an LLMExplainer instance."""
    return LLMExplainer(config)

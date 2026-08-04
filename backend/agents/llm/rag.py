"""
RAG (Retrieval-Augmented Generation) agent using FAISS plus a configured LLM provider.

Supports Groq by default and Ollama when ``LLM_PROVIDER=ollama`` is set.
Uses chat-style APIs so both providers can return content in a consistent shape.
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import logging
import os
import time

import httpx
import requests
from .ingest import Ingestor

logger = logging.getLogger(__name__)

# Default fallback when the configured LLM provider is unreachable
_FALLBACK_ANSWER = (
    "LLM service is currently unavailable. Please try again later or "
    "check the configured provider connectivity."
)


class LLMAgent:
    """LLM + RAG agent using sentence-transformers + FAISS for retrieval and provider-based generation.

    Features:
      - ingest_documents(docs): add docs to index
    - handle_request(..., action='query') -> runs retrieval + prompt + provider call

    Configuration (environment variables):
            - LLM_PROVIDER       – ``groq`` (default) or ``ollama``
            - GROQ_API_KEY       – Groq API key (required when provider is ``groq``)
            - GROQ_API_URL       – base URL (default ``https://api.groq.com/openai/v1``)
            - GROQ_MODEL         – model name (default ``llama-3.3-70b-versatile``)
            - OLLAMA_URL         – base URL (default ``http://localhost:11434/api/generate``)
            - OLLAMA_MODEL       – model name (default ``gpt-oss:20b``)
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self._ingestor: Optional[Ingestor] = None

        # LLM availability state (mirrors LLMExplainer pattern)
        self._available: Optional[bool] = None
        self._last_check: float = 0.0
        self._check_interval: float = float(os.environ.get("LLM_CHECK_INTERVAL", "10.0"))
        self._call_count: int = 0
        self._fallback_count: int = 0

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _get_provider(self) -> str:
        provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
        return "groq" if provider == "groq" else "ollama"

    def _get_model(self) -> str:
        if self._get_provider() == "groq":
            return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

    def _get_base_url(self) -> str:
        if self._get_provider() == "groq":
            return os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1").rstrip("/")
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        return url.replace("/api/generate", "").replace("/api/chat", "")

    def _get_groq_api_key(self) -> str:
        return os.environ.get("GROQ_API_KEY", "")

    def is_available(self) -> bool:
        """Check if the configured LLM provider is reachable (cached, re-checks periodically)."""
        now = time.time()
        if self._available is not None and (now - self._last_check) < self._check_interval:
            return self._available

        self._last_check = now
        provider = self._get_provider()
        base = self._get_base_url()
        try:
            if provider == "groq":
                api_key = self._get_groq_api_key()
                if not api_key:
                    logger.warning("RAG LLM Groq API key not set (GROQ_API_KEY).")
                    self._available = False
                    return self._available
                resp = requests.get(
                    f"{base}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5.0,
                )
                self._available = resp.status_code == 200
                if self._available:
                    logger.debug("RAG LLM Groq available at %s", base)
                else:
                    logger.warning("RAG LLM Groq /models returned %d", resp.status_code)
            else:
                resp = requests.get(f"{base}/api/tags", timeout=5.0)
                self._available = resp.status_code == 200
                if self._available:
                    logger.debug("RAG LLM Ollama available at %s", base)
                else:
                    logger.warning("RAG LLM Ollama /api/tags returned %d", resp.status_code)
        except Exception as exc:
            self._available = False
            logger.warning("RAG LLM availability check failed for %s: %s", provider, exc)

        return self._available

    def _mark_provider_unavailable(self, exc: Exception, label: str) -> None:
        """Temporarily mark the configured provider unavailable for retryable failures."""
        provider = self._get_provider()
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

        if should_backoff:
            self._available = False
            self._last_check = time.time()
            logger.warning(
                "%s [%s] — marking %s unavailable for %.0fs",
                label,
                f"HTTP {status_code}" if status_code is not None else exc_type,
                provider,
                self._check_interval,
            )
        else:
            logger.warning(
                "%s [%s] — keeping %s available",
                label,
                f"HTTP {status_code}" if status_code is not None else exc_type,
                provider,
            )

    # ------------------------------------------------------------------
    # Ingestor
    # ------------------------------------------------------------------

    def _ensure_ingestor(self):
        if self._ingestor is None:
            self._ingestor = Ingestor(model_name=self.embedding_model_name, index_path=self.index_path)

    def ingest_documents(self, docs: List[Dict[str, Any]], persist: bool = False):
        """Docs: list of {id, text, meta}"""
        self._ensure_ingestor()
        self._ingestor.ingest(docs, persist=persist)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
        blocks = [f"Context {i+1}: {d['text']}\nSource: {d.get('meta', {})}" for i, (d, _) in enumerate(retrieved)]
        context_str = "\n\n".join(blocks)
        return (
            "You are an assistant. Use the following context to answer the question.\n\n"
            f"{context_str}\n\nQuestion: {question}\nAnswer concisely and include citations."
        )

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------

    def _extract_text(self, result: Any) -> str:
        """Extract answer text from a provider chat response.

        Handles ``message.content``, ``message.thinking``, and the legacy
        ``response`` field so that both Groq- and Ollama-shaped responses work.
        """
        if not isinstance(result, dict):
            return ""
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    text = str(msg.get("content") or "")
                    if text:
                        return text.strip()
        msg = result.get("message")
        if isinstance(msg, dict):
            text = str(msg.get("content") or "")
            if not text:
                text = str(msg.get("thinking") or "")
                if text:
                    logger.debug("RAG LLM used thinking field (no content)")
            if text:
                return text.strip()
        # Legacy /api/generate compatibility
        text = str(result.get("response") or "")
        return text.strip()

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Synchronous Ollama call via /api/chat."""
        model = self._get_model()
        base = self._get_base_url()
        url = f"{base}/api/chat"

        read_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "60.0"))
        connect_timeout = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        logger.debug(
            "RAG LLM sync request: POST %s model=%s prompt_len=%d",
            url, model, len(prompt),
        )
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
            elapsed = time.time() - t0
            logger.debug("RAG LLM response status=%d elapsed=%.2fs", r.status_code, elapsed)
            r.raise_for_status()
            result = r.json()
            text = self._extract_text(result)
            self._call_count += 1
            if text:
                logger.info("RAG LLM answer (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
            else:
                logger.warning("RAG LLM returned no text (elapsed=%.2fs, keys=%s)", elapsed, list(result.keys()))
            return {"answer": text, "raw": result}
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            self._mark_provider_unavailable(exc, "RAG LLM sync call failed")
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}
        except Exception as exc:
            logger.error("RAG LLM unexpected error: %s", exc, exc_info=True)
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}

    def _call_groq(self, prompt: str) -> Dict[str, Any]:
        """Synchronous Groq call via /chat/completions."""
        model = self._get_model()
        base = self._get_base_url()
        url = f"{base}/chat/completions"
        api_key = self._get_groq_api_key()

        read_timeout = float(os.environ.get("GROQ_TIMEOUT", "30.0"))
        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        logger.debug(
            "RAG LLM sync request: POST %s model=%s prompt_len=%d",
            url, model, len(prompt),
        )
        t0 = time.time()
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=(connect_timeout, read_timeout),
            )
            elapsed = time.time() - t0
            logger.debug("RAG LLM response status=%d elapsed=%.2fs", r.status_code, elapsed)
            r.raise_for_status()
            result = r.json()
            text = self._extract_text(result)
            self._call_count += 1
            if text:
                logger.info("RAG LLM answer (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
            else:
                logger.warning("RAG LLM returned no text (elapsed=%.2fs, keys=%s)", elapsed, list(result.keys()))
            return {"answer": text, "raw": result}
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            json.JSONDecodeError,
        ) as exc:
            self._mark_provider_unavailable(exc, "RAG LLM Groq sync call failed")
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}
        except Exception as exc:
            logger.error("RAG LLM Groq unexpected error: %s", exc, exc_info=True)
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}

    async def _call_ollama_async(self, prompt: str) -> Dict[str, Any]:
        """Async Ollama call via /api/chat (avoids thread-pool starvation)."""
        model = self._get_model()
        base = self._get_base_url()
        url = f"{base}/api/chat"

        read_timeout = float(os.environ.get("OLLAMA_TIMEOUT", "60.0"))
        connect_timeout = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5.0"))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )

        logger.debug(
            "RAG LLM async request: POST %s model=%s prompt_len=%d",
            url, model, len(prompt),
        )
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                elapsed = time.time() - t0
                logger.debug("RAG LLM async response status=%d elapsed=%.2fs", resp.status_code, elapsed)
                resp.raise_for_status()
                result = resp.json()
                text = self._extract_text(result)
                self._call_count += 1
                if text:
                    logger.info("RAG LLM async answer (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
                else:
                    logger.warning("RAG LLM async returned no text (elapsed=%.2fs)", elapsed)
                return {"answer": text, "raw": result}
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            json.JSONDecodeError,
        ) as exc:
            self._mark_provider_unavailable(exc, "RAG LLM async call failed")
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}
        except Exception as exc:
            logger.error("RAG LLM async unexpected error: %s", exc, exc_info=True)
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}

    async def _call_groq_async(self, prompt: str) -> Dict[str, Any]:
        """Async Groq call via /chat/completions."""
        model = self._get_model()
        base = self._get_base_url()
        url = f"{base}/chat/completions"
        api_key = self._get_groq_api_key()

        read_timeout = float(os.environ.get("GROQ_TIMEOUT", "30.0"))
        connect_timeout = float(os.environ.get("GROQ_CONNECT_TIMEOUT", "5.0"))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )

        logger.debug(
            "RAG LLM async request: POST %s model=%s prompt_len=%d",
            url, model, len(prompt),
        )
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
                elapsed = time.time() - t0
                logger.debug("RAG LLM async response status=%d elapsed=%.2fs", resp.status_code, elapsed)
                resp.raise_for_status()
                result = resp.json()
                text = self._extract_text(result)
                self._call_count += 1
                if text:
                    logger.info("RAG LLM async answer (%d chars, %.2fs): %s", len(text), elapsed, text[:120])
                else:
                    logger.warning("RAG LLM async returned no text (elapsed=%.2fs)", elapsed)
                return {"answer": text, "raw": result}
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            json.JSONDecodeError,
        ) as exc:
            self._mark_provider_unavailable(exc, "RAG LLM Groq async call failed")
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}
        except Exception as exc:
            logger.error("RAG LLM Groq async unexpected error: %s", exc, exc_info=True)
            self._fallback_count += 1
            return {"answer": _FALLBACK_ANSWER, "error": str(exc)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_request(self, session_id: str, action: str, args: Dict[str, Any], context: Dict[str, Any]):
        if action in ("query", "chat", None):
            question = args.get("question") or args.get("q") or args.get("prompt")
            if not question:
                raise ValueError("LLMAgent requires 'question' in args")
            provider = self._get_provider()

            # Check availability first
            if not self.is_available():
                self._fallback_count += 1
                logger.warning(
                    "RAG LLM unavailable — returning fallback (fallback #%d)",
                    self._fallback_count,
                )
                return {"answer": _FALLBACK_ANSWER, "retrieved": [], "llm_available": False}

            # Retrieve context documents
            retrieved = []
            if self._ingestor is not None:
                retrieved = self._ingestor.query(question, top_k=5)

            prompt = self._build_prompt(question, retrieved)

            # Prefer native async to avoid thread-pool starvation
            try:
                asyncio.get_running_loop()
                if provider == "groq":
                    res = await self._call_groq_async(prompt)
                else:
                    res = await self._call_ollama_async(prompt)
            except RuntimeError:
                # No running loop — use sync call
                if provider == "groq":
                    res = self._call_groq(prompt)
                else:
                    res = self._call_ollama(prompt)

            return {
                "answer": res.get("answer", ""),
                "retrieved": [r[0] for r in retrieved],
                "llm_available": True,
                "provider": provider,
            }

        raise ValueError(f"Unsupported LLMAgent action: {action}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return internal state for debugging."""
        return {
            "available": self._available,
            "provider": self._get_provider(),
            "model": self._get_model(),
            "base_url": self._get_base_url(),
            "call_count": self._call_count,
            "fallback_count": self._fallback_count,
            "check_interval": self._check_interval,
        }

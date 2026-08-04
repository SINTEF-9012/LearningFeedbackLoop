"""
IntegratedAgent — fan-out + synthesize across sub-agents.

Per the plan (Agent I / IMPLEMENTATION_PLAN_INTEGRATED_AGENT_PREREQS.txt):
calls retriever / monitoring / analytics / stoppage concurrently with per-
sub-agent timeout, degrades gracefully when any of them is unavailable, and
optionally produces a short synthesis via the LLM agent.

The retriever is the intern integration slot — this agent simply calls
whatever sub-agent is registered under the name ``retriever`` in the
router registry. Do not hardcode knowledge of its internals here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Sub-agents fanned out per query. Order matters only for deterministic
# output; execution is concurrent.
_DEFAULT_SUB_AGENTS = ("retriever", "monitoring", "analytics", "stoppage")


class IntegratedAgent:
    """Composite agent that dispatches to multiple sub-agents in parallel.

    Actions supported:
    - ``query`` (default): fan-out the question to each configured sub-agent,
      collect their responses, and optionally produce a combined synthesis.

    Input args:
        question / q / prompt: str  (required)
        sub_agents: Optional[List[str]]  (override which sub-agents to call)
        synthesize: Optional[bool]       (default True; off → skip LLM)
        per_agent_timeout_s: Optional[float]  (default 8.0)

    Output:
        {
            "sub_results": {name: {...} | {"error": str, "degraded": True}},
            "degraded": [name, ...],           # sub-agents that failed/timed out
            "synthesis": Optional[str],        # LLM combined answer, or None
            "synthesis_source": "llm"|"concat"|"skipped"|"unavailable",
        }
    """

    def __init__(
        self,
        *,
        get_agent: Optional[Callable[[str], Any]] = None,
        llm_agent: Optional[Any] = None,
        sub_agents: Optional[List[str]] = None,
        default_timeout_s: float = 8.0,
    ) -> None:
        # get_agent is a late-bound lookup so the registry can be populated
        # after this agent is instantiated (matches how the router starts
        # agents lazily).
        self._get_agent = get_agent
        self._llm = llm_agent
        self._sub_agent_names = tuple(sub_agents) if sub_agents else _DEFAULT_SUB_AGENTS
        self._default_timeout_s = float(default_timeout_s)

    async def start(self) -> None:  # noqa: D401 - lifecycle hook
        """No-op: sub-agents are started by the router."""
        return None

    # ------------------------------------------------------------------
    # Registry hookup
    # ------------------------------------------------------------------

    def _resolve_agent(self, name: str) -> Optional[Any]:
        if self._get_agent is None:
            return None
        try:
            return self._get_agent(name)
        except Exception:
            return None

    def _resolve_llm(self) -> Optional[Any]:
        if self._llm is not None:
            return self._llm
        return self._resolve_agent("llm.rag")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        session_id: str,
        action: Optional[str],
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if action not in (None, "query", "chat"):
            return {"error": f"Unsupported action: {action}"}

        question = (
            args.get("question")
            or args.get("q")
            or args.get("prompt")
            or ""
        )
        if not isinstance(question, str) or not question.strip():
            return {"error": "IntegratedAgent requires 'question' / 'q' / 'prompt' in args"}

        names = list(args.get("sub_agents") or self._sub_agent_names)
        timeout_s = float(args.get("per_agent_timeout_s") or self._default_timeout_s)
        do_synth = bool(args.get("synthesize", True))

        sub_results, degraded = await self._fanout(
            names, session_id=session_id, question=question, args=args,
            context=context, timeout_s=timeout_s,
        )

        synthesis, synth_source = await self._synthesize(
            question=question,
            sub_results=sub_results,
            enabled=do_synth,
        )

        return {
            "sub_results": sub_results,
            "degraded": degraded,
            "synthesis": synthesis,
            "synthesis_source": synth_source,
        }

    async def _fanout(
        self,
        names: List[str],
        *,
        session_id: str,
        question: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        timeout_s: float,
    ) -> tuple[Dict[str, Any], List[str]]:
        # Prepare one coroutine per configured sub-agent (skip missing ones).
        tasks: List[asyncio.Task] = []
        task_names: List[str] = []

        for name in names:
            agent = self._resolve_agent(name)
            if agent is None:
                tasks.append(asyncio.create_task(_absent_agent(name)))
                task_names.append(name)
                continue

            coro = self._call_sub_agent(
                agent, name,
                session_id=session_id,
                question=question,
                args=args,
                context=context,
                timeout_s=timeout_s,
            )
            tasks.append(asyncio.create_task(coro))
            task_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sub_results: Dict[str, Any] = {}
        degraded: List[str] = []
        for name, res in zip(task_names, results):
            if isinstance(res, BaseException):
                sub_results[name] = {"error": str(res), "degraded": True}
                degraded.append(name)
                continue
            if isinstance(res, dict) and res.get("degraded"):
                degraded.append(name)
            sub_results[name] = res

        return sub_results, degraded

    async def _call_sub_agent(
        self,
        agent: Any,
        name: str,
        *,
        session_id: str,
        question: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        timeout_s: float,
    ) -> Dict[str, Any]:
        sub_args = dict(args)
        sub_args.setdefault("question", question)
        sub_args.setdefault("q", question)
        try:
            coro = agent.handle_request(session_id, "query", sub_args, context)
            result = await asyncio.wait_for(coro, timeout=timeout_s)
            if not isinstance(result, dict):
                result = {"result": result}
            return result
        except asyncio.TimeoutError:
            logger.warning("IntegratedAgent: sub-agent '%s' timed out after %.1fs", name, timeout_s)
            return {"error": f"timeout after {timeout_s:.1f}s", "degraded": True}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("IntegratedAgent: sub-agent '%s' failed: %s", name, exc)
            return {"error": str(exc), "degraded": True}

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        *,
        question: str,
        sub_results: Dict[str, Any],
        enabled: bool,
    ) -> tuple[Optional[str], str]:
        if not enabled:
            return None, "skipped"

        llm = self._resolve_llm()
        if llm is None:
            return _concat_synthesis(sub_results), "concat"

        # Only call the LLM if it reports availability — avoids burning the
        # user's timeout budget on a dead service.
        is_avail = getattr(llm, "is_available", None)
        try:
            if callable(is_avail) and not is_avail():
                return _concat_synthesis(sub_results), "unavailable"
        except Exception:
            return _concat_synthesis(sub_results), "unavailable"

        prompt = _build_synthesis_prompt(question, sub_results)
        try:
            res = await llm.handle_request("integrated", "query", {"question": prompt}, {})
        except Exception as exc:
            logger.warning("IntegratedAgent: synthesis LLM call failed: %s", exc)
            return _concat_synthesis(sub_results), "unavailable"

        answer = ""
        if isinstance(res, dict):
            answer = str(res.get("answer") or res.get("result") or "").strip()
        if not answer:
            return _concat_synthesis(sub_results), "unavailable"
        return answer, "llm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _absent_agent(name: str) -> Dict[str, Any]:
    return {"error": f"sub-agent '{name}' not registered", "degraded": True}


def _concat_synthesis(sub_results: Dict[str, Any]) -> Optional[str]:
    """Deterministic no-LLM fallback: concatenate per-agent summaries."""
    lines: List[str] = []
    for name, res in sub_results.items():
        if not isinstance(res, dict):
            continue
        if res.get("degraded"):
            lines.append(f"- {name}: unavailable ({res.get('error', 'unknown')})")
            continue
        summary = (
            res.get("answer")
            or res.get("summary")
            or res.get("result")
            or _short_repr(res)
        )
        lines.append(f"- {name}: {summary}")
    if not lines:
        return None
    return "\n".join(lines)


def _short_repr(d: Dict[str, Any], limit: int = 200) -> str:
    keys = [k for k in d.keys() if k not in ("error", "degraded")]
    if not keys:
        return "(empty)"
    head = ", ".join(str(k) for k in keys[:6])
    return f"keys=[{head}]" if len(head) <= limit else f"keys=[{head[:limit]}...]"


def _build_synthesis_prompt(question: str, sub_results: Dict[str, Any]) -> str:
    blocks: List[str] = []
    for name, res in sub_results.items():
        if not isinstance(res, dict) or res.get("degraded"):
            continue
        text = (
            res.get("answer")
            or res.get("summary")
            or res.get("result")
            or ""
        )
        if text:
            blocks.append(f"[{name}]\n{text}")
    context_block = "\n\n".join(blocks) if blocks else "(no sub-agent data available)"
    return (
        "You are synthesizing answers from multiple expert sub-agents on a "
        "CNC monitoring system. Combine the information below into a short, "
        "factual answer to the user's question. Do not invent facts that are "
        "not in the sub-agent output. Flag any disagreement between sources.\n\n"
        f"User question: {question}\n\n"
        f"Sub-agent outputs:\n{context_block}\n\n"
        "Answer:"
    )

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, Optional

from backend.events import bus


class LiveRunnerBase:
    """Shared async execution shell for in-process experiment runners."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._channel = f"experiment.{self.run_id}"
        self._t0 = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._logger = logging.getLogger(self.__class__.__module__)

    async def execute(self) -> Dict[str, Any]:
        """Run the experiment in an executor and stream progress events."""
        self._t0 = time.time()
        self._loop = asyncio.get_running_loop()
        result: Dict[str, Any] = {}
        try:
            result = await self._loop.run_in_executor(None, self._run_sync)
            await self._emit(
                "done",
                "completed",
                self._success_message(),
                pct=100,
                detail=self._success_detail(result),
            )
        except Exception as exc:
            tb = traceback.format_exc()
            await self._emit("error", "error", str(exc), detail={"traceback": tb})
            result = {"success": False, "error": str(exc), "traceback": tb}
        return result

    def _run_sync(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _success_message(self) -> str:
        return "Experiment finished"

    def _success_detail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        detail = {"success": True}
        if isinstance(result, dict):
            detail.update(result.get("detail", {}))
        return detail

    def _event_payload(
        self,
        phase: str,
        status: str,
        message: str,
        *,
        pct: float = 0,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "phase": phase,
            "status": status,
            "message": message,
            "pct": round(pct, 1),
            "elapsed_s": round(time.time() - self._t0, 2),
        }
        if detail:
            payload["detail"] = detail
        return payload

    def _emit_sync(
        self,
        phase: str,
        status: str,
        message: str,
        *,
        pct: float = 0,
        detail: Optional[Dict[str, Any]] = None,
    ):
        """Synchronously schedule a progress event from the executor thread."""
        evt = self._event_payload(phase, status, message, pct=pct, detail=detail)
        self._logger.info("[%s] %s - %s (%.0f%%)", self.run_id, phase, message, pct)
        try:
            if self._loop is None:
                raise RuntimeError("runner event loop is not initialized")
            future = asyncio.run_coroutine_threadsafe(bus.publish(self._channel, evt), self._loop)
            future.result(timeout=5)
        except Exception:
            self._logger.debug("Failed to emit progress event %s/%s", phase, status, exc_info=True)

    async def _emit(
        self,
        phase: str,
        status: str,
        message: str,
        *,
        pct: float = 0,
        detail: Optional[Dict[str, Any]] = None,
    ):
        """Emit a progress event from the main async loop."""
        evt = self._event_payload(phase, status, message, pct=pct, detail=detail)
        await bus.publish(self._channel, evt)
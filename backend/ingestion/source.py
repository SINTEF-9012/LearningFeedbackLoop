"""Source abstractions for live and simulated ingestion."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Protocol


class StreamSource(Protocol):
    """Minimal protocol for a session-bound data source."""

    name: str

    def start(self, session_id: str, *, startup_delay: float = 0.0) -> asyncio.Task:
        ...

    async def run(self, session_id: str) -> None:
        ...

    async def stop(self, session_id: str) -> None:
        ...

    def status(self, session_id: str) -> Dict[str, Any]:
        ...
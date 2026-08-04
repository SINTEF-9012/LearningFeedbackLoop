"""Session-backed source that replays uploaded/demo signal files."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import time
from typing import Any, Dict

from backend.events import publish_feature
from backend.metadata_utils import get_sample_frequency
from backend.routers.dependencies import DEFAULT_SAMPLES_PER_TICK, DEFAULT_SPEED


logger = logging.getLogger(__name__)


class SimulatedFileSource:
    """Replay an in-memory uploaded/demo session as a live stream."""

    name = "simulated_file"

    def __init__(self, sessions: Dict[str, Dict[str, Any]]):
        self._sessions = sessions

    def _status_block(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions[session_id]
        status = session.setdefault(
            "source_status",
            {
                "kind": self.name,
                "connected": False,
                "last_frame_ts": None,
                "lag_ms": 0.0,
                "dropped": 0,
            },
        )
        status["kind"] = self.name
        return status

    def start(self, session_id: str, *, startup_delay: float = 0.0) -> asyncio.Task:
        session = self._sessions[session_id]
        session["source_name"] = self.name
        session["_stream_source"] = self

        async def _runner() -> None:
            if startup_delay > 0:
                await asyncio.sleep(startup_delay)
            await self.run(session_id)

        task = asyncio.create_task(_runner())
        session["task"] = task
        return task

    async def stop(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["running"] = False
        task = session.get("task")
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def status(self, session_id: str) -> Dict[str, Any]:
        return dict(self._status_block(session_id))

    async def run(self, session_id: str) -> None:
        session = self._sessions[session_id]
        session["source_name"] = self.name
        session["_stream_source"] = self
        session.pop("last_error", None)
        status = self._status_block(session_id)
        status["connected"] = True

        cfg = session["config"]
        data = session["data"]
        metadata = session["metadata"]

        fs = get_sample_frequency(metadata)
        sample_period = 1.0 / fs
        channels = cfg.get("channels") or list(data.keys())
        speed = float(cfg.get("speed", DEFAULT_SPEED))
        samples_per_tick = max(1, int(cfg.get("samples_per_tick", DEFAULT_SAMPLES_PER_TICK)))

        pos = int(session.get("position", 0))
        if not channels:
            raise ValueError("No channels available to process")
        n_max = min(len(data[ch]) for ch in channels)

        start_wall = time.perf_counter()
        start_pos = pos
        last_speed = speed
        last_samples_per_tick = samples_per_tick
        next_emit_time = start_wall

        try:
            while session.get("running", False) and pos < n_max:
                if session.get("paused", False):
                    await asyncio.sleep(0.1)
                    start_wall = time.perf_counter()
                    start_pos = pos
                    next_emit_time = start_wall
                    continue

                cfg_now = session.get("config", {})
                speed = float(cfg_now.get("speed", DEFAULT_SPEED))
                samples_per_tick = max(1, int(cfg_now.get("samples_per_tick", DEFAULT_SAMPLES_PER_TICK)))
                if speed != last_speed or samples_per_tick != last_samples_per_tick:
                    last_speed = speed
                    last_samples_per_tick = samples_per_tick
                    start_wall = time.perf_counter()
                    start_pos = pos
                    next_emit_time = start_wall

                i0 = pos
                i1 = min(pos + samples_per_tick, n_max)
                if i0 >= i1:
                    break

                if samples_per_tick == 1:
                    t = i0 / fs
                    frame: Dict[str, Any] = {"t": t, "i": i0, "fs": fs}
                    for channel in channels:
                        arr = data[channel]
                        frame[channel] = arr[i0] if i0 < len(arr) else None
                else:
                    t0 = i0 / fs
                    t1 = (i1 - 1) / fs
                    frame = {"t0": t0, "t1": t1, "i0": i0, "i1": i1, "fs": fs}
                    for channel in channels:
                        arr = data[channel]
                        frame[channel] = arr[i0:i1]

                pos = i1
                session["position"] = pos

                for queue in list(session.get("subscribers", [])):
                    await queue.put(frame)

                try:
                    payload = {
                        "type": "time",
                        "session_id": session_id,
                        "position": pos,
                        "frame": frame,
                        "metadata": deepcopy(metadata),
                    }
                    source_name = metadata.get("source")
                    if isinstance(source_name, str) and source_name:
                        payload["source"] = source_name
                    await publish_feature(session_id, payload)
                except Exception:
                    logger.warning("Feature publish failed (time); continuing stream", exc_info=True)

                samples_emitted = pos - start_pos
                target_elapsed = (samples_emitted * sample_period) / max(speed, 1e-9)
                next_emit_time = start_wall + target_elapsed
                now = time.perf_counter()
                delay = next_emit_time - now
                status["last_frame_ts"] = time.time()
                status["lag_ms"] = max(0.0, (now - next_emit_time) * 1000.0)
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

            session["running"] = False
            status["connected"] = False
            eos = {"eos": True, "fs": fs, "final_i": pos}
            for queue in list(session.get("subscribers", [])):
                try:
                    await queue.put(eos)
                except Exception:
                    pass

        except Exception as exc:
            session["last_error"] = str(exc)
            session["running"] = False
            status["connected"] = False
            logger.exception("SimulatedFileSource crashed for session %s", session_id)
        finally:
            session["task"] = None
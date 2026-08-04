"""1 Hz case-data source backed by the real casedata CSV layout."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.agents.processing.dataset_loader import CHANNEL_GROUPS, KEY_COLUMNS, DatasetLoader, pd
from backend.events import publish_feature
from backend.ingestion.schema import FrameEnvelope


logger = logging.getLogger(__name__)


def _selected_group_columns(group_name: str, columns: list[str]) -> list[str]:
    if group_name == "vibration":
        return ["timestamp"] + [column for column in columns if column != "timestamp"]

    selected = [column for column in KEY_COLUMNS[group_name] if column in columns]
    return ["timestamp", *selected]


class SimulatedCasedataSource:
    """Replay a merged 1 Hz case-data operation through the feature bus."""

    name = "simulated_casedata"

    def __init__(
        self,
        sessions: Dict[str, Dict[str, Any]],
        *,
        casedata_root: str | Path,
        operation_id: str,
        case_dir: Optional[str] = None,
        tolerance_seconds: float = 1.0,
    ):
        if pd is None:
            raise RuntimeError("pandas is required for SimulatedCasedataSource")
        self._sessions = sessions
        self._root = Path(casedata_root)
        self._operation_id = operation_id
        self._case_dir = case_dir
        self._tolerance_seconds = tolerance_seconds
        self._merged_cache = None
        self._operation_meta: Dict[str, Any] | None = None

    @classmethod
    def resolve_operation_id(
        cls,
        casedata_root: str | Path,
        operation_id: Optional[str] = None,
        case_dir: Optional[str] = None,
    ) -> str:
        if pd is None:
            raise RuntimeError("pandas is required for SimulatedCasedataSource")
        loader = DatasetLoader(casedata_root)
        if operation_id:
            loader.get_operation(operation_id, case=case_dir)
            return operation_id
        operations = loader.list_operations(case=case_dir)
        if not operations:
            if case_dir:
                raise ValueError(f"No casedata operations found under {casedata_root} for case {case_dir}")
            raise ValueError(f"No casedata operations found under {casedata_root}")
        return operations[0].operation_id

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
                "case_dir": self._case_dir,
                "operation_id": self._operation_id,
            },
        )
        status["kind"] = self.name
        status["case_dir"] = self._case_dir
        status["operation_id"] = self._operation_id
        return status

    def _load_rows(self):
        if self._merged_cache is not None:
            return self._merged_cache, self._operation_meta or {}

        loader = DatasetLoader(self._root)
        operation = loader.get_operation(self._operation_id, case=self._case_dir)
        frames = []
        for friendly in CHANNEL_GROUPS:
            file_path = operation.channel_files.get(friendly)
            if file_path is None or not file_path.exists():
                continue
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            available = _selected_group_columns(friendly, list(df.columns))
            if len(available) <= 1:
                continue
            frames.append(
                df[available]
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"])
                .reset_index(drop=True)
            )

        if not frames:
            raise ValueError(f"No casedata frames available for operation {self._operation_id}")

        merged = frames[0]
        tolerance = pd.Timedelta(seconds=self._tolerance_seconds)
        for frame in frames[1:]:
            merged = pd.merge_asof(
                merged,
                frame,
                on="timestamp",
                direction="nearest",
                tolerance=tolerance,
            )

        value_columns = [col for col in merged.columns if col != "timestamp"]
        merged = merged.dropna(how="all", subset=value_columns).reset_index(drop=True)
        self._merged_cache = merged
        self._operation_meta = {
            "operation_id": operation.operation_id,
            "case_dir": operation.case_dir,
            "tool_id": operation.tool_id,
        }
        return merged, self._operation_meta

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

    def session_data(self) -> Tuple[Dict[str, list[float]], Dict[str, Any]]:
        rows, meta = self._load_rows()
        data: Dict[str, list[float]] = {}
        for column in rows.columns:
            if column == "timestamp":
                continue
            series = rows[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            clean = series.astype(float).ffill().bfill().fillna(0.0)
            data[column] = clean.tolist()

        metadata: Dict[str, Any] = {
            "sample_frequency": 1.0,
            "source": self.name,
            "casedata": {
                **meta,
                "root": str(self._root),
            },
        }
        return data, metadata

    def session_time_axis_unix(self) -> list[float]:
        rows, _ = self._load_rows()
        return [float(ts.timestamp()) for ts in rows["timestamp"].tolist()]

    async def run(self, session_id: str) -> None:
        session = self._sessions[session_id]
        session["source_name"] = self.name
        session["_stream_source"] = self
        session.pop("last_error", None)
        status = self._status_block(session_id)
        status["connected"] = True

        rows, meta = self._load_rows()
        speed = float(session.get("config", {}).get("speed", 1.0))
        pos = int(session.get("position", 0) or 0)
        start_wall = time.perf_counter()
        start_pos = pos
        last_speed = speed
        next_emit_time = start_wall
        warmup_samples = max(
            0,
            int(session.get("inference_config", {}).get("window_samples", 0) or 0),
        )
        warmup_until = min(len(rows), pos + warmup_samples)

        try:
            while session.get("running", False) and pos < len(rows):
                if session.get("paused", False):
                    await asyncio.sleep(0.1)
                    start_wall = time.perf_counter()
                    start_pos = pos
                    next_emit_time = start_wall
                    continue

                cfg_now = session.get("config", {})
                speed = float(cfg_now.get("speed", 1.0))
                if speed != last_speed:
                    last_speed = speed
                    start_wall = time.perf_counter()
                    start_pos = pos
                    next_emit_time = start_wall

                row = rows.iloc[pos]
                ts = row["timestamp"]
                signals: Dict[str, float] = {}
                for column, value in row.items():
                    if column == "timestamp" or value is None:
                        continue
                    if isinstance(value, bool):
                        continue
                    try:
                        if pd.isna(value):
                            continue
                    except Exception:
                        pass
                    if isinstance(value, (int, float)):
                        signals[column] = float(value)

                frame = {
                    "t": float(pos),
                    "i": int(pos),
                    "fs": 1.0,
                    "ts_unix": ts.timestamp(),
                    "timestamp": ts.isoformat(),
                    **signals,
                }
                session["position"] = pos + 1

                for queue in list(session.get("subscribers", [])):
                    await queue.put(frame)

                envelope = FrameEnvelope(
                    kind="tag_sample",
                    session_id=session_id,
                    ts_unix=ts.timestamp(),
                    position=pos,
                    fs=1.0,
                    source=self.name,
                    signals=signals,
                    metadata=meta,
                )
                await publish_feature(session_id, envelope)

                samples_emitted = (pos + 1) - start_pos
                target_elapsed = samples_emitted / max(speed, 1e-9)
                next_emit_time = start_wall + target_elapsed
                now = time.perf_counter()
                delay = next_emit_time - now
                status["last_frame_ts"] = ts.timestamp()
                status["lag_ms"] = max(0.0, (now - next_emit_time) * 1000.0)
                pos += 1

                # Front-load the first inference window so the UI can show
                # live model outputs shortly after a large casedata session
                # finishes loading instead of waiting another full window in
                # real time.
                if pos < warmup_until:
                    await asyncio.sleep(0)
                    continue
                if warmup_until and pos == warmup_until:
                    warmup_until = 0
                    start_wall = time.perf_counter()
                    start_pos = pos
                    next_emit_time = start_wall
                    await asyncio.sleep(0)
                    continue

                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

            session["running"] = False
            status["connected"] = False
            eos = {"eos": True, "fs": 1.0, "final_i": pos}
            for queue in list(session.get("subscribers", [])):
                try:
                    await queue.put(eos)
                except Exception:
                    pass

        except Exception as exc:
            session["last_error"] = str(exc)
            session["running"] = False
            status["connected"] = False
            logger.exception("SimulatedCasedataSource crashed for session %s", session_id)
        finally:
            session["task"] = None
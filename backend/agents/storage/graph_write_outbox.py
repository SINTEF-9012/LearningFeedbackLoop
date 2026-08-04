from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class GraphWriteIntent:
    kind: str
    payload: Dict[str, Any]
    created_at: str = field(default_factory=_now_iso)
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphWriteIntent":
        return cls(
            kind=str(data.get("kind") or ""),
            payload=dict(data.get("payload") or {}),
            created_at=str(data.get("created_at") or _now_iso()),
            sequence=int(data.get("sequence") or 0),
        )


class GraphWriteOutbox:
    """Append-only JSONL outbox for deferred Neo4j write intents."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cursor_path = self.path.with_suffix(self.path.suffix + ".cursor")
        self._lock = threading.Lock()
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    self._sequence = max(self._sequence, int(record.get("sequence") or 0))
            except OSError:
                pass

    def append(self, intent: GraphWriteIntent) -> GraphWriteIntent:
        with self._lock:
            self._sequence += 1
            intent.sequence = self._sequence
            payload = json.dumps(intent.to_dict(), separators=(",", ":"), default=str)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
            return intent

    def _read_cursor(self) -> int:
        try:
            return int(self.cursor_path.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    def _write_cursor(self, sequence: int) -> None:
        tmp = self.cursor_path.with_suffix(self.cursor_path.suffix + ".tmp")
        tmp.write_text(str(int(sequence)), encoding="utf-8")
        os.replace(tmp, self.cursor_path)

    def iter_pending(self) -> Iterable[GraphWriteIntent]:
        if not self.path.exists():
            return
        cursor = self._read_cursor()
        # Fast path: nothing to replay. Avoid opening and JSON-parsing the
        # whole file (which can grow to hundreds of MB over many sessions)
        # every time the cursor has already caught up to the last append.
        if self._sequence > 0 and cursor >= self._sequence:
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sequence = int(record.get("sequence") or 0)
                if sequence > cursor:
                    yield GraphWriteIntent.from_dict(record)

    def mark_acked(self, sequence: int) -> None:
        if sequence > 0:
            self._write_cursor(sequence)

    def pending_count(self) -> int:
        return sum(1 for _ in self.iter_pending())

    def compact(self) -> int:
        """Rewrite the on-disk log keeping only unacked (pending) intents.

        Stops the append-only file from growing without bound across long
        sessions. Sequence numbers are preserved so an existing cursor stays
        valid; when nothing is pending the file and cursor are both reset to
        zero so a stale-high cursor can never suppress future intents after a
        restart. Returns the number of intents retained.
        """
        with self._lock:
            if not self.path.exists():
                return 0
            cursor = self._read_cursor()
            retained: list[str] = []
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            record = json.loads(stripped)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if int(record.get("sequence") or 0) > cursor:
                            retained.append(stripped)
            except OSError:
                return -1
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                if retained:
                    handle.write("\n".join(retained) + "\n")
            os.replace(tmp, self.path)
            if not retained:
                self._sequence = 0
                self._write_cursor(0)
            return len(retained)

    def maybe_compact(self, *, min_bytes: int = 5_000_000) -> int:
        """Compact only once the file has grown past ``min_bytes``.

        Cheap to call after every flush: a bare ``stat()`` when the log is
        small, a full rewrite only when it has actually grown large.
        """
        try:
            if not self.path.exists() or self.path.stat().st_size < min_bytes:
                return 0
        except OSError:
            return 0
        return self.compact()

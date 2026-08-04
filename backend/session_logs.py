from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SESSION_LOG_DIR = _PROJECT_ROOT / "data" / "sessions" / "demo_logs"


def session_log_path(session_id: str) -> Path:
    return _SESSION_LOG_DIR / f"{session_id}.jsonl"


def append_session_log(session_id: str, payload: Dict[str, Any]) -> Path:
    path = session_log_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    return path
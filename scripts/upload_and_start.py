"""Create a session, upload a JSON file, and start playback.

Demo notes
- For the realtime UI demo, sessions can complete before you attach.
- Use a slow speed (e.g. 0.02) and/or start paused so the UI can connect.

Usage:
    python scripts/upload_and_start.py test_data/sample_session.json
    python scripts/upload_and_start.py test_data/sample_session.json --speed 0.02 --samples-per-tick 8 --start-paused
"""

import argparse
import sys

import requests

BASE = "http://localhost:8000"

def create_session(*, base: str, speed: float, samples_per_tick: int, start_paused: bool):
    r = requests.post(
        f"{base}/sessions",
        json={
            "interval_ms": 100,
            "channels": None,
            "mode": "time",
            "speed": float(speed),
            "samples_per_tick": int(samples_per_tick),
            "start_paused": bool(start_paused),
        },
    )
    r.raise_for_status()
    return r.json()["session_id"]

def upload(*, base: str, session_id: str, file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": f}
        r = requests.post(f"{base}/sessions/{session_id}/upload", files=files)
    r.raise_for_status()
    return r.json()

def start(*, base: str, session_id: str):
    r = requests.post(f"{base}/sessions/{session_id}/start")
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a session, upload data, and start playback")
    parser.add_argument("path", help="Path to JSON session file")
    parser.add_argument("--url", default=BASE, help="Base API URL (default: http://localhost:8000)")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.02,
        help="Playback speed multiplier (1.0=real-time, 0.02=50x slower)",
    )
    parser.add_argument(
        "--samples-per-tick",
        type=int,
        default=8,
        help="How many samples to emit per tick (smaller = smoother, bigger = lower overhead)",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start session in paused state so the UI can attach before streaming",
    )
    args = parser.parse_args()

    path = args.path
    base = str(args.url).rstrip("/")
    sid = create_session(base=base, speed=args.speed, samples_per_tick=args.samples_per_tick, start_paused=args.start_paused)
    print("Created session:", sid)
    print("Uploading...", upload(base=base, session_id=sid, file_path=path))
    print("Starting playback...", start(base=base, session_id=sid))
    if args.start_paused:
        print("Session is STARTED but PAUSED. Resume via POST /sessions/{sid}/resume or the UI controls.")
    print("Done. Use ws listener or visualizer with session id:", sid)

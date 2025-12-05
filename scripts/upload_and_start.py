"""Create a session, upload a JSON file, and start playback.

Usage: python scripts/upload_and_start.py sample_session.json
"""
import sys
import requests

BASE = "http://localhost:8000"

def create_session():
    r = requests.post(f"{BASE}/sessions", json={"interval_ms": 100, "channels": None, "mode": "time"})
    r.raise_for_status()
    return r.json()["session_id"]

def upload(session_id: str, file_path: str):
    with open(file_path, "rb") as f:
        files = {"file": f}
        r = requests.post(f"{BASE}/sessions/{session_id}/upload", files=files)
    r.raise_for_status()
    return r.json()

def start(session_id: str):
    r = requests.post(f"{BASE}/sessions/{session_id}/start")
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/upload_and_start.py sample_session.json")
        sys.exit(1)
    path = sys.argv[1]
    sid = create_session()
    print("Created session:", sid)
    print("Uploading...", upload(sid, path))
    print("Starting playback...", start(sid))
    print("Done. Use ws listener or visualizer with session id:", sid)

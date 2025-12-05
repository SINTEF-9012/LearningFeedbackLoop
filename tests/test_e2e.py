import time
import json
import requests
import pytest

BASE = "http://localhost:8000"


def server_available():
    try:
        r = requests.get(f"{BASE}/sessions")
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not server_available(), reason="Server not available on localhost:8000")
def test_create_upload_start_and_stream():
    # create session
    r = requests.post(f"{BASE}/sessions", json={"interval_ms": 50, "channels": None, "mode": "time"})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # upload sample file
    with open("test_data/sample_session.json", "rb") as f:
        files = {"file": f}
        ru = requests.post(f"{BASE}/sessions/{sid}/upload", files=files)
    assert ru.status_code == 200

    # start playback
    rs = requests.post(f"{BASE}/sessions/{sid}/start")
    assert rs.status_code == 200

    # wait briefly for server to emit frames
    time.sleep(0.5)

    # ask compute agent for a quick analyze (non-blocking)
    payload = {
        "agent": "compute",
        "action": "analyze",
        "args": {"channel": "A", "start": 0, "end": 10}
    }
    ra = requests.post(f"{BASE}/agent/dispatch/{sid}", json=payload)
    assert ra.status_code == 200
    assert ra.json().get("ok", True) is True

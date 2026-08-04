"""Tiny CLI to dispatch agent requests to the server.

Usage examples:
  python scripts/agent_cli.py <session_id> online start
  python scripts/agent_cli.py <session_id> compute amplitudes '{"request": {"window": {"t_min": 0, "t_max": 1}}}'
"""
import sys
import json
import requests

BASE = "http://localhost:8000"

def dispatch(session_id, agent, action, args=None):
    payload = {"agent": agent, "action": action, "args": args or {}}
    r = requests.post(f"{BASE}/agent/dispatch/{session_id}", json=payload)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: agent_cli.py <session_id> <agent> <action> [args-json]")
        raise SystemExit(1)
    sid = sys.argv[1]
    agent = sys.argv[2]
    action = sys.argv[3]
    args = None
    if len(sys.argv) > 4:
        args = json.loads(sys.argv[4])
    print(dispatch(sid, agent, action, args))

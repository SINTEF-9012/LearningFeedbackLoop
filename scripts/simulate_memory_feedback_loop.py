"""Synthetic memory+feedback demo driver (no playback required).

This script exercises the *most important* part of the demo: memories + feedback
changing pattern priors in real time.

It repeatedly:
  1) POSTs a synthetic event to /agent/memory/events
  2) Optionally confirms/dismisses the created memory
  3) Prints the top priors occasionally

Run:
  ./.venv/bin/python scripts/simulate_memory_feedback_loop.py --url http://localhost:8000

Then open the UI and watch the priors chart update.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any, Dict, List, Optional

import requests


def _now_session_id() -> str:
    return f"synthetic_{int(time.time() * 1000)}"


def _post_json(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _patch_json(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.patch(f"{base_url}{path}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_json(base_url: str, path: str) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _reset_priors(base_url: str) -> None:
    _post_json(base_url, "/agent/memory/scorer/reset-priors", {})


def _pick_patterns(pool: List[str], *, k: int) -> List[str]:
    if not pool:
        return []
    k = max(1, min(k, len(pool)))
    return random.sample(pool, k=k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    ap.add_argument("--session", default=None, help="Session id to use (default: synthetic_<ts>)")
    ap.add_argument("--rate-hz", type=float, default=4.0, help="Events per second")
    ap.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    ap.add_argument("--fs", type=float, default=1000.0, help="Sampling rate used in time_range")
    ap.add_argument("--window", type=int, default=1000, help="Synthetic window size (samples)")
    ap.add_argument("--patterns-per-event", type=int, default=2, help="How many patterns to attach")
    ap.add_argument(
        "--feedback-every",
        type=int,
        default=None,
        help="If set, send feedback every N events (overrides --feedback-prob)",
    )
    ap.add_argument("--feedback-prob", type=float, default=0.7, help="Probability of sending feedback")
    ap.add_argument("--confirm-prob", type=float, default=0.6, help="Given feedback, probability of confirm vs dismiss")
    ap.add_argument("--print-priors-every", type=int, default=10, help="Print priors every N events")
    ap.add_argument("--reset-priors", action="store_true", help="Reset priors before starting")

    # Keep the pool small and interpretable for demo.
    ap.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help="Add a pattern key to the pool (repeatable)",
    )

    args = ap.parse_args()

    base_url = args.url.rstrip("/")
    session_id = args.session or _now_session_id()

    patterns_pool = args.patterns or [
        "CHATTER_DETECTED",
        "ANOMALY_HIGH",
        "SPECTRAL_PEAK_425Hz",
        "SPECTRAL_PEAK_512Hz",
        "RATIO_Fx_Fy:>5",
        "RATIO_Fz_Fx:>2",
    ]

    if args.reset_priors:
        print("Resetting priors...")
        _reset_priors(base_url)

    print(f"Base URL: {base_url}")
    print(f"Session:  {session_id}")
    print(f"Rate:     {args.rate_hz} Hz")
    print(f"Pool:     {patterns_pool}")

    dt = 1.0 / max(args.rate_hz, 1e-9)
    start = time.time()

    i1 = 0
    event_n = 0

    while True:
        if args.duration is not None and (time.time() - start) >= args.duration:
            break

        i0 = max(0, i1 - int(args.window))

        payload = {
            "session_id": session_id,
            "time_range": {
                "i0": i0,
                "i1": i1,
                "t0": float(i0 / args.fs),
                "t1": float(i1 / args.fs),
                "fs": float(args.fs),
            },
            "pattern_keys": _pick_patterns(patterns_pool, k=int(args.patterns_per_event)),
            "channels": ["Fx", "Fy", "Fz"],
            "external_signals": {
                # a non-zero signal can help drive score rules, depending on config.
                "synthetic_score": float(random.random()),
            },
        }

        try:
            res = _post_json(base_url, "/agent/memory/events", payload)
        except Exception as e:
            print(f"POST /agent/memory/events failed: {e}")
            time.sleep(1.0)
            continue

        event_n += 1
        i1 += int(max(1, args.window // 4))

        mem_id = res.get("memory_id")
        action = res.get("action")
        score = res.get("significance_score")

        print(f"event#{event_n:04d} action={action} score={score} memory_id={mem_id}")

        # Optionally apply feedback to drive priors.
        should_feedback = False
        if mem_id:
            if args.feedback_every is not None:
                n = max(1, int(args.feedback_every))
                should_feedback = (event_n % n) == 0
            else:
                should_feedback = random.random() < float(args.feedback_prob)

        if should_feedback:
            do_confirm = random.random() < float(args.confirm_prob)
            fb_action = "confirm" if do_confirm else "dismiss"
            try:
                _patch_json(
                    base_url,
                    f"/agent/memory/{mem_id}/feedback",
                    {"action": fb_action, "user_id": "synthetic", "reason": "demo", "comment": ""},
                )
                print(f"  feedback: {fb_action}")
            except Exception as e:
                print(f"  feedback failed: {e}")

        if args.print_priors_every and (event_n % int(args.print_priors_every) == 0):
            try:
                pri = _get_json(base_url, "/agent/memory/scorer/priors?limit=10").get("priors") or []
                print("  top priors:")
                for p in pri[:10]:
                    print(f"    {p.get('pattern')}: {p.get('prior')}")
            except Exception as e:
                print(f"  priors fetch failed: {e}")

        # Rate limit
        time.sleep(dt)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

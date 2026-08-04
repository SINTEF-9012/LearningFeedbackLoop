#!/usr/bin/env python3
"""Live plot of memory-system alerts (significance over time).

Connects to the memory alerts websocket and plots significance score as alerts
arrive. This is a simple visualization to demonstrate the memory-first
feedback loop: as you confirm/dismiss memories (feedback), future alerts and
scores should shift.

Usage:
  # Global alerts (all sessions)
  python scripts/visualize_memory_alerts.py --session all

  # Single session
  python scripts/visualize_memory_alerts.py --session <session_id>

  # Custom base URL
  python scripts/visualize_memory_alerts.py --url http://localhost:8000 --session all

Notes:
  - Alerts websocket is served under the agents router:
      ws://<host>/agent/memory/alerts/{session_id}
  - Use session_id="all" to receive all alerts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import websockets


def _clip_text(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")


def _action_color(action: str) -> str:
    a = (action or "").lower()
    if a == "critical":
        return "#d62728"  # red
    if a == "alert":
        return "#ff7f0e"  # orange
    if a == "store":
        return "#1f77b4"  # blue
    return "#7f7f7f"  # gray


def _http_to_ws(url: str) -> str:
    url = url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    # If user already passed ws:// or wss://, keep it.
    return url


async def _listen_and_plot(
    ws_url: str,
    *,
    window: int,
    feed_size: int,
    duration_s: Optional[float],
    redraw_hz: float,
) -> None:
    xs: Deque[float] = deque(maxlen=window)
    ys: Deque[float] = deque(maxlen=window)
    labels: Deque[str] = deque(maxlen=window)
    colors: Deque[str] = deque(maxlen=window)
    feed: Deque[Dict[str, object]] = deque(maxlen=feed_size)

    start = time.time()

    plt.ion()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_feed = fig.add_subplot(gs[0, 1])

    (line,) = ax.plot([], [], linewidth=1.5, alpha=0.7)
    scatter = ax.scatter([], [], s=28)

    # Threshold guides (match default config)
    ax.axhline(0.30, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.60, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.85, color="gray", linestyle=":", linewidth=1)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("seconds")
    ax.set_ylabel("significance score")
    ax.set_title("Memory alerts (score timeline)")
    ax.grid(True, linestyle=":", alpha=0.35)

    ax_feed.set_axis_off()
    ax_feed.set_title("Live alert feed", fontsize=11)

    last_action_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    stats = {"total": 0, "alert": 0, "critical": 0, "store": 0, "ignore": 0}
    last_redraw = 0.0
    redraw_min_dt = 0.0 if redraw_hz <= 0 else (1.0 / redraw_hz)

    async with websockets.connect(ws_url) as ws:
        while True:
            if duration_s is not None and (time.time() - start) > duration_s:
                break

            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            # Message schema comes from AlertDispatcher.SignificantEventAlert.to_dict()
            # (server sends to_json()).
            significance = (msg.get("significance") or {})
            score = significance.get("score")
            action = significance.get("action")
            session_id = msg.get("session_id")
            event_id = msg.get("event_id")
            reasons = significance.get("reasons") or []
            patterns = msg.get("patterns") or []
            summary = msg.get("summary")
            ts = msg.get("timestamp")

            if score is None:
                continue

            t = time.time() - start
            xs.append(float(t))
            ys.append(float(score))
            labels.append(str(action or ""))
            colors.append(_action_color(str(action or "")))

            stats["total"] += 1
            a_norm = str(action or "").lower()
            if a_norm in stats:
                stats[a_norm] += 1

            feed.appendleft(
                {
                    "t": float(t),
                    "timestamp": ts,
                    "action": action,
                    "score": float(score),
                    "session_id": session_id,
                    "event_id": event_id,
                    "patterns": patterns,
                    "reasons": reasons,
                    "summary": summary,
                }
            )

            # Print a concise log line.
            print(f"t={t:7.2f}s score={float(score):.3f} action={action} session={session_id} id={event_id}")

            # Throttle redraws to keep UI responsive.
            now = time.time()
            if redraw_min_dt and (now - last_redraw) < redraw_min_dt:
                continue
            last_redraw = now

            # Update plot (timeline)
            if len(xs) >= 2:
                x_list = list(xs)
                y_list = list(ys)
                line.set_data(x_list, y_list)
                ax.set_xlim(max(0.0, x_list[0]), x_list[-1] + 0.25)
                scatter.set_offsets(list(zip(x_list, y_list)))
                scatter.set_color(list(colors))
            elif len(xs) == 1:
                scatter.set_offsets([(xs[0], ys[0])])
                scatter.set_color([colors[0]])
                ax.set_xlim(0.0, xs[0] + 0.25)

            last_action_text.set_text(
                "\n".join(
                    [
                        f"last: {action}  score={float(score):.3f}",
                        f"session: {session_id}",
                        f"counts: total={stats['total']}  alert={stats['alert']}  critical={stats['critical']}  store={stats['store']}",
                    ]
                )
            )

            # Update feed panel
            lines = []
            for i, ev in enumerate(list(feed)[:feed_size], 1):
                a = str(ev.get("action") or "")
                sc = float(ev.get("score") or 0.0)
                sid = str(ev.get("session_id") or "")
                pats = ev.get("patterns") or []
                rs = ev.get("reasons") or []
                summ = _clip_text(ev.get("summary"), 90)
                pats_preview = _clip_text(", ".join([str(p) for p in pats[:4]]), 60)
                r0 = _clip_text(rs[0] if rs else "", 70)
                lines.append(f"{i:02d}. {a:8s}  {sc:0.3f}  sid={_clip_text(sid, 14)}")
                if pats_preview:
                    lines.append(f"    patterns: {pats_preview}")
                if r0:
                    lines.append(f"    reason:   {r0}")
                if summ:
                    lines.append(f"    summary:  {summ}")

            ax_feed.clear()
            ax_feed.set_axis_off()
            ax_feed.set_title("Live alert feed", fontsize=11)
            ax_feed.text(0.0, 1.0, "\n".join(lines) if lines else "(no alerts yet)", va="top", family="monospace", fontsize=9)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    ap.add_argument("--session", default="all", help="Session ID or 'all'")
    ap.add_argument("--window", type=int, default=200, help="Max points in rolling plot")
    ap.add_argument("--feed-size", type=int, default=12, help="Number of recent alerts to show in feed")
    ap.add_argument(
        "--redraw-hz",
        type=float,
        default=15.0,
        help="Max UI redraw rate (set 0 to redraw every message)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run until Ctrl+C)",
    )
    args = ap.parse_args()

    ws_base = _http_to_ws(args.url)
    ws_url = f"{ws_base}/agent/memory/alerts/{args.session}"

    print(f"Connecting: {ws_url}")
    try:
        asyncio.run(
            _listen_and_plot(
                ws_url,
                window=args.window,
                feed_size=args.feed_size,
                duration_s=args.duration,
                redraw_hz=args.redraw_hz,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

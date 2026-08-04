#!/usr/bin/env python3
"""Poll and visualize memory scorer priors over time.

This is meant to demonstrate the operator feedback loop:
- Confirming/dismissing memories shifts pattern priors.
- Those priors persist and influence future significance scoring.

The script polls:
  GET /agent/memory/scorer/priors
and plots the top-K priors as time series.

Usage:
  # Start server first:
  #   uvicorn backend.app:app --reload --port 8000

  python scripts/visualize_memory_priors.py --url http://localhost:8000 --top-k 10

Tips:
  - Run this alongside scripts/demo_memory_feedback.py.
  - After confirmations/dismissals, you should see the plotted priors drift.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests


def _poll_priors(base_url: str, limit: int) -> List[Tuple[str, float]]:
    base_url = base_url.rstrip("/")
    r = requests.get(f"{base_url}/agent/memory/scorer/priors", params={"limit": limit}, timeout=10)
    r.raise_for_status()
    payload = r.json()
    priors = payload.get("priors") or []
    out: List[Tuple[str, float]] = []
    for item in priors:
        try:
            out.append((str(item.get("pattern")), float(item.get("prior"))))
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    ap.add_argument("--top-k", type=int, default=10, help="Number of priors to display")
    ap.add_argument("--interval", type=float, default=2.0, help="Polling interval (seconds)")
    ap.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive plotting (print/save only).",
    )
    ap.add_argument(
        "--print",
        dest="print_priors",
        action="store_true",
        help="Print the top-K priors each poll (useful in headless environments).",
    )
    ap.add_argument(
        "--save-dir",
        default=None,
        help="Directory to write a continuously-updated priors.png (useful in headless environments).",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Minimum prior change to record a new datapoint",
    )
    ap.add_argument(
        "--append-on-change",
        action="store_true",
        help="Only append points when priors change (reduces flat lines)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=300,
        help="Rolling max datapoints to keep per pattern",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run until Ctrl+C)",
    )
    ap.add_argument(
        "--mode",
        choices=["bar", "timeseries", "both"],
        default="both",
        help="Visualization mode",
    )
    args = ap.parse_args()

    backend = str(plt.get_backend() or "").lower()
    has_display = bool(os.environ.get("DISPLAY")) or sys.platform.startswith("win")
    is_noninteractive_backend = ("agg" in backend) and ("tk" not in backend) and ("qt" not in backend)
    headless = bool(args.no_gui) or (is_noninteractive_backend and not has_display)

    save_dir: Optional[Path] = Path(args.save_dir).expanduser() if args.save_dir else None
    if headless and (save_dir is None) and (not args.print_priors):
        # In headless mode, default to printing so users still see learning.
        args.print_priors = True

    start = time.time()

    series_x: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.window))
    series_y: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=args.window))
    last_seen: Dict[str, float] = {}

    if not headless:
        plt.ion()
    if args.mode == "bar":
        fig, ax_bar = plt.subplots(figsize=(10, 6))
        ax_ts = None
    elif args.mode == "timeseries":
        fig, ax_ts = plt.subplots(figsize=(10, 6))
        ax_bar = None
    else:
        fig, (ax_bar, ax_ts) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1, 2])

    if headless:
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / "priors.png"
            print(f"Headless mode detected (matplotlib backend={plt.get_backend()}).")
            print(f"Saving plot snapshots to: {out_path}")
        else:
            print(f"Headless mode detected (matplotlib backend={plt.get_backend()}).")
            print("Printing priors to stdout (use --save-dir to write priors.png).")

    if ax_bar is not None:
        ax_bar.set_xlim(0.0, 1.0)
        ax_bar.set_xlabel("prior")
        ax_bar.set_title("Current pattern priors (top-K)")
        ax_bar.axvline(0.5, color="gray", linestyle="--", linewidth=1)

    lines: Dict[str, any] = {}

    def ensure_line(pattern: str):
        if ax_ts is None or pattern in lines:
            return
        (line,) = ax_ts.plot([], [], linewidth=2, marker="o", markersize=3, label=pattern)
        lines[pattern] = line
        # Keep legend readable for small K only.
        if len(lines) <= 12:
            ax_ts.legend(loc="upper left", fontsize="small")

    try:
        while True:
            if args.duration is not None and (time.time() - start) > args.duration:
                break

            now = time.time()
            t = now - start

            priors = _poll_priors(args.url, limit=max(args.top_k, 1))
            if priors:
                priors = priors[: args.top_k]
                top_patterns = [p for (p, _) in priors]

                if args.print_priors:
                    joined = " | ".join([f"{p}={v:.3f}" for (p, v) in priors])
                    print(f"t={t:7.2f}s  {joined}")

                # BAR VIEW: redraw as a horizontal bar chart for clarity.
                if ax_bar is not None:
                    ax_bar.clear()
                    ax_bar.set_xlim(0.0, 1.0)
                    ax_bar.set_xlabel("prior")
                    ax_bar.set_title("Current pattern priors (top-K)")
                    ax_bar.axvline(0.5, color="gray", linestyle="--", linewidth=1)
                    # plot highest first (top of chart)
                    patterns = list(reversed(top_patterns))
                    values = [v for (_, v) in priors]
                    values = list(reversed(values))
                    y = list(range(len(patterns)))
                    ax_bar.barh(y, values, color="#4C78A8")
                    ax_bar.set_yticks(y)
                    ax_bar.set_yticklabels(patterns, fontsize="small")
                    ax_bar.grid(axis="x", linestyle=":", alpha=0.4)

                # TIMESERIES VIEW: record + plot either every poll or only when changed.
                if ax_ts is not None:
                    ax_ts.set_ylim(0.0, 1.0)
                    ax_ts.set_xlabel("seconds")
                    ax_ts.set_ylabel("prior")
                    ax_ts.set_title("Prior changes over time")
                    ax_ts.axhline(0.5, color="gray", linestyle="--", linewidth=1)

                    for pattern in top_patterns:
                        ensure_line(pattern)

                    for pattern, prior in priors:
                        prior_f = float(prior)
                        prev = last_seen.get(pattern)
                        last_seen[pattern] = prior_f
                        if args.append_on_change and prev is not None and abs(prior_f - prev) < args.epsilon:
                            continue
                        series_x[pattern].append(float(t))
                        series_y[pattern].append(prior_f)

                    # Update plotted data.
                    for pattern in top_patterns:
                        xs = list(series_x[pattern])
                        ys = list(series_y[pattern])
                        if not xs:
                            continue
                        line = lines.get(pattern)
                        if line is not None:
                            line.set_data(xs, ys)

                    # Keep x-limits moving.
                    newest = t
                    oldest = max(0.0, newest - max(args.interval * args.window, 10.0))
                    ax_ts.set_xlim(oldest, newest + 0.25)

                fig.canvas.draw()
                if not headless:
                    fig.canvas.flush_events()
                    plt.pause(0.01)
                elif save_dir is not None:
                    # Continuously overwrite a single file so users can open/refresh it.
                    out_path = save_dir / "priors.png"
                    fig.savefig(out_path, dpi=150, bbox_inches="tight")

            time.sleep(max(0.0, args.interval))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

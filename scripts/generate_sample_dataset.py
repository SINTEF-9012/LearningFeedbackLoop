#!/usr/bin/env python3
"""Generate a small synthetic dataset in the layout ``DatasetLoader`` expects.

The repository ships no real machining data. This script writes a compact,
fully synthetic dataset so the streaming pipeline, the memory/feedback loop and
the UI can be exercised straight after cloning::

    python scripts/generate_sample_dataset.py

Output (default ``test_data/sample_dataset/``)::

    <root>/CASE_A - MACHINE_A1 - demo/OF00001/{axis_power,vibration,energy,machine_state}.csv

The signals are deliberately simple but not flat: a steady cutting regime with
an injected chatter episode partway through, so significance scoring and the
alerting path actually fire. Nothing here is derived from any real machine —
adjust the constants below freely.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "test_data" / "sample_dataset"

SAMPLE_HZ = 1.0
N_SAMPLES = 600
# Chatter episode occupies this slice of the run.
EVENT_START, EVENT_END = 380, 460


def _t(start: datetime, i: int) -> str:
    return (start + timedelta(seconds=i / SAMPLE_HZ)).isoformat().replace("+00:00", "Z")


def _in_event(i: int) -> bool:
    return EVENT_START <= i < EVENT_END


def write_dataset(root: Path, seed: int = 20260803) -> None:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    op_dir = root / "CASE_A - MACHINE_A1 - demo" / "OF00001"
    op_dir.mkdir(parents=True, exist_ok=True)

    def rows(header: list[str], make):
        out = [header]
        for i in range(N_SAMPLES):
            out.append(make(i, rng))
        return out

    def axis_power(i, r):
        base = 0.0 if i < 30 else 41.0
        spike = 9.0 if _in_event(i) else 0.0
        return [
            _t(start, i),
            1 if i >= 30 else 0,
            round(base + spike + r.gauss(0, 1.1), 3),
            round(base * 0.31 + r.gauss(0, 0.5), 3),
            round(base * 0.29 + r.gauss(0, 0.5), 3),
            round(base * 0.27 + r.gauss(0, 0.5), 3),
            round(base * 0.18 + r.gauss(0, 0.4), 3),
        ]

    def vibration(i, r):
        sev = 1.5 + 0.25 * math.sin(i / 40.0) + r.gauss(0, 0.12)
        chat = 0
        amp = freq = 0.0
        if _in_event(i):
            sev += 4.4
            chat = 1
            amp = round(3.0 + r.gauss(0, 0.35), 3)
            freq = round(880 + r.gauss(0, 22), 1)
        return [
            _t(start, i), chat, chat, amp, amp,
            freq, freq,
            round(max(sev, 0.0), 3),
            round(max(sev * 0.92, 0.0), 3),
        ]

    def energy(i, r):
        active = (0.4 if i < 30 else 12.6) + (2.4 if _in_event(i) else 0.0)
        active += r.gauss(0, 0.28)
        return [
            _t(start, i),
            round(active, 3),
            round(active * 1.14, 3),
            round(min(max(0.88 + r.gauss(0, 0.02), 0.0), 1.0), 4),
            round(active * 0.42, 3),
        ]

    def machine_state(i, r):
        cutting = i >= 30
        return [
            _t(start, i),
            round(820.0 if cutting else 0.0, 1),
            850.0 if cutting else 0.0,
            round(2400 + r.gauss(0, 12) if cutting else 0.0, 1),
            2400 if cutting else 0,
            round(38.5 + i * 0.004 + r.gauss(0, 0.15), 2),
            round(21.5 + r.gauss(0, 0.08), 2),
            7,
            "DEMO_PART_01",
            "AUTO",
        ]

    files = {
        "axis_power.csv": rows(
            ["timestamp", "Operation_Status", "Power_Spindle", "Power_X1",
             "Power_X2", "Power_Y", "Power_Z"], axis_power),
        "vibration.csv": rows(
            ["timestamp", "Chatter_Detection_OnOff_X", "Chatter_Detection_OnOff_Y",
             "Chatter_Detection_Amplitude_X", "Chatter_Detection_Amplitude_Y",
             "Chatter_Detection_Frequency_X", "Chatter_Detection_Frequency_Y",
             "Vibration_Severity_X", "Vibration_Severity_Y"], vibration),
        "energy.csv": rows(
            ["timestamp", "Power_Active", "Power_Apparent", "Power_Factor",
             "Power_Reactive"], energy),
        "machine_state.csv": rows(
            ["timestamp", "Feed_Rate_Actual", "Feed_Rate_Commanded",
             "Spindle_Speed_Actual", "Spindle_Speed_Commanded",
             "Temperature_Head", "Temperature_Room", "Tool_Number",
             "Program_Name", "Operation_Mode"], machine_state),
    }

    for name, data in files.items():
        with (op_dir / name).open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(data)
        print(f"wrote {op_dir / name} ({len(data) - 1} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    write_dataset(args.root, args.seed)
    print(f"\nSynthetic dataset ready at {args.root}")
    print("Point the loader at it:  DatasetLoader('%s')" % args.root)


if __name__ == "__main__":
    main()

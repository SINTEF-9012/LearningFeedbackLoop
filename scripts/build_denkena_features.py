#!/usr/bin/env python3
"""Build a channel-agnostic feature CSV from the Denkena milling-wear HDF5 set.

Phase 2b of docs/PUBLIC_DATASET_INTEGRATION_PLAN_2026-07-17.md. Each ``.h5`` is one
milling run with a measured flank-wear label VB (µm); this emits **one feature row per
run** into ``data/breakage_patterns/denkena_features.csv``.

Design (per the 2026-07-24 adaptability steer): the extractor is **channel-agnostic**.
It computes a standard time-domain statistics block for *every* signal channel it finds,
and an extra spectral block for high-rate channels (the 25 kHz dynamometer force). New
machines with different or extra channels drop in without code changes — a channel simply
becomes NaN on runs/machines that lack it, and the downstream transfer step keeps only the
features that are informative across every machine (`informative_features`, ISS-62).

Cross-machine note (verified 2026-07-24): the three machines are NOT channel-identical —
M1 reports its axis drives as ``torque_axis_*`` while M2/M3 report ``force_axis_*``. Both
are captured here; the leave-one-machine-out transfer excludes them automatically because
they are absent (NaN) on the other machines. The honest shared basis is the 3-axis 25 kHz
``force_sensor_*``, ``torque_spindle``, and ``position_control_deviation_axis_*``.

Labels: the continuous ``wear`` (VB, µm) is written as-is; the worn/normal threshold is
**swept downstream** (run_denkena_transfer.py), so no binary label is baked in here.

Usage
-----
    python scripts/build_denkena_features.py                 # all 6,418 runs
    python scripts/build_denkena_features.py --limit 200     # quick smoke
    python scripts/build_denkena_features.py --jobs 8        # parallelise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("h5py is required: ./.venv/bin/pip install h5py")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "public" / "denkena_wear"
OUT_CSV = ROOT / "data" / "breakage_patterns" / "denkena_features.csv"

LABEL_KEYS = ("wear", "machine", "tool", "run", "cumulated_tool_contact_time")
# Channels at or above this length get an FFT spectral block (the 25 kHz force
# channels; the 500 Hz machine channels stay time-domain only). Length-based so a
# new high-rate sensor on a future machine automatically qualifies.
HIGH_RATE_MIN_SAMPLES = 20_000
HF_CUTOFF_HZ = 1_000.0  # "high frequency" split for the spectral energy ratio


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def time_stats(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Standard time-domain statistics for one channel. Channel-agnostic."""
    s = np.asarray(sig, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {}
    rms = float(np.sqrt(np.mean(s ** 2)))
    peak = float(np.max(np.abs(s)))
    std = float(np.std(s))
    p25, p50, p75 = (float(v) for v in np.percentile(s, [25, 50, 75]))
    # linear slope over the run (trend), normalised by sample index
    idx = np.arange(s.size, dtype=float)
    slope = float(np.polyfit(idx, s, 1)[0]) if s.size > 2 else 0.0
    out = {
        f"{prefix}_mean": float(np.mean(s)),
        f"{prefix}_std": std,
        f"{prefix}_min": float(np.min(s)),
        f"{prefix}_max": float(np.max(s)),
        f"{prefix}_range": float(np.max(s) - np.min(s)),
        f"{prefix}_rms": rms,
        f"{prefix}_abs_mean": float(np.mean(np.abs(s))),
        f"{prefix}_median": p50,
        f"{prefix}_iqr": p75 - p25,
        f"{prefix}_skew": _safe(_skew(s, std)),
        f"{prefix}_kurtosis": _safe(_kurtosis(s, std)),
        f"{prefix}_crest": _safe(peak / rms) if rms > 1e-12 else 0.0,
        f"{prefix}_slope": slope,
    }
    return out


def _skew(s: np.ndarray, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return float(np.mean(((s - s.mean()) / std) ** 3))


def _kurtosis(s: np.ndarray, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return float(np.mean(((s - s.mean()) / std) ** 4) - 3.0)


def spectral_stats(sig: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    """FFT-based features for a high-rate channel (wear-sensitive spectral content)."""
    s = np.asarray(sig, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 16 or fs <= 0:
        return {}
    s = s - s.mean()
    freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(s))
    power = mag ** 2
    total = float(power.sum()) + 1e-12
    hf = float(power[freqs > HF_CUTOFF_HZ].sum())
    centroid = float((freqs * power).sum() / total)
    # spread around the centroid
    spread = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
    dom = int(np.argmax(power[1:]) + 1)  # skip DC
    # spectral flatness (geometric/arithmetic mean of power) — tonal vs broadband
    pos = power[power > 0]
    flatness = float(np.exp(np.mean(np.log(pos))) / (pos.mean() + 1e-12)) if pos.size else 0.0
    return {
        f"{prefix}_hf_energy_ratio": hf / total,
        f"{prefix}_spec_centroid": centroid,
        f"{prefix}_spec_spread": spread,
        f"{prefix}_dom_freq": float(freqs[dom]),
        f"{prefix}_dom_amp": float(mag[dom]),
        f"{prefix}_spec_flatness": flatness,
    }


def _channel_prefix(h5_path: str) -> str:
    """`signals_sensor/force_sensor_x` -> `force_sensor_x`."""
    return h5_path.split("/")[-1]


def _sample_rate(times: np.ndarray) -> float:
    t = np.asarray(times, dtype=float).ravel()
    if t.size < 2:
        return 0.0
    span = float(t[-1] - t[0])
    return (t.size - 1) / span if span > 0 else 0.0


def extract_file(path: Path) -> Optional[Dict[str, Any]]:
    """Return one feature row (dict) for a single run, or None on failure."""
    try:
        with h5py.File(path, "r") as h:
            row: Dict[str, Any] = {"filename": path.name}
            # labels
            for k in LABEL_KEYS:
                key = f"labels/{k}"
                if key in h:
                    row[k] = float(np.asarray(h[key]).ravel()[0])
            # sample rates from the two time channels
            fs_by_group: Dict[str, float] = {}
            for grp, tkey in (("signals_machine", "signals_machine/time_machine"),
                              ("signals_sensor", "signals_sensor/time_sensor")):
                if tkey in h:
                    fs_by_group[grp] = _sample_rate(np.asarray(h[tkey]))

            def visit(name: str, obj: Any) -> None:
                if not hasattr(obj, "shape"):
                    return
                if name.startswith("labels/") or name.endswith(("time_machine", "time_sensor")):
                    return
                prefix = _channel_prefix(name)
                data = np.asarray(obj).ravel()
                row.update(time_stats(data, prefix))
                grp = name.split("/")[0]
                fs = fs_by_group.get(grp, 0.0)
                if data.size >= HIGH_RATE_MIN_SAMPLES and fs > 0:
                    row.update(spectral_stats(data, fs, prefix))

            h.visititems(visit)
            return row
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.h5"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no .h5 files in {data_dir} — download per the integration plan")
    print(f"extracting features from {len(files)} runs in {data_dir.relative_to(ROOT)} …")

    rows: List[Dict[str, Any]] = []
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(extract_file, files, chunksize=16), 1):
                if r:
                    rows.append(r)
                if i % 500 == 0:
                    print(f"  … {i}/{len(files)}", flush=True)
    else:
        for i, path in enumerate(files, 1):
            r = extract_file(path)
            if r:
                rows.append(r)
            if i % 500 == 0:
                print(f"  … {i}/{len(files)}", flush=True)

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    # cast the integer-ish label columns
    for k in ("machine", "tool", "run"):
        if k in df.columns:
            df[k] = df[k].astype("Int64")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    feat_cols = [c for c in df.columns
                 if c not in (*LABEL_KEYS, "filename") and df[c].dtype != "Int64"]
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}: {len(df)} rows x {len(df.columns)} cols "
          f"({len(feat_cols)} feature cols)")
    if "machine" in df.columns and "wear" in df.columns:
        print("  per machine:")
        for m, g in df.groupby("machine"):
            nvb = g["wear"]
            print(f"    M{m}: {len(g)} runs, VB {nvb.min():.0f}–{nvb.max():.0f} µm, "
                  f"tools {sorted(g['tool'].dropna().unique().tolist())}")
    # report cross-machine feature availability (the transfer basis)
    if "machine" in df.columns:
        machines = df["machine"].dropna().unique()
        common = [c for c in feat_cols
                  if all(df.loc[df["machine"] == m, c].notna().any() and
                         df.loc[df["machine"] == m, c].nunique(dropna=True) > 1
                         for m in machines)]
        print(f"  features informative on ALL {len(machines)} machines "
              f"(the honest transfer basis): {len(common)} of {len(feat_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

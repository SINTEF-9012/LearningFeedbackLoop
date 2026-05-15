"""On-machine OF (Fabrication Order) replay pipeline.

For each Komatsu/Goimek/WWR OF folder we have a handful of CSVs sampled at
~1 Hz, identified by a 6-character suffix:

    Komatsu  *_QRJWHE.csv or *_6XLMCH.csv  -> vibration peaks (model input)
             *_TYZBPS.csv                  -> Tool_Number, Spindle/Feed cmd
             *_BXCZ3M.csv                  -> Operation_Mode (cutting mask)
    Goimek   *_7N4ZJ8.csv                  -> vibration peaks (TBD)
    WWR      *_7DTZHE.csv                  -> vibration peaks (TBD)

This module focuses on Komatsu: find the active-cutting windows in BXCZ3M,
merge nearby ones, then yield aligned model inputs by stepping through the
peaks file and matching the closest TYZBPS row for each timestamp.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .tool_list import find_komatsu_tool_list, load_komatsu_tool_list


# ---------------------------------------------------------------------------
# Machine configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MachineSpec:
    name: str                       # display name
    folder_prefix: str              # e.g. "Komatsu - FLP-7152"
    vibration_suffixes: tuple[str, ...]   # CSV suffixes that hold the peaks
    tyzbps_suffix: str | None = "TYZBPS"  # tool/spindle/feed file
    bxcz3m_suffix: str | None = "BXCZ3M"  # operation-mode file
    tool_list_loader: str = "komatsu"     # which parser to use


MACHINES: dict[str, MachineSpec] = {
    "komatsu-7152": MachineSpec(
        name="Komatsu FLP-7152 (TXG_REGRHM)",
        folder_prefix="Komatsu - FLP-7152 - TXG_REGRHM",
        vibration_suffixes=("QRJWHE", "6XLMCH"),
    ),
    "komatsu-7153": MachineSpec(
        name="Komatsu FLP-7153 (SLG_E8AJ1C)",
        folder_prefix="Komatsu - FLP-7153 - SLG_E8AJ1C",
        vibration_suffixes=("6XLMCH", "QRJWHE"),
    ),
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def workspace_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def machine_folder(spec: MachineSpec) -> str:
    return os.path.join(workspace_root(), spec.folder_prefix)


def list_machines() -> list[dict]:
    out = []
    for mid, spec in MACHINES.items():
        ofs = list_ofs(mid)
        out.append({"id": mid, "name": spec.name, "n_ofs": len(ofs), "available": bool(ofs)})
    return out


def list_ofs(machine_id: str) -> list[str]:
    """List OFs that have usable TYZBPS data.

    An OF is excluded when its TYZBPS file is missing or its
    ``Spindle_Speed_Commanded`` / ``Feed_Rate_Commanded`` columns are
    entirely empty — those values are required to compute the model's
    ``[d, z, n, f, vf]`` inputs and to convert FFT peak frequencies to
    spindle-relative units. We accept the small upfront cost of reading
    those two columns from each TYZBPS so the UI never offers an OF that
    can't actually be replayed.
    """
    spec = MACHINES[machine_id]
    folder = machine_folder(spec)
    if not os.path.isdir(folder):
        return []
    candidates = sorted(
        d for d in os.listdir(folder)
        if d.startswith("OF") and os.path.isdir(os.path.join(folder, d))
    )
    usable: list[str] = []
    for of in candidates:
        if _of_has_commanded_columns(machine_id, of):
            usable.append(of)
    return usable


@lru_cache(maxsize=128)
def _of_has_commanded_columns(machine_id: str, of: str) -> bool:
    paths = find_of_files(machine_id, of)
    p = paths.get("tyzbps")
    if not p:
        return False
    try:
        df = pd.read_csv(p, usecols=lambda c: c in ("Spindle_Speed_Commanded", "Feed_Rate_Commanded"))
    except (ValueError, OSError):
        return False
    if "Spindle_Speed_Commanded" not in df.columns or df["Spindle_Speed_Commanded"].notna().sum() == 0:
        return False
    if "Feed_Rate_Commanded" not in df.columns or df["Feed_Rate_Commanded"].notna().sum() == 0:
        return False
    return True


def _find_of_file(of_dir: str, suffix: str) -> str | None:
    matches = glob.glob(os.path.join(of_dir, f"*_{suffix}.csv"))
    return matches[0] if matches else None


def find_of_files(machine_id: str, of: str) -> dict[str, str | None]:
    spec = MACHINES[machine_id]
    of_dir = os.path.join(machine_folder(spec), of)
    vib = None
    for suf in spec.vibration_suffixes:
        vib = _find_of_file(of_dir, suf)
        if vib:
            break
    return {
        "vibration": vib,
        "tyzbps": _find_of_file(of_dir, spec.tyzbps_suffix) if spec.tyzbps_suffix else None,
        "bxcz3m": _find_of_file(of_dir, spec.bxcz3m_suffix) if spec.bxcz3m_suffix else None,
    }


# ---------------------------------------------------------------------------
# Window detection
# ---------------------------------------------------------------------------

# Minimum number of *consecutive* TYZBPS rows whose spindle speed must exceed
# ``MIN_SPINDLE_MULTIPLIER`` × file-wide non-NaN mean for a same-tool run to
# qualify as a cutting window. Below this we assume the spindle is only
# briefly spinning up / coasting and not actually cutting.
MIN_HIGH_SPINDLE_CONSEC = 10
MIN_SPINDLE_MULTIPLIER = 2.0


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the length of the longest contiguous True run in ``mask``."""
    if mask.size == 0:
        return 0
    # Find run lengths via diff on int view.
    best = run = 0
    for v in mask:
        if v:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def detect_cutting_windows(machine_id: str, of: str) -> list[dict]:
    """Return list of valid cutting windows for a Komatsu OF.

    A window is the **entire contiguous run** of TYZBPS timestamps on which:

      1. ``Tool_Number`` is unchanged across the run,
      2. that tool number resolves in the Komatsu tool list to a known
         diameter (mm) **and** number of inserts (teeth),
      3. ``Spindle_Speed_Commanded`` and ``Feed_Rate_Commanded`` are both
         non-NaN (after a forward-fill that only fills *after* the first real
         observation — leading rows before the first measurement stay NaN
         and are excluded).

    A qualifying run must additionally contain **at least
    ``MIN_HIGH_SPINDLE_CONSEC`` consecutive timepoints** with spindle speed
    above ``MIN_SPINDLE_MULTIPLIER`` × the file-wide non-NaN mean spindle
    speed. The full same-tool run is emitted as the window (not just the
    high-spindle subsegment), so the model sees the spin-up and slow-down
    parts as well.

    Each window dict carries ``start``, ``end``, ``duration_sec``,
    ``tool_number``, ``diameter_mm``, ``n_inserts`` and ``n_rows`` for UI
    display.
    """
    spec = MACHINES[machine_id]
    paths = find_of_files(machine_id, of)
    if not paths["tyzbps"]:
        return []

    df = pd.read_csv(
        paths["tyzbps"],
        usecols=lambda c: c in (
            "Tool_Number",
            "Spindle_Speed_Commanded",
            "Feed_Rate_Commanded",
            "timestamp",
        ),
    )
    if not {"Tool_Number", "Spindle_Speed_Commanded", "Feed_Rate_Commanded"}.issubset(df.columns):
        return []

    df["ts"] = _parse_ts(df["timestamp"])
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # File-wide mean spindle speed over non-NaN raw rows.
    raw_spindle = df["Spindle_Speed_Commanded"].astype(float)
    mean_spindle = float(raw_spindle.dropna().mean()) if raw_spindle.notna().any() else float("nan")
    if not np.isfinite(mean_spindle) or mean_spindle <= 0:
        return []
    threshold = MIN_SPINDLE_MULTIPLIER * mean_spindle

    # Forward-fill only: leading rows with no observation yet remain NaN and
    # are filtered out by the validity mask below.
    tool = df["Tool_Number"].ffill()
    spindle = df["Spindle_Speed_Commanded"].astype(float).ffill()
    feed = df["Feed_Rate_Commanded"].astype(float).ffill()

    # Tool-list lookup for the diameter/teeth check.
    if spec.tool_list_loader == "komatsu":
        tl_path = find_komatsu_tool_list(workspace_root())
        tool_list = load_komatsu_tool_list(tl_path) if tl_path else {}
    else:
        tool_list = {}

    # Per-row validity: tool resolvable, diameter+teeth known, spindle+feed
    # non-NaN.
    valid = np.zeros(len(df), dtype=bool)
    tool_int_arr = np.full(len(df), -1, dtype=np.int64)
    for i, (t, sp, fd) in enumerate(zip(tool, spindle, feed)):
        if pd.isna(t) or pd.isna(sp) or pd.isna(fd):
            continue
        try:
            ti = int(t)
        except (TypeError, ValueError):
            continue
        if ti == 0:
            continue
        info = tool_list.get(ti)
        if not info:
            continue
        if not info.get("diameter_mm") or not info.get("n_inserts"):
            continue
        valid[i] = True
        tool_int_arr[i] = ti

    if not valid.any():
        return []

    # Group contiguous runs where validity is True AND Tool_Number is
    # unchanged. A change in tool number ends the current run.
    high = (spindle.to_numpy() > threshold) & valid
    windows: list[dict] = []
    n = len(df)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        tnum = tool_int_arr[i]
        while j + 1 < n and valid[j + 1] and tool_int_arr[j + 1] == tnum:
            j += 1
        # Run [i, j]. Qualify it on the high-spindle consecutive criterion.
        if _longest_true_run(high[i : j + 1]) >= MIN_HIGH_SPINDLE_CONSEC:
            info = tool_list.get(int(tnum), {})
            start_ts = df["ts"].iloc[i]
            end_ts = df["ts"].iloc[j]
            windows.append({
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "duration_sec": (end_ts - start_ts).total_seconds(),
                "tool_number": int(tnum),
                "diameter_mm": info.get("diameter_mm"),
                "n_inserts": info.get("n_inserts"),
                "n_rows": int(j - i + 1),
            })
        i = j + 1

    return windows


# ---------------------------------------------------------------------------
# OF data loading for streaming
# ---------------------------------------------------------------------------

@dataclass
class OFStream:
    """All the per-OF data needed for a streamed inference pass.

    The vibration peaks file (QRJWHE/6XLMCH) drives the time axis. For every
    one of its timestamps we lazily look up the closest row in TYZBPS for the
    tool number and the spindle/feed commands, and in BXCZ3M for the current
    operation mode (only used for UI display).
    """
    vib: pd.DataFrame          # QRJWHE/6XLMCH with parsed ts
    tyzbps: pd.DataFrame       # TYZBPS with parsed ts
    bxcz3m: pd.DataFrame       # BXCZ3M with parsed ts
    tool_list: dict[int, dict]


def load_of_stream(machine_id: str, of: str) -> OFStream:
    spec = MACHINES[machine_id]
    paths = find_of_files(machine_id, of)
    if not paths["vibration"]:
        raise FileNotFoundError(f"No vibration CSV in {machine_id}/{of}")
    if not paths["tyzbps"]:
        raise FileNotFoundError(f"No TYZBPS CSV in {machine_id}/{of}")
    if not paths["bxcz3m"]:
        raise FileNotFoundError(f"No BXCZ3M CSV in {machine_id}/{of}")

    vib = pd.read_csv(paths["vibration"])
    vib["ts"] = _parse_ts(vib["timestamp"])
    vib = vib.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # TYZBPS rows are extremely sparse — each row populates only the field
    # that changed at that timestamp. We forward+back-fill so every row
    # carries the last known value of each slow-changing column. We require
    # the controller's commanded spindle speed and feed rate to be present
    # somewhere in the file; OFs missing either column are unusable because
    # the model needs the program-intent (commanded) values, not the noisy
    # actual readings, to match training.
    tyzbps_wanted = [
        "Tool_Number",
        "Spindle_Speed_Commanded",
        "Feed_Rate_Commanded",
        "timestamp",
    ]
    tyzbps = pd.read_csv(
        paths["tyzbps"],
        usecols=lambda c: c in tyzbps_wanted,
    )
    if "Spindle_Speed_Commanded" not in tyzbps.columns or tyzbps["Spindle_Speed_Commanded"].notna().sum() == 0:
        raise ValueError(
            f"{machine_id}/{of}: Spindle_Speed_Commanded column missing or entirely empty in TYZBPS"
        )
    if "Feed_Rate_Commanded" not in tyzbps.columns or tyzbps["Feed_Rate_Commanded"].notna().sum() == 0:
        raise ValueError(
            f"{machine_id}/{of}: Feed_Rate_Commanded column missing or entirely empty in TYZBPS"
        )
    tyzbps["ts"] = _parse_ts(tyzbps["timestamp"])
    tyzbps = tyzbps.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    for c in ("Tool_Number", "Spindle_Speed_Commanded", "Feed_Rate_Commanded"):
        tyzbps[c] = tyzbps[c].ffill().bfill()
    # Rename to the simple names the rest of the pipeline expects.
    tyzbps = tyzbps.rename(columns={
        "Spindle_Speed_Commanded": "Spindle_Speed",
        "Feed_Rate_Commanded": "Feed_Rate",
    })

    bxcz3m = pd.read_csv(paths["bxcz3m"], usecols=["Operation_Mode", "timestamp"])
    bxcz3m["ts"] = _parse_ts(bxcz3m["timestamp"])
    bxcz3m = bxcz3m.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    bxcz3m["Operation_Mode"] = bxcz3m["Operation_Mode"].ffill().bfill()

    # Eagerly merge tool / spindle / feed / op-mode onto the vibration
    # timeline once, so per-step extraction is just a row lookup. Using
    # ``direction="backward"`` matches "the last known value at or before the
    # vibration timestamp", which is the right semantics for forward-filled
    # CNC state.
    vib = pd.merge_asof(
        vib, tyzbps[["ts", "Tool_Number", "Spindle_Speed", "Feed_Rate"]],
        on="ts", direction="backward",
    )
    vib = pd.merge_asof(
        vib, bxcz3m[["ts", "Operation_Mode"]],
        on="ts", direction="backward",
    )

    if spec.tool_list_loader == "komatsu":
        path = find_komatsu_tool_list(workspace_root())
        if not path:
            raise FileNotFoundError("Komatsu tool list xlsx not found")
        tool_list = load_komatsu_tool_list(path)
    else:
        tool_list = {}

    return OFStream(vib=vib, tyzbps=tyzbps, bxcz3m=bxcz3m, tool_list=tool_list)


# ---------------------------------------------------------------------------
# Step-wise extraction
# ---------------------------------------------------------------------------

PEAK_COLS_X_AMP = [f"Vibration_Peak_{k}_X_Amplitude" for k in range(1, 6)]
PEAK_COLS_X_FRQ = [f"Vibration_Peak_{k}_X_Frequency" for k in range(1, 6)]
PEAK_COLS_Y_AMP = [f"Vibration_Peak_{k}_Y_Amplitude" for k in range(1, 6)]
PEAK_COLS_Y_FRQ = [f"Vibration_Peak_{k}_Y_Frequency" for k in range(1, 6)]


def slice_by_window(stream: OFStream, start_iso: str, end_iso: str) -> np.ndarray:
    """Return the indices into ``stream.vib`` whose timestamps fall in the window."""
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    ts = stream.vib["ts"]
    return np.where((ts >= start) & (ts <= end))[0]


def extract_step(stream: OFStream, vib_row: int, k_peaks: int = 5,
                 f_max_rel: float | None = 12.0) -> dict:
    """Pull everything needed for one model timestep at ``stream.vib[vib_row]``.

    Returns a dict with ``pairs`` (C, K, 2) -- (f_rel, amp) -- and ``params``
    ``[d, z, n, f, vf]`` matching the training JSONs (``f`` is feed per tooth,
    computed as ``vf / (z * n)``). ``valid`` is False when the tool / spindle
    / feed cannot be resolved (in which case the caller should skip the step
    or pass an all-zero pair tensor through the model).

    The vibration dataframe has already been enriched in ``load_of_stream``
    with ``Tool_Number``, ``Spindle_Speed``, ``Feed_Rate`` and
    ``Operation_Mode`` columns via asof-merges, so this function is just a
    row lookup.
    """
    vib = stream.vib.iloc[vib_row]
    ts = vib["ts"]

    tool_num_raw = vib.get("Tool_Number")
    try:
        tool_num = int(tool_num_raw) if pd.notna(tool_num_raw) else None
    except (TypeError, ValueError):
        tool_num = None

    tool_info = stream.tool_list.get(tool_num) if tool_num is not None else None
    diameter = (tool_info or {}).get("diameter_mm")
    teeth = (tool_info or {}).get("n_inserts")

    spindle = vib.get("Spindle_Speed")
    feed = vib.get("Feed_Rate")
    spindle = float(spindle) if pd.notna(spindle) else None
    feed = float(feed) if pd.notna(feed) else None

    op_mode = vib.get("Operation_Mode")
    op_mode = float(op_mode) if pd.notna(op_mode) else None

    # Peaks -> (C=2, K, 2) with (f_rel, amp); zeros for missing
    pairs = np.zeros((2, k_peaks, 2), dtype=np.float32)
    valid = bool(spindle and spindle > 0 and diameter and teeth)
    if valid:
        fg = spindle / 60.0  # Hz
        for ci, (amp_cols, frq_cols) in enumerate(
            [(PEAK_COLS_X_AMP, PEAK_COLS_X_FRQ),
             (PEAK_COLS_Y_AMP, PEAK_COLS_Y_FRQ)]
        ):
            entries = []
            for ac, fc in zip(amp_cols[:k_peaks], frq_cols[:k_peaks]):
                a = vib.get(ac)
                f = vib.get(fc)
                if pd.isna(a) or pd.isna(f) or f <= 0:
                    continue
                f_rel = float(f) / fg
                if f_max_rel is not None and f_rel > f_max_rel:
                    continue
                entries.append((f_rel, float(a)))
            entries.sort(key=lambda x: x[0])
            for j, (fr, am) in enumerate(entries[:k_peaks]):
                pairs[ci, j, 0] = fr
                pairs[ci, j, 1] = am

    # Params [d, z, n, f, vf]; f = vf / (z * n)  (feed per tooth, mm/tooth)
    if valid:
        d_val = float(diameter)
        z_val = float(teeth)
        n_val = float(spindle)
        vf_val = float(feed) if feed is not None else 0.0
        if z_val > 0 and n_val > 0 and vf_val > 0:
            f_val = vf_val / (z_val * n_val)
        else:
            f_val = 0.0
        params = np.array([d_val, z_val, n_val, f_val, vf_val], dtype=np.float32)
    else:
        params = np.zeros(5, dtype=np.float32)

    return {
        "ts": ts.isoformat(),
        "pairs": pairs,
        "params": params,
        "tool_number": tool_num,
        "tool_description": (tool_info or {}).get("description"),
        "diameter_mm": diameter,
        "n_inserts": teeth,
        "spindle_rpm": spindle,
        "feed_rate": feed,
        "operation_mode": op_mode,
        "valid": valid,
    }

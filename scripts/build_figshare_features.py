#!/usr/bin/env python3
"""Build a tool-life feature CSV from the figshare 14-tool run-to-failure milling set.

Phase 2a of docs/PUBLIC_DATASET_INTEGRATION_PLAN_2026-07-17.md — the breakage-claim
rescue. The paper's breakage transfer is at chance because Site_a_line2 has only two
sessions; this dataset has **14 tools run to failure** (one machine, 968 cycles), giving
14 leave-one-tool-out folds for a properly-powered pre-failure transfer test.

The article ships a cycle-level aggregate table (`FeatureAndMetadata_Milling.csv`:
20 channels × 6 stats per cycle + tool-life labels), so no raw-signal processing is
needed for a first result. This script cleans it into
`data/breakage_patterns/figshare14_features.csv` with:
  - metadata: tool (TollIndex), cycle, cycle_to_failure, cycle_to_failure_norm, tool_type
  - features: the 120 sensor aggregates ONLY.

Deliberately EXCLUDED from features: the process/setup parameters (ADOC, RDOC,
HardnessMean, ToolHolderLength). They are fixed per tool, so as features they would leak
tool identity into a leave-one-tool-out split — a silent shortcut. They are dropped, not
kept as metadata-features.

Labels stay continuous (`cycle_to_failure_norm`, 1 at fresh → 0 at failure); the
pre-failure threshold is swept downstream (run_figshare_transfer.py), mirroring the
Denkena VB-threshold sweep.

Quirk handled: the source CSV has a junk first row, ';' separators, and **mixed decimal
formats** — metadata columns use comma-decimals ("0,53"), sensor columns use dot-decimals.

Usage
-----
    python scripts/build_figshare_features.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "public" / "figshare14" / "FeatureAndMetadata_Milling.csv"
OUT = ROOT / "data" / "breakage_patterns" / "figshare14_features.csv"

# process/setup params — carried as metadata, NEVER as features (would leak tool id)
PROCESS_META = ["MillingToolType", "ADOC", "RDOC", "HardnessMean", "ToolHolderLength"]
ID_META = ["FileName", "NumberOfCycle", "SampleIndex", "TollIndex",
           "CycleToFailure", "CycleToFailureNormalized"]


def _to_float(series: pd.Series) -> pd.Series:
    """Parse a possibly comma-decimal string column to float."""
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _slug(col: str) -> str:
    """`Accelerometer - Spindle +Y - min` -> `accelerometer_spindle_py_min`.

    Splits on the ' - ' stat separator; the last part is the statistic, the rest is the
    channel. Preserves axis sign (+ -> p, - -> m) so +Y and -Y stay distinct.
    """
    parts = col.split(" - ")
    stat = parts[-1]
    chan = "_".join(parts[:-1])
    s = chan.replace("+", "p").replace("-", "m")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return f"{s}_{stat.strip().lower()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        sys.exit(f"missing {src} — run scripts/fetch_public_datasets.py --dataset figshare14")

    raw = pd.read_csv(src, sep=";", skiprows=1, dtype=str)
    feat_cols = [c for c in raw.columns if " - " in c]

    # build every column first, then concat once (avoids DataFrame fragmentation)
    cols = {
        "tool": pd.to_numeric(raw["TollIndex"], errors="coerce").astype("Int64"),
        "cycle": pd.to_numeric(raw["NumberOfCycle"], errors="coerce").astype("Int64"),
        "cycle_to_failure": pd.to_numeric(raw["CycleToFailure"], errors="coerce"),
        "cycle_to_failure_norm": _to_float(raw["CycleToFailureNormalized"]),
        "tool_type": pd.to_numeric(raw["MillingToolType"], errors="coerce").astype("Int64"),
    }
    # features: the 120 sensor aggregates ONLY
    for c in feat_cols:
        cols[_slug(c)] = _to_float(raw[c])
    out = pd.DataFrame(cols)
    out = out.dropna(subset=["tool", "cycle_to_failure_norm"]).reset_index(drop=True)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    n_feat = len([c for c in out.columns if c not in
                  ("tool", "cycle", "cycle_to_failure", "cycle_to_failure_norm", "tool_type")])
    try:
        shown = dst.relative_to(ROOT)
    except ValueError:
        shown = dst
    print(f"wrote {shown}: {len(out)} cycles x {len(out.columns)} cols ({n_feat} sensor features)")
    print(f"  tools: {sorted(out['tool'].dropna().astype(int).unique().tolist())} "
          f"(process params ADOC/RDOC/hardness/holder EXCLUDED from features — leak tool id)")
    per = out.groupby("tool").size()
    print(f"  cycles/tool: min={per.min()} max={per.max()} median={int(per.median())}")
    # pre-failure class balance preview at a few thresholds
    print("  pre_break balance (cycle_to_failure_norm <= τ):")
    for tau in (0.1, 0.2, 0.3):
        pb = (out["cycle_to_failure_norm"] <= tau)
        usable = sum(1 for _, g in out.groupby("tool")
                     if (g["cycle_to_failure_norm"] <= tau).nunique() > 1)
        print(f"    τ={tau}: {int(pb.sum())}/{len(out)} pre_break, usable tools (both classes): {usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

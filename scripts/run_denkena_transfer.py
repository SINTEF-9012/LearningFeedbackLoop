#!/usr/bin/env python3
"""Denkena leave-one-machine-out wear-transfer, swept over the VB worn/normal threshold.

Phase 2b of docs/PUBLIC_DATASET_INTEGRATION_PLAN_2026-07-17.md — the controlled RQ4
experiment. Three 5-axis milling centres with **identical setup, parameters and
material**, so the machine is the only variable: train on two machines, freeze, score
the held-out machine. Because flank wear VB is *measured* (µm), the worn/normal label is
a threshold we **sweep** (D1) rather than a weak heuristic — the sweep is the measured-
label counterpart to the weak-label sensitivity analysis in the pilot experiments.

The transfer machinery is reused verbatim from run_transfer_analysis.py (RandomForest /
Logistic / IsolationForest, train-calibrated operating points, block-bootstrap CI). Only
the feature basis differs: Denkena supplies its own channel-agnostic features, and we keep
those informative across **all three** machines — which automatically drops the axis-drive
channel that is torque on M1 but force on M2/M3 (see build_denkena_features.py).

Usage
-----
    python scripts/run_denkena_transfer.py
    python scripts/run_denkena_transfer.py --thresholds 50 75 100 125 150
    python scripts/run_denkena_transfer.py --out data/denkena_transfer_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_transfer_analysis import (  # reuse the exact transfer harness
    informative_features,
    transfer_eval,
    _present,
)

CSV = ROOT / "data" / "breakage_patterns" / "denkena_features.csv"
META_COLS = {"filename", "wear", "machine", "tool", "run", "cumulated_tool_contact_time"}
DEFAULT_THRESHOLDS = [50, 75, 100, 125, 150]  # µm; VB≈150 is the dataset's end-of-life
POS_LABEL = "worn"

# Physically-motivated condition-monitoring channels: the dynamometer force, the
# spindle torque, and the axis position-control deviations. Excludes tool_position_*
# (toolpath geometry — recorded, but not a wear signal). The --basis physical ablation
# restricts to these to show the transfer result is force-dominated, not leaking from
# toolpath geometry.
PHYSICAL_PREFIXES = ("force_sensor_", "torque_spindle", "position_control_deviation_")


def transfer_basis(df: pd.DataFrame, kind: str = "all") -> List[str]:
    """Feature columns informative on EVERY machine — the honest shared basis.

    A feature present only on some machines (e.g. the torque/force axis-drive split)
    is non-constant on those but all-NaN on the others, so it is excluded here and can
    never leak a machine-identifying signal into the transfer. ``kind='physical'``
    further restricts to the condition-monitoring channels (`PHYSICAL_PREFIXES`).
    """
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if kind == "physical":
        feat_cols = [c for c in feat_cols if c.startswith(PHYSICAL_PREFIXES)]
    machines = sorted(df["machine"].dropna().unique())
    basis = []
    for c in feat_cols:
        ok = True
        for m in machines:
            col = df.loc[df["machine"] == m, c]
            if not (col.notna().any() and col.nunique(dropna=True) > 1):
                ok = False
                break
        if ok:
            basis.append(c)
    return basis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                    help="VB (µm) worn/normal thresholds to sweep")
    ap.add_argument("--basis", choices=["all", "physical"], default="all",
                    help="'all' = every shared feature; 'physical' = condition-monitoring "
                         "channels only (force/torque/position-deviation, excludes toolpath geometry)")
    ap.add_argument("--out", default="data/denkena_transfer_results.json")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"missing {csv} — run scripts/build_denkena_features.py first")
    df = pd.read_csv(csv)
    df = df[df["machine"].notna() & df["wear"].notna()].copy()

    basis = transfer_basis(df, kind=args.basis)
    machines = sorted(df["machine"].dropna().astype(int).unique())
    print(f"Denkena wear transfer — {len(df)} runs, machines {machines}, "
          f"basis='{args.basis}': {len(basis)} features informative on all machines "
          f"(of {len([c for c in df.columns if c not in META_COLS])} extracted).")
    print(f"Leave-one-machine-out, frozen, no eval-time feedback. VB sweep: {args.thresholds} µm\n")

    store = {"n_runs": int(len(df)), "machines": machines, "basis": args.basis,
             "n_features": len(basis), "features": basis, "sweep": {}}

    for tau in args.thresholds:
        d = df.copy()
        d["label"] = (d["wear"] >= tau).map({True: POS_LABEL, False: "normal"})
        pos = int((d["label"] == POS_LABEL).sum())
        # need both classes present on every machine, else that fold is dropped for AUC
        per_m = d.groupby("machine")["label"].apply(lambda s: s.nunique())
        usable = [int(m) for m, k in per_m.items() if k > 1]
        res = transfer_eval(d, "machine", POS_LABEL, basis)
        _present(f"VB≥{tau:.0f}µm = worn  ({pos}/{len(d)} worn, "
                 f"usable machines: {usable})", res, ops=False)
        store["sweep"][f"vb{tau:.0f}"] = {
            "threshold_um": tau,
            "n_worn": pos,
            "n_total": int(len(d)),
            "usable_machines": usable,
            "results": res,
        }

    out_arg = args.out
    # keep the physical-basis run from clobbering the default 'all' results
    if args.basis != "all" and out_arg == "data/denkena_transfer_results.json":
        out_arg = f"data/denkena_transfer_{args.basis}.json"
    out = (ROOT / out_arg) if not Path(out_arg).is_absolute() else Path(out_arg)
    out.write_text(json.dumps(store, indent=2, default=str))
    print(f"\nStored: {out}")

    # compact sweep summary: RandomForest pooled AUC vs threshold
    print("\nSUMMARY — RandomForest pooled transfer AUC vs worn threshold:")
    for tau in args.thresholds:
        cell = store["sweep"][f"vb{tau:.0f}"]["results"]
        rf = next((v for k, v in cell.items() if "RandomForest" in k), {})
        auc = rf.get("pooled_auc")
        ci = rf.get("auc_ci95")
        print(f"  VB≥{tau:>3.0f}µm  worn={store['sweep'][f'vb{tau:.0f}']['n_worn']:>4}  "
              f"AUC={auc}  CI={ci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

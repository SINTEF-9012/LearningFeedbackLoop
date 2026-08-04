#!/usr/bin/env python3
"""Figshare leave-one-tool-out pre-failure transfer, swept over the pre-break threshold.

Phase 2a of docs/PUBLIC_DATASET_INTEGRATION_PLAN_2026-07-17.md — the breakage-claim
rescue. The paper's breakage transfer is at chance on Site_a_line2's two sessions; here
14 tools run to failure give 14 leave-one-tool-out folds, a properly-powered test of
whether a pre-failure signature learned on 13 tools generalises to an unseen 14th.

Complements Denkena (wear-STATE classification, leave-one-machine-out): this is a
pre-failure PREDICTION task (label = "in the last τ fraction of tool life") on measured
tool-life ground truth, transferred across tools on one machine.

Reuses the transfer harness verbatim (transfer_eval / informative_features / _present):
train on 13 tools, freeze, score the held-out tool; pooled AUC + block-bootstrap CI
clustered by tool. Only sensor-aggregate features enter (the loader already excludes the
tool-fixed process params that would leak tool identity).

Usage
-----
    python scripts/run_figshare_transfer.py
    python scripts/run_figshare_transfer.py --thresholds 0.1 0.2 0.3
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

from scripts.run_transfer_analysis import informative_features, transfer_eval, _present

CSV = ROOT / "data" / "breakage_patterns" / "figshare14_features.csv"
META_COLS = {"tool", "cycle", "cycle_to_failure", "cycle_to_failure_norm", "tool_type"}
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.3]  # fraction of tool life = "pre-break"
POS_LABEL = "pre_break"


def feature_basis(df: pd.DataFrame) -> List[str]:
    """Sensor-aggregate columns informative across the pooled data."""
    return informative_features(df, [c for c in df.columns if c not in META_COLS])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                    help="pre-break = cycle_to_failure_norm <= τ (fraction of life)")
    ap.add_argument("--out", default="data/figshare_transfer_results.json")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"missing {csv} — run scripts/build_figshare_features.py first")
    df = pd.read_csv(csv)
    df = df[df["tool"].notna() & df["cycle_to_failure_norm"].notna()].copy()
    basis = feature_basis(df)
    tools = sorted(df["tool"].dropna().astype(int).unique())
    print(f"Figshare pre-failure transfer — {len(df)} cycles, {len(tools)} tools, "
          f"{len(basis)} sensor features.")
    print(f"Leave-one-tool-out, frozen, no eval-time feedback. pre-break sweep (τ = fraction "
          f"of life): {args.thresholds}\n")

    store = {"n_cycles": int(len(df)), "tools": tools, "n_features": len(basis),
             "features": basis, "sweep": {}}
    for tau in args.thresholds:
        d = df.copy()
        d["label"] = (d["cycle_to_failure_norm"] <= tau).map({True: POS_LABEL, False: "normal"})
        pos = int((d["label"] == POS_LABEL).sum())
        per_t = d.groupby("tool")["label"].nunique()
        usable = [int(t) for t, k in per_t.items() if k > 1]
        res = transfer_eval(d, "tool", POS_LABEL, basis)
        _present(f"pre-break = last {tau:.0%} of life  ({pos}/{len(d)} pre-break, "
                 f"{len(usable)} usable tools)", res, ops=False)
        store["sweep"][f"tau{tau}"] = {
            "threshold_frac": tau, "n_pre_break": pos, "n_total": int(len(d)),
            "usable_tools": usable, "results": res,
        }

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text(json.dumps(store, indent=2, default=str))
    print(f"\nStored: {out}")

    print("\nSUMMARY — RandomForest pooled leave-one-tool-out AUC vs pre-break threshold:")
    for tau in args.thresholds:
        cell = store["sweep"][f"tau{tau}"]
        rf = next((v for k, v in cell["results"].items() if "RandomForest" in k), {})
        print(f"  last {tau:>5.0%} of life  pre_break={cell['n_pre_break']:>4}  "
              f"AUC={rf.get('pooled_auc')}  CI={rf.get('auc_ci95')}  "
              f"(tools={len(cell['usable_tools'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

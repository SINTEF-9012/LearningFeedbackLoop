#!/usr/bin/env python3
"""Turn the Denkena feedback result into MaaS CapabilityEvidence objects (RQ5 construction).

This closes part of the paper's RQ5 gap — evidence-object *generation* — on **real,
measured-label public data**, demonstrating the exact transformation the loop performs:
per-(plant, context, capability) operator confirm/dismiss counts → an aggregate, volume-
shrunk `CapabilityEvidence` record. It reuses `backend/agents/maas.build_evidence` unchanged,
so this is the real exporter, not a mock.

Grounded inputs — no fabricated detection numbers:
  - The confirm/dismiss tally per machine is the **measured** performance of the
    feedback-calibrated wear model on a held-out eval set (TP = worn tools the operator
    confirmed, FP = false wear-flags dismissed), from the actual Denkena experiment.
  - Context is the real dataset metadata (solid-carbide 4-flute end mill, cast iron
    600-3/S, 5-axis milling centre).

Provenance honesty (per project convention): the Denkena machines are a **separate research
dataset** (Leibniz Universität Hannover), NOT machines in the MaaS catalogue. So each record
uses honest `DENKENA-M{1,2,3}` plant ids with `declared=False` — a capability **measured
from data, not previously declared**. We do NOT map Denkena onto a catalogue supplier's
plant. The catalogue-linked declared→measured upgrade + CO2 weighting is already demonstrated
honestly for a real SITE_A machine in `run_maas_evidence_export.py`; it is not re-faked here.

Usage
-----
    python scripts/export_denkena_evidence.py            # τ=100, physical basis, K=20
    python scripts/export_denkena_evidence.py --tau 100 --budget 20 --basis physical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backend.agents.maas import build_evidence
from backend.agents.maas.evidence_exporter import write_evidence
from scripts.run_denkena_transfer import transfer_basis
from scripts.run_transfer_analysis import _threshold_for_recall

CSV = ROOT / "data" / "breakage_patterns" / "denkena_features.csv"
OUT = ROOT / "data" / "maas_evidence" / "denkena_capability_evidence.json"
CAPABILITY = "Tool-wear monitoring"
# Real dataset metadata (Mendeley zpxs87bjt8 description).
CONTEXT = {"machine_family": "5_axis_milling_center",
           "tool_type": "solid_carbide_end_mill_4flute_tin_tialn",
           "material": "cast_iron_600-3"}


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=5,
                                  class_weight="balanced", random_state=42)


def machine_tally(df: pd.DataFrame, basis: list, tau: float, budget: int,
                  target_recall: float) -> list:
    """For each held-out machine, deploy the feedback-calibrated model and count the
    operator's confirm/dismiss on the frozen eval set → an evidence aggregate."""
    df = df.copy()
    df["y"] = (df["wear"] >= tau).astype(int)
    machines = sorted(df["machine"].dropna().astype(int).unique())
    X = np.nan_to_num(df[basis].to_numpy(dtype=float))
    y = df["y"].to_numpy()
    mach = df["machine"].astype(int).to_numpy()
    rng = np.random.RandomState(42)

    aggregates = []
    for held in machines:
        tr = mach != held
        te_all = np.where(mach == held)[0]
        scaler = StandardScaler().fit(X[tr])
        model = _rf().fit(scaler.transform(X[tr]), y[tr])
        # feedback: draw `budget` oracle responses from the held machine to calibrate
        perm = rng.permutation(te_all)
        fb_idx, eval_idx = perm[:budget], perm[budget:]
        s_fb = model.predict_proba(scaler.transform(X[fb_idx]))[:, 1]
        if len(set(y[fb_idx].tolist())) < 2:
            continue
        thr = _threshold_for_recall(y[fb_idx], s_fb, target_recall)
        # deploy on the frozen eval; operator adjudicates each wear-flag
        s_ev = model.predict_proba(scaler.transform(X[eval_idx]))[:, 1]
        flagged = s_ev >= thr
        confirmed = int(np.sum(flagged & (y[eval_idx] == 1)))   # true worn, confirmed
        dismissed = int(np.sum(flagged & (y[eval_idx] == 0)))   # false flag, dismissed
        aggregates.append({
            "plant_id": f"DENKENA-M{held}",
            "context": CONTEXT,
            "capability": CAPABILITY,
            "confirmed": confirmed,
            "dismissed": dismissed,
            "event_id": None,
        })
    return aggregates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--tau", type=float, default=100.0)
    ap.add_argument("--budget", type=int, default=20, help="oracle confirmations for calibration")
    ap.add_argument("--target-recall", type=float, default=0.80)
    ap.add_argument("--basis", choices=["all", "physical"], default="physical")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"missing {csv} — run scripts/build_denkena_features.py first")
    df = pd.read_csv(csv)
    df = df[df["machine"].notna() & df["wear"].notna()].copy()
    basis = transfer_basis(df, kind=args.basis)

    aggregates = machine_tally(df, basis, args.tau, args.budget, args.target_recall)
    # catalogue=None, dpp=None: Denkena is not a catalogue plant — declared stays False,
    # no CO2 attached. This is the honest "measured, not declared" construction.
    records = build_evidence(aggregates, catalogue=None, dpp=None, window_days=90)
    out = Path(args.out)
    n = write_evidence(records, out)

    print("=" * 78)
    print(f"  DENKENA FEEDBACK → MaaS CAPABILITY EVIDENCE  (τ={args.tau:.0f}µm, "
          f"basis={args.basis}, K={args.budget}, target recall {args.target_recall})")
    print("=" * 78)
    print(f"  Capability: '{CAPABILITY}'  (measured from confirm/dismiss on measured-wear labels)")
    print(f"  Context: {CONTEXT}")
    print(f"  {'plant':>12} {'confirmed':>10} {'dismissed':>10} {'confirm_rate':>13} {'confidence':>11} {'declared':>9}")
    for e in records:
        print(f"  {e.plant_id:>12} {e.confirmed:>10} {e.dismissed:>10} "
              f"{e.confirm_rate:>13} {e.confidence:>11} {str(e.declared):>9}")
    print("-" * 78)
    print(f"  Wrote {n} evidence record(s) → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print("  Aggregate-only: confirm/dismiss counts + volume-shrunk confidence — never raw")
    print("  signals or per-run data. declared=False is honest: these research machines are")
    print("  not in the MaaS catalogue, so the capability is MEASURED, not previously declared.")
    print("  (Catalogue-linked declared→measured + CO2 is shown for a real SITE_A plant in")
    print("   run_maas_evidence_export.py — not re-faked onto Denkena here.)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

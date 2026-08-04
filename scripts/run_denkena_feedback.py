#!/usr/bin/env python3
"""Denkena feedback experiment: operator confirmations calibrate a cross-machine wear model.

Design + reasoning: docs/DENKENA_FEEDBACK_PLAN_2026-07-24.md. The Phase 2b transfer showed a
RandomForest wear classifier ranks worn-vs-normal on an unseen machine at AUC 0.91–0.99, but a probe
found its score scale shifts per machine, so a threshold set on the training machines catches ~0 % of
worn tools on a new one. This experiment tests whether a small budget of **measured** operator
confirmations (oracle: worn iff VB ≥ τ) on the new machine fixes that — i.e. whether feedback's value
is **operating-point calibration** (expected: large) rather than **ranking** (expected: flat), on
measured labels.

Protocol (leave-one-machine-out, per held-out machine, N_SPLITS random pool/eval splits):
  - train RandomForest on the other two machines, freeze;
  - split the held-out machine into a feedback POOL and a disjoint frozen EVAL set;
  - draw K oracle responses from the pool; three operating-point strategies compared on EVAL:
      (a) naive       — threshold from the training machines (no feedback);
      (b) source-cal  — isotonic calibration fit on the training machines, then threshold (still no
                        target-machine info — pre-empts "just calibrate the classifier");
      (c) feedback    — threshold re-set from the K oracle responses;
  - calibration metric: recall-error |achieved − target| and precision on EVAL;
  - adaptation metric: refit on train + K held-out labels, eval AUC (confirms ranking stays flat).

Usage
-----
    python scripts/run_denkena_feedback.py
    python scripts/run_denkena_feedback.py --tau 100 --target-recall 0.80 --budgets 0 5 10 20 40
    python scripts/run_denkena_feedback.py --noise 0.0 0.1 0.2      # RQ3 robustness sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from scripts.run_denkena_transfer import transfer_basis, META_COLS
from scripts.run_transfer_analysis import _threshold_for_recall

CSV = ROOT / "data" / "breakage_patterns" / "denkena_features.csv"
DEFAULT_BUDGETS = [0, 3, 5, 10, 20, 40]


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=5,
                                  class_weight="balanced", random_state=42)


def _recall_precision(y: np.ndarray, s: np.ndarray, thr: float) -> Tuple[float, float]:
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return recall, precision


def _draw_feedback(pool_idx: np.ndarray, y: np.ndarray, k: int,
                   rng: np.random.RandomState, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (chosen indices, oracle labels) for K responses. Labels are the true
    (measured) label with a fraction `noise` flipped — a mis-grading operator."""
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = rng.permutation(pool_idx)[:k]
    labels = y[order].copy()
    if noise > 0:
        flip = rng.random(len(labels)) < noise
        labels[flip] = 1 - labels[flip]
    return order, labels


def run(df: pd.DataFrame, basis: List[str], tau: float, target_recall: float,
        budgets: List[int], noise: float, n_splits: int) -> Dict:
    df = df.copy()
    df["y"] = (df["wear"] >= tau).astype(int)
    machines = sorted(df["machine"].dropna().astype(int).unique())
    X = np.nan_to_num(df[basis].to_numpy(dtype=float))
    y = df["y"].to_numpy()
    mach = df["machine"].astype(int).to_numpy()

    # accumulate per-(K) metrics across machines x splits
    acc = {K: {"recall_err_naive": [], "recall_err_source": [], "recall_err_fb": [],
               "prec_fb": [], "adapt_auc": [], "base_auc": []} for K in budgets}

    for held in machines:
        tr = mach != held
        te_all = np.where(mach == held)[0]
        if y[tr].sum() < 5 or y[te_all].sum() < 5:
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        model = _rf().fit(Xtr, y[tr])
        s_tr = model.predict_proba(Xtr)[:, 1]
        thr_naive = _threshold_for_recall(y[tr], s_tr, target_recall)

        # source-side isotonic calibration (fit on TRAIN only — no target info)
        iso = IsotonicRegression(out_of_bounds="clip").fit(s_tr, y[tr])
        s_tr_cal = iso.predict(s_tr)
        thr_source = _threshold_for_recall(y[tr], s_tr_cal, target_recall)

        Xte_all = scaler.transform(X[te_all])
        s_te_all = model.predict_proba(Xte_all)[:, 1]
        yte_all = y[te_all]

        for split in range(n_splits):
            rng = np.random.RandomState(1000 + split)
            # stratified 50/50 pool/eval on the held-out machine
            pos = te_all[yte_all == 1]; neg = te_all[yte_all == 0]
            rng.shuffle(pos); rng.shuffle(neg)
            pool = np.concatenate([pos[: len(pos) // 2], neg[: len(neg) // 2]])
            evalix = np.concatenate([pos[len(pos) // 2:], neg[len(neg) // 2:]])
            if y[evalix].sum() == 0 or y[evalix].sum() == len(evalix):
                continue
            # eval scores from the frozen source model
            s_ev = model.predict_proba(scaler.transform(X[evalix]))[:, 1]
            s_ev_cal = iso.predict(s_ev)
            y_ev = y[evalix]
            base_auc = float(roc_auc_score(y_ev, s_ev))

            for K in budgets:
                fb_idx, fb_lab = _draw_feedback(pool, y, K, rng, noise)
                # (a) naive, (b) source-cal — independent of K, but recorded per K for a flat line
                re_naive = abs(_recall_precision(y_ev, s_ev, thr_naive)[0] - target_recall)
                re_source = abs(_recall_precision(y_ev, s_ev_cal, thr_source)[0] - target_recall)
                # (c) feedback: threshold from the K oracle responses (needs both classes)
                if K > 0 and len(set(fb_lab.tolist())) > 1:
                    s_fb = model.predict_proba(scaler.transform(X[fb_idx]))[:, 1]
                    thr_fb = _threshold_for_recall(fb_lab, s_fb, target_recall)
                    rec_fb, prec_fb = _recall_precision(y_ev, s_ev, thr_fb)
                    re_fb = abs(rec_fb - target_recall)
                else:
                    re_fb, prec_fb = re_naive, float("nan")  # K=0 falls back to naive

                # adaptation: refit on train + K held-out labels, measure eval AUC
                if K > 0 and len(set(fb_lab.tolist())) > 1:
                    X_aug = np.vstack([X[tr], X[fb_idx]])
                    y_aug = np.concatenate([y[tr], fb_lab])
                    sc2 = StandardScaler().fit(X_aug)
                    m2 = _rf().fit(sc2.transform(X_aug), y_aug)
                    adapt_auc = float(roc_auc_score(
                        y_ev, m2.predict_proba(sc2.transform(X[evalix]))[:, 1]))
                else:
                    adapt_auc = base_auc

                acc[K]["recall_err_naive"].append(re_naive)
                acc[K]["recall_err_source"].append(re_source)
                acc[K]["recall_err_fb"].append(re_fb)
                acc[K]["prec_fb"].append(prec_fb)
                acc[K]["adapt_auc"].append(adapt_auc)
                acc[K]["base_auc"].append(base_auc)

    def _summ(v: List[float]) -> Dict:
        a = np.array([x for x in v if np.isfinite(x)], dtype=float)
        if a.size == 0:
            return {"mean": None, "n": 0}
        return {"mean": round(float(a.mean()), 4),
                "ci95": [round(float(np.percentile(a, 2.5)), 4),
                         round(float(np.percentile(a, 97.5)), 4)],
                "n": int(a.size)}

    out = {"tau_um": tau, "target_recall": target_recall, "noise": noise,
           "machines": machines, "n_splits": n_splits, "curve": {}}
    for K in budgets:
        out["curve"][str(K)] = {m: _summ(acc[K][m]) for m in acc[K]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--tau", type=float, default=100.0, help="worn threshold VB (µm)")
    ap.add_argument("--target-recall", type=float, default=0.80)
    ap.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0],
                    help="oracle label-flip fractions to sweep (RQ3 robustness)")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--out", default="data/denkena_feedback_results.json")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"missing {csv} — run scripts/build_denkena_features.py first")
    df = pd.read_csv(csv)
    df = df[df["machine"].notna() & df["wear"].notna()].copy()
    basis = transfer_basis(df)
    print(f"Denkena feedback — {len(df)} runs, machines {sorted(df['machine'].dropna().astype(int).unique())}, "
          f"{len(basis)} shared features, τ={args.tau}µm, target recall {args.target_recall}\n")

    store = {"tau_um": args.tau, "target_recall": args.target_recall,
             "n_features": len(basis), "noise_sweep": {}}
    for noise in args.noise:
        res = run(df, basis, args.tau, args.target_recall, args.budgets, noise, args.splits)
        store["noise_sweep"][str(noise)] = res
        print(f"== noise={noise:.2f} — recall-error vs feedback budget K (lower is better) ==")
        print(f"  {'K':>4} {'naive(a)':>10} {'source-cal(b)':>14} {'feedback(c)':>12} "
              f"{'prec@fb':>9} {'adapt AUC':>10}")
        for K in args.budgets:
            c = res["curve"][str(K)]
            def g(m):
                v = c[m]["mean"]; return f"{v:.3f}" if v is not None else "  -  "
            print(f"  {K:>4} {g('recall_err_naive'):>10} {g('recall_err_source'):>14} "
                  f"{g('recall_err_fb'):>12} {g('prec_fb'):>9} {g('adapt_auc'):>10}")
        print()

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text(json.dumps(store, indent=2, default=str))
    print(f"Stored: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

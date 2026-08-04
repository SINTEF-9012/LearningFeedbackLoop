#!/usr/bin/env python3
"""Transfer analysis & best-config search (stoppage + breakage).

Answers the decisive question from the Defender/Adversary debate
(docs/EXPERIMENT_DEBATE_CONCLUSION_2026-06-16.md): does a model trained on some
operations GENERALISE to a held-out operation **with no eval-time feedback**?
This converts the within-operation AAD result (which was online-fit to the eval
op's own oracle labels, ~AUC 0.86) into an honest transfer number.

Protocol: leave-one-group-out (operation for stoppage, session for breakage).
Train on N-1 groups, FREEZE, score the held-out group. Report per-fold AUC and a
pooled out-of-fold AUC with a block-bootstrap CI (clustered by group). No
eval-time feedback anywhere — pure transfer.

Configs compared (the deployable cores of the exploration configs):
  - IsolationForest (one-class)  = the faithful pipeline's seed anomaly model
  - Logistic (pattern indicators)= the AAD combiner's transferable form (stoppage)
  - Logistic (28 features)        = linear supervised baseline
  - RandomForest (28 features)    = the ablation's supervised core

Usage:  python scripts/run_transfer_analysis.py [--out data/transfer_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from backend.agents.processing.classical_models import FEATURE_NAMES
from backend.agents.processing.window_stats import block_bootstrap_metric_ci

POS = "pre_stoppage"  # set per-dataset


def _pattern_indicator_matrix(df: pd.DataFrame) -> tuple:
    """Fired-pattern indicator matrix for the stoppage dataset (the AAD's input)."""
    from backend.agents.experiment.config import ExperimentConfig, PATTERN_KEYS
    from backend.agents.experiment.evaluator import _detect_patterns_batch
    cfg = ExperimentConfig()
    fired = _detect_patterns_batch(df, cfg)  # list[list[str]] per row
    keys = list(PATTERN_KEYS)
    X = np.zeros((len(df), len(keys)), dtype=float)
    kidx = {k: i for i, k in enumerate(keys)}
    for r, ks in enumerate(fired):
        for k in ks:
            if k in kidx:
                X[r, kidx[k]] = 1.0
    return X, keys


def informative_features(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Drop constant/all-NaN columns — they carry no signal but inflate the count.

    Site_a_line2 has all 28 FEATURE_NAMES present but 9 are constant; counting those
    as features overstates what the model actually had to work with.
    """
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def _models(n_feats: int) -> Dict[str, Callable]:
    """Model zoo. The feature count is taken from the data, not hardcoded: only a
    subset of FEATURE_NAMES exists on any given dataset (13/28 on stoppage), so a
    fixed "(28 features)" label misreports the headline result (ISS-62).
    """
    return {
        "IsolationForest (one-class seed)": lambda: ("oneclass", IsolationForest(
            n_estimators=200, contamination=0.1, random_state=42)),
        f"Logistic ({n_feats} features)": lambda: ("supervised", LogisticRegression(
            max_iter=1000, class_weight="balanced")),
        f"RandomForest ({n_feats} features)": lambda: ("supervised", RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=42)),
    }


def _fit_score(kind, model, Xtr, ytr, Xte) -> tuple:
    """Fit, then return (train_scores, test_scores). Train scores are used to
    CALIBRATE the operating-point threshold (so the threshold is never set on the
    held-out data — the honest deployment setting)."""
    if kind == "oneclass":
        model.fit(Xtr[ytr == 0])               # normal-only, like the seed model
        return -model.score_samples(Xtr), -model.score_samples(Xte)
    sc = StandardScaler().fit(Xtr)
    model.fit(sc.transform(Xtr), ytr)
    return (model.predict_proba(sc.transform(Xtr))[:, 1],
            model.predict_proba(sc.transform(Xte))[:, 1])


def _threshold_for_recall(y, s, target_recall) -> float:
    """Lowest threshold achieving >= target_recall on (y, s) — calibrated on TRAIN."""
    y = np.asarray(y); s = np.asarray(s)
    P = max(1, int(y.sum()))
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    recall = tp / P
    idx = np.argmax(recall >= target_recall)
    if recall[idx] < target_recall:
        return float(s.min() - 1e-9)  # alert on everything
    return float(s[order][idx])


def transfer_eval(df: pd.DataFrame, group_col: str, pos_label: str,
                  feature_cols: List[str], extra_models: Dict = None) -> Dict:
    df = df.copy()
    y = (df["label"] == pos_label).astype(int).to_numpy()
    groups = [g for g in sorted(df[group_col].astype(str).unique())]
    Xfeat = np.nan_to_num(df[feature_cols].to_numpy(dtype=float))

    extra_models = extra_models or {}
    model_specs = dict(_models(len(feature_cols)))
    # entries ending in "__X" carry an alternate feature matrix, not a maker
    for k, v in extra_models.items():
        if not k.endswith("__X"):
            model_specs[k] = v

    target_recalls = (0.80, 0.90)
    cadence_s = 60.0  # one decision per 60 s window for the FA/hour conversion

    results = {}
    for name, maker in model_specs.items():
        Xsrc = extra_models.get(name + "__X")
        X = Xsrc if Xsrc is not None else Xfeat
        fold_aucs, oof_score, oof_true, oof_group = [], [], [], []
        # operating-point confusion accumulated across folds, threshold calibrated
        # on TRAIN to hit each target recall (never on the held-out data).
        opconf = {R: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for R in target_recalls}
        for held in groups:
            te_mask = df[group_col].astype(str).values == held
            tr_mask = ~te_mask
            yte = y[te_mask]; ytr = y[tr_mask]
            if yte.sum() == 0 or yte.sum() == len(yte):
                continue  # need both classes in held-out group for AUC
            kind, model = maker()
            s_tr, s_te = _fit_score(kind, model, X[tr_mask], ytr, X[te_mask])
            fold_aucs.append((held, float(roc_auc_score(yte, s_te))))
            oof_score.extend(s_te.tolist()); oof_true.extend(yte.tolist())
            oof_group.extend([held] * int(te_mask.sum()))
            for R in target_recalls:
                tau = _threshold_for_recall(ytr, s_tr, R)  # calibrate on train
                pred = s_te >= tau
                opconf[R]["tp"] += int(np.sum(pred & (yte == 1)))
                opconf[R]["fp"] += int(np.sum(pred & (yte == 0)))
                opconf[R]["tn"] += int(np.sum(~pred & (yte == 0)))
                opconf[R]["fn"] += int(np.sum(~pred & (yte == 1)))
        if not oof_true:
            results[name] = {"mean_fold_auc": None, "pooled_auc": None}
            continue
        # operating-point metrics at each target recall (train-calibrated threshold)
        op_metrics = {}
        for R, c in opconf.items():
            denom_p = c["tp"] + c["fp"]; denom_n = c["fp"] + c["tn"]; denom_r = c["tp"] + c["fn"]
            fp_rate = (c["fp"] / denom_n) if denom_n else 0.0
            op_metrics[f"recall@{R:.0%}"] = {
                "achieved_recall": round(c["tp"] / denom_r, 3) if denom_r else None,
                "precision": round(c["tp"] / denom_p, 3) if denom_p else None,
                "fp_rate_per_window": round(fp_rate, 3),
                "false_alarms_per_hour@60s": round(fp_rate * 3600.0 / cadence_s, 2),
            }
        pooled, lo, hi = block_bootstrap_metric_ci(
            oof_true, oof_score, oof_group,
            metric_fn=lambda yt, ys: roc_auc_score(yt, ys) if len(set(yt)) > 1 else 0.5,
            n_boot=2000)
        results[name] = {
            "mean_fold_auc": round(float(np.mean([a for _, a in fold_aucs])), 3),
            "pooled_auc": round(pooled, 3),
            "auc_ci95": [round(lo, 3), round(hi, 3)],
            "per_fold": {g: round(a, 3) for g, a in fold_aucs},
            "operating_points": op_metrics,
        }
    return results


def _present(title: str, res: Dict, ops: bool = False) -> None:
    print("\n" + "=" * 92)
    print(f"  TRANSFER (leave-one-group-out, frozen, NO eval-time feedback) — {title}")
    print("=" * 92)
    print(f"  {'config':<34}{'mean-fold AUC':>14}{'pooled AUC':>12}{'95% CI (block)':>22}")
    print("  " + "-" * 82)
    order = sorted(res.items(), key=lambda kv: -(kv[1].get('pooled_auc') or 0))
    for name, r in order:
        if r.get("pooled_auc") is None:
            print(f"  {name:<34}{'n/a':>14}"); continue
        ci = f"[{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]"
        print(f"  {name:<34}{r['mean_fold_auc']:>14.3f}{r['pooled_auc']:>12.3f}{ci:>22}")
    if ops:
        print("\n  OPERATING POINTS (threshold calibrated on TRAIN; FA/hour assumes 1 decision / 60 s):")
        print(f"  {'config':<34}{'target':>8}{'precision':>11}{'achv.recall':>12}{'FA/hour':>10}")
        print("  " + "-" * 73)
        for name, r in order:
            for tgt, m in (r.get("operating_points") or {}).items():
                print(f"  {name:<34}{tgt:>8}{(m['precision'] if m['precision'] is not None else float('nan')):>11.3f}"
                      f"{(m['achieved_recall'] if m['achieved_recall'] is not None else float('nan')):>12.3f}"
                      f"{m['false_alarms_per_hour@60s']:>10.2f}")
    print("=" * 92)


def _stoppage_csv_for_gap(gap: int) -> Path:
    base = ROOT / "data" / "breakage_patterns"
    return base / ("stoppage_features.csv" if gap == 0 else f"stoppage_features_gap{gap}s.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/transfer_results.json")
    ap.add_argument("--gaps", type=int, nargs="+", default=[0],
                    help="prediction gaps (s) to sweep for the stoppage horizon (needs the balanced gap CSVs)")
    args = ap.parse_args()

    store = {"stoppage_horizon": {}}

    # ---- STOPPAGE: LOO by operation, swept over prediction gap (lead-time) ----
    for gap in args.gaps:
        csv = _stoppage_csv_for_gap(gap)
        if not csv.exists():
            print(f"  [skip gap={gap}s] missing {csv.name} — generate with "
                  f"`extract_pre_stoppage_patterns.py --gap {gap}`")
            continue
        sdf = pd.read_csv(csv)
        avail_s = [c for c in FEATURE_NAMES if c in sdf.columns]
        feat_s = informative_features(sdf, avail_s)
        Xpat, pat_keys = _pattern_indicator_matrix(sdf)
        extra = {
            "Logistic (pattern indicators = AAD form)": lambda: ("supervised", LogisticRegression(
                max_iter=1000, class_weight="balanced")),
            "Logistic (pattern indicators = AAD form)__X": Xpat,
        }
        s_res = transfer_eval(sdf, "operation_id", "pre_stoppage", feat_s, extra_models=extra)
        mode = "DETECTION (onset)" if gap == 0 else f"PREDICTION (lead-time {gap}s)"
        _present(f"STOPPAGE {mode} — {sdf['operation_id'].nunique()} ops, "
                 f"{len(feat_s)}/{len(FEATURE_NAMES)} feats", s_res, ops=True)
        store["stoppage_horizon"][f"gap{gap}s"] = {
            "n": len(sdf),
            "n_features_available": len(avail_s),
            "n_features_used": len(feat_s),
            "features_used": feat_s,
            "results": s_res,
        }

    # ---- BREAKAGE: LOO by session (Site_a_line2) ----
    bdf = pd.read_csv(ROOT / "data" / "breakage_patterns" / "site_a_line2_features.csv")
    avail_b = [c for c in FEATURE_NAMES if c in bdf.columns]
    feat_b = informative_features(bdf, avail_b)
    b_res = transfer_eval(bdf, "operation_id", "pre_break", feat_b)
    _present(f"BREAKAGE (Site_a_line2, {bdf['operation_id'].nunique()} sessions, "
             f"{len(feat_b)}/{len(FEATURE_NAMES)} feats)", b_res, ops=True)
    store["breakage"] = {"n": len(bdf), "groups": sorted(bdf['operation_id'].unique()),
                         "n_features_available": len(avail_b),
                         "n_features_used": len(feat_b),
                         "features_used": feat_b,
                         "results": b_res}

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text(json.dumps(store, indent=2, default=str))
    print(f"\nStored: {out}")


if __name__ == "__main__":
    main()

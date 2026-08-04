#!/usr/bin/env python3
"""Drive the REAL LFL loop end-to-end on Denkena measured-wear data (live-loop wiring).

The offline experiments (run_denkena_transfer/feedback) validate the mechanisms with
scikit-learn scripts. This script instead pushes the same measured-label data through
the **actual system pipeline** --- `MemoryEventOrchestrator.process_event` → significance
scoring → SQLite `MemoryStore` → operator feedback via `MemoryFeedbackHandler` →
pattern-prior update → MaaS `build_evidence` --- to show the described architecture
processes real data and produces the artifacts the paper claims (memory records,
feedback-updated priors, evidence objects), not just sklearn numbers.

It runs fully in-process (no live server, Neo4j, LLM, SINDIT or torch): SQLite store,
classical/harmonic scorers and LLM explanations disabled. Honest scope: the significance
scorer is event-oriented (stoppage/chatter); for slow wear we feed the **wear model's
probability** as the classical anomaly signal (`external_signals["anomaly_detector_score"]`,
the architecture's classical-model integration point) and use `always_store` so every run
becomes a memory to adjudicate. The demonstration is that the *loop* runs on this data and
learns from feedback --- not that the stoppage heuristics are meaningful for wear.

Usage
-----
    python scripts/run_denkena_live_loop.py                 # held-out M3, 40 runs
    python scripts/run_denkena_live_loop.py --held 3 --n 60 --tau 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backend.agents.core.schemas import PatternKey, PatternType, TimeRange
from backend.agents.core.context import CuttingContext
from backend.agents.memory.orchestrator import (
    MemoryEvent, OrchestratorConfig, create_orchestrator,
)
from backend.agents.memory.feedback import FeedbackAction, MemoryFeedbackRequest
from backend.agents.storage.store import MemoryStore
from backend.agents.maas import build_evidence
from scripts.run_denkena_transfer import transfer_basis

CSV = ROOT / "data" / "breakage_patterns" / "denkena_features.csv"
WEAR_PATTERN = "FORCE_RMS_RISE"          # a Denkena wear pattern (see domain_packs/denkena.yaml)
CONTEXT_META = {"machine_family": "5_axis_milling_center",
                "tool_type": "solid_carbide_end_mill_4flute_tin_tialn",
                "material": "cast_iron_600-3"}


def _extract_prior(pri: dict, pattern: str) -> dict:
    """Pull the learned prior + confirm/dismiss counts for `pattern` from the
    persisted priors JSON (global and any context-keyed entry)."""
    needle = pattern.lower()

    def _match(d):
        return {k: v for k, v in (d or {}).items() if needle in k.lower()}

    out = {"global_prior": _match(pri.get("pattern_priors", {})),
           "global_counts": _match(pri.get("feedback_counts", {})),
           "by_context_prior": {}, "by_context_counts": {}}
    for ctx, pd_ in (pri.get("pattern_priors_by_context", {}) or {}).items():
        m = _match(pd_)
        if m:
            out["by_context_prior"][ctx] = m
    for ctx, cd in (pri.get("feedback_counts_by_context", {}) or {}).items():
        m = _match(cd)
        if m:
            out["by_context_counts"][ctx] = m
    return out


def _wear_probabilities(df, basis, held, tau):
    """Train the wear RF on the other machines, return per-run P(worn) for held machine."""
    df = df.copy()
    df["y"] = (df["wear"] >= tau).astype(int)
    mach = df["machine"].astype(int).to_numpy()
    X = np.nan_to_num(df[basis].to_numpy(float))
    tr = mach != held
    sc = StandardScaler().fit(X[tr])
    m = RandomForestClassifier(n_estimators=200, max_depth=5,
                               class_weight="balanced", random_state=42).fit(sc.transform(X[tr]), df["y"].to_numpy()[tr])
    held_mask = mach == held
    probs = m.predict_proba(sc.transform(X[held_mask]))[:, 1]
    return df[held_mask].reset_index(drop=True), probs


def _event_from_run(row, prob, basis, idx) -> MemoryEvent:
    """Build a MemoryEvent for one milling run, wear prob as the classical anomaly signal."""
    ctx = CuttingContext(
        tool_id=f"T{int(row['tool'])}",
        tool_type="end_mill",
        workpiece_material="cast_iron",
        machine_type="5_axis_milling_center",
    )
    # The wear model has raised a wear-pattern alert on this run.
    patterns = [PatternKey(pattern_type=PatternType.CUSTOM, key=WEAR_PATTERN,
                           confidence=float(prob), fault_type="tool_wear",
                           channel="force_sensor_x")]
    raw = {k: float(row[k]) for k in basis if pd.notna(row[k])}
    return MemoryEvent(
        session_id=f"denkena_M{int(row['machine'])}",
        time_range=TimeRange(i0=int(idx) * 110250, i1=int(idx) * 110250 + 110250,
                             t0=float(idx) * 4.4, t1=float(idx) * 4.4 + 4.4, fs=25000.0),
        patterns=patterns,
        cutting_context=ctx,
        external_signals={"anomaly_detector_score": float(prob),
                          "classical_alert": bool(prob >= 0.5)},
        raw_metrics=raw,
        metadata={"machine": int(row["machine"]), "tool": int(row["tool"]),
                  "wear_um": float(row["wear"]), "wear_prob": float(prob)},
    )


async def drive(df, basis, held, n, tau):
    scratch = Path(tempfile.mkdtemp(prefix="denkena_loop_"))
    store = MemoryStore(db_path=str(scratch / "memories.db"), enable_ann=False)
    config = OrchestratorConfig(
        priors_path=str(scratch / "pattern_priors.json"),
        use_classical_models=False,      # no seed-model training
        enable_harmonic_scorer=False,    # no torch
        generate_explanations=False,     # no LLM
        always_store=True,               # every run becomes a memory to adjudicate
        dispatch_alerts=False,           # no WS clients in-process
    )
    orch = create_orchestrator(store, config)

    held_df, probs = _wear_probabilities(df, basis, held, tau)
    worn = (held_df["wear"] >= tau).to_numpy()
    # The operator reviews the highest-risk flagged runs. Because the score scale shifts
    # per machine, a fixed 0.5 threshold is miscalibrated (the RQ3 finding), so we rank.
    # To exercise BOTH feedback directions we take the highest-scoring runs of each true
    # class: real wear (→ confirm) and the model's false alarms (→ dismiss).
    order = np.argsort(-probs)
    tp = [i for i in order if worn[i]]          # highest-scoring true wear
    fp = [i for i in order if not worn[i]]      # highest-scoring false alarms
    n_fp = min(len(fp), n // 3)                 # false alarms are rarer at the top
    chosen = tp[: n - n_fp] + fp[:n_fp]
    np.random.RandomState(0).shuffle(chosen)

    stored = confirmed = dismissed = alerts = 0
    prior_key = WEAR_PATTERN
    for step, i in enumerate(chosen):
        row = held_df.loc[i]
        prob = float(probs[i])
        ev = _event_from_run(row, prob, basis, step)
        res = await orch.process_event(ev)
        if res.memory_id:
            stored += 1
            if res.significant:
                alerts += 1
            action = FeedbackAction.CONFIRM if bool(worn[i]) else FeedbackAction.DISMISS
            await orch.feedback_handler.process_feedback(
                res.memory_id, MemoryFeedbackRequest(action=action, user_id="operator"))
            confirmed += int(action == FeedbackAction.CONFIRM)
            dismissed += int(action == FeedbackAction.DISMISS)

    # read the learned prior from the persisted store (all context-keyed values)
    prior = None
    priors_file = scratch / "pattern_priors.json"
    if priors_file.exists():
        try:
            pri = json.loads(priors_file.read_text())
            # find any entry whose pattern matches WEAR_PATTERN
            prior = _extract_prior(pri, WEAR_PATTERN)
        except Exception:
            prior = None

    # export MaaS evidence from the loop's own confirm/dismiss tally
    agg = [{"plant_id": f"DENKENA-M{held}", "context": CONTEXT_META,
            "capability": "Tool-wear monitoring",
            "confirmed": confirmed, "dismissed": dismissed, "event_id": None}]
    evidence = build_evidence(agg, catalogue=None, dpp=None, window_days=90)

    return {
        "held_machine": held, "runs_processed": len(chosen), "memories_stored": stored,
        "alerts": alerts, "confirmed": confirmed, "dismissed": dismissed,
        "learned_prior": prior, "evidence": [e.to_dict() for e in evidence],
        "scratch": str(scratch),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held", type=int, default=3, help="held-out machine (1/2/3)")
    ap.add_argument("--n", type=int, default=40, help="number of runs to process")
    ap.add_argument("--tau", type=float, default=100.0, help="worn threshold VB (µm)")
    ap.add_argument("--out", default="data/denkena_live_loop_result.json")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    df = df[df["machine"].notna() & df["wear"].notna()].copy()
    basis = transfer_basis(df, kind="physical")

    r = asyncio.run(drive(df, basis, args.held, args.n, args.tau))

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text(json.dumps(r, indent=2, default=str))

    print("=" * 74)
    print(f"  DENKENA → LIVE LFL LOOP  (held-out machine M{r['held_machine']}, "
          f"in-process: SQLite store, no LLM/Neo4j/torch)")
    print("=" * 74)
    print(f"  event → significance → memory store → feedback → prior → evidence")
    print(f"  runs processed        : {r['runs_processed']}")
    print(f"  memories stored        : {r['memories_stored']}   (alerts: {r['alerts']})")
    print(f"  operator feedback      : {r['confirmed']} confirmed / {r['dismissed']} dismissed "
          f"(vs measured VB≥{args.tau:.0f}µm)")
    lp = r["learned_prior"] or {}
    gp = lp.get("global_prior") or {}
    cp = lp.get("by_context_prior") or {}
    print(f"  learned prior '{WEAR_PATTERN}' (0.5 = neutral, moves toward confirm rate):")
    print(f"     global : {gp if gp else 'none'}")
    for ctx, val in cp.items():
        print(f"     context [{ctx}]: {val}")
    if not gp and not cp:
        print("     (no prior movement recorded — see scratch priors file)")
    ev = r["evidence"][0] if r["evidence"] else {}
    print(f"  evidence object        : confirm_rate={ev.get('confirm_rate')} "
          f"confidence={ev.get('confidence')} declared={ev.get('declared')}")
    print("-" * 74)
    print(f"  The real orchestrator/store/feedback/evidence code processed measured-label")
    print(f"  wear data end-to-end. Stored to {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

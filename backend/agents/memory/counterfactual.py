"""Counterfactual "what did operator feedback change" summary (plan 2.4).

Turns a sequential-replay snapshot (produced by
``scripts/run_breakage_sequential.py --output ...``) into a compact
before/after summary the operator UI can render: how many alerts operator
feedback removed, how many breakage episodes were still caught, and how many
false-alarm episodes it silenced.

This serves a **measured** counterfactual on the validated Site_a_line2 breakage
case study — it is not a live re-run (re-running trains seed models per request
and takes ~30 s). The framing in the UI must say "measured on the case study",
not "your live session". Live per-session counterfactuals are future work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default snapshot: the full-stack result (calibration + co-occurrence gating +
# scoped feedback) over all 8 Site_a_line2 sessions.
DEFAULT_SNAPSHOT = (
    Path("data/experiment_snapshots/sequential_2026-07-07/seq_v2_all_cooccurrence.json")
)


def _arm_totals(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    alerts = sum(int(r.get("alerts", 0)) for r in rows)
    fp = sum(int(r.get("fp_alerts", 0)) for r in rows)
    tp = sum(int(r.get("tp_alerts", 0)) for r in rows)
    broken_alerted = sum(1 for r in rows if r.get("n_pre_break") and r.get("alerts"))
    broken_total = sum(1 for r in rows if r.get("n_pre_break"))
    healthy_alerted = sum(1 for r in rows if not r.get("n_pre_break") and r.get("alerts"))
    return {
        "alerts": alerts,
        "false_alerts": fp,
        "true_alerts": tp,
        "broken_episodes_alerted": broken_alerted,
        "broken_episodes_total": broken_total,
        "healthy_episodes_alerting": healthy_alerted,
    }


def summarize_counterfactual(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the before/after summary from a replay snapshot dict.

    Pure function — no I/O — so it is unit-testable and safe to reuse.
    """
    off_rows = snapshot.get("off", {}).get("rows", []) or []
    on_rows = snapshot.get("on", {}).get("rows", []) or []
    off = _arm_totals(off_rows)
    on = _arm_totals(on_rows)

    off_alerts = off["alerts"]
    burden_reduction = (
        (off_alerts - on["alerts"]) / off_alerts if off_alerts else 0.0
    )
    fp_reduction = (
        (off["false_alerts"] - on["false_alerts"]) / off["false_alerts"]
        if off["false_alerts"] else 0.0
    )
    return {
        "off": off,
        "on": on,
        "burden_reduction": round(burden_reduction, 4),
        "false_alarm_reduction": round(fp_reduction, 4),
        # Coverage is preserved iff feedback did not drop any broken episode.
        "coverage_preserved": on["broken_episodes_alerted"] >= off["broken_episodes_alerted"],
        "session": snapshot.get("session"),
        "adjudicate": snapshot.get("adjudicate"),
        "off_auc": snapshot.get("off", {}).get("auc"),
        "on_auc": snapshot.get("on", {}).get("auc"),
    }


def load_counterfactual_summary(
    snapshot_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load a snapshot and return its summary, or ``None`` if unavailable.

    Degrades to ``None`` (never raises) so a missing/corrupt snapshot simply
    hides the panel rather than breaking the page.
    """
    path = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    summary = summarize_counterfactual(snapshot)
    summary["source"] = str(path)
    summary["measured"] = True  # not a live re-run — see module docstring
    return summary

"""Read-only loader for the MaaS capability-evidence artifact (UI surface).

The evidence exporter (`evidence_exporter.build_evidence` / `write_evidence`)
writes a per-`(plant, context, capability)` `CapabilityEvidence` JSON — the
aggregate that "propagates up" to a MaaS matchmaking platform (declared→measured
capability, confirm-rate, volume-shrunk confidence, and CO₂ avoided per catch;
never raw signals or memory contents). This loads that artifact for a read-only
UI panel.

**Framing (must be preserved in the UI):** this is an *illustrative evidence
artifact*, not a live platform decision — there is no running MaaS matchmaking
service consuming it. See docs/DEMO_IMPROVEMENT_HANDOFF_2026-07-07.md §3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_EVIDENCE_DIR = Path("data/maas_evidence")
DEFAULT_EVIDENCE_PATH = _EVIDENCE_DIR / "capability_evidence_tool_wear.json"

# The three additional evidence facets written alongside capability by
# scripts/run_maas_evidence_export.py. Each is loaded with the same generic
# reader; a missing artifact simply yields None and the UI hides that panel.
FACET_PATHS: Dict[str, Path] = {
    "capability": DEFAULT_EVIDENCE_PATH,
    "fault": _EVIDENCE_DIR / "fault_lead_time_evidence.json",
    "availability": _EVIDENCE_DIR / "availability_evidence.json",
    "sustainability": _EVIDENCE_DIR / "sustainability_evidence.json",
}


def load_evidence_summary(path: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Load the capability-evidence artifact into a UI-friendly summary.

    Returns ``None`` (never raises) when the artifact is missing/corrupt so the
    UI simply hides the panel. Otherwise returns
    ``{records: [...], count, illustrative: True, source}``.
    """
    p = Path(path) if path else DEFAULT_EVIDENCE_PATH
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return None
    records: List[Dict[str, Any]]
    if isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        records = [raw]
    else:
        return None
    if not records:
        return None
    return {
        "records": records,
        "count": len(records),
        "illustrative": True,
        "source": str(p),
    }


def load_evidence_facets() -> Dict[str, Any]:
    """Load every evidence facet (capability, fault, availability, sustainability).

    Returns ``{facets: {name: summary|None}}`` where each summary matches the
    shape of ``load_evidence_summary``. Never raises; a missing facet is ``None``.
    """
    return {
        "facets": {name: load_evidence_summary(p) for name, p in FACET_PATHS.items()},
        "illustrative": True,
    }

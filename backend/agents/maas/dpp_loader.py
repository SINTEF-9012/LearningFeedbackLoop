"""Digital Product Passport (DPP) loader — per-part CO2/cost impact for feedback weighting.

The DPP (`data/supplementary_data/DPP_*.json`) gives a per-part Product Carbon Footprint
(PCF) decomposed across life-cycle stages A1-A3. This module exposes it as `PartImpact`
so the feedback loop can weight a confirmed catch by what acting/not-acting actually
protects: a confirmed breakage catch on a high-PCF part avoids more embodied CO2 than the
same catch on a cheap part. Pure read-only; degrades to None when no DPP is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # defensive: usecase normalisation is best-effort, never fatal to DPP loading
    from backend.agents.usecase import resolve_usecase
except Exception:  # pragma: no cover - fallback keeps the loader import-safe
    resolve_usecase = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PartImpact:
    """Embodied impact of one good part, from its DPP."""

    event_id: str
    company: str
    pcf_total_kg: float          # cradle-to-gate A1-A3
    pcf_processing_kg: float     # A3 (the manufacturing stage this loop influences)
    source: str
    usecase: str = "UNKNOWN"     # normalised usecase (SITE_A/SITE_B/SITE_C/...) for scoped lookup

    @property
    def co2_avoided_per_scrap_kg(self) -> float:
        """A confirmed catch that prevents scrapping a part avoids re-making it.

        The embodied carbon already spent on the work-in-progress is the cradle-to-gate
        figure; we credit the processing stage (A3) as the directly loop-attributable
        avoided re-work, and expose the total separately for cradle-to-gate framing.
        """
        return self.pcf_processing_kg


class DPPRegistry:
    """Loads DPP files and resolves per-part impact by event_id (or first available)."""

    def __init__(self) -> None:
        self._by_event: Dict[str, PartImpact] = {}
        self._by_usecase: Dict[str, List[PartImpact]] = {}

    @classmethod
    def from_dir(cls, directory: str | Path) -> "DPPRegistry":
        reg = cls()
        d = Path(directory)
        if not d.exists():
            return reg
        for f in sorted(d.glob("DPP_*.json")):
            try:
                reg._ingest(f)
            except Exception:
                continue  # defensive: a malformed DPP must not break the loop
        return reg

    def _ingest(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        records = raw.get("DPP") if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            return
        for rec in records:
            event_id = str(rec.get("event_id") or rec.get("product") or path.stem)
            company = str(rec.get("company") or "unknown")
            usecase = _normalise_usecase(rec.get("usecase") or company)
            pcfs = (((rec.get("carbonFootprint") or {}).get("ProductCarbonFootprints")) or [])
            total = _pcf_for(pcfs, "A1-A3")
            a3 = _pcf_for(pcfs, "A3")
            if total is None and a3 is None:
                continue
            impact = PartImpact(
                event_id=event_id,
                company=company,
                pcf_total_kg=float(total if total is not None else a3),
                pcf_processing_kg=float(a3 if a3 is not None else 0.0),
                source=path.name,
                usecase=usecase,
            )
            self._by_event[event_id] = impact
            self._by_usecase.setdefault(usecase, []).append(impact)

    def resolve(self, event_id: Optional[str] = None) -> Optional[PartImpact]:
        """Resolve by event_id, else the first available part.

        NOTE: this is intentionally *not* usecase-scoped — it preserves the
        original behaviour used by the feedback impact-weighting path. For any
        operator-facing / per-usecase surface use ``resolve_for_usecase`` so a
        SITE_C event never silently receives a SITE_A part's carbon.
        """
        if event_id and event_id in self._by_event:
            return self._by_event[event_id]
        # fall back to the first available part for the same company / any part
        return next(iter(self._by_event.values()), None)

    def resolve_for_usecase(
        self, usecase: Optional[str], event_id: Optional[str] = None
    ) -> Optional[PartImpact]:
        """Resolve a part impact strictly within one usecase.

        Returns ``None`` when no DPP exists for that usecase — never falls back
        across usecases (that would attribute e.g. SITE_A carbon to a SITE_C event).
        ``event_id`` is honoured only if it belongs to the requested usecase.
        """
        norm = _normalise_usecase(usecase)
        if not norm:
            return None
        candidates = self._by_usecase.get(norm) or []
        if not candidates:
            return None
        if event_id:
            for impact in candidates:
                if impact.event_id == event_id:
                    return impact
        return candidates[0]

    def all(self) -> List[PartImpact]:
        return list(self._by_event.values())


def _normalise_usecase(value: Optional[str]) -> str:
    """Best-effort map a usecase/company hint to a canonical usecase code.

    Uses the shared ``resolve_usecase`` normaliser when available (so "site_a"
    -> SITE_A, "site_c" -> SITE_C, etc.); degrades to an upper-cased raw value otherwise.
    """
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN"
    if resolve_usecase is not None:
        try:
            resolved = resolve_usecase(usecase=raw, source=raw, fallback_generic=False)
            if resolved:
                return str(resolved)
        except Exception:
            pass
    return raw.upper()


def _pcf_for(pcfs: list, stage_suffix: str) -> Optional[float]:
    """Pick the PCF entry whose ProcessName ends with the given stage code."""
    for p in pcfs:
        name = str(p.get("ProcessName", "")).strip()
        if name.endswith(stage_suffix):
            try:
                return float(p.get("PCF in kg CO2eq"))
            except (TypeError, ValueError):
                return None
    return None

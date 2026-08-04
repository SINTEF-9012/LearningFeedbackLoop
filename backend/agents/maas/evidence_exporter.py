"""Capability-evidence exporter — feedback -> validated, context-conditioned MaaS records.

Converts per-context operator feedback (confirm/dismiss counts the local loop already
keeps) into `CapabilityEvidence`: the aggregate record MaaS ingests to upgrade a plant's
*declared* `machiningStabilityCapabilities` entry into a *measured*, evidence-backed one.
What flows up is the aggregate only — never raw signals or memory bodies.

Faithful to `docs/MAAS_FEEDBACK_CONTRIBUTION_2026-06-02.md` §6. Pure dict-in/dict-out;
no dependency on storage backends, so it runs on a schedule or on demand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .dpp_loader import DPPRegistry, PartImpact

# The only domain table to maintain: which fired patterns evidence which declared capability.
CAPABILITY_PATTERN_MAP: Dict[str, List[str]] = {
    "Vibration control": ["fault:chatter", "signature:modulated_tooth_passing_vibration"],
    "Tool-wear monitoring": ["fault:tool_breakage", "signature:hf_burst_periodicity_loss"],
    "Alignment control": ["fault:workpiece_slip", "signature:spindle_shift_phase_change"],
    "Thermal control": ["fault:thermal_drift"],
    "Defect detection": ["fault:surface_defect"],
}

# confidence shrinkage: confidence = n / (n + CONFIDENCE_PRIOR_N). Low feedback volume ->
# low confidence -> the plant is matched conservatively (the proposal's safeguard).
CONFIDENCE_PRIOR_N = 20.0


def capability_for_pattern(pattern: str) -> Optional[str]:
    for capability, patterns in CAPABILITY_PATTERN_MAP.items():
        if pattern in patterns:
            return capability
    return None


@dataclass
class CapabilityEvidence:
    supplier_id: str
    plant_id: str
    context: Dict[str, str]
    capability: str
    declared: bool                 # did the plant declare this capability in the catalogue?
    confirmed: int
    dismissed: int
    lead_time_s_median: Optional[float]
    window: str
    confidence: float
    confirm_rate: float
    realised_energy_kwh_per_good_part: Optional[float]
    realised_co2_kg_per_good_part: Optional[float]
    co2_avoided_kg_per_confirmed_catch: Optional[float] = None
    co2_avoided_kg_total: Optional[float] = None
    dpp_source: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FaultLeadTimeEvidence:
    """Per-fault confirmed/dismissed tallies and observed lead times for one capability.

    Skeleton facet (feedback-fed): groups the confirmed faults behind a declared
    capability with their median detection lead time, where measured. `lead_time_s_median`
    degrades to None when the loop did not record a pre-event horizon for that fault.
    """

    supplier_id: str
    plant_id: str
    context: Dict[str, str]
    capability: str
    faults: List[Dict]  # [{fault, confirmed, dismissed, lead_time_s_median}]
    window: str
    confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AvailabilityAdjustmentEvidence:
    """Declared availability adjusted by observed confirmed stoppages.

    Skeleton facet (stoppage-fed): `mean_hours_between_stoppages` is a direct
    observable. `availability_adjustment_pct` additionally needs a downtime-per-
    stoppage assumption (the detector observes stoppage onset, not duration); it is
    left None rather than fabricated when no assumption is supplied.
    """

    supplier_id: str
    plant_id: str
    context: Dict[str, str]
    declared_availability_pct: Optional[float]
    confirmed_stoppages: int
    operating_hours: float
    mean_hours_between_stoppages: Optional[float]
    availability_adjustment_pct: Optional[float]
    window: str
    confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)


# Any of the evidence facets share the JSON writer below.
EvidenceRecord = Union[CapabilityEvidence, FaultLeadTimeEvidence, AvailabilityAdjustmentEvidence]


@dataclass
class PlantCatalogue:
    """Read-only view of the MaaS plant catalogue (declared, static fields)."""

    plants: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PlantCatalogue":
        raw = json.loads(Path(path).read_text())
        records = raw if isinstance(raw, list) else list(raw.values())
        return cls(plants={str(p.get("plantId")): p for p in records if p.get("plantId")})

    def supplier_of(self, plant_id: str) -> Optional[str]:
        p = self.plants.get(plant_id)
        return str(p.get("supplierId")) if p else None

    def declares(self, plant_id: str, capability: str) -> bool:
        p = self.plants.get(plant_id) or {}
        return capability in (p.get("machiningStabilityCapabilities") or [])

    def energy_kwh_per_part(self, plant_id: str) -> Optional[float]:
        p = self.plants.get(plant_id) or {}
        v = p.get("averageEnergyConsumptionPerPartKwh")
        return float(v) if v is not None else None

    def co2_factor(self, plant_id: str) -> Optional[float]:
        p = self.plants.get(plant_id) or {}
        v = p.get("co2FactorKgPerKwh")
        return float(v) if v is not None else None

    def availability_pct(self, plant_id: str) -> Optional[float]:
        p = self.plants.get(plant_id) or {}
        v = p.get("availabilityNext6WeeksPercent")
        return float(v) if v is not None else None


def _confidence(n: int) -> float:
    return round(n / (n + CONFIDENCE_PRIOR_N), 3) if n > 0 else 0.0


def build_evidence(
    feedback_aggregates: Sequence[Mapping],
    *,
    catalogue: Optional[PlantCatalogue] = None,
    dpp: Optional[DPPRegistry] = None,
    window_days: int = 90,
) -> List[CapabilityEvidence]:
    """Aggregate per-(plant, context, capability) feedback into MaaS evidence records.

    Each `feedback_aggregates` item is a dict:
        {plant_id, context: {machine_family, tool_type, material},
         capability OR pattern, confirmed, dismissed, lead_times_s?: [..], event_id?}
    Confidence grows with feedback volume; CO2 fields are filled from the plant
    catalogue and the DPP when available, and omitted (None) otherwise.
    """
    window = f"{window_days}d"
    out: List[CapabilityEvidence] = []
    for agg in feedback_aggregates:
        plant_id = str(agg.get("plant_id") or "")
        capability = agg.get("capability") or capability_for_pattern(str(agg.get("pattern") or ""))
        if not plant_id or not capability:
            continue
        confirmed = int(agg.get("confirmed", 0))
        dismissed = int(agg.get("dismissed", 0))
        n = confirmed + dismissed
        lead = list(agg.get("lead_times_s") or [])
        context = {str(k): str(v) for k, v in (agg.get("context") or {}).items()}

        supplier_id = (catalogue.supplier_of(plant_id) if catalogue else None) or "UNKNOWN"
        declared = catalogue.declares(plant_id, capability) if catalogue else False
        energy = catalogue.energy_kwh_per_part(plant_id) if catalogue else None
        factor = catalogue.co2_factor(plant_id) if catalogue else None
        realised_co2 = round(energy * factor, 1) if (energy is not None and factor is not None) else None

        impact: Optional[PartImpact] = dpp.resolve(agg.get("event_id")) if dpp else None
        co2_per_catch = round(impact.co2_avoided_per_scrap_kg, 1) if impact else None
        co2_total = round(co2_per_catch * confirmed, 1) if co2_per_catch is not None else None

        out.append(CapabilityEvidence(
            supplier_id=supplier_id,
            plant_id=plant_id,
            context=context,
            capability=str(capability),
            declared=declared,
            confirmed=confirmed,
            dismissed=dismissed,
            lead_time_s_median=round(median(lead), 1) if lead else None,
            window=window,
            confidence=_confidence(n),
            confirm_rate=round(confirmed / n, 3) if n else 0.0,
            realised_energy_kwh_per_good_part=energy,
            realised_co2_kg_per_good_part=realised_co2,
            co2_avoided_kg_per_confirmed_catch=co2_per_catch,
            co2_avoided_kg_total=co2_total,
            dpp_source=impact.source if impact else None,
        ))
    return out


def build_fault_lead_time_evidence(
    fault_aggregates: Sequence[Mapping],
    *,
    catalogue: Optional[PlantCatalogue] = None,
    window_days: int = 90,
) -> List[FaultLeadTimeEvidence]:
    """Aggregate per-(plant, context, capability) confirmed faults and their lead times.

    Each `fault_aggregates` item is a dict:
        {plant_id, context: {...}, capability OR pattern,
         faults: [{fault, confirmed, dismissed, lead_times_s?: [..]}, ...]}

    Confidence grows with the total adjudicated fault events in the record. A fault
    with no recorded lead times reports `lead_time_s_median: null`.
    """
    window = f"{window_days}d"
    out: List[FaultLeadTimeEvidence] = []
    for agg in fault_aggregates:
        plant_id = str(agg.get("plant_id") or "")
        capability = agg.get("capability") or capability_for_pattern(str(agg.get("pattern") or ""))
        if not plant_id or not capability:
            continue
        context = {str(k): str(v) for k, v in (agg.get("context") or {}).items()}
        supplier_id = (catalogue.supplier_of(plant_id) if catalogue else None) or "UNKNOWN"

        faults: List[Dict] = []
        total = 0
        for f in (agg.get("faults") or []):
            confirmed = int(f.get("confirmed", 0))
            dismissed = int(f.get("dismissed", 0))
            total += confirmed + dismissed
            lead = list(f.get("lead_times_s") or [])
            faults.append({
                "fault": str(f.get("fault") or "unknown"),
                "confirmed": confirmed,
                "dismissed": dismissed,
                "lead_time_s_median": round(median(lead), 1) if lead else None,
            })
        if not faults:
            continue
        out.append(FaultLeadTimeEvidence(
            supplier_id=supplier_id,
            plant_id=plant_id,
            context=context,
            capability=str(capability),
            faults=faults,
            window=window,
            confidence=_confidence(total),
        ))
    return out


def build_availability_evidence(
    stoppage_aggregates: Sequence[Mapping],
    *,
    catalogue: Optional[PlantCatalogue] = None,
    assumed_downtime_h_per_stoppage: Optional[float] = None,
    window_days: int = 90,
) -> List[AvailabilityAdjustmentEvidence]:
    """Turn confirmed-stoppage counts into an availability-adjustment record.

    Each `stoppage_aggregates` item is a dict:
        {plant_id, context: {...}, confirmed_stoppages, operating_hours,
         assumed_downtime_h_per_stoppage?}

    `mean_hours_between_stoppages` is computed directly. The availability
    adjustment additionally needs a downtime-per-stoppage assumption (the stoppage
    detector observes onset, not duration); when none is supplied — per record or
    via the function argument — the adjustment is left None rather than fabricated.
    """
    window = f"{window_days}d"
    out: List[AvailabilityAdjustmentEvidence] = []
    for agg in stoppage_aggregates:
        plant_id = str(agg.get("plant_id") or "")
        if not plant_id:
            continue
        context = {str(k): str(v) for k, v in (agg.get("context") or {}).items()}
        confirmed = int(agg.get("confirmed_stoppages", 0))
        hours = float(agg.get("operating_hours", 0.0) or 0.0)
        supplier_id = (catalogue.supplier_of(plant_id) if catalogue else None) or "UNKNOWN"
        declared = catalogue.availability_pct(plant_id) if catalogue else None

        mhbs = round(hours / confirmed, 1) if confirmed > 0 and hours > 0 else None

        downtime = agg.get("assumed_downtime_h_per_stoppage", assumed_downtime_h_per_stoppage)
        adjustment: Optional[float] = None
        if downtime is not None and hours > 0:
            lost_fraction = min(confirmed * float(downtime) / hours, 1.0)
            adjustment = -round(lost_fraction * 100, 1)

        out.append(AvailabilityAdjustmentEvidence(
            supplier_id=supplier_id,
            plant_id=plant_id,
            context=context,
            declared_availability_pct=declared,
            confirmed_stoppages=confirmed,
            operating_hours=hours,
            mean_hours_between_stoppages=mhbs,
            availability_adjustment_pct=adjustment,
            window=window,
            confidence=_confidence(confirmed),
        ))
    return out


def write_evidence(records: Iterable[EvidenceRecord], path: str | Path) -> int:
    payload = [r.to_dict() for r in records]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    return len(payload)

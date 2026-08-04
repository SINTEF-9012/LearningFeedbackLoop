"""MaaS value-chain layer: turn operator feedback into validated, context-conditioned
capability evidence and CO2/cost-weighted impact for the matchmaking platform.

This is the upstream end of the learning feedback loop described in
`docs/MAAS_FEEDBACK_CONTRIBUTION_2026-06-02.md`: the local loop already captures
operator confirm/dismiss as per-context priors/counts; this package converts that
material into the `CapabilityEvidence` records MaaS consumes (never raw signals or
memory bodies) and consumes the Digital Product Passport (DPP) for impact weighting.
"""

from .dpp_loader import DPPRegistry, PartImpact
from .evidence_exporter import (
    CAPABILITY_PATTERN_MAP,
    AvailabilityAdjustmentEvidence,
    CapabilityEvidence,
    FaultLeadTimeEvidence,
    PlantCatalogue,
    build_availability_evidence,
    build_evidence,
    build_fault_lead_time_evidence,
    capability_for_pattern,
)

__all__ = [
    "DPPRegistry",
    "PartImpact",
    "CapabilityEvidence",
    "FaultLeadTimeEvidence",
    "AvailabilityAdjustmentEvidence",
    "PlantCatalogue",
    "CAPABILITY_PATTERN_MAP",
    "build_evidence",
    "build_fault_lead_time_evidence",
    "build_availability_evidence",
    "capability_for_pattern",
]

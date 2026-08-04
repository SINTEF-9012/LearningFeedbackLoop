"""
Cutting Context - Machining parameters for context-aware memory retrieval.

# ===========================================================================
# DRAFT/PROTOTYPE - Tag: [PROTOTYPE_LLM_MEMORY_V1]
# This module is a draft implementation for the LLM memory system.
# Expected to be refined or replaced as the system evolves.
# Key integration points marked with [INTEGRATION_POINT]
# ===========================================================================

Provides structured representation of cutting conditions for:
1. Context-based memory filtering
2. Similarity scoring between operations
3. Derived parameter computation (tooth passing freq, etc.)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, computed_field
from enum import Enum
import math


# [PROTOTYPE_LLM_MEMORY_V1] - Operating regime classification
class OperatingRegime(str, Enum):
    """Machining operation type - affects baseline expectations."""
    ROUGHING = "roughing"
    SEMI_FINISHING = "semi_finishing"
    FINISHING = "finishing"
    DRILLING = "drilling"
    TAPPING = "tapping"
    UNKNOWN = "unknown"


# [PROTOTYPE_LLM_MEMORY_V1] - Main context schema
class CuttingContext(BaseModel):
    """
    Machining parameters for context-based retrieval.
    
    Used for:
    - Filtering memories to contextually relevant subset
    - Computing derived parameters (tooth passing frequency)
    - Determining operating regime for normalization
    
    [INTEGRATION_POINT] Feature extractor should populate this from session metadata.
    """
    
    # Tool parameters
    tool_type: Optional[str] = None  # end_mill, ball_mill, drill, tap, etc.
    tool_diameter: Optional[float] = None  # mm
    num_teeth: Optional[int] = None  # z (number of flutes)
    tool_id: Optional[str] = None  # Tool identifier if tracked
    tool_length: Optional[float] = None  # mm
    tool_material: Optional[str] = None  # carbide, HSS, etc.
    
    # Cutting parameters (from NC program or sensors)
    spindle_speed: Optional[float] = None  # n (rpm)
    feed_rate: Optional[float] = None  # vf (mm/min)
    feed_per_tooth: Optional[float] = None  # fz (mm/tooth)
    cutting_speed: Optional[float] = None  # vc (m/min)
    axial_depth: Optional[float] = None  # ap (mm)
    radial_depth: Optional[float] = None  # ae (mm)
    
    # Material
    workpiece_material: Optional[str] = None  # steel, aluminum, titanium, etc.
    workpiece_hardness: Optional[float] = None  # HRC
    
    # Machine
    machine_id: Optional[str] = None
    machine_type: Optional[str] = None
    
    # Derived/classified
    operating_regime: Optional[OperatingRegime] = None
    
    # [PROTOTYPE_LLM_MEMORY_V1] - Raw metadata passthrough
    # For fields not yet formalized
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def tooth_passing_freq(self) -> Optional[float]:
        """
        Tooth passing frequency in Hz.
        f_tooth = (n * z) / 60
        
        Critical for chatter detection - harmonics of this frequency
        indicate regenerative chatter.
        """
        if self.spindle_speed and self.num_teeth:
            return (self.spindle_speed * self.num_teeth) / 60.0
        return None
    
    @property
    def spindle_freq(self) -> Optional[float]:
        """Spindle rotation frequency in Hz."""
        if self.spindle_speed:
            return self.spindle_speed / 60.0
        return None
    
    def classify_regime(self) -> OperatingRegime:
        """
        [PROTOTYPE_LLM_MEMORY_V1] - Simple heuristic regime classification.
        Should be replaced with proper classification based on domain rules.
        """
        if self.operating_regime:
            return self.operating_regime
        
        # Heuristic based on axial depth
        if self.axial_depth is not None:
            if self.axial_depth > 2.0:
                return OperatingRegime.ROUGHING
            elif self.axial_depth > 0.5:
                return OperatingRegime.SEMI_FINISHING
            else:
                return OperatingRegime.FINISHING
        
        # Heuristic based on tool type
        if self.tool_type:
            if "drill" in self.tool_type.lower():
                return OperatingRegime.DRILLING
            if "tap" in self.tool_type.lower():
                return OperatingRegime.TAPPING
        
        return OperatingRegime.UNKNOWN
    
    def matches(
        self, 
        other: "CuttingContext",
        tolerance: Optional["ContextTolerance"] = None
    ) -> Tuple[bool, float]:
        """
        Check if this context matches another within tolerance.
        
        Returns:
            (is_match, similarity_score)
        
        [PROTOTYPE_LLM_MEMORY_V1] - Simple matching, to be refined.
        """
        if tolerance is None:
            tolerance = get_default_tolerance()
        
        score = 0.0
        max_score = 0.0
        
        # Exact matches (high weight)
        if self.machine_type and other.machine_type:
            max_score += 2.0
            if self.machine_type == other.machine_type:
                score += 2.0
        
        if self.tool_type and other.tool_type:
            max_score += 2.0
            if self.tool_type == other.tool_type:
                score += 2.0
        
        if self.workpiece_material and other.workpiece_material:
            max_score += 2.0
            if self.workpiece_material.lower() == other.workpiece_material.lower():
                score += 2.0
        
        # Numeric ranges
        if self.spindle_speed and other.spindle_speed:
            max_score += 1.0
            ratio = min(self.spindle_speed, other.spindle_speed) / max(self.spindle_speed, other.spindle_speed)
            if ratio >= (1.0 - tolerance.spindle_speed_pct):
                score += ratio
        
        if self.cutting_speed and other.cutting_speed:
            max_score += 1.0
            ratio = min(self.cutting_speed, other.cutting_speed) / max(self.cutting_speed, other.cutting_speed)
            if ratio >= (1.0 - tolerance.cutting_speed_pct):
                score += ratio
        
        if self.axial_depth and other.axial_depth:
            max_score += 1.0
            ratio = min(self.axial_depth, other.axial_depth) / max(self.axial_depth, other.axial_depth)
            if ratio >= (1.0 - tolerance.depth_pct):
                score += ratio
        
        # Regime match
        if self.classify_regime() == other.classify_regime():
            score += 1.0
            max_score += 1.0
        elif max_score > 0:
            max_score += 1.0  # Penalize regime mismatch
        
        if max_score == 0:
            return (True, 0.5)  # No context to compare, neutral
        
        similarity = score / max_score
        is_match = similarity >= tolerance.min_similarity
        
        return (is_match, similarity)


# [PROTOTYPE_LLM_MEMORY_V1] - Tolerance configuration
class ContextTolerance(BaseModel):
    """Tolerance thresholds for context matching."""
    spindle_speed_pct: float = 0.15  # ±15%
    cutting_speed_pct: float = 0.15  # ±15%
    depth_pct: float = 0.30  # ±30%
    min_similarity: float = 0.5  # Minimum score to be considered a match


# Default tolerance instance (reusable to avoid repeated instantiation)
_DEFAULT_TOLERANCE: Optional[ContextTolerance] = None

def get_default_tolerance() -> ContextTolerance:
    """Get cached default tolerance instance."""
    global _DEFAULT_TOLERANCE
    if _DEFAULT_TOLERANCE is None:
        _DEFAULT_TOLERANCE = ContextTolerance()
    return _DEFAULT_TOLERANCE


# [PROTOTYPE_LLM_MEMORY_V1] - Factory function
def extract_context_from_metadata(metadata: Dict[str, Any]) -> CuttingContext:
    """
    Extract CuttingContext from session/file metadata.
    
    [INTEGRATION_POINT] This should be extended to handle different
    metadata formats from various data sources.
    """
    casedata = metadata.get("casedata", {})
    if isinstance(casedata, dict) and casedata:
        casedata_ctx = dict(casedata)
        nested_ctx = casedata.get("cutting_context")
        if isinstance(nested_ctx, dict):
            casedata_ctx.update(nested_ctx)

        regime_value = casedata_ctx.get("operating_regime") or casedata_ctx.get("regime")
        operating_regime = None
        if regime_value is not None and str(regime_value).strip():
            try:
                operating_regime = OperatingRegime(str(regime_value).strip().lower())
            except ValueError:
                # The machine reported a mode we can't map to a known regime
                # (e.g. a raw numeric Operation_Mode). Leave it unset rather than
                # surfacing a meaningless "unknown" downstream.
                operating_regime = None

        extra = dict(casedata_ctx.get("extra") or {})
        for key in ("root", "operation_id", "operation", "sample_frequency", "source"):
            value = casedata_ctx.get(key, metadata.get(key))
            if value is not None:
                extra[key] = value

        return CuttingContext(
            tool_type=casedata_ctx.get("tool_type") or casedata_ctx.get("type"),
            tool_diameter=casedata_ctx.get("tool_diameter") or casedata_ctx.get("diameter"),
            num_teeth=casedata_ctx.get("num_teeth") or casedata_ctx.get("z"),
            tool_id=casedata_ctx.get("tool_id"),
            tool_length=casedata_ctx.get("tool_length") or casedata_ctx.get("length"),
            tool_material=casedata_ctx.get("tool_material"),
            spindle_speed=casedata_ctx.get("spindle_speed") or casedata_ctx.get("spindle"),
            feed_rate=casedata_ctx.get("feed_rate") or casedata_ctx.get("feed"),
            feed_per_tooth=casedata_ctx.get("feed_per_tooth") or casedata_ctx.get("fz"),
            cutting_speed=casedata_ctx.get("cutting_speed") or casedata_ctx.get("vc"),
            axial_depth=casedata_ctx.get("axial_depth") or casedata_ctx.get("ap"),
            radial_depth=casedata_ctx.get("radial_depth") or casedata_ctx.get("ae"),
            workpiece_material=casedata_ctx.get("workpiece_material") or casedata_ctx.get("material"),
            workpiece_hardness=casedata_ctx.get("workpiece_hardness"),
            machine_id=casedata_ctx.get("machine_id"),
            machine_type=casedata_ctx.get("machine_type"),
            operating_regime=operating_regime,
            extra=extra,
        )

    # Handle MATLAB-style machining metadata
    machining = metadata.get("machining", {})
    file_header = metadata.get("file_header", {})
    
    ctx = CuttingContext(
        # From machining params
        num_teeth=machining.get("z"),
        axial_depth=machining.get("ap"),
        radial_depth=machining.get("ae"),
        cutting_speed=machining.get("vc"),
        spindle_speed=machining.get("n"),
        feed_per_tooth=machining.get("f"),
        feed_rate=machining.get("vf"),
        tool_type=machining.get("type"),
        
        # Extra fields
        extra={
            "d": machining.get("d"),  # tool diameter
            "fg": machining.get("fg"),  # expected Fg
            "fp": machining.get("fp"),  # expected Fp
            "break": machining.get("break"),  # breakage indicator
            "sample_frequency": file_header.get("SampleFrequency"),
        }
    )
    
    # Set tool diameter if available
    if machining.get("d"):
        ctx.tool_diameter = float(machining.get("d"))
    
    return ctx

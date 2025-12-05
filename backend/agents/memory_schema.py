"""Memory system data models and schema definitions.

This module defines the core data structures for storing and retrieving
operator-annotated time-series memories with pattern-based indexing.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class PatternType(str, Enum):
    """Types of pattern keys for memory indexing."""
    RATIO = "ratio"              # Ratio of two variables/channels
    SPECTRAL_PEAK = "spectral_peak"  # Peak frequency in PSD
    BAND_POWER = "band_power"    # Power in frequency band
    SPIKE_RATE = "spike_rate"    # Transient event rate
    RMS_JUMP = "rms_jump"        # RMS discontinuity
    ANOMALY = "anomaly"          # Anomaly detection label
    CLUSTER = "cluster"          # Numeric feature cluster ID
    CUSTOM = "custom"            # User-defined pattern


class PatternKey(BaseModel):
    """A symbolic pattern key derived from time-series metrics."""
    pattern_type: PatternType
    key: str  # e.g., "RATIO_Fx_Fy:2-4", "PSD_PEAK_512Hz:>0.5", "CLUSTER_17"
    confidence: float = 1.0  # Pattern match confidence/strength
    source_metric: Optional[str] = None  # Which metric produced this pattern
    
    # Optional indexable components for PatternIndex
    condition: Optional[str] = None  # Operating condition (e.g., "cutting", "idle")
    machine_type: Optional[str] = None  # Machine/equipment type
    fault_type: Optional[str] = None  # Fault/anomaly classification
    channel: Optional[str] = None  # Primary channel this pattern relates to
    additional: Optional[Dict[str, Any]] = None  # Extra indexable fields


class TimeRange(BaseModel):
    """Time/index window specification."""
    i0: int  # Start sample index
    i1: int  # End sample index (exclusive)
    t0: float  # Start time (seconds)
    t1: float  # End time (seconds)
    fs: float  # Sampling frequency


class NumericMetrics(BaseModel):
    """Computed numeric features for a time-series window."""
    # Time-domain statistics (per channel)
    means: Dict[str, float] = Field(default_factory=dict)
    stds: Dict[str, float] = Field(default_factory=dict)
    rms: Dict[str, float] = Field(default_factory=dict)
    peaks: Dict[str, float] = Field(default_factory=dict)
    mins: Dict[str, float] = Field(default_factory=dict)
    skewness: Dict[str, float] = Field(default_factory=dict)
    kurtosis: Dict[str, float] = Field(default_factory=dict)
    
    # Cross-channel metrics
    channel_ratios: Dict[str, float] = Field(default_factory=dict)  # e.g., {"Fx/Fy": 2.3}
    cross_correlations: Dict[str, float] = Field(default_factory=dict)
    
    # Spectral features
    dominant_freqs: Dict[str, float] = Field(default_factory=dict)  # Dominant frequency per channel
    spectral_centroids: Dict[str, float] = Field(default_factory=dict)
    band_powers: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # {channel: {band: power}}
    psd_peaks: Dict[str, List[Dict[str, float]]] = Field(default_factory=dict)  # {channel: [{freq, mag}]}
    
    # Transient/event metrics
    spike_counts: Dict[str, int] = Field(default_factory=dict)
    envelope_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Signal quality
    snr_estimates: Dict[str, float] = Field(default_factory=dict)
    nan_percentages: Dict[str, float] = Field(default_factory=dict)


# Alias for backward compatibility
MetricsSummary = NumericMetrics


class OnlineAgentSnapshot(BaseModel):
    """Snapshot of online agent state at memory creation time."""
    model_version: str = "unknown"
    predictions: Dict[str, Any] = Field(default_factory=dict)
    anomaly_scores: Dict[str, float] = Field(default_factory=dict)
    feature_vector: Optional[List[float]] = None
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RLSnapshot(BaseModel):
    """RL agent signals (placeholder for future implementation)."""
    reward: Optional[float] = None
    action_recommendation: Optional[str] = None
    q_values: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryProvenance(BaseModel):
    """Computation and data provenance metadata."""
    compute_version: str = "1.0"
    nfft: Optional[int] = None
    window_type: Optional[str] = None
    detrend: Optional[bool] = None
    overlap: Optional[float] = None
    computation_params: Dict[str, Any] = Field(default_factory=dict)
    data_source: Optional[str] = None  # File path or session reference


class Memory(BaseModel):
    """Core memory record containing annotated time-series window with context."""
    # Identity
    id: str  # UUID
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "operator"  # operator ID or "system"
    
    # Time-series window
    time_range: TimeRange
    channels: List[str]  # Channel names included
    
    # Human annotation
    annotation_text: str  # Operator's note/description
    tags: List[str] = Field(default_factory=list)  # Free-form tags
    label: Optional[str] = None  # Classification label if applicable
    
    # Computed context
    metrics: NumericMetrics
    pattern_keys: List[PatternKey]  # Derived symbolic patterns
    numeric_vector: Optional[List[float]] = None  # Dense feature vector for ANN
    text_embedding: Optional[List[float]] = None  # Embedding of annotation_text
    
    # Agent snapshots
    online_snapshot: Optional[OnlineAgentSnapshot] = None
    rl_snapshot: Optional[RLSnapshot] = None
    
    # Provenance
    provenance: MemoryProvenance
    
    # Metadata
    visibility: str = "active"  # "active" | "archived" | "deleted"
    related_memory_ids: List[str] = Field(default_factory=list)  # Links to related memories
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra fields
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""
    pattern: Optional[PatternKey] = None  # Pattern to match
    partial_pattern_match: bool = True  # Allow partial pattern matches
    similar_to_metrics: Optional[NumericMetrics] = None  # For ANN similarity search
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    time_range_min: Optional[datetime] = None
    time_range_max: Optional[datetime] = None
    limit: int = 10
    offset: int = 0


class MemoryQueryResult(BaseModel):
    """A single memory returned from a query with relevance score."""
    memory: Memory
    relevance_score: float  # Combined score from pattern/ANN/semantic matching
    match_reasons: List[str]  # Human-readable match explanations
    pattern_matches: List[PatternKey]  # Which patterns matched


class MemoryFilter(BaseModel):
    """Filtering criteria for memory queries."""
    session_ids: Optional[List[str]] = None
    created_by: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    time_range_min: Optional[datetime] = None
    time_range_max: Optional[datetime] = None
    visibility: str = "active"


# API Request/Response Models

class CreateMemoryRequest(BaseModel):
    """Request to create a new memory from an annotated window."""
    session_id: str
    time_range: TimeRange
    annotation_text: str
    tags: Optional[List[str]] = None
    label: Optional[str] = None
    created_by: str = "operator"
    
    # Computation options
    compute_metrics: bool = True
    include_online_snapshot: bool = True
    include_rl_snapshot: bool = False
    compute_patterns: bool = True
    
    # Pattern generation options
    pattern_discretization: str = "auto"  # "auto" | "fine" | "coarse"
    pattern_rules: Optional[Dict[str, Any]] = None


class CreateMemoryResponse(BaseModel):
    """Response after creating a memory."""
    memory_id: str
    pattern_keys: List[PatternKey]
    metrics_summary: Dict[str, Any]  # Brief summary of computed metrics
    message: str = "Memory created successfully"


class QueryMemoriesRequest(BaseModel):
    """Request to query memories by pattern/vector/semantic similarity."""
    # Query source (provide at least one)
    session_id: Optional[str] = None  # Query using a live session window
    time_range: Optional[TimeRange] = None  # If session_id provided
    query_text: Optional[str] = None  # Semantic text query
    query_vector: Optional[List[float]] = None  # Pre-computed numeric vector
    pattern_keys: Optional[List[str]] = None  # Explicit pattern keys to match
    
    # Retrieval parameters
    top_k: int = 10
    filters: Optional[MemoryFilter] = None
    use_pattern_matching: bool = True
    use_ann_matching: bool = True
    use_semantic_matching: bool = True
    
    # Ranking/boosting
    boost_pattern_matches: float = 2.0  # Multiplier for exact pattern matches
    recency_weight: float = 0.1  # Weight for recent memories
    
    # Analysis options
    include_llm_analysis: bool = False  # Whether to invoke LLM analysis
    analysis_prompt: Optional[str] = None  # Custom prompt for LLM


class QueryMemoriesResponse(BaseModel):
    """Response from memory query."""
    results: List[MemoryQueryResult]
    query_metrics: Optional[NumericMetrics] = None  # Metrics computed for query window
    query_patterns: Optional[List[PatternKey]] = None
    llm_analysis: Optional[str] = None  # LLM-generated analysis if requested
    total_matches: int
    retrieval_time_ms: float


class ListMemoriesRequest(BaseModel):
    """Request to list/browse memories with filtering."""
    filters: Optional[MemoryFilter] = None
    sort_by: str = "created_at"  # "created_at" | "relevance" | "session_id"
    sort_order: str = "desc"  # "asc" | "desc"
    offset: int = 0
    limit: int = 50


class ListMemoriesResponse(BaseModel):
    """Response for list memories."""
    memories: List[Memory]
    total_count: int
    offset: int
    limit: int


class DeleteMemoryRequest(BaseModel):
    """Request to soft-delete or archive a memory."""
    memory_id: str
    hard_delete: bool = False  # If False, sets visibility="deleted"


class UpdateMemoryRequest(BaseModel):
    """Request to update memory metadata (annotation, tags, label)."""
    memory_id: str
    annotation_text: Optional[str] = None
    tags: Optional[List[str]] = None
    label: Optional[str] = None
    visibility: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

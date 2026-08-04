"""Agent configuration and default registry.

Keep configuration simple and environment-friendly. Swap providers via environment
variables or by modifying this module during deployment.

# ===========================================================================
# [PROTOTYPE_LLM_MEMORY_V1] - Extended configuration for memory system
# ===========================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Load .env from repo root so `uvicorn backend.app:app` picks up SINDIT_ENABLED,
# NEO4J_*, etc. without the user having to export them. Silent no-op if
# python-dotenv is missing or no .env file exists.
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore

    _load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ============================================================================
# Agent Registry
# ============================================================================

AGENT_REGISTRY = {
    "compute": "ComputeAgent",
    "llm.rag": "LLMAgent",
    "online": "OnlineAgent",
    "retriever": "RetrieverAgent",
    "monitoring": "MonitoringAgent",
    "analytics": "AnalyticsAgent",
    "stoppage": "StoppagePredictor",
}


# ============================================================================
# Embedding & Index Configuration
# ============================================================================

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_ON_DISK = os.environ.get("FAISS_INDEX_ON_DISK", "false").lower() == "true"


# ============================================================================
# LLM Configuration
# ============================================================================

# Provider: "groq" (cloud, default) or "ollama" (local)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

# --- Ollama (local) ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30.0"))

# --- Groq (cloud) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "30.0"))

# --- LLM output guardrails [LLM_GUARDRAILS_V1] ---
# Tier-1 deterministic output rail over operator-facing LLM explanations.
# Defaults to enabled when explanations are generated; can be disabled
# independently. Tier 2 (semantic/NLI judge) is a separate, unimplemented hook.
LLM_GUARDRAILS_ENABLED = os.environ.get("LLM_GUARDRAILS_ENABLED", "true").lower() == "true"


# ============================================================================
# Storage Configuration
# ============================================================================

DATA_DIR = os.environ.get("REED_DATA_DIR", "data")
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", os.path.join(DATA_DIR, "memories.db"))
PATTERN_INDEX_PATH = os.environ.get("PATTERN_INDEX_PATH", os.path.join(DATA_DIR, "pattern_index.json"))
PATTERN_PRIORS_PATH = os.environ.get("PATTERN_PRIORS_PATH", os.path.join(DATA_DIR, "pattern_priors.json"))
MODEL_CONFIDENCE_PATH = os.environ.get(
    "MODEL_CONFIDENCE_PATH",
    os.path.join(os.path.dirname(PATTERN_PRIORS_PATH), "model_confidence.json"),
)
BOOTSTRAP_PATTERN_PRIORS = os.environ.get("BOOTSTRAP_PATTERN_PRIORS", "false").lower() == "true"

# Storage backend: "neo4j" (default) or "sqlite"
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "neo4j")


# ============================================================================
# Neo4j Configuration (used when STORAGE_BACKEND="neo4j")
# ============================================================================

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "changeme")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_CONNECT_TIMEOUT_S = float(os.environ.get("NEO4J_CONNECT_TIMEOUT_S", "5.0"))
NEO4J_MAX_POOL_SIZE = int(os.environ.get("NEO4J_MAX_POOL_SIZE", "50"))
NEO4J_TX_RETRY_S = float(os.environ.get("NEO4J_TX_RETRY_S", "5.0"))
NEO4J_GRAPH_OUTBOX_PATH = os.environ.get(
    "NEO4J_GRAPH_OUTBOX_PATH",
    os.path.join(DATA_DIR, "neo4j_graph_outbox.jsonl"),
)


# ============================================================================
# SINDIT Configuration (optional digital-twin context provider)
# ============================================================================

SINDIT_ENABLED = os.environ.get("SINDIT_ENABLED", "false").lower() == "true"
SINDIT_API_URL = os.environ.get("SINDIT_API_URL", "http://localhost:9017")
SINDIT_TIMEOUT_S = float(os.environ.get("SINDIT_TIMEOUT_S", "5.0"))


# ============================================================================
# Significance Thresholds
# ============================================================================

@dataclass
class SignificanceThresholds:
    """Configurable thresholds for significance scoring."""
    store_threshold: float = 0.3
    alert_threshold: float = 0.6
    critical_threshold: float = 0.85
    
    # Rule weights
    weight_classical_alert: float = 0.4
    weight_harmonic_alert: float = 0.1
    weight_pattern_rule: float = 0.25
    weight_anomaly_deviation: float = 0.2
    weight_historical_prior: float = 0.15
    
    chatter_ratio_threshold: float = 5.0
    anomaly_score_threshold: float = 0.7
    # Statistical-anomaly z threshold. 4.0 (was 3.0) so ordinary 3σ blips on a
    # high-throughput stream no longer trigger a store/alert on their own.
    anomaly_z_threshold: float = 4.0
    # Require ≥2 detector rules for an ALERT — a single lone rule below the
    # critical band caps at STORE. Opt-in (default off): it also suppresses
    # legitimate single-rule alerts (pattern corroboration within one rule,
    # prior-boosted). Enable with SIG_REQUIRE_MULTI_RULE_ALERT=1 to cut
    # single-model false alerts. A lone rule reaching critical always alerts.
    require_multi_rule_alert: bool = False

    # Evidence-count damping: low-support priors are pulled toward neutral (0.5).
    prior_evidence_damping_k: float = 20.0

    # Sub-neutral prior damping (ISS-13): allow sustained operator dismissals to
    # damp scores below neutral. Off by default; guarded by an evidence minimum,
    # a factor floor, and critical-band immunity (see scorer.SignificanceConfig).
    prior_allow_subneutral: bool = False
    prior_factor_floor: float = 0.85
    prior_subneutral_min_evidence: float = 3.0

    @classmethod
    def from_env(cls) -> "SignificanceThresholds":
        """Load thresholds from environment variables."""
        return cls(
            store_threshold=float(os.environ.get("SIG_STORE_THRESHOLD", "0.3")),
            alert_threshold=float(os.environ.get("SIG_ALERT_THRESHOLD", "0.6")),
            critical_threshold=float(os.environ.get("SIG_CRITICAL_THRESHOLD", "0.85")),
            weight_classical_alert=float(os.environ.get("SIG_WEIGHT_CLASSICAL", "0.4")),
            weight_harmonic_alert=float(os.environ.get("SIG_WEIGHT_HARMONIC", "0.1")),
            weight_pattern_rule=float(os.environ.get("SIG_WEIGHT_PATTERN", "0.25")),
            weight_anomaly_deviation=float(os.environ.get("SIG_WEIGHT_ANOMALY", "0.2")),
            weight_historical_prior=float(os.environ.get("SIG_WEIGHT_PRIOR", "0.15")),
            chatter_ratio_threshold=float(os.environ.get("CHATTER_RATIO_THRESHOLD", "5.0")),
            anomaly_score_threshold=float(os.environ.get("ANOMALY_SCORE_THRESHOLD", "0.7")),
            anomaly_z_threshold=float(os.environ.get("SIG_ANOMALY_Z_THRESHOLD", "4.0")),
            require_multi_rule_alert=os.environ.get("SIG_REQUIRE_MULTI_RULE_ALERT", "0").strip().lower() in ("1", "true", "yes"),
            prior_evidence_damping_k=float(os.environ.get("SIG_PRIOR_DAMPING_K", "20.0")),
            prior_allow_subneutral=os.environ.get("SIG_PRIOR_ALLOW_SUBNEUTRAL", "0").strip().lower() in ("1", "true", "yes"),
            prior_factor_floor=float(os.environ.get("SIG_PRIOR_FACTOR_FLOOR", "0.85")),
            prior_subneutral_min_evidence=float(os.environ.get("SIG_PRIOR_SUBNEUTRAL_MIN_EVIDENCE", "3.0")),
        )


# ============================================================================
# Memory System Configuration
# ============================================================================

@dataclass
class MemorySystemConfig:
    """Complete configuration for the memory system."""
    # Storage
    storage_backend: str = field(default_factory=lambda: STORAGE_BACKEND)  # "sqlite" | "neo4j"
    db_path: str = field(default_factory=lambda: MEMORY_DB_PATH)
    pattern_index_path: str = field(default_factory=lambda: PATTERN_INDEX_PATH)
    pattern_priors_path: str = field(default_factory=lambda: PATTERN_PRIORS_PATH)
    model_confidence_path: str = field(default_factory=lambda: MODEL_CONFIDENCE_PATH)
    bootstrap_pattern_priors: bool = field(default_factory=lambda: BOOTSTRAP_PATTERN_PRIORS)
    enable_ann: bool = True
    enable_embeddings: bool = False

    # Neo4j (only used when storage_backend="neo4j")
    neo4j_uri: str = field(default_factory=lambda: NEO4J_URI)
    neo4j_username: str = field(default_factory=lambda: NEO4J_USERNAME)
    neo4j_password: str = field(default_factory=lambda: NEO4J_PASSWORD)
    neo4j_database: str = field(default_factory=lambda: NEO4J_DATABASE)
    neo4j_connect_timeout_s: float = field(default_factory=lambda: NEO4J_CONNECT_TIMEOUT_S)
    neo4j_max_pool_size: int = field(default_factory=lambda: NEO4J_MAX_POOL_SIZE)
    neo4j_tx_retry_s: float = field(default_factory=lambda: NEO4J_TX_RETRY_S)
    neo4j_graph_outbox_path: str = field(default_factory=lambda: NEO4J_GRAPH_OUTBOX_PATH)

    # SINDIT (optional digital-twin context provider)
    sindit_enabled: bool = field(default_factory=lambda: SINDIT_ENABLED)
    sindit_api_url: str = field(default_factory=lambda: SINDIT_API_URL)
    sindit_timeout_s: float = field(default_factory=lambda: SINDIT_TIMEOUT_S)
    
    # LLM
    llm_provider: str = field(default_factory=lambda: LLM_PROVIDER)
    ollama_url: str = field(default_factory=lambda: OLLAMA_URL)
    ollama_model: str = field(default_factory=lambda: OLLAMA_MODEL)
    ollama_timeout: float = field(default_factory=lambda: OLLAMA_TIMEOUT)
    groq_api_key: str = field(default_factory=lambda: GROQ_API_KEY)
    groq_api_url: str = field(default_factory=lambda: GROQ_API_URL)
    groq_model: str = field(default_factory=lambda: GROQ_MODEL)
    groq_timeout: float = field(default_factory=lambda: GROQ_TIMEOUT)
    
    # Thresholds
    thresholds: SignificanceThresholds = field(default_factory=SignificanceThresholds.from_env)
    
    # Behavior
    generate_explanations: bool = False  # Off by default — requires a reachable configured LLM provider
    # Tier-1 deterministic output rail over LLM explanations. Effective only
    # when generate_explanations is on (no LLM text → nothing to guard).
    llm_guardrails_enabled: bool = field(default_factory=lambda: LLM_GUARDRAILS_ENABLED)
    require_llm: bool = False
    dispatch_alerts: bool = True
    top_k_similar: int = 5
    # When True (production default) the orchestrator builds a classical
    # seed model by reading `data/casedata/*.csv`, which can take tens of
    # seconds. Tests that don't need classical scoring should disable it.
    use_classical_models: bool = True
    # When True and no cached seed model exists, the ~30 s casedata
    # training runs on a background thread so orchestrator init does
    # not block. Detector omits the seed-model score until training
    # completes. Off by default to preserve deterministic production
    # startup semantics.
    lazy_seed_training: bool = False

    @classmethod
    def from_env(cls) -> "MemorySystemConfig":
        """Load complete configuration from environment."""
        return cls(
            storage_backend=STORAGE_BACKEND,
            db_path=MEMORY_DB_PATH,
            pattern_index_path=PATTERN_INDEX_PATH,
            pattern_priors_path=PATTERN_PRIORS_PATH,
            model_confidence_path=MODEL_CONFIDENCE_PATH,
            bootstrap_pattern_priors=BOOTSTRAP_PATTERN_PRIORS,
            enable_ann=os.environ.get("ENABLE_ANN_INDEX", "true").lower() == "true",
            enable_embeddings=os.environ.get("ENABLE_EMBEDDINGS", "false").lower() == "true",
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD,
            neo4j_database=NEO4J_DATABASE,
            neo4j_connect_timeout_s=NEO4J_CONNECT_TIMEOUT_S,
            neo4j_max_pool_size=NEO4J_MAX_POOL_SIZE,
            neo4j_tx_retry_s=NEO4J_TX_RETRY_S,
            neo4j_graph_outbox_path=NEO4J_GRAPH_OUTBOX_PATH,
            sindit_enabled=SINDIT_ENABLED,
            sindit_api_url=SINDIT_API_URL,
            sindit_timeout_s=SINDIT_TIMEOUT_S,
            llm_provider=LLM_PROVIDER,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            ollama_timeout=OLLAMA_TIMEOUT,
            groq_api_key=GROQ_API_KEY,
            groq_api_url=GROQ_API_URL,
            groq_model=GROQ_MODEL,
            groq_timeout=GROQ_TIMEOUT,
            thresholds=SignificanceThresholds.from_env(),
            generate_explanations=os.environ.get("GENERATE_EXPLANATIONS", "false").lower() == "true",
            llm_guardrails_enabled=LLM_GUARDRAILS_ENABLED,
            require_llm=os.environ.get("REQUIRE_LLM", "false").lower() == "true",
            dispatch_alerts=os.environ.get("DISPATCH_ALERTS", "true").lower() == "true",
            top_k_similar=int(os.environ.get("TOP_K_SIMILAR", "5")),
            use_classical_models=os.environ.get(
                "USE_CLASSICAL_MODELS", "true"
            ).lower() == "true",
            lazy_seed_training=os.environ.get(
                "LAZY_SEED_TRAINING", "false"
            ).lower() == "true",
        )


# Singleton config instance
_config: Optional[MemorySystemConfig] = None


def get_config() -> MemorySystemConfig:
    """Get the global memory system configuration."""
    global _config
    if _config is None:
        _config = MemorySystemConfig.from_env()
    return _config


def reset_config():
    """Reset config (for testing)."""
    global _config
    _config = None

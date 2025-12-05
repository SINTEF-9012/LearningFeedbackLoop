"""Memory System Implementation Plan

This document outlines the step-by-step implementation plan for the memory system.
"""

# ============================================================================
# PHASE 1: CORE INFRASTRUCTURE (Week 1)
# ============================================================================

"""
1.1 Create Memory Storage Backend
----------------------------------
Files to create:
- backend/agents/memory_store.py: Core storage abstraction
- backend/data/memories.db: SQLite database file (gitignored)

Components:
- MemoryStore class:
  - save_memory(memory: Memory) -> str
  - get_memory(memory_id: str) -> Memory
  - list_memories(filters: MemoryFilter, offset, limit) -> List[Memory]
  - update_memory(memory_id, updates: dict) -> bool
  - delete_memory(memory_id, hard_delete) -> bool
  
Database schema (SQLite):
  CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    created_by TEXT NOT NULL,
    time_range JSON NOT NULL,
    channels JSON NOT NULL,
    annotation_text TEXT NOT NULL,
    tags JSON,
    label TEXT,
    metrics JSON NOT NULL,
    pattern_keys JSON NOT NULL,
    numeric_vector BLOB,
    text_embedding BLOB,
    online_snapshot JSON,
    rl_snapshot JSON,
    provenance JSON NOT NULL,
    visibility TEXT DEFAULT 'active',
    related_memory_ids JSON,
    metadata JSON,
    UNIQUE(session_id, time_range)
  );
  
  CREATE INDEX idx_session_id ON memories(session_id);
  CREATE INDEX idx_created_at ON memories(created_at);
  CREATE INDEX idx_created_by ON memories(created_by);
  CREATE INDEX idx_visibility ON memories(visibility);
  CREATE INDEX idx_label ON memories(label);

Dependencies:
- sqlite3 (standard library)
- json for serialization

Testing:
- Unit tests for CRUD operations
- Test filtering and pagination
- Test concurrent writes
"""

"""
1.2 Create Pattern Index
-------------------------
Files to create:
- backend/agents/pattern_index.py: Pattern key indexing

Components:
- PatternIndex class:
  - insert(memory_id: str, patterns: List[PatternKey]) -> None
  - lookup(pattern_key: str) -> Set[str]  # memory IDs
  - lookup_multi(pattern_keys: List[str]) -> Dict[str, Set[str]]
  - remove(memory_id: str) -> None
  - save() -> None  # persist to disk
  - load() -> None  # load from disk

Implementation:
  # In-memory inverted index
  {
    "RATIO_Fx_Fy:2-4": {"mem_id_1", "mem_id_2", ...},
    "PSD_PEAK_512Hz:>0.5": {"mem_id_1", "mem_id_3", ...},
    ...
  }
  
  # Persistence: JSON file backend/data/pattern_index.json
  # On startup: load into memory
  # On update: append-only log + periodic full dump

Dependencies:
- json for persistence
- collections.defaultdict

Testing:
- Test insert/lookup with various pattern types
- Test persistence and reload
- Benchmark lookup speed (should be <1ms for 10K memories)
"""

"""
1.3 Create ANN Index Manager
-----------------------------
Files to create:
- backend/agents/ann_index.py: FAISS index management

Components:
- ANNIndex class:
  - build(vectors: np.ndarray, ids: List[str]) -> None
  - insert(vector: np.ndarray, memory_id: str) -> None
  - search(query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]
  - save(path: str) -> None
  - load(path: str) -> None

Implementation:
  # Two FAISS indices:
  # 1. Numeric feature vectors (dimension: ~50-100)
  # 2. Text embeddings (dimension: 384 for all-MiniLM-L6-v2)
  
  # Start with IndexFlatL2 (exact search) for prototyping
  # Later upgrade to IndexIVFFlat or HNSW for scale
  
  # Keep separate ID mapping: faiss_idx -> memory_id
  
Files:
- backend/data/ann_numeric.index
- backend/data/ann_text.index
- backend/data/ann_id_mapping.json

Dependencies:
- faiss-cpu (already in requirements.txt)
- numpy

Testing:
- Test insert and search with sample vectors
- Test persistence and reload
- Benchmark search speed (should be <10ms for 10K vectors)
"""

# ============================================================================
# PHASE 2: METRIC COMPUTATION (Week 1-2)
# ============================================================================

"""
2.1 Extend Computation Module
------------------------------
Files to modify:
- backend/computation.py: Add comprehensive metric extractors

New functions to add:
- compute_time_domain_metrics(window_data: Dict[str, np.ndarray]) -> Dict
  Returns: means, stds, rms, peaks, mins, skewness, kurtosis per channel
  
- compute_cross_channel_metrics(window_data: Dict[str, np.ndarray]) -> Dict
  Returns: channel_ratios, cross_correlations
  
- compute_spectral_metrics(window_data: Dict[str, np.ndarray], fs: float) -> Dict
  Returns: dominant_freqs, spectral_centroids, band_powers, psd_peaks
  Uses existing compute_rfft_multichannel under the hood
  
- compute_transient_metrics(window_data: Dict[str, np.ndarray], fs: float) -> Dict
  Returns: spike_counts, envelope_stats
  
- compute_signal_quality(window_data: Dict[str, np.ndarray]) -> Dict
  Returns: snr_estimates, nan_percentages

- compute_all_metrics(session: Dict, time_range: TimeRange) -> NumericMetrics
  Orchestrator function that calls all above and returns populated NumericMetrics

Dependencies:
- scipy.stats (skew, kurtosis)
- scipy.signal (find_peaks, hilbert for envelope)
- existing compute_rfft_multichannel

Testing:
- Unit tests with synthetic signals (sine, noise, chirp)
- Validate metric values against known ground truth
- Performance test: should compute metrics for 1000-sample window in <50ms
"""

"""
2.2 Pattern Generation Logic
-----------------------------
Files to create:
- backend/agents/pattern_generator.py: Derive patterns from metrics

Components:
- PatternGenerator class:
  - generate(metrics: NumericMetrics, options: dict) -> List[PatternKey]
  - _ratio_patterns(metrics) -> List[PatternKey]
  - _spectral_peak_patterns(metrics) -> List[PatternKey]
  - _band_power_patterns(metrics) -> List[PatternKey]
  - _spike_rate_patterns(metrics) -> List[PatternKey]
  - _anomaly_patterns(metrics) -> List[PatternKey]

Pattern generation rules (configurable):
  Ratio patterns:
    - For each pair of channels (A, B):
      ratio = metrics.rms[A] / metrics.rms[B]
      if ratio > 5: RATIO_A_B:>5
      elif ratio > 2: RATIO_A_B:2-5
      elif ratio > 0.5: RATIO_A_B:0.5-2
      else: RATIO_A_B:<0.5
  
  Spectral peak patterns:
    - For each channel, for each PSD peak:
      if peak['mag'] > threshold:
        freq_bin = round(peak['freq'] / bin_width) * bin_width
        SPECTRAL_PEAK_{freq_bin}Hz:>{mag_threshold}
  
  Band power patterns:
    - Define standard bands (e.g., 0-100Hz, 100-500Hz, 500-1000Hz)
    - For each band, for each channel:
      if band_power > threshold:
        BAND_POWER_{channel}_{band}:high
  
  Spike rate patterns:
    - Count spikes per second
    - if spike_rate > 10/s: SPIKE_RATE:>10
  
  Anomaly patterns:
    - If online agent anomaly_score > 0.8: ANOMALY_HIGH
    - If anomaly_score > 0.5: ANOMALY_MEDIUM

Configuration file:
- backend/agents/pattern_rules.yaml or .json
  Defines thresholds, discretization, and rules

Dependencies:
- pyyaml (optional, for config)

Testing:
- Test pattern generation with known metric values
- Verify discretization and bucketing
- Test pattern count explosion prevention (should generate <20 patterns per memory)
"""

"""
2.3 Feature Vector Engineering
-------------------------------
Files to create:
- backend/agents/feature_engineering.py: Create dense numeric vectors

Components:
- FeatureVectorizer class:
  - fit(memories: List[Memory]) -> None  # Learn normalization params
  - transform(metrics: NumericMetrics) -> np.ndarray  # Fixed-dim vector
  - fit_transform(memories) -> np.ndarray
  - save(path) / load(path)

Feature vector composition (~80 dimensions):
  - Time-domain: 8 stats × N channels = 16 (for 2 channels)
  - Cross-channel: 3 ratios + 3 correlations = 6
  - Spectral: 4 features × N channels = 8
  - Transients: 2 features × N channels = 4
  - Signal quality: 2 × N channels = 4
  - PSD band powers: 5 bands × N channels = 10
  Total: ~48 base features
  
  Add derived features:
  - log transforms of power metrics
  - ratios of ratios
  - PCA components (optional, for dimensionality reduction)

Normalization:
  - StandardScaler per feature (zero mean, unit variance)
  - Store scaler params in backend/data/feature_scaler.pkl

Dependencies:
- numpy
- scikit-learn (StandardScaler, optional PCA)
- joblib (for serialization)

Testing:
- Test with multiple sample windows
- Verify vector dimensionality is fixed
- Test normalization (mean≈0, std≈1)
"""

# ============================================================================
# PHASE 3: API ENDPOINTS (Week 2)
# ============================================================================

"""
3.1 Create Memory API Router
-----------------------------
Files to create:
- backend/agents/memory_api.py: FastAPI router with all endpoints

Components:
- FastAPI APIRouter instance
- Endpoint handlers (see memory_api_contract.py for specs):
  - POST /memories (create_memory)
  - POST /memories/query (query_memories)
  - GET /memories (list_memories)
  - GET /memories/{id} (get_memory)
  - PATCH /memories/{id} (update_memory)
  - DELETE /memories/{id} (delete_memory)
  - POST /memories/reindex (reindex_memories)
  - GET /memories/stats (get_stats)

Dependencies on other modules:
- memory_store.MemoryStore
- pattern_index.PatternIndex
- ann_index.ANNIndex
- computation.compute_all_metrics
- pattern_generator.PatternGenerator
- feature_engineering.FeatureVectorizer
- ingest.Ingestor (for text embeddings)
- llm_rag.LLMAgent (for analysis)

Error handling:
- Wrap all operations in try/except
- Return appropriate HTTP status codes
- Log errors with traceback

Testing:
- Integration tests using TestClient
- Test all CRUD operations
- Test query with various filter combinations
"""

"""
3.2 Integrate with Main App
----------------------------
Files to modify:
- backend/app.py: Mount memory router

Changes:
  from agents.memory_api import router as memory_router
  app.include_router(memory_router, prefix="/memories", tags=["memories"])

Startup tasks:
  @app.on_event("startup")
  async def startup_memory_system():
      # Initialize memory store
      # Load pattern index
      # Load ANN indices
      # Load feature scaler
      # Register background reindexing scheduler (optional)
"""

"""
3.3 Update LLMAgent for Memory Integration
-------------------------------------------
Files to modify:
- backend/agents/llm_rag.py

Changes:
- Add action="analyze_with_memories" to handle_request
- Accept retrieved_memories parameter in handle_request
- Extend _build_prompt to format memory context:
  
  def _build_rag_memory_prompt(self, query_metrics, retrieved_memories, question):
      memory_summaries = []
      for i, result in enumerate(retrieved_memories):
          mem = result.memory
          summary = f"Memory {i+1} (ID: {mem.id[:8]}...):\n"
          summary += f"  Session: {mem.session_id}, Time: {mem.time_range.t0:.2f}-{mem.time_range.t1:.2f}s\n"
          summary += f"  Label: {mem.label}, Tags: {', '.join(mem.tags)}\n"
          summary += f"  Operator note: {mem.annotation_text}\n"
          summary += f"  Key metrics: RMS={mem.metrics.rms}, Dominant freq={mem.metrics.dominant_freqs}\n"
          summary += f"  Pattern matches: {', '.join([p.key for p in result.pattern_matches])}\n"
          summary += f"  Relevance score: {result.relevance_score:.2f}\n"
          memory_summaries.append(summary)
      
      memories_text = "\n".join(memory_summaries)
      
      prompt = f'''You are a time-series analysis assistant. Compare the current window to historical events.

Current window metrics:
{json.dumps(query_metrics.dict(), indent=2)}

Historical memories (most relevant first):
{memories_text}

Question: {question}

Provide a concise analysis citing memory IDs and suggesting actions.'''
      return prompt

Testing:
- Test prompt formatting with sample memories
- Verify LLM responses cite memory IDs correctly
"""

# ============================================================================
# PHASE 4: ONLINE AGENT INTEGRATION (Week 3)
# ============================================================================

"""
4.1 Add OnlineAgent State Export
---------------------------------
Files to modify:
- backend/agents/online_agent.py

Changes:
- Add export_state(session_id: str) -> OnlineAgentSnapshot method:
  
  def export_state(self, session_id: str) -> OnlineAgentSnapshot:
      # Return current model state for given session
      # Includes predictions, anomaly scores, feature vector
      if self._model is None:
          return OnlineAgentSnapshot(model_version="not_started")
      
      # Get latest predictions (may need to track per-session state)
      snapshot = OnlineAgentSnapshot(
          model_version=self.model_name,
          predictions=self._last_predictions.get(session_id, {}),
          anomaly_scores=self._anomaly_scores.get(session_id, {}),
          feature_vector=self._last_features.get(session_id, []),
          confidence=self._confidences.get(session_id),
      )
      return snapshot

- Track per-session state in _run() loop
- Store last N predictions/features in deques per session

Testing:
- Test export_state returns valid snapshot
- Verify snapshot serialization to JSON
"""

"""
4.2 Publish Memory Events to Event Bus
---------------------------------------
Files to modify:
- backend/agents/memory_api.py

Changes:
- After successful memory creation, publish event:
  
  from events import publish_feature
  
  await publish_feature(session_id, {
      "type": "memory_created",
      "memory_id": memory.id,
      "session_id": session_id,
      "label": memory.label,
      "pattern_keys": [p.key for p in memory.pattern_keys],
  })

- OnlineAgent can subscribe to "memory_created" events for feedback learning
  (optional, Phase 5)
"""

# ============================================================================
# PHASE 5: ADVANCED FEATURES (Week 3-4)
# ============================================================================

"""
5.1 Clustering and Automatic Pattern Discovery
-----------------------------------------------
Files to create:
- backend/agents/memory_clustering.py

Components:
- MemoryClusterer class:
  - fit(memories: List[Memory]) -> Dict[int, List[str]]  # cluster_id -> memory_ids
  - assign_cluster(memory: Memory) -> int
  - get_cluster_patterns(cluster_id: int) -> List[PatternKey]

Algorithm:
- Use KMeans or HDBSCAN on numeric_vectors
- Assign cluster IDs as pattern keys: CLUSTER_{id}
- Periodically recluster (daily/weekly)
- Incremental assignment for new memories

Dependencies:
- scikit-learn (KMeans, HDBSCAN)
- joblib (persistence)

Trigger:
- POST /memories/reindex with recluster=true
- Background scheduled job

Testing:
- Test clustering on synthetic data
- Verify cluster stability
"""

"""
5.2 Explainability and Pattern Importance
------------------------------------------
Files to create:
- backend/agents/memory_explainer.py

Components:
- MemoryExplainer class:
  - explain_match(query_memory, matched_memory) -> str
  - feature_importance(memory) -> Dict[str, float]

Features:
- SHAP-style feature importance for match scoring
- Textual explanation of why memories matched
- Pattern frequency statistics

Dependencies:
- shap (optional)

Integration:
- Include explanation in QueryMemoriesResponse.match_reasons
"""

"""
5.3 Memory Lifecycle Management
--------------------------------
Files to modify:
- backend/agents/memory_api.py

Features:
- Auto-archive old memories (>6 months, configurable)
- Deduplication: detect near-duplicate memories and merge
- Memory quality scoring: flag low-quality annotations
- Batch import/export for backup

Endpoints:
- POST /memories/merge (merge duplicate memories)
- POST /memories/export (export to JSON/CSV)
- POST /memories/import (bulk import)

Background jobs:
- Daily deduplication check
- Monthly archival sweep
"""

"""
5.4 UI/CLI Tools
----------------
Files to create:
- scripts/memory_cli.py: CLI for operators

Commands:
- memory create <session_id> <i0> <i1> --annotation "..." --tags vibration
- memory query <session_id> <i0> <i1> --top-k 5 --analyze
- memory list --tags vibration --limit 10
- memory export --session <id> --output memories.json
- memory stats

Dependencies:
- click or argparse
- requests (for API calls)

Distribution:
- Install as CLI command: pip install -e .
- Add entry_point in setup.py
"""

# ============================================================================
# TESTING STRATEGY
# ============================================================================

"""
Unit Tests:
-----------
- backend/tests/test_memory_store.py: CRUD operations
- backend/tests/test_pattern_index.py: Index operations
- backend/tests/test_ann_index.py: ANN search
- backend/tests/test_pattern_generator.py: Pattern derivation
- backend/tests/test_feature_engineering.py: Vector generation
- backend/tests/test_computation.py: Metric computation

Integration Tests:
------------------
- backend/tests/test_memory_api.py: Full API workflow
- Create memory -> query -> verify results
- Test with real session data
- Test error cases (missing session, invalid ranges)

End-to-End Tests:
-----------------
- scripts/test_memory_workflow.py:
  1. Start server
  2. Upload session
  3. Create memories with annotations
  4. Query memories and verify retrieval
  5. Test LLM analysis

Performance Tests:
------------------
- Benchmark metric computation speed
- Benchmark pattern generation
- Benchmark ANN search with 1K, 10K, 100K memories
- Memory footprint tests

Target metrics:
- Memory creation: <500ms per memory
- Query (top-10): <100ms with 10K memories
- Storage: <1MB per memory (excluding raw data)
"""

# ============================================================================
# DEPENDENCIES TO ADD
# ============================================================================

"""
requirements.txt additions:
---------------------------
scipy>=1.9.0  # For stats, signal processing
scikit-learn>=1.2.0  # For feature scaling, clustering
joblib>=1.2.0  # For model persistence
pyyaml>=6.0  # For pattern rules config (optional)
click>=8.1.0  # For CLI tools (optional)

Already present:
- numpy
- faiss-cpu
- sentence-transformers
- pydantic
- fastapi
"""

# ============================================================================
# DEPLOYMENT CHECKLIST
# ============================================================================

"""
Before Production:
------------------
[ ] All unit tests passing
[ ] Integration tests passing
[ ] Performance benchmarks meet targets
[ ] Data persistence layer tested (SQLite -> Postgres for production)
[ ] Index persistence tested (crash recovery)
[ ] API security review (auth, rate limiting)
[ ] Error handling and logging comprehensive
[ ] Documentation updated (API docs, operator guides)
[ ] Backup/restore procedures documented
[ ] Monitoring and alerting configured
[ ] Memory retention policy defined
[ ] GDPR/data privacy compliance reviewed (if applicable)

Production Upgrades:
--------------------
- Replace SQLite with PostgreSQL for concurrency
- Replace in-memory PatternIndex with Redis
- Upgrade FAISS to HNSW index for scale
- Add distributed task queue (Celery) for background jobs
- Add authentication and authorization
- Add audit logging
- Add Prometheus metrics and Grafana dashboards
"""

# ============================================================================
# TIMELINE SUMMARY
# ============================================================================

"""
Week 1:
-------
✓ Schema and API contract (complete)
- Memory storage backend (SQLite)
- Pattern index
- ANN index manager
- Basic metric computation
- Pattern generation logic

Week 2:
-------
- Feature vector engineering
- API endpoints implementation
- LLMAgent integration
- Basic testing

Week 3:
-------
- OnlineAgent integration
- Event bus integration
- Clustering and pattern discovery
- CLI tools

Week 4:
-------
- Advanced features (explainability, lifecycle)
- Performance optimization
- Comprehensive testing
- Documentation
- Deployment preparation

Total: ~4 weeks for MVP
Additional 1-2 weeks for production hardening
"""

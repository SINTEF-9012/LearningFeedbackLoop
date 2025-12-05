"""Memory System API Contract and Endpoint Specifications.

This document defines the REST API endpoints for the memory system.
All endpoints are prefixed with /memories.

Base URL: http://localhost:8000/memories
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

"""
POST /memories
--------------
Create a new memory from an operator-annotated time-series window.

Request Body (JSON):
{
  "session_id": "1234567890",
  "time_range": {
    "i0": 1000,
    "i1": 2000,
    "t0": 0.244,
    "t1": 0.488,
    "fs": 4096.0
  },
  "annotation_text": "Unusual vibration during spindle ramp-up",
  "tags": ["vibration", "spindle", "anomaly"],
  "label": "tool_chatter",
  "created_by": "operator_alice",
  "compute_metrics": true,
  "include_online_snapshot": true,
  "compute_patterns": true,
  "pattern_discretization": "auto"
}

Response (200 OK):
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "pattern_keys": [
    {
      "pattern_type": "ratio",
      "key": "RATIO_Fx_Fy:2-4",
      "confidence": 0.95,
      "source_metric": "channel_ratios"
    },
    {
      "pattern_type": "spectral_peak",
      "key": "PSD_PEAK_512Hz:>0.5",
      "confidence": 0.87,
      "source_metric": "psd_peaks"
    }
  ],
  "metrics_summary": {
    "rms_Fx": 0.234,
    "dominant_freq_Fx": 512.0,
    "anomaly_score": 0.82
  },
  "message": "Memory created successfully"
}

Status Codes:
- 200: Success
- 400: Invalid request (missing required fields, invalid time range)
- 404: Session not found
- 500: Server error during computation

curl example:
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "1234567890",
    "time_range": {"i0": 1000, "i1": 2000, "t0": 0.244, "t1": 0.488, "fs": 4096.0},
    "annotation_text": "Unusual vibration during spindle ramp-up",
    "tags": ["vibration", "spindle"],
    "label": "tool_chatter"
  }'
"""

"""
POST /memories/query
--------------------
Query memories using pattern matching, ANN, and/or semantic search.
Optionally trigger LLM analysis of current window vs retrieved memories.

Request Body (JSON):
{
  "session_id": "1234567890",
  "time_range": {
    "i0": 5000,
    "i1": 6000,
    "t0": 1.22,
    "t1": 1.46,
    "fs": 4096.0
  },
  "query_text": "similar vibration patterns",
  "top_k": 5,
  "filters": {
    "tags": ["vibration"],
    "labels": ["tool_chatter", "spindle_issue"],
    "visibility": "active"
  },
  "use_pattern_matching": true,
  "use_ann_matching": true,
  "use_semantic_matching": true,
  "boost_pattern_matches": 2.0,
  "include_llm_analysis": true,
  "analysis_prompt": "Compare current window to historical events. Is this likely tool chatter?"
}

Response (200 OK):
{
  "results": [
    {
      "memory": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "session_id": "1234567890",
        "created_at": "2025-12-03T10:30:00Z",
        "annotation_text": "Unusual vibration during spindle ramp-up",
        "tags": ["vibration", "spindle"],
        "label": "tool_chatter",
        "metrics": { ... },
        "pattern_keys": [ ... ],
        ...
      },
      "relevance_score": 0.92,
      "match_reasons": [
        "Exact pattern match: RATIO_Fx_Fy:2-4",
        "ANN similarity: 0.87",
        "Semantic similarity: 0.79"
      ],
      "pattern_matches": [
        {
          "pattern_type": "ratio",
          "key": "RATIO_Fx_Fy:2-4",
          "confidence": 0.95
        }
      ]
    },
    ...
  ],
  "query_metrics": {
    "means": {"Fx": 0.12, "Fy": 0.05},
    "rms": {"Fx": 0.245, "Fy": 0.102},
    ...
  },
  "query_patterns": [
    {
      "pattern_type": "ratio",
      "key": "RATIO_Fx_Fy:2-4",
      "confidence": 0.93
    }
  ],
  "llm_analysis": "Based on 3 similar historical events (memory IDs: 550e8400..., abc123..., def456...), the current vibration pattern strongly resembles tool chatter observed during spindle ramp-up. All cases show Fx/Fy ratio of 2-4 and dominant frequency near 512 Hz. Recommended action: inspect tool condition and reduce feed rate.",
  "total_matches": 5,
  "retrieval_time_ms": 45.3
}

Status Codes:
- 200: Success
- 400: Invalid query parameters
- 404: Session not found (if session_id provided)
- 500: Server error

curl example:
curl -X POST http://localhost:8000/memories/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "1234567890",
    "time_range": {"i0": 5000, "i1": 6000, "t0": 1.22, "t1": 1.46, "fs": 4096.0},
    "top_k": 5,
    "include_llm_analysis": true
  }'
"""

"""
GET /memories
-------------
List and browse memories with optional filtering and pagination.

Query Parameters:
- session_ids: comma-separated session IDs (optional)
- tags: comma-separated tags (optional)
- labels: comma-separated labels (optional)
- created_by: comma-separated operator IDs (optional)
- time_range_min: ISO datetime (optional)
- time_range_max: ISO datetime (optional)
- visibility: "active" | "archived" | "deleted" (default: "active")
- sort_by: "created_at" | "session_id" (default: "created_at")
- sort_order: "asc" | "desc" (default: "desc")
- offset: int (default: 0)
- limit: int (default: 50, max: 1000)

Response (200 OK):
{
  "memories": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "session_id": "1234567890",
      "created_at": "2025-12-03T10:30:00Z",
      "annotation_text": "Unusual vibration during spindle ramp-up",
      "tags": ["vibration", "spindle"],
      "label": "tool_chatter",
      "time_range": { ... },
      "metrics": { ... },
      ...
    },
    ...
  ],
  "total_count": 127,
  "offset": 0,
  "limit": 50
}

Status Codes:
- 200: Success
- 400: Invalid query parameters

curl example:
curl "http://localhost:8000/memories?tags=vibration,spindle&limit=10&sort_order=desc"
"""

"""
GET /memories/{memory_id}
--------------------------
Retrieve a single memory by ID.

Path Parameters:
- memory_id: UUID string

Response (200 OK):
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "1234567890",
  "created_at": "2025-12-03T10:30:00Z",
  "annotation_text": "Unusual vibration during spindle ramp-up",
  "tags": ["vibration", "spindle"],
  "label": "tool_chatter",
  "time_range": { ... },
  "metrics": { ... },
  "pattern_keys": [ ... ],
  ...
}

Status Codes:
- 200: Success
- 404: Memory not found

curl example:
curl http://localhost:8000/memories/550e8400-e29b-41d4-a716-446655440000
"""

"""
PATCH /memories/{memory_id}
----------------------------
Update memory metadata (annotation, tags, label, visibility).

Path Parameters:
- memory_id: UUID string

Request Body (JSON) - all fields optional:
{
  "annotation_text": "Updated description with more detail",
  "tags": ["vibration", "spindle", "resolved"],
  "label": "tool_chatter_confirmed",
  "visibility": "archived",
  "metadata": {
    "resolution": "Tool replaced",
    "resolved_by": "operator_bob"
  }
}

Response (200 OK):
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory updated successfully"
}

Status Codes:
- 200: Success
- 400: Invalid update data
- 404: Memory not found

curl example:
curl -X PATCH http://localhost:8000/memories/550e8400-e29b-41d4-a716-446655440000 \
  -H "Content-Type: application/json" \
  -d '{
    "tags": ["vibration", "spindle", "resolved"],
    "visibility": "archived"
  }'
"""

"""
DELETE /memories/{memory_id}
-----------------------------
Delete or archive a memory.

Path Parameters:
- memory_id: UUID string

Query Parameters:
- hard_delete: bool (default: false) - if true, permanently delete; if false, soft-delete

Response (200 OK):
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory deleted successfully"
}

Status Codes:
- 200: Success
- 404: Memory not found

curl example:
curl -X DELETE "http://localhost:8000/memories/550e8400-e29b-41d4-a716-446655440000?hard_delete=false"
"""

"""
POST /memories/reindex
----------------------
Trigger background reindexing of pattern keys and ANN indices.
Admin/maintenance endpoint.

Request Body (JSON) - optional:
{
  "rebuild_patterns": true,
  "rebuild_ann": true,
  "rebuild_embeddings": true,
  "recluster": true
}

Response (202 Accepted):
{
  "job_id": "reindex-20251204-103045",
  "message": "Reindexing job started"
}

Status Codes:
- 202: Job accepted
- 500: Failed to start job

curl example:
curl -X POST http://localhost:8000/memories/reindex \
  -H "Content-Type: application/json" \
  -d '{"rebuild_patterns": true, "rebuild_ann": true}'
"""

"""
GET /memories/stats
-------------------
Get statistics about the memory store.

Response (200 OK):
{
  "total_memories": 1273,
  "active_memories": 1189,
  "archived_memories": 72,
  "deleted_memories": 12,
  "unique_sessions": 45,
  "unique_tags": 23,
  "unique_labels": 8,
  "top_patterns": [
    {"key": "RATIO_Fx_Fy:2-4", "count": 234},
    {"key": "PSD_PEAK_512Hz:>0.5", "count": 189}
  ],
  "storage_size_mb": 523.4,
  "index_update_time": "2025-12-04T09:15:32Z"
}

Status Codes:
- 200: Success

curl example:
curl http://localhost:8000/memories/stats
"""

# ============================================================================
# INTEGRATION NOTES
# ============================================================================

"""
Integration with existing agent system:

1. Memory endpoints will be mounted at /memories (separate from /agent/dispatch)
   
2. LLMAgent will be extended to:
   - Accept memory query results in handle_request
   - Format retrieved memories into RAG prompts
   - Support action="analyze_with_memories"

3. OnlineAgent integration:
   - POST /memories will call OnlineAgent.export_state() to capture snapshot
   - Memory events can optionally publish to event bus for online learning

4. Compute integration:
   - Memory creation uses existing compute_fg_fp_for_window_session_multi_ref
   - Additional metric extractors will be added to computation module

5. Event bus integration (optional):
   - Memory creation can publish "memory_created" event
   - Memory queries can publish "memory_retrieved" event for analytics

6. Authentication (future):
   - created_by field tracks operator identity
   - Access control per session/operator (not implemented in v1)
"""

# ============================================================================
# DATA FLOW DIAGRAMS
# ============================================================================

"""
Memory Creation Flow:
----------------------
1. Operator → POST /memories {session_id, time_range, annotation}
2. Memory API → Extract session data for window
3. Memory API → compute_metrics(window) → NumericMetrics
4. Memory API → derive_patterns(metrics) → List[PatternKey]
5. Memory API → OnlineAgent.export_state(session_id) → OnlineAgentSnapshot
6. Memory API → embed_text(annotation) → text_embedding vector
7. Memory API → MemoryStore.save(Memory record)
8. Memory API → PatternIndex.update(pattern_keys → memory_id)
9. Memory API → ANNIndex.insert(numeric_vector, memory_id)
10. Memory API → TextIndex.insert(text_embedding, memory_id)
11. Memory API → Response {memory_id, pattern_keys, summary}


Memory Query + Analysis Flow:
------------------------------
1. Client → POST /memories/query {session_id, time_range, include_llm_analysis=true}
2. Memory API → compute_metrics(query_window) → query_metrics, query_patterns
3. Memory API → PatternIndex.lookup(query_patterns) → exact_matches
4. Memory API → ANNIndex.search(query_vector, k=top_k) → ann_matches
5. Memory API → TextIndex.search(query_text, k=top_k) → semantic_matches
6. Memory API → merge_and_rank(exact, ann, semantic) → ranked_results
7. Memory API → build_rag_prompt(query_metrics, results)
8. Memory API → LLMAgent.handle_request(action='query', prompt) → llm_analysis
9. Memory API → Response {results, query_metrics, llm_analysis}


Background Reindexing Flow:
----------------------------
1. Admin → POST /memories/reindex
2. Memory API → schedule_background_job()
3. Background Worker → fetch all memories
4. Background Worker → recompute patterns/vectors (if schema changed)
5. Background Worker → rebuild indices from scratch
6. Background Worker → atomic swap old → new indices
7. Background Worker → mark job complete
"""

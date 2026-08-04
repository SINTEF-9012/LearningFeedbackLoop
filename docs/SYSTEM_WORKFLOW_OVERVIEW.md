# System Workflow Overview (backend)

## One-paragraph description
This project is a FastAPI backend that (1) ingests a time-series session payload, (2) replays it as a real-time stream over WebSockets, (3) computes FFT and domain-specific amplitudes, and (4) routes higher-level requests to “agents” (compute, online learner, LLM/RAG) while a separate memory subsystem can observe the live feature stream, score “significant” events, store them to SQLite, and broadcast alerts.

## Primary runtime components
- **FastAPI app**: [backend/app.py](../backend/app.py)
- **Session playback**: `playback_task()` in [backend/app.py](../backend/app.py)
- **FFT streaming**: `fft_stream_task()` in [backend/fft_streamer.py](../backend/fft_streamer.py)
- **Compute primitives**: [backend/computation.py](../backend/computation.py)
- **Event bus (in-memory pub/sub)**: [backend/events.py](../backend/events.py)
- **Agent router**: [backend/agents/router.py](../backend/agents/router.py)
- **Agents**:
  - Compute: [backend/agents/processing/compute.py](../backend/agents/processing/compute.py)
  - Online learner: [backend/agents/processing/online.py](../backend/agents/processing/online.py)
  - LLM/RAG: [backend/agents/llm/rag.py](../backend/agents/llm/rag.py)
- **Memory system**:
  - Init + lifecycle: [backend/agents/memory/init.py](../backend/agents/memory/init.py)
  - Feature→memory bridge: [backend/agents/memory/feature_stream_bridge.py](../backend/agents/memory/feature_stream_bridge.py)
  - Orchestrator (score→store→retrieve→explain→alert): [backend/agents/memory/orchestrator.py](../backend/agents/memory/orchestrator.py)
  - HTTP/WS API: [backend/agents/memory/router.py](../backend/agents/memory/router.py)
  - Storage: [backend/agents/storage/store.py](../backend/agents/storage/store.py)

## End-to-end workflow
1. **Create session**: `POST /sessions` creates a session record in `app.state.sessions`.
2. **Upload data**: `POST /sessions/{id}/upload` reads JSON and normalizes it via `preprocess_payload()` into:
   - `session["data"]`: `channel_name -> samples`
   - `session["metadata"]`: acquisition + machining metadata
   It also initializes FFT streaming config/state.
3. **Start playback**: `POST /sessions/{id}/start` schedules:
   - `playback_task(session_id)` (time-domain replay)
   - `fft_stream_task(session)` (frequency-domain stream driven by session position)
4. **Time-domain streaming**: `WS /streams/{id}` receives frames from per-client asyncio queues. Frames are also published to the in-memory feature bus via `publish_feature()`.
5. **FFT streaming**: `WS /sessions/{id}/fft` receives periodic FFT payloads derived from the most recent `nfft` samples.
6. **Agent dispatch**: `POST /agent/dispatch/{id}` selects an agent and injects the live `session` dict into args/context.
   - Compute agent calls domain compute utilities (e.g., fg/fp amplitudes).
   - Online agent can subscribe to the global feature bus and update an online model (River).
   - LLM/RAG agent can retrieve from an in-process vector store and call Ollama for generation.
7. **Memory-first learning loop** (app lifespan): On startup, the app initializes the memory system and starts a background task that subscribes to the feature bus:
   - The bridge converts feature events into `MemoryEvent`s.
   - The orchestrator scores significance, stores to SQLite, retrieves similar memories, optionally generates an explanation (LLM), and dispatches alerts.
   - Clients can subscribe to alerts via `WS /agent/memory/alerts/{session_id}`.

## Dataflow diagram

### High-level dataflow (ASCII)

```
          (HTTP)                     (WS) time frames
Client ─────────────► FastAPI app ─────────────────────────► /streams/{session}
  │                      │
  │                      │ publishes feature events
  │                      ▼
  │                In-memory bus (PubSub)
  │                  backend/events.py
  │                      │
  │          ┌───────────┴───────────┐
  │          │                       │
  │          ▼                       ▼
  │   OnlineAgent              Memory bridge
  │  (River model)        backend/agents/memory/feature_stream_bridge.py
  │          │                       │
  │          │                  Orchestrator
  │          │   score → store(SQLite) → retrieve → explain → alert
  │          │                       │
  │          ▼                       ▼
  │   (optional)              WS alerts endpoint
  │   anomaly/preds     /agent/memory/alerts/{session}
  │
  └────────────────► FFT streamer task ────────────────► /sessions/{session}/fft (WS)
                    backend/fft_streamer.py
```

### Agent request path

```
Client (HTTP) → POST /agent/dispatch/{session_id}
  → agents/router.py selects agent
  → injects live session dict into args/context
  → ComputeAgent / OnlineAgent / LLMAgent executes
  → returns JSON result
```

## What the system currently does NOT do (notable gaps)
- **No durable session store**: sessions and stream subscribers are in-memory (`app.state.sessions`), so restarts lose sessions and multi-worker deployments will not share state.
- **No authz/authn**: endpoints allow upload/dispatch without authentication.
- **Inconsistent sampling-rate usage**: some FFT/analyze utilities use `metadata["file_header"]["SampleFrequency"]`, others use `metadata["sample_frequency"]` or assume `d=1.0`.
- **Memory bridge metadata is stubby**: bridge’s `session_meta_cache` currently defaults `fs` (e.g., 10000 Hz) rather than reading actual session metadata.
- **Online learning is not “wired” to supervised labels**: the online agent expects `ev["label"]` sometimes, but the streaming pipeline doesn’t naturally emit labels.
- **RAG index is process-local**: the LLM/RAG ingestor has no dedicated HTTP endpoint here for persistent ingestion, and index persistence is optional.
- **Backpressure/slow clients**: queues can fill; the system mostly drops or waits, but there’s no explicit QoS policy per client.

## Learning feedback loop (memory-first learning) — detailed

### The “learning loop” in this codebase
This repo’s learning loop is not “model weights training” end-to-end; it’s primarily **operator-in-the-loop calibration** of what the system considers “significant”, plus **case-based recall** (retrieve similar memories). Concretely, it learns by updating **pattern priors** in the significance scorer and by accumulating a growing memory database.

### Step-by-step loop
1. **A feature event is emitted**
   - Time playback publishes `{"type":"time", ...}` events via `publish_feature()`.
   - FFT streamer publishes `{"type":"fft", ...}` events via `publish_feature()`.
   - Both go onto the in-memory bus in [backend/events.py](../backend/events.py).

2. **Bridge converts feature → MemoryEvent**
   - Background task in [backend/agents/memory/feature_stream_bridge.py](../backend/agents/memory/feature_stream_bridge.py) subscribes to the bus.
   - If upstream didn’t provide `metrics`/`patterns`, the bridge’s `DefaultFeatureExtractor` may add minimal metrics/patterns so events don’t get skipped.
   - It builds a `MemoryEvent` (time range, pattern keys, optional metrics/context).

3. **Orchestrator scores significance**
   - [backend/agents/memory/orchestrator.py](../backend/agents/memory/orchestrator.py) calls the scorer.
   - [backend/agents/memory/scorer.py](../backend/agents/memory/scorer.py) combines multiple “rules” into a composite score $0..1$:
     - external/classical alerts (if present)
     - pattern heuristics
     - anomaly deviation vs rolling baseline
     - learned historical prior (the “learning” piece)
   - The score maps to an action: `ignore | store | alert | critical`.

4. **If significant: store + retrieve + (optional) explain + alert**
   - Store: [backend/agents/storage/store.py](../backend/agents/storage/store.py) persists a `Memory` to SQLite and updates indices.
   - Retrieve: [backend/agents/memory/retriever.py](../backend/agents/memory/retriever.py) finds similar memories (context/pattern/vector).
   - Explain: [backend/agents/llm/explainer.py](../backend/agents/llm/explainer.py) can call Ollama; otherwise returns a deterministic fallback.
   - Alert: [backend/agents/memory/dispatcher.py](../backend/agents/memory/dispatcher.py) pushes a structured alert to websocket subscribers.

5. **Operator feedback closes the loop**
   - UI or script calls:
     - `PATCH /agent/memory/{memory_id}/feedback` (general)
     - or shortcuts: `POST /agent/memory/{memory_id}/confirm` / `POST /agent/memory/{memory_id}/dismiss`
   - Feedback updates:
     - the memory record (tags/labels/feedback stats)
     - and crucially: **pattern priors** via `SignificanceScorer.update_pattern_prior()`.
   - Priors are persisted to disk (configured via `PATTERN_PRIORS_PATH`). Next time a similar pattern appears, the historical-prior rule boosts or suppresses the composite score.

6. **Reconfiguration basis is updated from the same feedback signal**
   - The same confirm/dismiss evidence that updates priors and thresholds also updates the pattern-score inputs used by the reconfiguration composer.
   - This keeps proposal confidence coupled to accumulated feedback history, not only event-local confidence.

7. **Close the loop into a bounded reconfiguration proposal (optional runtime hook)**
   - For significant events with complete context, the reconfig composer emits deterministic parameter/tool actions plus a batch-scoped `RecipeEdit(target="next_unit")` when batch identity is present.
   - An optional LLM narration pass can translate that deterministic proposal into operator-facing language and justification.
   - Proposals are never auto-applied: `requires_operator_confirmation` remains true and operator decisions are append-only logged (`accept`, `modify`, `reject`).

### What “learning” looks like in practice
- After a `confirm`, the prior for the involved pattern keys moves toward 1.0 via an exponential moving average; after a `dismiss`, it moves toward 0.0.
- This changes the future: the same pattern set tends to cross `store/alert/critical` thresholds more easily (or less easily), even in a new session.

## How to demonstrate and visualize the feedback loop

### Option A (already exists): scripted demo
Use [scripts/demo_memory_feedback.py](../scripts/demo_memory_feedback.py). It walks through events → store/alert → feedback → show updated priors.

Typical flow:
1. Run server: `uvicorn backend.app:app --reload --port 8000`
2. Run demo: `python scripts/demo_memory_feedback.py --url http://localhost:8000 --pause`

### Option B (new): live alert visualization
Use [scripts/visualize_memory_alerts.py](../scripts/visualize_memory_alerts.py) to connect to the alerts websocket and plot significance score over time. This gives you a “heartbeat” view of when the system decides to alert/critical.

### Option C (new): polling priors visualization
Use [scripts/visualize_memory_priors.py](../scripts/visualize_memory_priors.py) to poll the scorer’s learned pattern priors and plot how they drift after feedback.

Typical flow:
1. Run server.
2. Start priors plotter: `python scripts/visualize_memory_priors.py --top-k 10`
3. Run feedback demo and confirm/dismiss a few events.
4. Watch priors move toward 1.0 (confirm) or 0.0 (dismiss).

Suggested demo recipe:
1. Start server.
2. Start alert plotter (global): `python scripts/visualize_memory_alerts.py --session all`
3. In another terminal, run [scripts/demo_memory_feedback.py](../scripts/demo_memory_feedback.py) or POST events to `/agent/memory/events`.
4. Confirm/dismiss some events; watch how subsequent alerts become more/less frequent and how scores shift.


**Project Overview**
- **Purpose:** FastAPI backend for streaming time-series data, computing FFTs and amplitudes, and routing high-level requests to pluggable agents (compute, RAG/LLM, online learners).
- Code lives under `backend/` and helper scripts are in `scripts/`.

**Quick Start**
- Create and activate your Python environment and install deps:
  - `conda activate emitter`
  - `pip install -r requirements.txt`
- Start the server from the repo root:
  - `uvicorn app:app --reload --port 8000 --app-dir backend`
- Optional: start Ollama (or set `OLLAMA_URL`) for the LLM/RAG agent.

**Scripts**
- `scripts/upload_and_start.py <file>` — create a session, upload JSON file, start playback.
- `scripts/ws_listen.py <session_id>` — print frames from the time-domain websocket stream.
- `scripts/visualize_stream.py <session_id>` — real-time matplotlib visualizer of first channel.
- `scripts/agent_cli.py <session_id> <agent> <action> [args-json]` — dispatch an agent request.

**Endpoints**
Below is a concise list of the HTTP+WebSocket endpoints implemented by the server. Use `http://localhost:8000` (default) as base.

- `POST /sessions`
  - Create a new session.
  - Body: JSON `SessionConfig` — e.g. `{ "interval_ms": 100, "channels": null, "mode": "time" }`
  - Response: `{ "session_id": "<id>", "ws": "/streams/<id>" }`

- `POST /sessions/{session_id}/upload`
  - Upload a JSON session payload (file form field `file`). Preprocesses and stores `data`, `metadata`, and `raw_file`.
  - Response: `{ "ok": true, "channels": [...], "metadata": {...} }`

- `POST /sessions/{session_id}/start`
  - Start playback for the session. Also starts the FFT streamer for the session (if not already running).
  - Response: `{ "ok": true }`

- `POST /sessions/{session_id}/pause`
  - Pause playback (sets `session["paused"] = True`).
  - Response: `{ "ok": true, "paused": true }`

- `POST /sessions/{session_id}/resume`
  - Resume playback (sets `session["paused"] = False`).
  - Response: `{ "ok": true, "paused": false }`

- `POST /sessions/{session_id}/replay`
  - Restart playback from the beginning of the session.
  - Body: `ReplayRequest` e.g. `{ "speed": 1.0 }` to set playback speed.
  - Behavior: resets `session["position"] = 0`, updates `session["config"]["speed"]` and `session["metadata"]["playback_speed"]`, cancels any running playback task and starts a new one.
  - Response: `{ "status": "restarted", "session_id": "...", "speed": 1.0 }`

- `GET /sessions`
  - List active session IDs. Response: `{ "sessions": ["id1", "id2"] }`

- `GET /sessions/{session_id}`
  - Get session info: config, channels, metadata, running flag, raw_file.

- `GET /sessions/{session_id}/metadata`
  - Returns session metadata and timesteps played: `{ "metadata": {...}, "timesteps_played": N }`

- `GET /sessions/{session_id}/download?format={json|csv}`
  - Download the portion of data already played (position). Defaults to JSON; CSV option returns a streaming CSV response.

- `POST /sessions/{session_id}/analyze` (simple POST variant)
  - Analyze a numeric range: query args `start`, `end`, `channel` (POST form or JSON). Returns `freqs` and `spectrum`.

- `GET /sessions/{session_id}/analyze?channel=...&start=...&end=...` (GET variant)
  - Query a channel range and receive FFT spectrum for that slice.

- `POST /sessions/{session_id}/fft2`
  - Compute FFT for the most recent window using `window_size` in the body (JSON). Returns per-channel freqs/magnitudes.

- `POST /sessions/{session_id}/fft`
  - Compute FFT for a requested time window across multiple variables (body: `min_time`, `max_time`, `variables`).

- `POST /sessions/{session_id}/amplitudes/fg-fp`
  - High-level amplitude estimator endpoint (calls `compute_fg_fp_for_window_session_multi_ref`). Body: compute request with `window`, `channels`, `options`.

- `POST /agent/dispatch/{session_id}`
  - Dispatch a request to a registered agent (compute, `llm.rag`, `online`, ...).
  - Body example: `{ "agent": "compute", "action": "amplitudes", "args": { "request": { ... } }, "stream": false }`
  - The router injects a live session reference into the agent `args` as `args.session` and into the `context`.

- `POST /sessions/{session_id}/fft/start` and `POST /sessions/{session_id}/fft/stop` (router)
  - Start/stop the background FFT streaming task for the session.

- WebSocket endpoints
  - `ws://<host>:8000/streams/{session_id}` — time-domain frames (text JSON messages per tick).
  - `ws://<host>:8000/sessions/{session_id}/fft` — FFT frames (JSON messages from FFT streamer).

- `POST /query-time-series/`
  - Convenience endpoint to query a global DataFrame (`data`) timeseries slice and forward to an LLM. Expects a `TimeRange` body `{start, end, prompt}`.

**Can I restart a session playback in the current session?**
- Yes. Use `POST /sessions/{session_id}/replay` to restart playback from the beginning. This endpoint resets the session position to `0`, updates the playback speed if provided, cancels any currently running playback task, and starts a fresh playback task for the same session. You can also pause/resume with `POST /sessions/{id}/pause` and `POST /sessions/{id}/resume`.

**Testing & Interaction**
- Use the provided `scripts/` utilities for common flows:
  - `python scripts/upload_and_start.py test_data/sample_session.json` — create, upload, start.
  - `python scripts/ws_listen.py <session_id>` — print frames from the time-domain WS.
  - `python scripts/visualize_stream.py <session_id>` — real-time plot of the first channel.
  - `python scripts/agent_cli.py <session_id> online start` — start the online learner agent for the server-wide feature stream.
- For ad-hoc HTTP calls use `curl`, `httpie`, or Postman. Example restart/replay:
  - `curl -X POST "http://localhost:8000/sessions/<session_id>/replay" -H 'Content-Type: application/json' -d '{"speed":1.0}'`

**Notes & Next Steps**
- Agent endpoints are pluggable and the router injects the live session reference into each agent call; agents can operate directly on `session["data"]`, `session["metadata"]`, and playback state.
- The current server stores sessions and events in-memory. For production durability and scaling, replace the in-memory `sessions` and `events` bus with persistent stores (Redis, Postgres, Kafka) and add authentication/authorization on upload/agent endpoints.

If you'd like, I can add a Postman collection, expand the README with request/response examples for each endpoint, or wire automatic indexing of uploaded files into the RAG index.

**Curl Examples (copy-paste)**

Below are ready-to-run `curl` examples for the most commonly used endpoints. Replace `<session_id>` and file paths as needed.

- Create a session
```bash
curl -s -X POST http://localhost:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"interval_ms":100, "channels": null, "mode":"time"}' | jq
```

- Upload a session file
```bash
SESSION_ID=<session_id>
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data/sample_session.json" | jq
```

- Start playback
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/start" | jq
```

- Pause / Resume
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/pause" | jq
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/resume" | jq
```

- Restart (replay) from beginning with speed
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/replay" \
  -H 'Content-Type: application/json' \
  -d '{"speed":1.0}' | jq
```

- List sessions / Get session info
```bash
curl -s http://localhost:8000/sessions | jq
curl -s http://localhost:8000/sessions/${SESSION_ID} | jq
```

- Download played portion (JSON or CSV)
```bash
curl -s "http://localhost:8000/sessions/${SESSION_ID}/download?format=json" -o played.json
curl -s "http://localhost:8000/sessions/${SESSION_ID}/download?format=csv" -o played.csv
```

- Analyze (GET variant)
```bash
curl -s "http://localhost:8000/sessions/${SESSION_ID}/analyze?channel=A&start=0&end=100" | jq
```

- Compute FFT (window recent)
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/fft2" \
  -H 'Content-Type: application/json' \
  -d '{"window_size":1024}' | jq
```

- Compute FFT (time window / variables)
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/fft" \
  -H 'Content-Type: application/json' \
  -d '{"min_time":0, "max_time":1, "variables":["A","B"]}' | jq
```

- Amplitude endpoint (fg/fp)
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/amplitudes/fg-fp" \
  -H 'Content-Type: application/json' \
  -d '{"window":{"t_min":0.0,"t_max":0.5},"channels":["A"]}' | jq
```

- Control FFT streamer
```bash
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/fft/start" | jq
curl -s -X POST "http://localhost:8000/sessions/${SESSION_ID}/fft/stop" | jq
```

- Agent dispatch examples
```bash
# Start online agent for server-wide features
curl -s -X POST "http://localhost:8000/agent/dispatch/${SESSION_ID}" \
  -H 'Content-Type: application/json' \
  -d '{"agent":"online","action":"start","args":{}}' | jq

# Compute amplitudes via compute agent
curl -s -X POST "http://localhost:8000/agent/dispatch/${SESSION_ID}" \
  -H 'Content-Type: application/json' \
  -d '{"agent":"compute","action":"amplitudes","args":{"request":{"window":{"t_min":0,"t_max":0.5}}}}' | jq

# Ask RAG/LLM agent (requires Ollama running)
curl -s -X POST "http://localhost:8000/agent/dispatch/${SESSION_ID}" \
  -H 'Content-Type: application/json' \
  -d '{"agent":"llm.rag","action":"query","args":{"question":"Describe anomalies near t=0.1s"}}' | jq
```

- WebSocket (time-domain)
```bash
# using wscat (npm) or websocat
wscat -c ws://localhost:8000/streams/${SESSION_ID}
# or
websocat ws://localhost:8000/streams/${SESSION_ID}
```

- WebSocket (FFT)
```bash
wscat -c ws://localhost:8000/sessions/${SESSION_ID}/fft
```


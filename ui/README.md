# tmplfl UI

This is a desktop-ready realtime UI (implemented with React/TypeScript + uPlot).

## Prereqs
- Node.js 18+ and npm

On Ubuntu/WSL (quick):
- `sudo apt-get update && sudo apt-get install -y nodejs npm`

If you prefer version management, install nvm and use `nvm install --lts`.

## Run
This demo has 3 moving parts: backend API, a running session stream (optional), and the UI.

### 1) Start the backend API
From repo root:
- `./.venv/bin/uvicorn backend.app:app --reload --port 8000`

Notes:
- The backend enables CORS for `http://localhost:5173` so the dev UI can call it.

### 2) Create a session + upload data + start playback
In another terminal (repo root):
- `./.venv/bin/python scripts/upload_and_start.py test_data/sample_session.json --speed 0.02 --samples-per-tick 8 --start-paused`

This prints a `session_id`.

### 2a) Recommended: mock "real run" UI driver (stream + alerts + feedback)
If you want a *reliable*, end-to-end UI demo that creates a real playback session
and also injects high-signal memory events (so the Alerts panel lights up), run:
- `./.venv/bin/python scripts/demo_ui_mock_run.py test_data/sample_session.json --start-paused --reset-priors`

If your repo env sets `REQUIRE_LLM=true` but you want to exercise the UI without
an LLM provider, use:
- `./.venv/bin/python scripts/demo_ui_mock_run.py test_data/sample_session.json --start-paused --reset-priors --disable-llm`

It prints a `session_id`. Select that in the UI.

### 2b) (Alternative) Playback-free synthetic demo (memories + feedback + priors)
If you want to focus on the memory/feedback learning loop without any session playback, run:
- `./.venv/bin/python scripts/simulate_memory_feedback_loop.py --reset-priors --rate-hz 4 --feedback-every 5`

This posts synthetic events to `POST /agent/memory/events` and periodically applies feedback. The UI will still show priors updating in real time.

### 3) Start the UI
From repo root:
- `cd ui`
- `npm install`
- `npm run dev`

Open the UI at `http://localhost:5173`.

### 4) Use the feedback loop
In the UI:
- Select the `session_id` (dropdown).
- Click `Resume` in the "Playback controls" row.
- Watch the stream update (left plot).
- Drag on the plot to select a window.
- Click "Create memory" (POSTs to `/agent/memory/capture`).
- Select the new memory on the right and click Confirm/Dismiss.
- Observe priors changing in the "Priors" panel.

If you're using the playback-free synthetic demo, you can skip the stream/plot steps and just watch the priors panel react to incoming feedback.

Playback controls:
- `Pause` calls `POST /sessions/{session_id}/pause`
- `Resume` calls `POST /sessions/{session_id}/resume`
- `Replay` calls `POST /sessions/{session_id}/replay` with the given speed

## What it does
- Connects to `WS /streams/{session_id}` for the time stream.
- Connects to `WS /agent/memory/alerts/{session_id}` for memory alerts.
- Lists memories for the session and shows detail + feedback + traces.
- Shows priors and lets you Confirm/Dismiss a memory.
- Lets you drag-select a window on the plot and create a memory via:
  - `POST /agent/memory/capture`

## Notes
- Packaging as a true desktop binary (Tauri) is the next step; this UI code is written so it can be wrapped without rewrites.

## Troubleshooting
- UI shows no sessions: backend not running or wrong API base URL (use Apply button).
- Stream is blank: session may be paused/finished; use UI Resume/Replay; run `scripts/demo_ui_mock_run.py --start-paused`.
- Alerts are empty: memory events may not be significant; use `scripts/demo_ui_mock_run.py` or `scripts/demo_memory_feedback.py`.
- Alerts are not human-readable / badge shows `fallback`: the backend couldn't reach the configured LLM provider or the configured model is missing.
  - For Groq, set `GROQ_API_KEY` and verify `GROQ_MODEL`.
  - For Ollama, set `LLM_PROVIDER=ollama`, then start Ollama and ensure the model exists (e.g. `ollama pull $OLLAMA_MODEL`).
  - For demos, set `REQUIRE_LLM=true` in `.env` to fail fast if LLM isn't usable.
- CORS errors in browser console: ensure backend is on `http://localhost:8000` (or extend CORS allowlist in `backend/app.py`).

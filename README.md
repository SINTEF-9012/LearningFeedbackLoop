# Learning Feedback Loop (LFL)

A research prototype for **operator-in-the-loop anomaly handling on CNC machining
lines**. It streams sensor time-series, scores events for significance, explains
the interesting ones in plain language, and — crucially — learns from what the
operator says back.

The core idea is that the last step is the valuable one. Detectors produce
alerts; shop floors ignore alerts that are usually wrong. This system treats the
operator's *confirm* / *dismiss* as the ground-truth signal and feeds it back
into scoring, retrieval and pattern discovery, so the alert stream re-orders
itself around what that particular site actually cares about.

> **Status: research prototype.** It demonstrates the loop end to end and has
> been exercised on public milling datasets. It is **not** a validated or
> production-hardened product, and none of its outputs should be relied on for
> safety or process decisions. See [Scope and honest limitations](#scope-and-honest-limitations).

---

## How the loop works

```
   sensor stream ──► significance scoring ──► similar past events + operator history
                                │                        │
                                ▼                        ▼
                          explanation (LLM) ◄──── live machine/tool context
                                │
                                ▼
                       operator: confirm / dismiss
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  pattern priors        ground truth for          new pattern
   & thresholds           retraining               discovery
```

1. An event fires, from either pattern heuristics or a model's anomaly score.
2. The orchestrator retrieves **similar past events and their feedback history**
   (confirm/dismiss rates, learned priors).
3. It adds live machine/tool context, and optionally grounding from a document
   knowledge graph built from machine documentation.
4. An LLM turns that bundle into an operator-facing explanation. With no LLM
   configured it falls back to a deterministic template — the loop still works.
5. The operator confirms or dismisses.
6. Feedback updates pattern priors, is stored as a training label, and confirmed
   events that match no known pattern are clustered into **candidate new
   patterns**.

Nothing in the loop acts autonomously. Reconfiguration output is a *proposal*
that always requires operator confirmation.

---

## Quick start

`pyproject.toml` declares Python `>=3.9`; development and CI run on **3.12**,
which is what the Docker image uses. Node 18+ for the UI.

```bash
# 1. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. UI dependencies
cd ui && npm install && cd ..

# 3. Configuration (every variable is documented in .env.example)
cp .env.example .env

# 4. Generate the synthetic sample dataset
python scripts/generate_sample_dataset.py
```

Run the backend and the UI in two terminals:

```bash
# Terminal 1 — API on :8000 (SQLite storage, no Docker needed)
STORAGE_BACKEND=sqlite uvicorn backend.app:app --reload --port 8000

# Terminal 2 — UI on :5173
cd ui && npm run dev
```

Open <http://localhost:5173>. `STORAGE_BACKEND=sqlite` keeps everything local;
switch to Neo4j when you want the graph features.

### With Neo4j

```bash
docker compose --profile core up -d neo4j
STORAGE_BACKEND=neo4j NEO4J_URI=bolt://localhost:7687 \
NEO4J_USERNAME=neo4j NEO4J_PASSWORD=changeme \
uvicorn backend.app:app --reload --port 8000
```

---

## Bringing your own data

**No real machining data ships with this repository.** `scripts/generate_sample_dataset.py`
writes a small synthetic dataset so everything runs out of the box.

To use your own, lay it out like this — channel groups are identified from **CSV
column headers**, not filenames, so no vendor-specific naming is assumed:

```
<root>/<case>/<operation>/axis_power.csv
                          vibration.csv
                          energy.csv
                          machine_state.csv
```

Operation directories are named `OF<number>`. Each CSV needs a `timestamp`
column plus any subset of the columns declared in `KEY_COLUMNS` in
[`backend/agents/processing/dataset_loader.py`](backend/agents/processing/dataset_loader.py);
the group sharing the most columns wins. Extend that mapping to teach the
loader about new sensor groups.

Sensor semantics are configured declaratively in `domain_packs/*.yaml`, which
map channel names onto semantic roles (primary vibration, spindle power, …).
**Adding a new machine type means adding a YAML file, not editing Python.**

---

## Architecture

| Area | Where | What it does |
|---|---|---|
| Streaming core | `backend/app.py`, `fft_streamer.py` | Session upload, time-domain replay, FFT streaming over WebSocket |
| Agents | `backend/agents/` | Pluggable compute / online-learning / RAG / retrieval / monitoring / analytics agents behind one dispatch API |
| Memory & feedback loop | `backend/agents/memory/` | Significance scoring, retrieval of similar events, explanation assembly, feedback handling |
| Pattern learning | `backend/agents/patterns/` | Fault signatures, pattern registry, discovery of new clusters from confirmed events |
| Domain config | `domain_packs/*.yaml` | Declarative channel-role and fault-indicator mapping |
| Storage | `backend/agents/storage/` | SQLite or Neo4j behind one protocol |
| Frontend | `ui/` | React + TypeScript + Vite operator and analysis surfaces |

Deeper writeups: [`docs/SYSTEM_WORKFLOW_OVERVIEW.md`](docs/SYSTEM_WORKFLOW_OVERVIEW.md)
(the streaming/memory pipeline) and
[`DOCS_GRAPH_ARCHITECTURE_GUIDE.md`](DOCS_GRAPH_ARCHITECTURE_GUIDE.md)
(the documentation knowledge graph).

### Optional integrations

Every heavy or external dependency is optional and degrades gracefully — a
missing one disables its feature and logs a warning rather than breaking start-up.

- **LLM** — Groq or Ollama (`LLM_PROVIDER`). Without one, explanations fall back
  to deterministic templates.
- **Digital twin (SINDIT)** — set `SINDIT_API_URL` and `SINDIT_ENABLED=true` to
  enrich events with asset/tool context from an external instance. No twin
  service is bundled.
- **Neo4j** — graph-backed memory and the document knowledge graph.
- **PyTorch / FAISS** — harmonic CNN scoring and approximate nearest-neighbour
  retrieval.

---

## Testing

```bash
pytest                  # backend suite
mypy                    # gradual type checking (not clean yet — see CONTRIBUTING)
cd ui && npm run build  # tsc + vite build
cd ui && npm run test   # vitest
```

Tests needing a site-specific tool-master dataset skip themselves automatically
(see `tests/conftest.py`); end-to-end tests skip when no server is running. The
suite is green on a clean clone with no data.

---

## Scope and honest limitations

- **Not validated in production.** The loop is demonstrated end to end; it has
  not been shown to improve outcomes on a real production line.
- **Feedback improves *ranking*, not calibration.** Learned priors reliably
  change *what gets shown first*. They do not turn the score into a calibrated
  probability, and the absolute numbers should not be read as one.
- **Labels are weak.** The datasets this was developed against have sparse and
  partly derived labels, so evaluation leans on weak supervision rather than
  clean ground truth.
- **LLM output is explanatory, not authoritative.** Explanations are generated
  from a retrieved, bounded context and are constrained to it, but they can
  still be wrong. They are a reading aid for the operator, not a decision.
- **Carbon figures are illustrative.** Where the code models CO₂, it reports
  what is *at stake* under stated assumptions. It does not measure or claim
  savings.
- **Nothing is autonomous.** Every proposal requires operator confirmation.

---

## Licence

MIT — see [LICENSE](LICENSE).

Public datasets referenced by the validation scripts keep their own licences and
attribution; `scripts/fetch_public_datasets.py` documents the source and licence
of each.

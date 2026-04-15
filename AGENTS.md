# AGENTS.md – ToolBreak Harmonic Pipeline GUI

## Overview

A full-stack application for training, testing, and interactively simulating a CNN-based tool breakage detection model that operates on harmonic features extracted from accelerometer data during milling operations.

**Architecture:**
- **Backend:** Python / FastAPI (port 8000) — data loading, harmonic computation, PyTorch model training, inference, WebSocket streaming
- **Frontend:** React / TypeScript / Vite (port 5173) — interactive GUI with Plotly.js charts

---

## Quick Start

### 1. Install Python dependencies

```bash
cd /path/to/LFL_classicaldemo
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Launch both servers

```bash
bash start.sh
```

This starts the backend (port 8000) and frontend (port 5173) together.

### 4. Open the GUI

Navigate to **http://localhost:5173** in your browser.

#### Manual start (alternative)

```bash
# Terminal 1: Backend
uvicorn backend.app:app --port 8000

# Terminal 2: Frontend (from frontend/ directory)
cd frontend && npx vite@5
```

---

## Features

### Training Tab
- Select which data folders to include in training
- Configure train/test split ratio (slider)
- Define a multi-stage learning rate schedule (add/remove stages)
- Set batch size
- View live training loss curve that updates every 0.5 seconds
- See data split summary (counts of OK / broken samples)
- Stop training early at any time

### Simulation Tab
- Pick any JSON data file from the available folders
- Stream harmonic time points through the model at an adjustable speed (1–200 pts/s)
- **Play / Pause / Resume / Stop** transport controls
- **Top chart:** Combined signal (`w · harmonics`) + P(broke) on a twin y-axis
- **Bottom chart:** All 21 individual harmonic magnitudes (toggle visibility via legend)
- Real-time progress bar and current prediction badge
- Speed can be changed mid-stream

### Evaluation Tab
- Evaluate on the **test set** from the last training split, or select **specific folders**
- **Window position** slider: choose where in each time series the CNN window is placed (0% = start, 100% = end)
- Results include:
  - Accuracy, F1 scores
  - Confusion matrix (interactive heatmap)
  - Probability distribution histogram (OK vs Broke)
  - Full classification report table
  - Per-sample results table with correct/incorrect highlighting

### Config Tab
- **Data directory** — path to the folder containing JSON data subfolders
- **FFT Window Size** — samples per FFT window (default 4096 = 1 s at 4096 Hz)
- **FFT Step** — stride between FFT windows (default 1024 = 75% overlap)
- **Harmonic Multipliers** — which multiples of spindle frequency to extract (default: 1, 2, 3, 4, 6, 8, 10)
- **CNN Window Size** — number of harmonic time steps fed to the CNN (default 16)
- **Conv Channels** — channel count per conv layer (default: 16, 16)
- **FC Hidden Size** — fully connected hidden layer width (default 32)
- **Kernel Size** — conv kernel size (default 5)

Changes take effect on the next training run. Click **Save Config** to push to the server.

---

## Project Structure

```
LFL_classicaldemo/
├── backend/
│   ├── __init__.py
│   ├── app.py            # FastAPI endpoints + WebSocket
│   ├── data.py           # JSON data loading
│   ├── harmonics.py      # Sliding-window FFT harmonic extraction
│   ├── model.py          # HarmonicBreakNet (PyTorch)
│   └── trainer.py        # Background training thread
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts    # Proxies /api and /ws to backend
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx        # Shell layout with sidebar nav
│       ├── api.ts         # REST API client
│       ├── types.ts       # TypeScript interfaces
│       ├── ui.tsx         # Shared UI components (Card, Field, Plotly hook)
│       ├── index.css
│       └── components/
│           ├── ConfigPanel.tsx
│           ├── TrainingPanel.tsx
│           ├── SimulationPanel.tsx
│           └── EvaluationPanel.tsx
├── requirements.txt
├── start.sh              # Launch both servers
├── .gitignore
└── AGENTS.md             # This file
```

---

## Data Format

Each JSON file in the data directory contains:
- **Cutting parameters:** `d` (diameter), `z` (teeth), `ap`, `ae`, `n` (RPM), `f`, `vf`
- **Label:** `break` (boolean)
- **Signals:** `Channel_1`, `Channel_2`, `Channel_3` each with a `Signal` array (accelerometer X/Y/Z)

The harmonic pipeline extracts magnitudes at multiples of the spindle frequency (`n/60` Hz) from each channel via sliding-window FFT, producing a `(T, 3 × n_harmonics)` feature matrix per sample.

---

## Model Architecture (HarmonicBreakNet)

1. **Learnable W matrix** `(n_features × 7)`: Maps the 7 cutting parameters to per-harmonic-channel weights
2. **Weighted combination:** `combined(t) = harmonics(t) · (params @ W^T)` — collapses multi-channel harmonics to a scalar time series
3. **1D Conv blocks:** Each block = Conv1d → BatchNorm → ReLU → AvgPool(2) → Dropout
4. **FC head:** Two hidden layers with BatchNorm + Dropout → single logit output
5. **Loss:** BCEWithLogitsLoss

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/config` | GET | Current pipeline configuration |
| `/api/config` | POST | Update configuration |
| `/api/data/folders` | GET | List data folders with file counts |
| `/api/data/files/{folder}` | GET | List JSON files with metadata |
| `/api/train/start` | POST | Start background training |
| `/api/train/status` | GET | Training progress (epoch, loss history) |
| `/api/train/stop` | POST | Stop training early |
| `/api/evaluate` | POST | Batch evaluation on test set or folders |
| `/ws/simulate` | WebSocket | Stream simulation for a single file |

### WebSocket Protocol (`/ws/simulate`)

**Client → Server:**
```json
{"action": "start", "file_path": "Folder/0.json", "speed": 10}
{"action": "pause"}
{"action": "resume"}
{"action": "set_speed", "speed": 50}
{"action": "stop"}
```

**Server → Client:**
```json
{"type": "init", "total_steps": 163, "harm_labels": [...], "cnn_window": 16, "broke": true, ...}
{"type": "step", "t": 0, "harmonics": [...], "combined": 0.42, "prob": null}
{"type": "step", "t": 15, "harmonics": [...], "combined": 0.31, "prob": 0.73}
{"type": "done"}
```

---

## Typical Workflow

1. **Config** — Verify data directory is detected, adjust FFT/CNN params if needed
2. **Training** — Select folders, set LR schedule, click Train, watch loss curve
3. **Evaluation** — Click "Run Evaluation" on the test set to see accuracy and confusion matrix
4. **Simulation** — Pick an interesting file, hit Play, watch the model's predictions evolve in real-time

import asyncio
import json as json_lib
import os
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .data import PARAM_KEYS, load_samples, resolve_data_dir
from .harmonics import compute_harmonics_with_mag
from .model import HarmonicBreakNet
from .trainer import Trainer

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    data_dir: str = "../lfl/testdata"
    fft_window: int = 4096
    fft_step: int = 4096
    harm_mults: list[int] = [1, 2, 3, 4, 6, 8, 10]
    cnn_window: int = 16
    conv_channels: list[int] = [16, 16]
    fc_hidden: int = 32
    kernel_size: int = 5


class TrainRequest(BaseModel):
    folders: list[str]
    test_split: float = 0.2
    lr_schedule: list[dict]
    batch_size: int = 16


class EvalRequest(BaseModel):
    source: str = "test_set"          # "test_set" or "folders"
    folders: list[str] = []
    window_position: float = 0.5


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.config = PipelineConfig()
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model: Optional[HarmonicBreakNet] = None
        self.trainer = Trainer()
        self.train_samples: list[dict] = []
        self.test_samples: list[dict] = []


state = AppState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ToolBreak Harmonic Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    cfg = state.config.model_dump()
    cfg["data_dir"] = resolve_data_dir(state.config.data_dir)
    cfg["device"] = str(state.device)
    cfg["model_loaded"] = state.model is not None
    return cfg


@app.post("/api/config")
def set_config(config: PipelineConfig):
    # Validate: cnn_window must survive all pooling layers
    t = config.cnn_window
    for _ in config.conv_channels:
        t = t // 2
    if t < 1:
        return {"error": f"cnn_window={config.cnn_window} too small for {len(config.conv_channels)} pool layers"}
    state.config = config
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/data/folders")
def list_folders():
    data_dir = resolve_data_dir(state.config.data_dir)
    if not os.path.isdir(data_dir):
        return {"folders": [], "data_dir": data_dir, "error": f"Not found: {data_dir}"}
    folders = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith(".")
    )
    result = []
    import glob
    for folder in folders:
        fp = os.path.join(data_dir, folder)
        n_files = len(glob.glob(os.path.join(fp, "*.json")))
        result.append({"name": folder, "n_files": n_files})
    return {"folders": result, "data_dir": data_dir}


@app.get("/api/data/files/{folder}")
def list_files(folder: str):
    import glob
    data_dir = resolve_data_dir(state.config.data_dir)
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        return {"files": [], "error": "Folder not found"}
    files = []
    for jf in sorted(glob.glob(os.path.join(folder_path, "*.json"))):
        with open(jf) as fh:
            d = json_lib.load(fh)
        chans = sorted(k for k in d if k.startswith("Channel_"))[:3]
        n_samples = len(d[chans[0]]["Signal"]) if chans else 0
        files.append({
            "name": os.path.basename(jf),
            "broke": bool(d.get("break", False)),
            "n_samples": n_samples,
            "params": {k: d.get(k, 0) for k in PARAM_KEYS},
        })
    return {"files": files}


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

@app.post("/api/train/start")
def start_training(req: TrainRequest):
    if state.trainer.running:
        return {"error": "Training already in progress"}

    cfg = state.config
    data_dir = resolve_data_dir(cfg.data_dir)

    # If no model exists, we need to create one and prepare data
    need_new_model = state.model is None

    if need_new_model or not state.train_samples:
        samples = load_samples(data_dir, req.folders)
        if len(samples) < 2:
            return {"error": f"Need ≥2 samples, got {len(samples)}"}

        # Compute harmonics (XYZ + magnitude channel)
        for s in samples:
            fg = s["params"][4] / 60.0
            s["harmonics"] = compute_harmonics_with_mag(
                s["accel"], fg, cfg.harm_mults,
                fft_win=cfg.fft_window, fft_step=cfg.fft_step,
            )

        # Filter out samples too short after FFT
        samples = [s for s in samples if s["harmonics"].shape[0] >= cfg.cnn_window]
        if len(samples) < 2:
            return {"error": "Not enough samples with sufficient harmonic time steps"}

        # Train/test split
        labels = [s["broke"] for s in samples]
        n_classes = len(set(labels))
        kwargs = dict(test_size=req.test_split, random_state=42)
        if n_classes >= 2:
            kwargs["stratify"] = labels
        train_idx, test_idx = train_test_split(range(len(samples)), **kwargs)

        state.train_samples = [samples[i] for i in train_idx]
        state.test_samples = [samples[i] for i in test_idx]

    if need_new_model:
        # Create model. n_harm_features = 4 * n_mults: X, Y, Z, |accel|.
        n_harm_features = len(cfg.harm_mults) * 4
        state.model = HarmonicBreakNet(
            n_harm_features=n_harm_features,
            n_params=len(PARAM_KEYS),
            cnn_window=cfg.cnn_window,
            conv_channels=cfg.conv_channels,
            fc_hidden=cfg.fc_hidden,
            ks=cfg.kernel_size,
        ).to(state.device)

        # Standardization stats from the training split only (avoid test leakage).
        train_params = np.stack([s["params"] for s in state.train_samples])
        state.model.set_param_stats(
            torch.tensor(train_params.mean(axis=0), dtype=torch.float32),
            torch.tensor(train_params.std(axis=0), dtype=torch.float32),
        )

    state.trainer.start(
        model=state.model,
        train_samples=state.train_samples,
        lr_schedule=req.lr_schedule,
        cnn_window=cfg.cnn_window,
        device=state.device,
        batch_size=req.batch_size,
        reset_history=need_new_model,
    )

    return {
        "status": "started" if need_new_model else "continued",
        "n_train": len(state.train_samples),
        "n_test": len(state.test_samples),
        "n_broke_train": sum(s["broke"] for s in state.train_samples),
        "n_broke_test": sum(s["broke"] for s in state.test_samples),
        "device": str(state.device),
        "n_params": sum(p.numel() for p in state.model.parameters()),
    }


@app.get("/api/train/status")
def training_status():
    return state.trainer.status()


@app.post("/api/train/stop")
def stop_training():
    state.trainer.stop()
    return {"status": "stopping"}


@app.post("/api/train/reset")
def reset_model():
    if state.trainer.running:
        state.trainer.stop()
    state.model = None
    state.train_samples = []
    state.test_samples = []
    state.trainer = Trainer()
    return {"status": "reset"}


@app.get("/api/train/test_files")
def get_test_files():
    """Return the list of files in the current test set."""
    if not state.test_samples:
        return {"files": []}
    files = []
    for s in state.test_samples:
        folder = os.path.basename(os.path.dirname(s["file"]))
        fname = os.path.basename(s["file"])
        files.append({
            "name": fname,
            "folder": folder,
            "path": f"{folder}/{fname}",
            "broke": s["broke"],
            "n_samples": s["accel"].shape[0],
            "params": {k: float(v) for k, v in zip(PARAM_KEYS, s["params"].tolist())},
        })
    return {"files": files}


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------

def _evaluate_samples(samples: list[dict], cfg: PipelineConfig, window_position: float):
    """Run inference on a list of samples (must have 'harmonics' key)."""
    model = state.model
    if model is None:
        return None, "No model loaded. Train first."

    model.eval()
    results = []
    inference_times: list[float] = []
    with torch.no_grad():
        for s in samples:
            T = s["harmonics"].shape[0]
            if T < cfg.cnn_window:
                continue
            max_start = T - cfg.cnn_window
            start = int(max_start * window_position)
            hw = s["harmonics"][start : start + cfg.cnn_window]
            ht = torch.tensor(hw).unsqueeze(0).to(state.device)
            pt = torch.tensor(s["params"]).unsqueeze(0).to(state.device)

            t0 = time.perf_counter()
            logit = model(ht, pt).item()
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000)

            prob = torch.sigmoid(torch.tensor(logit)).item()

            folder = os.path.basename(os.path.dirname(s["file"]))
            filename = os.path.basename(s["file"])
            results.append({
                "file": f"{folder}/{filename}",
                "true_label": int(s["broke"]),
                "predicted": 1 if prob > 0.5 else 0,
                "probability": round(prob, 4),
            })

    if not results:
        return None, "No samples with enough time steps"

    true_labels = [r["true_label"] for r in results]
    preds = [r["predicted"] for r in results]

    cm = confusion_matrix(true_labels, preds, labels=[0, 1]).tolist()
    report = classification_report(
        true_labels, preds, target_names=["OK", "Broke"], output_dict=True, zero_division=0,
    )
    acc = accuracy_score(true_labels, preds)

    return {
        "results": results,
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": round(acc, 4),
        "n_samples": len(results),
        "avg_inference_ms": round(sum(inference_times) / len(inference_times), 3) if inference_times else 0,
    }, None


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    if state.model is None:
        return {"error": "No model loaded. Train first."}

    cfg = state.config

    if req.source == "test_set":
        if not state.test_samples:
            return {"error": "No test set available. Train first."}
        result, err = _evaluate_samples(state.test_samples, cfg, req.window_position)
    else:
        data_dir = resolve_data_dir(cfg.data_dir)
        samples = load_samples(data_dir, req.folders)
        for s in samples:
            fg = s["params"][4] / 60.0
            s["harmonics"] = compute_harmonics_with_mag(
                s["accel"], fg, cfg.harm_mults,
                fft_win=cfg.fft_window, fft_step=cfg.fft_step,
            )
        result, err = _evaluate_samples(samples, cfg, req.window_position)

    if err:
        return {"error": err}
    return result


# ---------------------------------------------------------------------------
# WebSocket: streaming simulation
# ---------------------------------------------------------------------------

@app.websocket("/ws/simulate")
async def simulate(websocket: WebSocket):
    await websocket.accept()

    try:
        start_data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    if start_data.get("action") != "start":
        await websocket.send_json({"type": "error", "message": "Expected start action"})
        await websocket.close()
        return

    if state.model is None:
        await websocket.send_json({"type": "error", "message": "No model loaded. Train first."})
        await websocket.close()
        return

    control = {
        "paused": False,
        "stopped": False,
        "speed": start_data.get("speed", 5),
    }

    async def reader():
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if action == "pause":
                    control["paused"] = True
                elif action == "resume":
                    control["paused"] = False
                elif action == "stop":
                    control["stopped"] = True
                    break
                elif action == "set_speed":
                    control["speed"] = max(1, data.get("speed", 5))
        except WebSocketDisconnect:
            control["stopped"] = True

    reader_task = asyncio.create_task(reader())

    try:
        cfg = state.config
        data_dir = resolve_data_dir(cfg.data_dir)
        file_path = start_data["file_path"]
        full_path = os.path.join(data_dir, file_path)

        with open(full_path) as f:
            d = json_lib.load(f)

        chans = sorted(k for k in d if k.startswith("Channel_"))[:3]
        accel = np.column_stack([d[ch]["Signal"] for ch in chans]).astype(np.float32)
        params = np.array([d.get(k, 0) for k in PARAM_KEYS], dtype=np.float32)
        broke = bool(d.get("break", False))

        fg = params[4] / 60.0
        z = int(params[1])
        # Full input the model sees: X, Y, Z, |accel| harmonics concatenated.
        harmonics_full = compute_harmonics_with_mag(
            accel, fg, cfg.harm_mults,
            fft_win=cfg.fft_window, fft_step=cfg.fft_step,
        )
        n_h = len(cfg.harm_mults)
        harmonics = harmonics_full[:, : 3 * n_h]   # XYZ portion (frontend chart)
        mag_harmonics = harmonics_full[:, 3 * n_h :]  # magnitude portion (frontend chart)

        T = harmonics_full.shape[0]
        if T == 0:
            await websocket.send_json({"type": "error", "message": "No harmonic time steps"})
            return

        state.model.eval()
        W_np = state.model.W.detach().cpu().numpy()
        b_np = state.model.b.detach().cpu().numpy()
        mean_np = state.model.param_mean.detach().cpu().numpy()
        std_np = state.model.param_std.detach().cpu().numpy()
        params_std = (params - mean_np) / std_np
        w_vec = params_std @ W_np.T + b_np

        ch_names = ["X", "Y", "Z"]
        harm_labels = [f"{ch}·{m}×fg" for ch in ch_names for m in cfg.harm_mults]
        mag_harm_labels = [f"Mag·{m}×fg" for m in cfg.harm_mults]

        await websocket.send_json({
            "type": "init",
            "total_steps": T,
            "n_features": int(harmonics_full.shape[1]),
            "harm_labels": harm_labels,
            "mag_harm_labels": mag_harm_labels,
            "harm_mults": cfg.harm_mults,
            "spindle_freq": float(fg),
            "z": z,
            "cnn_window": cfg.cnn_window,
            "broke": broke,
            "params": {k: float(v) for k, v in zip(PARAM_KEYS, params.tolist())},
            "file": file_path,
            "w_vec": w_vec.tolist(),
        })

        for t in range(T):
            if control["stopped"]:
                break

            while control["paused"] and not control["stopped"]:
                await asyncio.sleep(0.05)

            if control["stopped"]:
                break

            combined = float(harmonics_full[t] @ w_vec)

            prob = None
            inference_ms = None
            if t >= cfg.cnn_window - 1:
                with torch.no_grad():
                    hw = harmonics_full[t - cfg.cnn_window + 1 : t + 1]
                    ht = torch.tensor(hw).unsqueeze(0).to(state.device)
                    pt = torch.tensor(params).unsqueeze(0).to(state.device)
                    t0 = time.perf_counter()
                    logit = state.model(ht, pt).item()
                    t1 = time.perf_counter()
                    inference_ms = round((t1 - t0) * 1000, 3)
                    prob = float(torch.sigmoid(torch.tensor(logit)).item())

            await websocket.send_json({
                "type": "step",
                "t": t,
                "harmonics": harmonics[t].tolist(),
                "mag_harmonics": mag_harmonics[t].tolist(),
                "combined": combined,
                "prob": prob,
                "inference_ms": inference_ms,
            })

            speed = max(1, control["speed"])
            await asyncio.sleep(1.0 / speed)

        if not control["stopped"]:
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        reader_task.cancel()

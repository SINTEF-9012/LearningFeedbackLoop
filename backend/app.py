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
from .harmonics import (
    CHANNEL_NAMES,
    DEFAULT_CHANNELS,
    PAIR_FEATURE_DIM,
    compute_peak_pairs,
)
from .model import HarmonicPairBreakNet
from .of_replay import (
    MACHINES,
    detect_cutting_windows,
    extract_step,
    find_of_files,
    list_machines,
    list_ofs,
    load_of_stream,
    slice_by_window,
    workspace_root,
)
from .trainer import Trainer

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    data_dir: str = "../lfl/testdata"
    fft_window: int = 4096
    fft_step: int = 4096
    sample_rate: float = 4096.0
    k_peaks: int = 5
    f_max_rel: float | None = 12.0   # ignore peaks above 12x spindle by default
    cnn_window: int = 16
    pair_embed_dim: int = 16
    conv_channels: list[int] = [16, 16]
    fc_hidden: int = 32
    kernel_size: int = 5


class TrainRequest(BaseModel):
    folders: list[str]
    test_split: float = 0.2
    val_split: float = 0.15
    lr_schedule: list[dict]
    batch_size: int = 16
    patience: int = 0
    n_windows: int = 1


class EvalRequest(BaseModel):
    source: str = "test_set"
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
        self.model: Optional[HarmonicPairBreakNet] = None
        self.trainer = Trainer()
        self.train_samples: list[dict] = []
        self.val_samples: list[dict] = []
        self.test_samples: list[dict] = []


state = AppState()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pairs(sample: dict, cfg: PipelineConfig) -> np.ndarray:
    """Compute the (T, C, K, 2) peak-pair tensor for a sample in place-style."""
    fg = float(sample["params"][PARAM_KEYS.index("n")]) / 60.0
    return compute_peak_pairs(
        sample["accel"],
        fg=fg,
        fft_win=cfg.fft_window,
        fft_step=cfg.fft_step,
        k_peaks=cfg.k_peaks,
        sample_rate=cfg.sample_rate,
        channels=DEFAULT_CHANNELS,
        f_max_rel=cfg.f_max_rel,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ToolBreak Pair-Input Pipeline")
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
    cfg["channel_names"] = CHANNEL_NAMES
    return cfg


@app.post("/api/config")
def set_config(config: PipelineConfig):
    t = config.cnn_window
    for _ in config.conv_channels:
        t = t // 2
    if t < 1:
        return {"error": f"cnn_window={config.cnn_window} too small for {len(config.conv_channels)} pool layers"}
    state.config = config
    return {"status": "ok"}


@app.get("/api/model/weights")
def get_model_weights():
    """Diagnostics for the parameter-conditioned per-pair encoder.

    The per-pair encoder's first linear layer has an effective weight matrix
    ``W_eff(p) = W0 + p @ M`` (shared across channels and peak slots) where:
      - ``W0`` of shape ``(D, 2)`` is the baseline reading of a (f_rel, amp)
        pair when the machine parameters sit at their training mean.
      - ``M`` of shape ``(D, n_params, 2)`` is the parameter modulation:
        slice ``M[:, p, :]`` says how cutting parameter ``p`` tilts the
        baseline reading of (f_rel, amp).

    We expose both so the UI can show per-parameter modulation heatmaps.
    """
    if state.model is None:
        return {"error": "No model loaded. Train first."}
    m = state.model
    enc = m.pair_encoder
    W0 = enc.W0.detach().cpu().numpy()         # (D, 2)
    M = enc.M.detach().cpu().numpy()           # (D, n_params, 2)
    b1 = enc.b1.detach().cpu().numpy()         # (D,)
    return {
        "pair_encoder_W0": W0.tolist(),
        "pair_encoder_M": M.tolist(),
        "pair_encoder_b1": b1.tolist(),
        "pair_input_labels": ["f_rel", "amp"],
        "pair_embed_dim": int(W0.shape[0]),
        "param_mean": m.param_mean.detach().cpu().numpy().tolist(),
        "param_std": m.param_std.detach().cpu().numpy().tolist(),
        "param_keys": PARAM_KEYS,
        "channel_names": CHANNEL_NAMES,
        "k_peaks": int(m.k_peaks),
        "n_channels": int(m.n_channels),
        "n_params": int(M.shape[1]),
    }


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

    need_new_model = state.model is None

    if need_new_model or not state.train_samples:
        samples = load_samples(data_dir, req.folders)
        if len(samples) < 2:
            return {"error": f"Need >=2 samples, got {len(samples)}"}

        for s in samples:
            s["pairs"] = _extract_pairs(s, cfg)

        samples = [s for s in samples if s["pairs"].shape[0] >= cfg.cnn_window]
        if len(samples) < 2:
            return {"error": "Not enough samples with sufficient time steps"}

        labels = [s["broke"] for s in samples]
        n_classes = len(set(labels))
        kwargs = dict(test_size=req.test_split, random_state=42)
        if n_classes >= 2:
            kwargs["stratify"] = labels
        trainval_idx, test_idx = train_test_split(range(len(samples)), **kwargs)

        trainval_samples = [samples[i] for i in trainval_idx]
        state.test_samples = [samples[i] for i in test_idx]

        if req.val_split > 0 and len(trainval_samples) >= 4:
            tv_labels = [s["broke"] for s in trainval_samples]
            v_kwargs = dict(test_size=req.val_split, random_state=42)
            if len(set(tv_labels)) >= 2:
                v_kwargs["stratify"] = tv_labels
            train_idx2, val_idx2 = train_test_split(range(len(trainval_samples)), **v_kwargs)
            state.train_samples = [trainval_samples[i] for i in train_idx2]
            state.val_samples = [trainval_samples[i] for i in val_idx2]
        else:
            state.train_samples = trainval_samples
            state.val_samples = []

    if need_new_model:
        state.model = HarmonicPairBreakNet(
            n_channels=len(DEFAULT_CHANNELS),
            k_peaks=cfg.k_peaks,
            pair_in_dim=PAIR_FEATURE_DIM,
            pair_embed_dim=cfg.pair_embed_dim,
            n_params=len(PARAM_KEYS),
            cnn_window=cfg.cnn_window,
            conv_channels=cfg.conv_channels,
            fc_hidden=cfg.fc_hidden,
            ks=cfg.kernel_size,
        ).to(state.device)

        train_params = np.stack([s["params"] for s in state.train_samples])
        state.model.set_param_stats(
            torch.tensor(train_params.mean(axis=0), dtype=torch.float32),
            torch.tensor(train_params.std(axis=0), dtype=torch.float32),
        )

    state.trainer.start(
        model=state.model,
        train_samples=state.train_samples,
        val_samples=state.val_samples,
        lr_schedule=req.lr_schedule,
        cnn_window=cfg.cnn_window,
        device=state.device,
        batch_size=req.batch_size,
        reset_history=need_new_model,
        patience=req.patience,
        n_windows=req.n_windows,
    )

    return {
        "status": "started" if need_new_model else "continued",
        "n_train": len(state.train_samples),
        "n_val": len(state.val_samples),
        "n_test": len(state.test_samples),
        "n_broke_train": sum(s["broke"] for s in state.train_samples),
        "n_broke_val": sum(s["broke"] for s in state.val_samples),
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
    state.val_samples = []
    state.test_samples = []
    state.trainer = Trainer()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(workspace_root(), "models")
_SAFE_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _safe_model_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Empty model name")
    if name.endswith(".pt"):
        name = name[:-3]
    if any(c not in _SAFE_NAME_CHARS for c in name):
        raise ValueError("Model name may only contain letters, digits, '-', '_' and '.'")
    return name


class SaveModelRequest(BaseModel):
    name: str


class LoadModelRequest(BaseModel):
    name: str


@app.get("/api/model/list")
def list_saved_models():
    if not os.path.isdir(MODELS_DIR):
        return {"models": []}
    entries = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if not f.endswith(".pt"):
            continue
        path = os.path.join(MODELS_DIR, f)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entries.append({
            "name": f[:-3],
            "size_bytes": st.st_size,
            "mtime": st.st_mtime,
        })
    return {"models": entries}


@app.post("/api/model/save")
def save_model(req: SaveModelRequest):
    if state.model is None:
        return {"error": "No model in memory. Train first."}
    try:
        name = _safe_model_name(req.name)
    except ValueError as e:
        return {"error": str(e)}
    os.makedirs(MODELS_DIR, exist_ok=True)
    cfg = state.config
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state.model.state_dict().items()},
        "config": cfg.model_dump(),
        "architecture": {
            "n_channels": len(DEFAULT_CHANNELS),
            "k_peaks": cfg.k_peaks,
            "pair_in_dim": PAIR_FEATURE_DIM,
            "pair_embed_dim": cfg.pair_embed_dim,
            "n_params": len(PARAM_KEYS),
            "cnn_window": cfg.cnn_window,
            "conv_channels": cfg.conv_channels,
            "fc_hidden": cfg.fc_hidden,
            "kernel_size": cfg.kernel_size,
        },
        "param_keys": PARAM_KEYS,
        "channel_names": [CHANNEL_NAMES[c] for c in DEFAULT_CHANNELS],
    }
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save(payload, path)
    return {"status": "saved", "name": name, "path": os.path.relpath(path, workspace_root())}


@app.post("/api/model/load")
def load_saved_model(req: LoadModelRequest):
    try:
        name = _safe_model_name(req.name)
    except ValueError as e:
        return {"error": str(e)}
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.isfile(path):
        return {"error": f"Model '{name}' not found"}
    payload = torch.load(path, map_location=state.device, weights_only=False)
    arch = payload["architecture"]
    cfg_dict = payload.get("config", {})
    # Refresh config from saved values so downstream evaluation matches.
    for k, v in cfg_dict.items():
        if hasattr(state.config, k):
            setattr(state.config, k, v)
    model = HarmonicPairBreakNet(
        n_channels=arch["n_channels"],
        k_peaks=arch["k_peaks"],
        pair_in_dim=arch["pair_in_dim"],
        pair_embed_dim=arch["pair_embed_dim"],
        n_params=arch["n_params"],
        cnn_window=arch["cnn_window"],
        conv_channels=arch["conv_channels"],
        fc_hidden=arch["fc_hidden"],
        kernel_size=arch["kernel_size"],
    ).to(state.device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    state.model = model
    return {"status": "loaded", "name": name}


@app.get("/api/train/test_files")
def get_test_files():
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
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_samples(samples: list[dict], cfg: PipelineConfig, window_position: float):
    model = state.model
    if model is None:
        return None, "No model loaded. Train first."

    model.eval()
    results = []
    inference_times: list[float] = []
    with torch.no_grad():
        for s in samples:
            T = s["pairs"].shape[0]
            if T < cfg.cnn_window:
                continue
            max_start = T - cfg.cnn_window
            start = int(max_start * window_position)
            pw = s["pairs"][start : start + cfg.cnn_window]
            pt_pairs = torch.tensor(pw).unsqueeze(0).to(state.device)
            pt_params = torch.tensor(s["params"]).unsqueeze(0).to(state.device)

            t0 = time.perf_counter()
            logit = model(pt_pairs, pt_params).item()
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
            s["pairs"] = _extract_pairs(s, cfg)
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

        fg = float(params[PARAM_KEYS.index("n")]) / 60.0

        pairs = compute_peak_pairs(
            accel,
            fg=fg,
            fft_win=cfg.fft_window,
            fft_step=cfg.fft_step,
            k_peaks=cfg.k_peaks,
            sample_rate=cfg.sample_rate,
            channels=DEFAULT_CHANNELS,
            f_max_rel=cfg.f_max_rel,
        )

        T = pairs.shape[0]
        if T == 0:
            await websocket.send_json({"type": "error", "message": "No time steps"})
            return

        state.model.eval()

        await websocket.send_json({
            "type": "init",
            "total_steps": T,
            "n_channels": len(DEFAULT_CHANNELS),
            "k_peaks": cfg.k_peaks,
            "channel_names": CHANNEL_NAMES,
            "spindle_freq": fg,
            "cnn_window": cfg.cnn_window,
            "broke": broke,
            "params": {k: float(v) for k, v in zip(PARAM_KEYS, params.tolist())},
            "file": file_path,
            "f_max_rel": cfg.f_max_rel,
        })

        for t in range(T):
            if control["stopped"]:
                break
            while control["paused"] and not control["stopped"]:
                await asyncio.sleep(0.05)
            if control["stopped"]:
                break

            prob = None
            inference_ms = None
            if t >= cfg.cnn_window - 1:
                with torch.no_grad():
                    pw = pairs[t - cfg.cnn_window + 1 : t + 1]
                    pt_pairs = torch.tensor(pw).unsqueeze(0).to(state.device)
                    pt_params = torch.tensor(params).unsqueeze(0).to(state.device)
                    t0 = time.perf_counter()
                    logit = state.model(pt_pairs, pt_params).item()
                    t1 = time.perf_counter()
                    inference_ms = round((t1 - t0) * 1000, 3)
                    prob = float(torch.sigmoid(torch.tensor(logit)).item())

            # pairs[t] is (C, K, 2). Send as nested lists; UI can plot per
            # channel: amplitude vs (f_rel * fg) for each peak.
            await websocket.send_json({
                "type": "step",
                "t": t,
                "pairs": pairs[t].tolist(),  # (C, K, 2)
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


# ---------------------------------------------------------------------------
# OF replay (real-machine streaming inference on Komatsu/Goimek/WWR data)
# ---------------------------------------------------------------------------

class OFWindowsRequest(BaseModel):
    machine_id: str
    of: str


@app.get("/api/of/machines")
def of_machines():
    return {"machines": list_machines()}


@app.get("/api/of/ofs/{machine_id}")
def of_ofs(machine_id: str):
    if machine_id not in MACHINES:
        return {"error": f"Unknown machine '{machine_id}'", "ofs": []}
    return {"ofs": list_ofs(machine_id)}


@app.post("/api/of/windows")
def of_windows(req: OFWindowsRequest):
    if req.machine_id not in MACHINES:
        return {"error": f"Unknown machine '{req.machine_id}'"}
    paths = find_of_files(req.machine_id, req.of)
    if not paths["tyzbps"]:
        return {"error": "Missing TYZBPS file in this OF"}
    wins = detect_cutting_windows(req.machine_id, req.of)
    return {
        "windows": wins,
        "files": {k: (os.path.basename(v) if v else None) for k, v in paths.items()},
    }


@app.websocket("/ws/of_replay")
async def of_replay_ws(websocket: WebSocket):
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

    machine_id = start_data.get("machine_id")
    of = start_data.get("of")
    start_iso = start_data.get("start")
    end_iso = start_data.get("end")
    if not (machine_id and of and start_iso and end_iso):
        await websocket.send_json({"type": "error", "message": "machine_id, of, start, end required"})
        await websocket.close()
        return

    control = {
        "paused": False,
        "stopped": False,
        "speed": start_data.get("speed", 50),
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
                    control["speed"] = max(1, data.get("speed", 50))
        except WebSocketDisconnect:
            control["stopped"] = True

    reader_task = asyncio.create_task(reader())

    try:
        cfg = state.config
        stream = load_of_stream(machine_id, of)
        idxs = slice_by_window(stream, start_iso, end_iso)
        if len(idxs) == 0:
            await websocket.send_json({"type": "error", "message": "Window contains no vibration rows"})
            return

        await websocket.send_json({
            "type": "init",
            "total_steps": int(len(idxs)),
            "n_channels": 2,
            "k_peaks": cfg.k_peaks,
            "channel_names": CHANNEL_NAMES,
            "cnn_window": cfg.cnn_window,
            "machine_id": machine_id,
            "of": of,
            "start": start_iso,
            "end": end_iso,
            "f_max_rel": cfg.f_max_rel,
        })

        state.model.eval()

        # Rolling buffer for the temporal context. We push every step regardless
        # of validity, with zero pairs for invalid (no-tool) rows.
        pair_buf: list[np.ndarray] = []
        last_tool = None

        for step_i, vib_row in enumerate(idxs):
            if control["stopped"]:
                break
            while control["paused"] and not control["stopped"]:
                await asyncio.sleep(0.05)
            if control["stopped"]:
                break

            s = extract_step(stream, int(vib_row), k_peaks=cfg.k_peaks, f_max_rel=cfg.f_max_rel)
            pair_buf.append(s["pairs"])
            if len(pair_buf) > cfg.cnn_window:
                pair_buf.pop(0)

            prob = None
            inference_ms = None
            if len(pair_buf) == cfg.cnn_window and s["valid"]:
                pw = np.stack(pair_buf, axis=0)  # (T, C, K, 2)
                with torch.no_grad():
                    pt_pairs = torch.tensor(pw).unsqueeze(0).to(state.device)
                    pt_params = torch.tensor(s["params"]).unsqueeze(0).to(state.device)
                    t0 = time.perf_counter()
                    logit = state.model(pt_pairs, pt_params).item()
                    t1 = time.perf_counter()
                    inference_ms = round((t1 - t0) * 1000, 3)
                    prob = float(torch.sigmoid(torch.tensor(logit)).item())

            tool_changed = (s["tool_number"] != last_tool)
            last_tool = s["tool_number"]

            await websocket.send_json({
                "type": "step",
                "t": step_i,
                "ts": s["ts"],
                "pairs": s["pairs"].tolist(),
                "prob": prob,
                "inference_ms": inference_ms,
                "tool_number": s["tool_number"],
                "tool_description": s["tool_description"],
                "diameter_mm": s["diameter_mm"],
                "n_inserts": s["n_inserts"],
                "spindle_rpm": s["spindle_rpm"],
                "feed_rate": s["feed_rate"],
                "operation_mode": s["operation_mode"],
                "valid": s["valid"],
                "tool_changed": bool(tool_changed and step_i > 0),
                "params": {k: float(v) for k, v in zip(PARAM_KEYS, s["params"].tolist())},
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

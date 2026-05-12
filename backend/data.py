import json
import os
import glob
import numpy as np

PARAM_KEYS = ["d", "z", "n", "f", "vf"]


def load_samples(data_dir: str, folders: list[str] | None = None) -> list[dict]:
    """Load accelerometer samples from JSON files in data_dir."""
    samples = []
    for folder in sorted(os.listdir(data_dir)):
        if folders and folder not in folders:
            continue
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for jf in sorted(glob.glob(os.path.join(folder_path, "*.json"))):
            with open(jf) as fh:
                d = json.load(fh)
            chans = sorted([k for k in d if k.startswith("Channel_")])[:3]
            if not chans:
                continue
            accel = np.column_stack([d[ch]["Signal"] for ch in chans]).astype(np.float32)
            params = np.array([d.get(k, 0) for k in PARAM_KEYS], dtype=np.float32)
            samples.append({
                "accel": accel,
                "params": params,
                "broke": bool(d.get("break", False)),
                "file": jf,
                "folder": folder,
            })
    return samples


def resolve_data_dir(configured: str) -> str:
    """Try to resolve the data directory from common locations."""
    candidates = [
        configured,
        os.path.join(os.getcwd(), "testdata"),
        os.path.join(os.getcwd(), "..", "lfl", "testdata"),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(p):
            return p
    return os.path.abspath(configured)

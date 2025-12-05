import asyncio
import traceback
import sys
import os
from typing import Any, Dict

# Add parent directory to path for sibling module imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import events


class OnlineAgent:
    """Online learner agent using River HoeffdingTree (lazy import).

    Subscribes to the global feature bus and updates an online model.
    """

    def __init__(self, model_name: str = "HoeffdingTree"):
        self.model_name = model_name
        self._task = None
        self._model = None
        self._running = False

    def _ensure_model(self):
        if self._model is None:
            try:
                from river import tree
            except Exception as e:
                raise RuntimeError("Please install 'river' to use OnlineAgent") from e
            # simple default classifier
            self._model = tree.HoeffdingTreeClassifier()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._running = True
        self._ensure_model()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run(self):
        q = events.subscribe_features()
        while self._running:
            try:
                ev = await q.get()
                # expect ev to be dict with 'frame' or 'payload' containing numeric info
                features = self._extract_features(ev)
                label = ev.get("label")
                if label is not None:
                    # supervised update
                    self._model.learn_one(features, label)
                else:
                    # unsupervised: we could predict
                    _ = self._model.predict_proba_one(features)
            except asyncio.CancelledError:
                break
            except Exception:
                traceback.print_exc()

    def _extract_features(self, ev: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        # prefer payload then frame
        data = ev.get("payload") or ev.get("frame") or {}
        if not isinstance(data, dict):
            return out
        for k, v in data.items():
            if k in ("type", "session_id"):
                continue
            try:
                # if it's a list/ndarray compute mean
                if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
                    import numpy as _np
                    arr = _np.asarray(v)
                    if arr.size == 0:
                        continue
                    out[f"{k}_mean"] = float(_np.mean(arr))
                    out[f"{k}_std"] = float(_np.std(arr))
                else:
                    out[k] = float(v)
            except Exception:
                # skip non-numeric
                continue
        return out

    # agent interface
    async def handle_request(self, session_id: str, action: str, args: Dict[str, Any], context: Dict[str, Any]):
        if action == "status":
            return {"running": self._running}
        if action == "start":
            await self.start()
            return {"ok": True}
        if action == "stop":
            await self.stop()
            return {"ok": True}
        raise ValueError(f"Unsupported OnlineAgent action: {action}")

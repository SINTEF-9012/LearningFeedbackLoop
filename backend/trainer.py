import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HarmonicWindowDataset(Dataset):
    """Flat (sample, window) dataset: one item = one window from one sample.

    Three modes:

    1. Random (default): each sample contributes ``n_windows`` items per
       epoch, each with a freshly-drawn random start in
       ``[0, T - cnn_window]``. Used for training; combined with
       ``DataLoader(shuffle=True)`` it distributes the windows from any one
       cut across multiple batches.

    2. ``deterministic=True``: each sample contributes exactly one item,
       the centered window. ``n_windows`` is ignored.

    3. ``all_windows=True``: each sample contributes *every* sliding window
       of length ``cnn_window`` (stride 1, ``T - cnn_window + 1`` items per
       sample). This is the natural choice for validation: every available
       window in each val series is evaluated, and the BCE/accuracy is the
       true mean across the whole val set's window distribution. Overrides
       ``deterministic`` and ``n_windows``.

    Samples with ``T < cnn_window`` are dropped at construction time.
    """

    def __init__(
        self,
        samples: list[dict],
        cnn_window: int,
        n_windows: int = 1,
        deterministic: bool = False,
        all_windows: bool = False,
    ):
        self.cnn_window = cnn_window
        self.deterministic = deterministic
        self.all_windows = all_windows
        if all_windows:
            self.n_windows = 1  # n_windows ignored; index map handles it
        elif deterministic:
            self.n_windows = 1
        else:
            self.n_windows = max(1, int(n_windows))
        # Filter out samples that can't fit a single window.
        self.samples = [s for s in samples if s["harmonics"].shape[0] >= cnn_window]
        if all_windows:
            # Build (sample_idx, start) index for every available window.
            self._index: list[tuple[int, int]] = []
            for si, s in enumerate(self.samples):
                T = s["harmonics"].shape[0]
                for start in range(T - cnn_window + 1):
                    self._index.append((si, start))
        else:
            self._index = None

    def __len__(self):
        if self.all_windows:
            return len(self._index)
        return len(self.samples) * self.n_windows

    def __getitem__(self, idx):
        if self.all_windows:
            si, start = self._index[idx]
            s = self.samples[si]
        else:
            s = self.samples[idx // self.n_windows]
            T = s["harmonics"].shape[0]
            if self.deterministic:
                start = (T - self.cnn_window) // 2
            else:
                start = int(np.random.randint(0, T - self.cnn_window + 1))
        return {
            "harmonics": s["harmonics"][start : start + self.cnn_window],
            "params": s["params"],
            "broke": s["broke"],
        }


def window_collate(batch):
    """Stack a batch of (window, params, label) dicts into tensors."""
    if not batch:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    harms = np.stack([b["harmonics"] for b in batch])
    params = np.stack([b["params"] for b in batch])
    labels = np.array([float(b["broke"]) for b in batch], dtype=np.float32)
    return (
        torch.tensor(harms),
        torch.tensor(params),
        torch.tensor(labels, dtype=torch.float32),
    )


class Trainer:
    """Manages background model training with status reporting."""

    def __init__(self):
        self.running = False
        self.should_stop = False
        self.history: list[float] = []
        self.val_history: list[float] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_stage = 0
        self.total_stages = 0
        # Early stopping / best-weights bookkeeping. best_epoch is 1-indexed
        # to match what we display in the UI; -1 means "no best yet".
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.epochs_since_improve: int = 0
        self.early_stopped: bool = False
        self._best_state: dict | None = None
        self._thread: threading.Thread | None = None
        # Persisted across calls so Adam's momentum (m, v) survives between
        # successive "Train" clicks. Recreating Adam each time would reset the
        # moment estimates and the first step would become ~lr * sign(grad),
        # which can knock a converged model out of a sharp minimum and send
        # the loss back to ~log(2). We rebuild only when explicitly reset
        # (new model / new training session).
        self._optimizer: torch.optim.Optimizer | None = None

    def reset_optimizer(self):
        self._optimizer = None

    def start(
        self,
        model: nn.Module,
        train_samples: list[dict],
        lr_schedule: list[dict],
        cnn_window: int,
        device: torch.device,
        batch_size: int = 16,
        reset_history: bool = True,
        val_samples: list[dict] | None = None,
        patience: int = 0,
        n_windows: int = 1,
    ):
        if self.running:
            return
        self.running = True
        self.should_stop = False
        self.early_stopped = False
        if reset_history:
            self.history = []
            self.val_history = []
            self.current_epoch = 0
            # A fresh training session also gets a fresh optimizer and a fresh
            # best-weights tracker.
            self._optimizer = None
            self.best_val_loss = float("inf")
            self.best_epoch = -1
            self.epochs_since_improve = 0
            self._best_state = None
        self.total_epochs = self.current_epoch + sum(s["epochs"] for s in lr_schedule)
        self.total_stages = len(lr_schedule)
        self.current_stage = 0

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(model, train_samples, lr_schedule, cnn_window, device, batch_size, val_samples, patience, n_windows),
            daemon=True,
        )
        self._thread.start()

    def _train_loop(self, model, train_samples, lr_schedule, cnn_window, device, batch_size, val_samples=None, patience=0, n_windows=1):
        try:
            dataset = HarmonicWindowDataset(train_samples, cnn_window, n_windows=n_windows)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=window_collate,
                num_workers=0,
            )
            val_loader = None
            if val_samples:
                # Validation enumerates every available sliding window of every
                # val sample (stride 1). The val loss is then the true mean BCE
                # over the entire val window distribution, not a noisy random
                # sample of it.
                val_loader = DataLoader(
                    HarmonicWindowDataset(val_samples, cnn_window, all_windows=True),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=window_collate,
                    num_workers=0,
                )
            loss_fn = nn.BCEWithLogitsLoss()

            for stage_idx, stage in enumerate(lr_schedule):
                if self.should_stop:
                    break
                self.current_stage = stage_idx + 1

                # Reuse a persistent Adam to keep momentum across stages and
                # across separate calls to start(); just update the lr.
                if self._optimizer is None:
                    self._optimizer = torch.optim.Adam(model.parameters(), lr=stage["lr"])
                else:
                    for pg in self._optimizer.param_groups:
                        pg["lr"] = stage["lr"]
                optimizer = self._optimizer

                for _epoch in range(stage["epochs"]):
                    if self.should_stop:
                        break
                    model.train()
                    epoch_loss, n_b = 0.0, 0
                    for harmonics, params, labels in loader:
                        if harmonics.numel() == 0:
                            continue
                        harmonics = harmonics.to(device)
                        params = params.to(device)
                        labels = labels.to(device)
                        loss = loss_fn(model(harmonics, params), labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_b += 1
                    avg = epoch_loss / max(n_b, 1)
                    self.history.append(avg)

                    # Validation loss (no grad, eval mode for stable BN stats).
                    if val_loader is not None:
                        model.eval()
                        v_loss, v_n = 0.0, 0
                        with torch.no_grad():
                            for harmonics, params, labels in val_loader:
                                if harmonics.numel() == 0:
                                    continue
                                harmonics = harmonics.to(device)
                                params = params.to(device)
                                labels = labels.to(device)
                                v_loss += loss_fn(model(harmonics, params), labels).item()
                                v_n += 1
                        v_avg = v_loss / max(v_n, 1)
                        self.val_history.append(v_avg)

                        # Track best-so-far weights and check patience.
                        if v_avg < self.best_val_loss:
                            self.best_val_loss = v_avg
                            self.best_epoch = self.current_epoch + 1
                            self.epochs_since_improve = 0
                            # Snapshot weights on CPU so they are safe to
                            # restore later regardless of device transitions.
                            self._best_state = {
                                k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()
                            }
                        else:
                            self.epochs_since_improve += 1
                            if patience > 0 and self.epochs_since_improve >= patience:
                                self.current_epoch += 1
                                self.early_stopped = True
                                self.should_stop = True
                                break

                    self.current_epoch += 1
        finally:
            # Restore best-on-validation weights so the trained model the user
            # keeps is the one that generalised best, not whatever it ended at.
            if self._best_state is not None:
                try:
                    model.load_state_dict({
                        k: v.to(device) for k, v in self._best_state.items()
                    })
                except Exception:
                    # Architecture mismatch or similar — leave model as-is.
                    pass
            self.running = False

    def stop(self):
        self.should_stop = True

    def status(self) -> dict:
        return {
            "running": self.running,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "history": self.history,
            "val_history": self.val_history,
            "best_val_loss": (
                None if self.best_val_loss == float("inf") else self.best_val_loss
            ),
            "best_epoch": self.best_epoch,
            "epochs_since_improve": self.epochs_since_improve,
            "early_stopped": self.early_stopped,
        }

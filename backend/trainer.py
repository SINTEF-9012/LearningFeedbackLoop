import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PairWindowDataset(Dataset):
    """Flat (sample, window) dataset for the pair-input model.

    Each sample carries ``s["pairs"]`` of shape (T, C, K, 2). A dataset item
    is a single CNN window — a contiguous slice of length ``cnn_window`` along
    the time axis — together with its parameters and label.

    Three modes (same semantics as the previous harmonic dataset):
      * Random (default): each sample contributes ``n_windows`` items per
        epoch, each with a fresh random start. Combined with shuffled loading
        this distributes windows from one cut across many batches.
      * ``deterministic=True``: the centred window per sample (n_windows=1).
      * ``all_windows=True``: every stride-1 window per sample, useful for
        validation so the val loss reflects the true full-window distribution.
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
            self.n_windows = 1
        elif deterministic:
            self.n_windows = 1
        else:
            self.n_windows = max(1, int(n_windows))
        self.samples = [s for s in samples if s["pairs"].shape[0] >= cnn_window]
        if all_windows:
            self._index: list[tuple[int, int]] | None = []
            for si, s in enumerate(self.samples):
                T = s["pairs"].shape[0]
                for start in range(T - cnn_window + 1):
                    self._index.append((si, start))
        else:
            self._index = None

    def __len__(self):
        if self.all_windows and self._index is not None:
            return len(self._index)
        return len(self.samples) * self.n_windows

    def __getitem__(self, idx):
        if self.all_windows and self._index is not None:
            si, start = self._index[idx]
            s = self.samples[si]
        else:
            s = self.samples[idx // self.n_windows]
            T = s["pairs"].shape[0]
            if self.deterministic:
                start = (T - self.cnn_window) // 2
            else:
                start = int(np.random.randint(0, T - self.cnn_window + 1))
        return {
            "pairs": s["pairs"][start : start + self.cnn_window],  # (W, C, K, 2)
            "params": s["params"],
            "broke": s["broke"],
        }


def pair_collate(batch):
    if not batch:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    pairs = np.stack([b["pairs"] for b in batch])     # (B, W, C, K, 2)
    params = np.stack([b["params"] for b in batch])
    labels = np.array([float(b["broke"]) for b in batch], dtype=np.float32)
    return (
        torch.tensor(pairs),
        torch.tensor(params),
        torch.tensor(labels, dtype=torch.float32),
    )


class Trainer:
    """Background training thread for the pair-input model."""

    def __init__(self):
        self.running = False
        self.should_stop = False
        self.history: list[float] = []
        self.val_history: list[float] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_stage = 0
        self.total_stages = 0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.epochs_since_improve: int = 0
        self.early_stopped: bool = False
        self._best_state: dict | None = None
        self._thread: threading.Thread | None = None
        # Persisted across calls so Adam's momentum survives between
        # successive "Train" clicks (avoids loss-spike on resume).
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
            args=(model, train_samples, lr_schedule, cnn_window, device,
                  batch_size, val_samples, patience, n_windows),
            daemon=True,
        )
        self._thread.start()

    def _train_loop(self, model, train_samples, lr_schedule, cnn_window,
                    device, batch_size, val_samples=None, patience=0, n_windows=1):
        try:
            dataset = PairWindowDataset(train_samples, cnn_window, n_windows=n_windows)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=pair_collate,
                num_workers=0,
            )
            val_loader = None
            if val_samples:
                # Validation uses ONE deterministic centred window per sample
                # so the val loss matches the test-set evaluation protocol
                # (which also uses a single window per sample). Using
                # ``all_windows=True`` here would over-count each sample by
                # ~(T - cnn_window + 1) heavily-correlated windows and bias
                # best-epoch selection away from what test actually measures.
                val_loader = DataLoader(
                    PairWindowDataset(val_samples, cnn_window, deterministic=True),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=pair_collate,
                    num_workers=0,
                )
            loss_fn = nn.BCEWithLogitsLoss()
            # For validation we want a sample-mean BCE that doesn't depend on
            # batch boundaries (the default reduction averages per batch, so
            # the final smaller batch is over-weighted). ``reduction="sum"``
            # + divide by the total number of val samples is exact.
            val_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

            for stage_idx, stage in enumerate(lr_schedule):
                if self.should_stop:
                    break
                self.current_stage = stage_idx + 1

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
                    for pairs, params, labels in loader:
                        if pairs.numel() == 0:
                            continue
                        pairs = pairs.to(device)
                        params = params.to(device)
                        labels = labels.to(device)
                        loss = loss_fn(model(pairs, params), labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_b += 1
                    avg = epoch_loss / max(n_b, 1)
                    self.history.append(avg)

                    if val_loader is not None:
                        model.eval()
                        v_loss, v_n = 0.0, 0
                        with torch.no_grad():
                            for pairs, params, labels in val_loader:
                                if pairs.numel() == 0:
                                    continue
                                pairs = pairs.to(device)
                                params = params.to(device)
                                labels = labels.to(device)
                                v_loss += val_loss_fn(model(pairs, params), labels).item()
                                v_n += int(labels.numel())
                        v_avg = v_loss / max(v_n, 1)
                        self.val_history.append(v_avg)

                        if v_avg < self.best_val_loss:
                            self.best_val_loss = v_avg
                            self.best_epoch = self.current_epoch + 1
                            self.epochs_since_improve = 0
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
            if self._best_state is not None:
                try:
                    model.load_state_dict({
                        k: v.to(device) for k, v in self._best_state.items()
                    })
                except Exception:
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

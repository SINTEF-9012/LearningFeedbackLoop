import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HarmonicDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(cnn_window: int, n_windows: int = 1):
    """Create a collate function that extracts random windows from harmonic series."""

    def collate(batch):
        harms, params, labels = [], [], []
        for s in batch:
            T = s["harmonics"].shape[0]
            if T < cnn_window:
                continue
            for _ in range(n_windows):
                start = np.random.randint(0, T - cnn_window + 1)
                harms.append(s["harmonics"][start : start + cnn_window])
                params.append(s["params"])
                labels.append(float(s["broke"]))
        if not harms:
            return torch.empty(0), torch.empty(0), torch.empty(0)
        return (
            torch.tensor(np.stack(harms)),
            torch.tensor(np.stack(params)),
            torch.tensor(labels, dtype=torch.float32),
        )

    return collate


class Trainer:
    """Manages background model training with status reporting."""

    def __init__(self):
        self.running = False
        self.should_stop = False
        self.history: list[float] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_stage = 0
        self.total_stages = 0
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
    ):
        if self.running:
            return
        self.running = True
        self.should_stop = False
        if reset_history:
            self.history = []
            self.current_epoch = 0
            # A fresh training session also gets a fresh optimizer.
            self._optimizer = None
        self.total_epochs = self.current_epoch + sum(s["epochs"] for s in lr_schedule)
        self.total_stages = len(lr_schedule)
        self.current_stage = 0

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(model, train_samples, lr_schedule, cnn_window, device, batch_size),
            daemon=True,
        )
        self._thread.start()

    def _train_loop(self, model, train_samples, lr_schedule, cnn_window, device, batch_size):
        try:
            dataset = HarmonicDataset(train_samples)
            collate_fn = make_collate(cnn_window)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
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
                    self.current_epoch += 1
        finally:
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
        }

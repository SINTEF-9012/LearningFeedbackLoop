import torch
import torch.nn as nn


class HarmonicBreakNet(nn.Module):
    """CNN classifier that learns to weight harmonic features via cutting parameters.

    Architecture (from ToolBreak.ipynb Pipeline 2):
      1. Learnable W matrix: params (B,7) → w (B, n_features) used to combine harmonics.
      2. Conv1d blocks with BatchNorm, ReLU, AvgPool(2), Dropout.
      3. Two FC hidden layers with BatchNorm and Dropout → logit output.
    """

    def __init__(
        self,
        n_harm_features: int = 21,
        n_params: int = 5,
        cnn_window: int = 16,
        conv_channels: list[int] | None = None,
        fc_hidden: int = 32,
        ks: int = 5,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 16]

        self.n_harm_features = n_harm_features
        self.cnn_window = cnn_window
        self.conv_channels_cfg = conv_channels

        self.W = nn.Parameter(torch.randn(n_harm_features, n_params) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_harm_features))

        # Standardization stats for cutting parameters (filled via set_param_stats).
        # Stored as buffers so they move with .to(device) and serialize with state_dict.
        self.register_buffer("param_mean", torch.zeros(n_params))
        self.register_buffer("param_std", torch.ones(n_params))

        pad = ks // 2
        layers: list[nn.Module] = []
        ch = 1
        for ch_out in conv_channels:
            layers += [
                nn.Conv1d(ch, ch_out, ks, padding=pad, padding_mode="replicate"),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(),
                nn.AvgPool1d(2),
                nn.Dropout(0.3),
            ]
            ch = ch_out
        self.conv = nn.Sequential(*layers)

        # Compute temporal dimension after pooling
        t_out = cnn_window
        for _ in conv_channels:
            t_out = t_out // 2

        self.fc1 = nn.Linear(ch * t_out, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def set_param_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set standardization stats for cutting parameters (call after train split)."""
        std = std.clone()
        std[std < 1e-8] = 1.0
        self.param_mean.copy_(mean.to(self.param_mean.device))
        self.param_std.copy_(std.to(self.param_std.device))

    def forward(self, harmonics: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            harmonics: (B, T, n_features)
            params: (B, n_params)  -- raw cutting parameters; standardized internally.
        Returns:
            logits: (B,)
        """
        params = (params - self.param_mean) / self.param_std
        w = params @ self.W.T + self.b                     # (B, n_features)
        x = torch.einsum("btc,bc->bt", harmonics, w)       # (B, T)
        x = x.unsqueeze(1)                                  # (B, 1, T)
        x = self.conv(x)                                    # (B, ch, T')
        x = x.flatten(1)
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).squeeze(-1)

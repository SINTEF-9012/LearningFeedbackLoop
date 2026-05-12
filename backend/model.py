"""Pair-input tool-break classifier.

Architecture (DeepSets-style per-pair encoder + temporal CNN):

  Input pairs:    (B, T, C, K, 2)   last dim = (f_rel, amp)
  Cutting params: (B, n_params)     standardised internally

  1. Per-pair encoder (shared MLP across all (B, T, C, K)):
        (f_rel, amp)  ->  embedding of dim D_pair
     This is permutation-equivariant in K, which lets the model see each
     spectral peak through the same lens regardless of which slot it landed
     in. It also handles padded zero-pairs naturally (their embedding is
     fixed by the shared weights and is identical for every padding slot).

  2. Aggregate over K (sum) per (B, T, C):
        peaks(t, c)  ->  channel embedding of dim D_pair
     Sum (rather than mean) keeps "more energy => bigger embedding" cues.
     Padded zeros add an offset that the network can absorb in its bias.

  3. Stack channels:  (B, T, C * D_pair)  =  (B, T, F)

  4. Temporal Conv1d stack: (B, F, T) -> (B, F', T')
     Conv1d -> BatchNorm -> ReLU -> AvgPool(2) -> Dropout, repeated.

  5. FC head with cutting-parameter conditioning:
        flatten conv output, concat standardised params, two MLP layers,
        single logit.

The model is ``torch.compile``-friendly and is small enough to train on MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HarmonicPairBreakNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 2,
        k_peaks: int = 5,
        pair_in_dim: int = 2,
        pair_embed_dim: int = 16,
        n_params: int = 5,
        cnn_window: int = 16,
        conv_channels: list[int] | None = None,
        fc_hidden: int = 32,
        ks: int = 5,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 16]

        self.n_channels = n_channels
        self.k_peaks = k_peaks
        self.pair_in_dim = pair_in_dim
        self.pair_embed_dim = pair_embed_dim
        self.cnn_window = cnn_window
        self.conv_channels_cfg = conv_channels

        # Standardisation buffers for cutting parameters.
        self.register_buffer("param_mean", torch.zeros(n_params))
        self.register_buffer("param_std", torch.ones(n_params))

        # Per-pair encoder: shared 2-layer MLP over (f_rel, amp).
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_in_dim, pair_embed_dim),
            nn.ReLU(),
            nn.Linear(pair_embed_dim, pair_embed_dim),
            nn.ReLU(),
        )

        # Temporal CNN over the per-channel embedding stream.
        per_step_dim = n_channels * pair_embed_dim
        pad = ks // 2
        layers: list[nn.Module] = []
        ch = per_step_dim
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

        t_out = cnn_window
        for _ in conv_channels:
            t_out = t_out // 2
        flat = ch * t_out

        # FC head, conditioned on standardised cutting parameters.
        self.fc1 = nn.Linear(flat + n_params, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def set_param_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        std = std.clone()
        std[std < 1e-8] = 1.0
        self.param_mean.copy_(mean.to(self.param_mean.device))
        self.param_std.copy_(std.to(self.param_std.device))

    def forward(self, pairs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs: (B, T, C, K, 2) — (f_rel, amp) per peak per channel per step.
            params: (B, n_params) raw cutting parameters; standardised inside.
        Returns:
            logits: (B,)
        """
        B, T, C, K, F = pairs.shape
        # Encode every pair through the shared MLP. Reshape so the encoder sees
        # a flat batch of pairs and we don't allocate any intermediate copies
        # of (f_rel, amp) per pair.
        e = self.pair_encoder(pairs.reshape(B * T * C * K, F))
        e = e.reshape(B, T, C, K, self.pair_embed_dim)

        # DeepSets aggregation over K peaks (sum); preserves "louder => bigger".
        e = e.sum(dim=3)                    # (B, T, C, D)
        e = e.reshape(B, T, C * self.pair_embed_dim)  # (B, T, F)

        # Conv1d expects (B, F, T).
        x = e.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)

        # Concatenate standardised cutting parameters at the FC head.
        p = (params - self.param_mean) / self.param_std
        x = torch.cat([x, p], dim=1)

        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).squeeze(-1)

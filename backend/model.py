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
import torch.nn.functional as F


class ParamConditionedPairEncoder(nn.Module):
    """Per-pair MLP whose first linear layer is conditioned on machine params.

    The first layer's effective weight matrix is

        W_eff(p) = W0 + p @ M                  # shape (D, 2)

    where ``W0`` is a parameter-independent baseline and ``M`` of shape
    ``(D, n_params, 2)`` carries the per-parameter modulation. Standardised
    machine parameters ``p`` (shape ``(B, n_params)``, zero-mean) thus tilt the
    encoder's reading of each (f_rel, amp) pair, but the encoder still does
    something sensible at the parameter centroid (``p ≈ 0``) thanks to ``W0``.

    The second linear layer (``D -> D``) is parameter-independent.

    Forward signature: ``(pairs, params_std) -> embedding`` where
        pairs: (B, T, C, K, 2)
        params_std: (B, n_params)
        embedding: (B, T, C, K, D)
    """

    def __init__(self, pair_in_dim: int, pair_embed_dim: int, n_params: int):
        super().__init__()
        self.pair_in_dim = pair_in_dim
        self.pair_embed_dim = pair_embed_dim
        self.n_params = n_params

        # Baseline first-layer weights (~Xavier-ish for fan-in=2).
        self.W0 = nn.Parameter(torch.randn(pair_embed_dim, pair_in_dim) * (1.0 / pair_in_dim ** 0.5))
        # Parameter modulation: small so training starts near the baseline
        # encoder and M grows only if data-driven gradients say it should.
        self.M = nn.Parameter(torch.randn(pair_embed_dim, n_params, pair_in_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(pair_embed_dim))

        # Second layer is a plain Linear; ReLUs in forward.
        self.linear2 = nn.Linear(pair_embed_dim, pair_embed_dim)

    def forward(self, pairs: torch.Tensor, params_std: torch.Tensor) -> torch.Tensor:
        # Effective per-sample first-layer weight: (B, D, 2). This is the same
        # for every (t, c, k) in the sample — the machine parameters are a
        # per-sample property, so the parameter-conditioned read of (f_rel,
        # amp) is fixed across all pairs in a sample (and across all time
        # steps in the window).
        W_eff = torch.einsum("bp,dpi->bdi", params_std, self.M) + self.W0
        # out[b, t, c, k, d] = sum_i W_eff[b, d, i] * pairs[b, t, c, k, i].
        h = torch.einsum("bdi,btcki->btckd", W_eff, pairs) + self.b1
        h = F.relu(h)
        # Second linear layer — no ReLU afterwards. Leaving the output of the
        # per-pair encoder linear-in-its-features gives the downstream
        # DeepSets sum + Conv1d more room to add or subtract contributions
        # without a non-negativity bottleneck.
        h = self.linear2(h)
        return h


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
        self.n_params = n_params
        self.cnn_window = cnn_window
        self.conv_channels_cfg = conv_channels

        # Standardisation buffers for cutting parameters.
        self.register_buffer("param_mean", torch.zeros(n_params))
        self.register_buffer("param_std", torch.ones(n_params))

        # Per-pair encoder is now parameter-conditioned at its first layer.
        # Machine parameters reshape how each (f_rel, amp) is read; they no
        # longer enter only at the FC head.
        self.pair_encoder = ParamConditionedPairEncoder(
            pair_in_dim=pair_in_dim,
            pair_embed_dim=pair_embed_dim,
            n_params=n_params,
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

        # FC head NO LONGER takes raw params — they already shaped the encoder.
        self.fc1 = nn.Linear(flat, fc_hidden)
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
        B, T, C, K, Fdim = pairs.shape

        # Standardise once and feed the same vector to the encoder.
        params_std = (params - self.param_mean) / self.param_std

        # Per-pair, parameter-conditioned encoding.
        e = self.pair_encoder(pairs, params_std)        # (B, T, C, K, D)

        # DeepSets aggregation over K peaks.
        e = e.sum(dim=3)                                # (B, T, C, D)
        e = e.reshape(B, T, C * self.pair_embed_dim)     # (B, T, F)

        # Conv1d expects (B, F, T).
        x = e.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)

        # No param concat here: machine parameters were already injected via
        # the parameter-conditioned first layer of the per-pair encoder.
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x).squeeze(-1)

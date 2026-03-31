"""Flexible state encoder supporting latent-vector, pixel, and embedding inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _LatentEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int], embed_dim: int):
        super().__init__()
        self.net = _mlp([state_dim] + hidden_sizes + [embed_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PixelEncoder(nn.Module):
    """Nature-DQN-style CNN for (B, C, H, W) float32 inputs in [0, 1]."""

    _POOL_SIZE = (4, 4)

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(self._POOL_SIZE),
        )
        flat_dim = 64 * self._POOL_SIZE[0] * self._POOL_SIZE[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, embed_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) float32 [0, 1]
        return self.fc(self.conv(x).flatten(1))


class _EmbeddingEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class StateEncoder(nn.Module):
    """Flexible state encoder projecting heterogeneous inputs to a common embedding.

    Parameters
    ----------
    mode:
        ``"latent"``    — flat state vector (B, state_dim) → MLP → (B, embed_dim).
        ``"pixel"``     — pixel observation (B, C, H, W) float32 [0,1] → CNN → (B, embed_dim).
        ``"embedding"`` — pre-computed embedding (B, input_dim) → linear → (B, embed_dim).
    embed_dim:
        Output dimensionality shared by all modes.
    state_dim:
        Required for ``mode="latent"``.
    hidden_sizes:
        MLP hidden layer sizes. Only used when ``mode="latent"``.
        Defaults to ``[256]``.
    in_channels:
        Required for ``mode="pixel"``. Number of input channels.
    input_dim:
        Required for ``mode="embedding"``.

    Notes
    -----
    **Pixel convention**: this encoder expects channel-first float32 tensors
    normalised to [0, 1], *not* the HWC uint8 format used by LocalCNN in
    ``fuzzy_actions/dqn.py``.  Normalisation and permutation are the
    caller's responsibility.
    """

    def __init__(
        self,
        mode: str,
        embed_dim: int,
        state_dim: int | None = None,
        hidden_sizes: list[int] | None = None,
        in_channels: int | None = None,
        input_dim: int | None = None,
    ):
        super().__init__()
        assert mode in ("latent", "pixel", "embedding"), (
            f"mode must be 'latent', 'pixel', or 'embedding'; got '{mode}'"
        )
        self.mode = mode
        self.embed_dim = embed_dim

        if mode == "latent":
            assert state_dim is not None, "state_dim required for mode='latent'"
            self._net = _LatentEncoder(state_dim, hidden_sizes or [256], embed_dim)
        elif mode == "pixel":
            assert in_channels is not None, "in_channels required for mode='pixel'"
            self._net = _PixelEncoder(in_channels, embed_dim)
        else:
            assert input_dim is not None, "input_dim required for mode='embedding'"
            self._net = _EmbeddingEncoder(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

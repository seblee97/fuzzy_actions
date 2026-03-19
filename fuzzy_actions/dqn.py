"""Q-network architectures for DQN.

Exports
-------
QNetwork            -- flat MLP for symbolic observations (train_dqn.py)
TwoStreamQNetwork   -- LocalCNN + MapNet merged by MLP (train_dqn_modular.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Flat MLP (used by train_dqn.py with symbolic observations)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron Q-network.

    Maps a flat observation vector to per-action Q-values.

    Args:
        obs_dim: Dimensionality of the (flat) observation.
        n_actions: Number of discrete actions.
        hidden_sizes: Sizes of the hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Two-stream image network (used by train_dqn_modular.py)
# ---------------------------------------------------------------------------

class LocalCNN(nn.Module):
    """CNN encoder for the partial local observation (room pixels).

    Uses the same three-layer architecture as the Nature DQN paper, followed
    by ``AdaptiveAvgPool2d`` so that the output size is independent of the
    exact input dimensions (useful when room sizes vary slightly, or when the
    ``PixelDownscaleWrapper`` is applied with different scale factors).

    Input pixels are expected as ``(B, H, W, C)`` uint8; normalisation to
    ``[0, 1]`` float is performed inside ``forward``.

    Args:
        obs_shape: ``(H, W, C)`` shape of *one* local observation.
        embed_dim: Dimensionality of the output embedding vector.
    """

    # Fixed spatial size that the adaptive pool targets.
    _POOL_SIZE = (4, 4)

    def __init__(self, obs_shape: tuple[int, int, int], embed_dim: int = 256) -> None:
        super().__init__()
        c = obs_shape[2]  # number of channels (3 for RGB)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(self._POOL_SIZE),
        )
        flat_dim = 64 * self._POOL_SIZE[0] * self._POOL_SIZE[1]  # 1024
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) — may be uint8 or float
        x = x.float() / 255.0          # normalise to [0, 1]
        x = x.permute(0, 3, 1, 2)      # (B, C, H, W) for Conv2d
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class MapNet(nn.Module):
    """CNN encoder for the global map image.

    The map image is small (one coloured cell per room, ``map_cell_size``
    pixels wide), so a lightweight two-layer CNN is sufficient.  Like
    ``LocalCNN``, ``AdaptiveAvgPool2d`` decouples the embedding size from the
    exact map dimensions, allowing different ``n_rooms`` counts without
    changing the network definition.

    Input pixels are expected as ``(B, H, W, C)`` uint8.

    Args:
        obs_shape: ``(H, W, C)`` shape of *one* map image.
        embed_dim: Dimensionality of the output embedding vector.
    """

    _POOL_SIZE = (4, 4)

    def __init__(self, obs_shape: tuple[int, int, int], embed_dim: int = 64) -> None:
        super().__init__()
        c = obs_shape[2]

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(self._POOL_SIZE),
        )
        flat_dim = 32 * self._POOL_SIZE[0] * self._POOL_SIZE[1]  # 512
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class TwoStreamQNetwork(nn.Module):
    """Two-stream Q-network for partial observability + global map.

    Architecture::

        local_obs  (H_room, W_room, 3) uint8
            └── LocalCNN ──────────────────┐
                                           ├── cat ── MLP head ── Q(s,a)
        map_image  (H_map,  W_map,  3) uint8
            └── MapNet  ──────────────────┘

    The two streams are processed independently and their embeddings are
    concatenated before being passed through a shared MLP to produce Q-values.

    Args:
        local_obs_shape: ``(H, W, C)`` of the local room-pixel observation.
        map_obs_shape: ``(H, W, C)`` of the global map image.
        n_actions: Number of discrete actions.
        local_embed_dim: Output dimensionality of ``LocalCNN``.
        map_embed_dim: Output dimensionality of ``MapNet``.
        hidden_sizes: Hidden layer sizes of the MLP head.
    """

    def __init__(
        self,
        local_obs_shape: tuple[int, int, int],
        map_obs_shape: tuple[int, int, int],
        n_actions: int,
        local_embed_dim: int = 256,
        map_embed_dim: int = 64,
        hidden_sizes: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.local_cnn = LocalCNN(local_obs_shape, local_embed_dim)
        self.map_net = MapNet(map_obs_shape, map_embed_dim)

        layers: list[nn.Module] = []
        in_dim = local_embed_dim + map_embed_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        local_obs: torch.Tensor,
        map_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values.

        Args:
            local_obs: ``(B, H, W, C)`` uint8 room-pixel observation.
            map_obs: ``(B, H, W, C)`` uint8 global map image.

        Returns:
            ``(B, n_actions)`` float Q-values.
        """
        local_feat = self.local_cnn(local_obs)
        map_feat = self.map_net(map_obs)
        return self.head(torch.cat([local_feat, map_feat], dim=1))

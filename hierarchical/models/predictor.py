"""SimSiam / BYOL predictor MLP (online branch only)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Two-layer MLP predictor used by SimSiam and BYOL on the online branch.

    Architecture::

        z  →  Linear  →  BN  →  ReLU  →  Linear  →  p

    Parameters
    ----------
    z_dim:
        Input (and output) dimensionality — same as the projection head output.
    hidden_dim:
        Hidden layer width.
    """

    def __init__(self, z_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

"""Simple MLP prediction head used by the JEPA-style loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Two-layer MLP: Linear → BN → ReLU → Linear.

    Used for both:
    - ``pred_x``: forward model output → predicted e(s2)  (in_dim=embed_dim, out_dim=embed_dim)
    - ``pred_z``: e(s1) → predicted z                     (in_dim=embed_dim, out_dim=z_dim)

    Parameters
    ----------
    in_dim:  Input dimensionality.
    out_dim: Output dimensionality.
    hidden_dim: Hidden layer size.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

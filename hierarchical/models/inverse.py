"""Inverse model: encodes the latent action z between two states."""

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


class InverseModel(nn.Module):
    """Estimates the latent action z between two encoded states.

    Architecture::

        [x1 ; x2]  →  MLP  →  z

    Parameters
    ----------
    embed_dim:
        Dimensionality of encoder output vectors.
    z_dim:
        Dimensionality of the latent action z.
    hidden_sizes:
        MLP hidden layer sizes.  Defaults to ``[512, 256]``.
    """

    def __init__(
        self,
        embed_dim: int,
        z_dim: int,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        hidden = hidden_sizes or [512, 256]
        self.net = _mlp([2 * embed_dim] + hidden + [z_dim])

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x1, x2:
            (B, embed_dim) encoded start / end states.
        context:
            Optional context tensor N — ignored by this implementation,
            reserved for subclasses that condition on task/variant info.

        Returns
        -------
        z: (B, z_dim) latent action.
        """
        return self.net(torch.cat([x1, x2], dim=-1))

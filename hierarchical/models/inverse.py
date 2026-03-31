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

        [enc_s1 ; enc_s2]  →  MLP  →  z  →  projection head  →  z_proj

    ``z`` is the latent action used downstream (forward model, decoder).
    ``z_proj`` is the extra projection used *only* for contrastive losses
    (following the SimCLR convention: projection head discarded at inference).

    Parameters
    ----------
    embed_dim:
        Dimensionality of encoder output vectors.
    z_dim:
        Dimensionality of the latent action z.
    hidden_sizes:
        MLP hidden layer sizes for the inverse network.
        Defaults to ``[512, 256]``.
    proj_hidden_dim:
        Hidden size of the projection head.
    proj_dim:
        Output size of the projection head (contrastive embedding space).
    """

    def __init__(
        self,
        embed_dim: int,
        z_dim: int,
        hidden_sizes: list[int] | None = None,
        proj_hidden_dim: int = 256,
        proj_dim: int = 128,
    ):
        super().__init__()
        hidden = hidden_sizes or [512, 256]
        self.inverse = _mlp([2 * embed_dim] + hidden + [z_dim])
        self.projector = nn.Sequential(
            nn.Linear(z_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_dim),
        )

    def forward(
        self, enc_s1: torch.Tensor, enc_s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (z, z_proj).

        Parameters
        ----------
        enc_s1, enc_s2:
            (B, embed_dim) encoded start / end states.

        Returns
        -------
        z:      (B, z_dim) latent action for downstream use.
        z_proj: (B, proj_dim) projected representation for contrastive loss.
        """
        z = self.inverse(torch.cat([enc_s1, enc_s2], dim=-1))
        z_proj = self.projector(z)
        return z, z_proj

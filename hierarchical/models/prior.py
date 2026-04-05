"""Prior model: predicts the latent action z from the start state alone."""

from __future__ import annotations

import torch
import torch.nn as nn


class Prior(nn.Module):
    """Point-estimate prior p(z | x1).

    Maps the encoded start state x1 to a predicted latent action z_prior.
    Trained via a prior loss (e.g. MSE against the stop-gradient z from the
    inverse model) without back-propagating into the inverse model.

    Architecture::

        x1  →  Linear → BN → ReLU → Linear  →  z_prior

    Parameters
    ----------
    embed_dim:
        Dimensionality of the encoder output (input to this network).
    z_dim:
        Dimensionality of the latent action space (output).
    hidden_dim:
        Hidden layer width.
    """

    def __init__(self, embed_dim: int, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        x1: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x1:
            (B, embed_dim) encoded start state.
        context:
            Optional context tensor N — ignored by this implementation,
            reserved for subclasses that condition on task/variant info.

        Returns
        -------
        (B, z_dim) predicted latent action.
        """
        return self.net(x1)

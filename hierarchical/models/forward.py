"""Deterministic forward model: predicts enc(s2) from enc(s1) and z."""

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


class ForwardModel(nn.Module):
    """Predicts enc(s2) given enc(s1) and the latent action z.

    Architecture::

        [enc_s1 ; z]  →  MLP  →  enc_s2_pred

    The forward loss is computed between enc_s2_pred and the *actual*
    enc_s2 produced by the encoder (with gradients stopped on the target).

    Parameters
    ----------
    embed_dim:
        Dimensionality of state embeddings (both input and output).
    z_dim:
        Dimensionality of the latent action z.
    hidden_sizes:
        Hidden layer sizes. Defaults to ``[512, 256]``.
    """

    def __init__(
        self,
        embed_dim: int,
        z_dim: int,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        hidden = hidden_sizes or [512, 256]
        self.net = _mlp([embed_dim + z_dim] + hidden + [embed_dim])

    def forward(self, enc_s1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict enc(s2).

        Parameters
        ----------
        enc_s1: (B, embed_dim)
        z:      (B, z_dim)

        Returns
        -------
        (B, embed_dim) predicted embedding of s2.
        """
        return self.net(torch.cat([enc_s1, z], dim=-1))

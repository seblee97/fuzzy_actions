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


class NConditionedForwardModel(nn.Module):
    """Forward model conditioned on N (timestep gap between s1 and s2).

    Architecture::

        [enc_s1 ; z ; n_embed(N)]  →  MLP  →  enc_s2_pred

    N is embedded via a small learned linear projection before concatenation.

    Parameters
    ----------
    embed_dim:
        Dimensionality of state embeddings.
    z_dim:
        Dimensionality of the latent action z.
    n_embed_dim:
        Dimensionality of the N embedding.  Defaults to 16.
    hidden_sizes:
        Hidden layer sizes.  Defaults to ``[512, 256]``.
    """

    def __init__(
        self,
        embed_dim: int,
        z_dim: int,
        n_embed_dim: int = 16,
        hidden_sizes: list[int] | None = None,
    ):
        super().__init__()
        self.n_embed = nn.Linear(1, n_embed_dim)
        hidden = hidden_sizes or [512, 256]
        self.net = _mlp([embed_dim + z_dim + n_embed_dim] + hidden + [embed_dim])

    def forward(
        self,
        x1: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x1:      (B, embed_dim)
        z:       (B, z_dim)
        context: (B,) or (B, 1) integer or float timestep gap N. Required.

        Returns
        -------
        (B, embed_dim) predicted embedding of s2.
        """
        if context is None:
            raise ValueError("NConditionedForwardModel requires context (N).")
        n = context.float().view(-1, 1)
        n_emb = self.n_embed(n)
        return self.net(torch.cat([x1, z, n_emb], dim=-1))


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

    def forward(
        self,
        x1: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict enc(s2).

        Parameters
        ----------
        x1:      (B, embed_dim)
        z:       (B, z_dim)
        context: Optional context tensor N — ignored by this implementation,
                 reserved for subclasses that condition on task/variant info.

        Returns
        -------
        (B, embed_dim) predicted embedding of s2.
        """
        return self.net(torch.cat([x1, z], dim=-1))

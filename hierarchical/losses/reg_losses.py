"""Concrete regularisation loss implementations."""

from __future__ import annotations

import torch

from hierarchical.losses.base import AbstractRegLoss


class NullRegLoss(AbstractRegLoss):
    """No regularisation — returns zero."""

    def forward(
        self,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return z.new_tensor(0.0)


class L2RegLoss(AbstractRegLoss):
    """L2 penalty on the latent action: mean(||z||^2).

    Encourages z to stay close to the origin, preventing unbounded growth
    in the absence of other normalisation constraints.
    """

    def forward(
        self,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return z.pow(2).mean()

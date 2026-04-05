"""Concrete prior loss implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hierarchical.losses.base import AbstractPriorLoss


class NullPriorLoss(AbstractPriorLoss):
    """No prior loss — returns zero."""

    def forward(
        self,
        prior_out: torch.Tensor,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return prior_out.new_tensor(0.0)


class MSEPriorLoss(AbstractPriorLoss):
    """MSE between the prior prediction and the latent action.

    Internal stop_grad names
    ------------------------
    ``"z"``
        Detach z before computing the loss, so the inverse model is not
        updated by this loss.  This is the standard setting.
    """

    def forward(
        self,
        prior_out: torch.Tensor,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        sg = stop_grad or []
        if "z" in sg:
            z = z.detach()
        return F.mse_loss(prior_out, z)

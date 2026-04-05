"""Concrete forward loss implementations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchical.losses.base import AbstractForwardLoss


class NullForwardLoss(AbstractForwardLoss):
    """No forward loss — returns zero."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return pred.new_tensor(0.0)


class MSEForwardLoss(AbstractForwardLoss):
    """MSE between (optionally projected) forward prediction and target.

    An optional prediction head pred_x is applied to the forward model output
    before computing the loss.  This follows the JEPA convention: the head
    absorbs the mismatch between the forward model's output space and the
    encoder's embedding space, preventing trivial collapse.

    Internal stop_grad names
    ------------------------
    ``"target"``
        Detach the target embedding (x2) before computing the loss.  This is
        the standard JEPA setting and is the default behaviour when specified
        externally via the grad config ``stop_grad: ["x2"]``.  Provided here
        as an internal alias for completeness.

    Parameters
    ----------
    embed_dim:
        Dimensionality of encoder embeddings (input and output of pred_x).
    use_predictor:
        If ``True`` (default), attach a two-layer MLP prediction head.
    hidden_dim:
        Hidden dimensionality of the prediction head.
    """

    def __init__(
        self,
        embed_dim: int,
        use_predictor: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if use_predictor:
            self.predictor: nn.Module | None = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.predictor = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        sg = stop_grad or []
        if self.predictor is not None:
            pred = self.predictor(pred)
        if "target" in sg:
            target = target.detach()
        return F.mse_loss(pred, target)

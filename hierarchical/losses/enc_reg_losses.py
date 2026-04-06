"""Concrete encoder regularisation loss implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hierarchical.losses.base import AbstractEncRegLoss


class NullEncRegLoss(AbstractEncRegLoss):
    """No encoder regularisation — returns zero."""

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return x1.new_tensor(0.0)


class VICRegVarLoss(AbstractEncRegLoss):
    """VICReg-style variance hinge on encoder embeddings.

    Penalises any embedding dimension whose per-batch standard deviation
    falls below ``target_std``:

        L_var = mean_over_dims(max(0, target_std - std(x, dim=0)))

    Applied to both x1 and x2 and averaged.  This directly prevents the
    encoder from collapsing all inputs to the same point on the hypersphere.

    Internal stop_grad names
    ------------------------
    None recognised — both x1 and x2 always receive gradients.

    Parameters
    ----------
    target_std:
        Minimum desired per-dimension standard deviation.  Defaults to 1.0
        (standard VICReg value; appropriate when embeddings are not L2
        normalised).  With L2 normalisation a smaller value such as 0.1
        may be more appropriate.
    """

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        def _var_loss(x: torch.Tensor) -> torch.Tensor:
            std = x.std(dim=0)  # (embed_dim,)
            return F.relu(self.target_std - std).mean()

        return (_var_loss(x1) + _var_loss(x2)) / 2

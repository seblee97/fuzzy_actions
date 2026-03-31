"""SimSiam loss (negative cosine similarity with stop-gradient)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):
    """Negative cosine similarity loss for SimSiam.

    Expects ``p`` to be the online branch prediction (with gradients) and
    ``z`` to be the stop-gradient target.  The caller is responsible for
    calling ``.detach()`` on ``z`` before passing it here, or using a
    symmetric formulation::

        loss = 0.5 * (simsiam(p_a, z_b.detach()) + simsiam(p_b, z_a.detach()))

    Reference: Chen & He, 2021 — "Exploring Simple Siamese Representation Learning"
    """

    def forward(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        p: (B, D) online branch prediction (predictor output), with gradients.
        z: (B, D) stop-gradient target.

        Returns
        -------
        Scalar loss (negative cosine similarity, in [-1, 0]).
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z.detach(), dim=-1)
        return -(p * z).sum(dim=-1).mean()

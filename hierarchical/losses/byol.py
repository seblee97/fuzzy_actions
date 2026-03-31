"""BYOL regression loss (normalised MSE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """BYOL target regression loss: 2 − 2 · cosine_similarity.

    Equivalent to MSE between L2-normalised vectors.  The caller passes
    ``online_pred = predictor(z_proj_online)`` and
    ``target_proj = ema_z_proj.detach()``; stop-gradient on the target
    branch is enforced inside this loss.

    Reference: Grill et al., 2020 — "Bootstrap Your Own Latent"
    """

    def forward(
        self, online_pred: torch.Tensor, target_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        online_pred: (B, D) predictor(online_z_proj), with gradients.
        target_proj: (B, D) EMA target z_proj (will be detached internally).

        Returns
        -------
        Scalar loss ∈ [0, 4].
        """
        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj.detach(), dim=-1)
        return 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()

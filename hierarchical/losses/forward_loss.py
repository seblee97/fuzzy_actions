"""Forward model reconstruction loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardLoss(nn.Module):
    """Reconstruction loss between predicted enc(s2) and actual enc(s2).

    The target enc(s2) is always detached (stop-gradient) so the encoder
    receives gradient only from the contrastive loss on the opposite view.

    Parameters
    ----------
    mode:
        ``"mse"``    — mean squared error in embedding space.
        ``"cosine"`` — 1 − cosine similarity (operates on L2-normalised vectors).
    """

    def __init__(self, mode: str = "mse"):
        super().__init__()
        assert mode in ("mse", "cosine"), f"mode must be 'mse' or 'cosine'; got '{mode}'"
        self.mode = mode

    def forward(self, s2_pred: torch.Tensor, enc_s2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s2_pred: (B, embed_dim) forward model prediction.
        enc_s2:  (B, embed_dim) actual encoder output for s2 (detached internally).

        Returns
        -------
        Scalar loss.
        """
        target = enc_s2.detach()
        if self.mode == "mse":
            return F.mse_loss(s2_pred, target)
        else:
            s2_pred = F.normalize(s2_pred, dim=-1)
            target = F.normalize(target, dim=-1)
            return (1 - (s2_pred * target).sum(dim=-1)).mean()

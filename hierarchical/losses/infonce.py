"""NT-Xent / InfoNCE contrastive loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """NT-Xent (InfoNCE) loss with in-batch negatives.

    Treats ``(z_a[i], z_b[i])`` as a positive pair; all other combinations
    within the batch serve as negatives.  Both views are L2-normalised before
    similarity computation.

    Parameters
    ----------
    temperature:
        Softmax temperature τ.  Smaller values produce sharper distributions.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Parameters
        ----------
        z_a, z_b:
            (B, D) projected representations from the two views.  Positive
            pairs are at matching indices.

        Returns
        -------
        Scalar loss.
        """
        B = z_a.shape[0]
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # Stack into (2B, D) and compute full similarity matrix
        reps = torch.cat([z_a, z_b], dim=0)           # (2B, D)
        sim = torch.mm(reps, reps.T) / self.temperature  # (2B, 2B)

        # Mask self-similarity on the diagonal
        mask = torch.eye(2 * B, device=z_a.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive targets: row i → column i+B, row i+B → column i
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z_a.device),
            torch.arange(0, B, device=z_a.device),
        ])  # (2B,)

        return F.cross_entropy(sim, labels)

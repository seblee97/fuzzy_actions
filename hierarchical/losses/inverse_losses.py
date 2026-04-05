"""Concrete inverse loss implementations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchical.losses.base import AbstractInverseLoss


class NullInverseLoss(AbstractInverseLoss):
    """No inverse loss — returns zero."""

    needs_pairs: bool = False

    def forward(
        self,
        z: torch.Tensor,
        z_pos: torch.Tensor | None = None,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        return z.new_tensor(0.0)


class InfoNCEInverseLoss(AbstractInverseLoss):
    """NT-Xent (InfoNCE) contrastive loss over latent actions.

    Owns an internal projection head (z → z_proj) following the SimCLR
    convention.  The projection head is trained alongside the loss and its
    parameters are included in the optimizer automatically.

    Internal stop_grad names
    ------------------------
    ``"keys"``
        One-sided (MoCo-style) formulation: only the anchor z is a
        gradient-connected query.  Positives and all in-batch negatives are
        treated as a detached key bank.  Specify via grad config::

            {"inverse": {"stop_grad": ["keys"]}}

    Parameters
    ----------
    z_dim:
        Dimensionality of the incoming latent action z.
    proj_dim:
        Output dimensionality of the projection head.
    proj_hidden_dim:
        Hidden dimensionality of the projection head.
    temperature:
        Softmax temperature τ.
    """

    needs_pairs: bool = True

    def __init__(
        self,
        z_dim: int,
        proj_dim: int = 128,
        proj_hidden_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(z_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        z_pos: torch.Tensor | None = None,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        if z_pos is None:
            raise ValueError("InfoNCEInverseLoss requires z_pos (needs_pairs=True).")

        sg = stop_grad or []

        if "keys" in sg:
            # One-sided: anchor z is the query (gradients flow).
            # Positives and all in-batch negatives form a detached key bank.
            z_q = F.normalize(self.projector(z), dim=-1)             # (B, D)  grads flow
            with torch.no_grad():
                z_k_pos = F.normalize(self.projector(z_pos), dim=-1) # (B, D)  detached
            z_k_neg = z_q.detach()                                    # (B, D)  other anchors, detached

            B = z_q.shape[0]
            keys = torch.cat([z_k_neg, z_k_pos], dim=0)              # (2B, D)
            sim  = torch.mm(z_q, keys.T) / self.temperature           # (B, 2B)
            sim[:, :B] = sim[:, :B].masked_fill(
                torch.eye(B, device=z.device, dtype=torch.bool), float("-inf")
            )
            labels = torch.arange(B, 2 * B, device=z.device)
            return F.cross_entropy(sim, labels)

        else:
            # Symmetric SimCLR: both branches receive gradients equally.
            z_a = F.normalize(self.projector(z),     dim=-1)
            z_b = F.normalize(self.projector(z_pos), dim=-1)

            B = z_a.shape[0]
            reps = torch.cat([z_a, z_b], dim=0)                      # (2B, D)
            sim  = torch.mm(reps, reps.T) / self.temperature          # (2B, 2B)
            sim  = sim.masked_fill(
                torch.eye(2 * B, device=z.device, dtype=torch.bool), float("-inf")
            )
            labels = torch.cat([
                torch.arange(B, 2 * B, device=z.device),
                torch.arange(0, B,     device=z.device),
            ])
            return F.cross_entropy(sim, labels)

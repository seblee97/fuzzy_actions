"""Exponential moving average (EMA) target network for BYOL."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAUpdater:
    """Maintains an EMA copy of an online model for use as a BYOL target.

    Parameters
    ----------
    online:
        The online (gradient-updated) network.
    target:
        The EMA (momentum-updated) network. Should be a deep copy of *online*
        at initialisation; use :meth:`from_online` to create both at once.
    decay:
        EMA decay coefficient τ.  Target params are updated as::

            θ_target ← τ · θ_target + (1 − τ) · θ_online
    """

    def __init__(self, online: nn.Module, target: nn.Module, decay: float = 0.996):
        self.online = online
        self.target = target
        self.decay = decay
        # Target is never gradient-updated.
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @classmethod
    def from_online(cls, online: nn.Module, decay: float = 0.996) -> "EMAUpdater":
        """Deep-copy *online* to create the target, then return an EMAUpdater."""
        target = copy.deepcopy(online)
        return cls(online, target, decay)

    @torch.no_grad()
    def step(self) -> None:
        """Update target parameters: θ_target ← τ·θ_target + (1−τ)·θ_online."""
        tau = self.decay
        for p_online, p_target in zip(
            self.online.parameters(), self.target.parameters()
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

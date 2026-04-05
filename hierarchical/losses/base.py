"""Abstract base classes for each loss slot in the hierarchical model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractInverseLoss(nn.Module, ABC):
    """Loss on the latent action z produced by the inverse model.

    Contrastive implementations (e.g. InfoNCE) require a positive pair
    ``z_pos`` from a second transition of the same type; set
    ``needs_pairs = True`` so the training loop fetches the second sample.

    Non-contrastive implementations (e.g. regression) set
    ``needs_pairs = False`` and ignore ``z_pos``.

    The ``stop_grad`` list is forwarded from the experiment's grad config and
    may contain loss-internal names (e.g. ``"keys"`` for InfoNCE).  Each
    concrete implementation documents which names it recognises.
    """

    needs_pairs: bool = False

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        z_pos: torch.Tensor | None = None,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z:
            (B, z_dim) latent action from the anchor transition.
        z_pos:
            (B, z_dim) latent action from a positive transition of the same
            type.  Required when ``needs_pairs=True``, else ignored.
        stop_grad:
            List of loss-internal tensor names to detach.  Each concrete
            implementation documents the names it recognises.

        Returns
        -------
        Scalar loss.
        """


class AbstractForwardLoss(nn.Module, ABC):
    """Loss between the forward model prediction and the target embedding."""

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred:
            (B, embed_dim) forward model output.
        target:
            (B, embed_dim) encoder output for s2.
        stop_grad:
            Loss-internal names to detach.  Each implementation documents
            the names it recognises.

        Returns
        -------
        Scalar loss.
        """


class AbstractPriorLoss(nn.Module, ABC):
    """Loss between the prior prediction p(z|x1) and the actual z."""

    @abstractmethod
    def forward(
        self,
        prior_out: torch.Tensor,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prior_out:
            Output of the Prior model given x1.
        z:
            (B, z_dim) actual latent action from the inverse model.
        stop_grad:
            Loss-internal names to detach.  Each implementation documents
            the names it recognises.

        Returns
        -------
        Scalar loss.
        """


class AbstractRegLoss(nn.Module, ABC):
    """Regularisation loss applied directly to z."""

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        stop_grad: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z:
            (B, z_dim) latent action.
        stop_grad:
            Loss-internal names to detach.  Each implementation documents
            the names it recognises.

        Returns
        -------
        Scalar loss.
        """

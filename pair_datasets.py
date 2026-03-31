"""Pair-construction datasets for hierarchical inverse-model training.

Each dataset wraps a ``MazeOracleDataset`` and yields dicts of the form::

    {
        "s1_a": (state_dim,)  — anchor start state
        "s2_a": (state_dim,)  — anchor end state
        "s1_b": (state_dim,)  — positive start state
        "s2_b": (state_dim,)  — positive end state
    }

for use with ``train_hierarchical.py``.

State representation
--------------------
By default, each state at timestep ``t`` is a one-hot encoded flat vector::

    state = [phase_onehot (5) | material_onehot (4) | room_onehot (n_rooms)]

giving ``state_dim = 5 + 4 + n_rooms``.  This can be overridden by
subclassing and overriding ``_extract_state``.

Variants
--------
RoomTransitionPairDataset
    Transitions defined by room boundaries (room_label changes).
    Positive pair: another (s1, s2) where the agent made the same
    room-to-room hop (same from_room_id, same to_room_id).

PhaseTransitionPairDataset
    Transitions defined by phase boundaries (phase_label changes).
    Positive pair: another transition with the same (from_phase, to_phase).

WindowPairDataset
    Fixed-length windows over trajectories.  Positive pair: another window
    where the starting state is in the same phase (and optionally room).
    Suitable for SimSiam / BYOL where negatives are not required.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Hashable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from maze_dataset import MazeOracleDataset


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PairDataset(Dataset, ABC):
    """Abstract base for pair-construction datasets.

    Parameters
    ----------
    root:
        Path to the dataset directory passed to ``MazeOracleDataset``.
    maze_dataset:
        Pre-built ``MazeOracleDataset``.  Mutually exclusive with *root*;
        one of the two must be provided.
    state_mode:
        ``"latent"`` — one-hot (phase | material | room) flat vector.
        ``"pixel"``  — room pixel observation ``(C, H, W)`` float32 [0, 1].
        When ``"pixel"``, the underlying dataset is always opened in
        ``variant="full"`` and the pixel cache is prepared automatically.
    min_group_size:
        Minimum number of transitions of the same type required to include
        them in the index.  Transitions whose type has fewer than
        ``min_group_size`` examples are silently dropped (no valid positive
        can be formed otherwise).
    seed:
        RNG seed for reproducible positive sampling.
    """

    def __init__(
        self,
        root: str | None = None,
        maze_dataset: MazeOracleDataset | None = None,
        state_mode: str = "latent",
        min_group_size: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        assert state_mode in ("latent", "pixel"), (
            f"state_mode must be 'latent' or 'pixel'; got '{state_mode}'"
        )
        self.state_mode = state_mode
        variant = "full" if state_mode == "pixel" else "actions_only"

        if maze_dataset is not None:
            self.ds = maze_dataset
        elif root is not None:
            self.ds = MazeOracleDataset(root=root, variant=variant)
        else:
            raise ValueError("Provide either 'root' or 'maze_dataset'.")
        self.min_group_size = min_group_size
        self._rng = random.Random(seed)

        n_rooms: int = self.ds.metadata["n_rooms"]
        self.n_phases: int = self.ds.n_phases
        self.n_materials: int = self.ds.n_materials
        self.n_rooms: int = n_rooms

        if state_mode == "latent":
            self.state_dim: int = self.n_phases + self.n_materials + n_rooms
            self.pixel_shape: tuple | None = None
        else:
            # Prepare pixel cache (one-time cost) and detect pixel shape
            self.ds.prepare_pixels()
            self.ds._ensure_pixel_cache()
            sample = self.ds._room_pixels_mmap[0, 0]  # (H, W, C) uint8
            H, W, C = sample.shape
            self.pixel_shape = (C, H, W)  # channel-first for encoder
            self.state_dim = None

        # Build flat index and type-group lookup
        raw_index: list[tuple[int, int, int]] = self._build_index()
        type_keys: list[Hashable] = [
            self._transition_type(ti, t1, t2) for ti, t1, t2 in raw_index
        ]

        # Group by type, keep only groups large enough for positive sampling
        groups: dict[Hashable, list[int]] = defaultdict(list)
        for pos, key in enumerate(type_keys):
            groups[key].append(pos)

        valid_positions: set[int] = set()
        for members in groups.values():
            if len(members) >= min_group_size:
                valid_positions.update(members)

        # Re-index to only valid transitions
        self.index: list[tuple[int, int, int]] = [
            raw_index[i] for i in sorted(valid_positions)
        ]
        remapped = {old: new for new, old in enumerate(sorted(valid_positions))}

        self.type_groups: dict[Hashable, list[int]] = defaultdict(list)
        for old_pos, key in enumerate(type_keys):
            if old_pos in valid_positions:
                self.type_groups[key].append(remapped[old_pos])

        self.item_type: list[Hashable] = [
            self._transition_type(ti, t1, t2) for ti, t1, t2 in self.index
        ]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_index(self) -> list[tuple[int, int, int]]:
        """Return all valid (traj_idx, t1, t2) transition tuples."""

    @abstractmethod
    def _transition_type(self, traj_idx: int, t1: int, t2: int) -> Hashable:
        """Return a hashable key grouping transitions of the same type."""

    # ------------------------------------------------------------------
    # State extraction (override to use a different state representation)
    # ------------------------------------------------------------------

    def _extract_state(self, traj_idx: int, t: int) -> torch.Tensor:
        """Extract state at timestep t.

        ``"latent"`` mode: one-hot (phase | material | room) flat vector.
        ``"pixel"`` mode:  room pixel frame (C, H, W) float32 in [0, 1].
        """
        if self.state_mode == "pixel":
            import numpy as np
            frame = np.array(self.ds._room_pixels_mmap[traj_idx, t])  # (H, W, C) uint8
            return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        phase = self.ds.phase_labels[traj_idx, t]
        material = self.ds.material_labels[traj_idx, t]
        room = self.ds.room_labels[traj_idx, t]
        return torch.cat([
            F.one_hot(phase, self.n_phases).float(),
            F.one_hot(material, self.n_materials).float(),
            F.one_hot(room, self.n_rooms).float(),
        ])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_a, t1_a, t2_a = self.index[idx]
        key = self.item_type[idx]

        # Sample a distinct positive from the same type group
        candidates = self.type_groups[key]
        pos_idx = idx
        while pos_idx == idx and len(candidates) > 1:
            pos_idx = self._rng.choice(candidates)
        traj_b, t1_b, t2_b = self.index[pos_idx]

        return {
            "s1_a": self._extract_state(traj_a, t1_a),
            "s2_a": self._extract_state(traj_a, t2_a),
            "s1_b": self._extract_state(traj_b, t1_b),
            "s2_b": self._extract_state(traj_b, t2_b),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n_types = len(self.type_groups)
        sizes = [len(v) for v in self.type_groups.values()]
        return (
            f"{self.__class__.__name__}: "
            f"{len(self.index)} transitions, "
            f"{n_types} types, "
            f"group sizes min={min(sizes)} max={max(sizes)} mean={sum(sizes)/n_types:.1f}"
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class RoomTransitionPairDataset(PairDataset):
    """Pairs defined by room-boundary transitions.

    A transition ``(t1, t2)`` is the step where the agent moves from one
    room into another: ``t1`` is the last timestep in the old room,
    ``t2 = t1 + 1`` is the first timestep in the new room.

    Transition type: ``(from_room_id, to_room_id)``.
    Positive pair: another transition with the same room hop.

    Parameters
    ----------
    maze_dataset, min_group_size, seed:
        See :class:`PairDataset`.
    """

    def _build_index(self) -> list[tuple[int, int, int]]:
        index = []
        room = self.ds.room_labels  # (N, T)
        N, T = room.shape
        for i in range(N):
            for t in range(1, T):
                if room[i, t] != room[i, t - 1]:
                    index.append((i, t - 1, t))
        return index

    def _transition_type(self, traj_idx: int, t1: int, t2: int) -> tuple:
        from_room = int(self.ds.room_labels[traj_idx, t1])
        to_room = int(self.ds.room_labels[traj_idx, t2])
        return (from_room, to_room)


class PhaseTransitionPairDataset(PairDataset):
    """Pairs defined by phase-boundary transitions.

    A transition ``(t1, t2)`` is the step where the agent's phase changes:
    ``t1`` is the last timestep of the old phase, ``t2 = t1 + 1`` is the
    first of the new phase.

    Transition type: ``(from_phase_id, to_phase_id)``.
    Positive pair: another transition with the same phase hop.

    Parameters
    ----------
    maze_dataset, min_group_size, seed:
        See :class:`PairDataset`.
    """

    def _build_index(self) -> list[tuple[int, int, int]]:
        index = []
        phase = self.ds.phase_labels  # (N, T)
        N, T = phase.shape
        for i in range(N):
            for t in range(1, T):
                if phase[i, t] != phase[i, t - 1]:
                    index.append((i, t - 1, t))
        return index

    def _transition_type(self, traj_idx: int, t1: int, t2: int) -> tuple:
        from_phase = int(self.ds.phase_labels[traj_idx, t1])
        to_phase = int(self.ds.phase_labels[traj_idx, t2])
        return (from_phase, to_phase)


class WindowPairDataset(PairDataset):
    """Pairs defined by fixed-length sliding windows over trajectories.

    Suitable for SimSiam / BYOL where no explicit negatives are needed.
    Each item is a window of length ``window_len``; the anchor state ``s1``
    is the first timestep and ``s2`` is the last.

    Transition type: the ``(phase_id, room_id)`` at the window's first
    timestep, so that positive pairs share the same coarse context.

    Parameters
    ----------
    maze_dataset:
        Source dataset.
    window_len:
        Length of each window in timesteps.
    stride:
        Step between consecutive window starts. Defaults to ``window_len``
        (non-overlapping windows).
    min_group_size, seed:
        See :class:`PairDataset`.
    """

    def __init__(
        self,
        root: str | None = None,
        maze_dataset: MazeOracleDataset | None = None,
        state_mode: str = "latent",
        window_len: int = 10,
        stride: int | None = None,
        min_group_size: int = 2,
        seed: int = 42,
    ):
        self._window_len = window_len
        self._stride = stride if stride is not None else window_len
        super().__init__(
            root=root, maze_dataset=maze_dataset, state_mode=state_mode,
            min_group_size=min_group_size, seed=seed,
        )

    def _build_index(self) -> list[tuple[int, int, int]]:
        index = []
        N, T = self.ds.actions.shape
        for i in range(N):
            t = 0
            while t + self._window_len - 1 < T:
                index.append((i, t, t + self._window_len - 1))
                t += self._stride
        return index

    def _transition_type(self, traj_idx: int, t1: int, t2: int) -> tuple:
        phase = int(self.ds.phase_labels[traj_idx, t1])
        room = int(self.ds.room_labels[traj_idx, t1])
        return (phase, room)

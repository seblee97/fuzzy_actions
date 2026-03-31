"""Pair-construction datasets for hierarchical inverse-model training.

Each dataset wraps a ``MazeOracleDataset`` and yields dicts of the form::

    {
        "s1_a": state tensor  — anchor start state
        "s2_a": state tensor  — anchor end state
        "s1_b": state tensor  — positive start state
        "s2_b": state tensor  — positive end state
    }

for use with ``train_hierarchical.py``.

Pair semantics
--------------
A "transition" is not a single boundary step but a full room-to-room (or
phase-to-phase) journey.  The index stores the *time ranges* of the source
and destination segments:

    (traj_idx, src_start, src_end, dst_start, dst_end)

At each ``__getitem__`` call, ``t1`` is sampled uniformly from
``[src_start, src_end]`` and ``t2`` from ``[dst_start, dst_end]``.  This
means every call to the same index entry can return a different (s1, s2)
pair, giving broad coverage of the transition without storing every frame
explicitly.

Positive pairs: two index entries with the same transition type (e.g. the
same room-to-room hop) drawn from *different* trajectories (or different
visits to the same room pair in the same trajectory).

State representation
--------------------
``state_mode="latent"``
    One-hot (phase | material | room) flat vector.
    ``state_dim = n_phases + n_materials + n_rooms``

``state_mode="pixel"``
    Room pixel frame ``(C, H, W)`` float32 in [0, 1].
    Requires the pixel cache to be built (done automatically on first use).

Variants
--------
RoomTransitionPairDataset
    Segments trajectories by room visits.  Each consecutive
    (room_A_visit, room_B_visit) pair is one index entry.

PhaseTransitionPairDataset
    Same, but segmented by phase labels.

WindowPairDataset
    Fixed-length sliding windows.  ``s1`` is the first frame of the window,
    ``s2`` the last.  Positive = another window starting in the same
    (phase, room).  Suitable for SimSiam / BYOL.
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
# Helpers
# ---------------------------------------------------------------------------

def _segment_by_label(labels_1d) -> list[tuple[int, int, int]]:
    """Split a 1-D label sequence into (label, t_start, t_end) segments.

    Both ``t_start`` and ``t_end`` are inclusive.
    """
    T = len(labels_1d)
    segments = []
    current = int(labels_1d[0])
    t_start = 0
    for t in range(1, T):
        v = int(labels_1d[t])
        if v != current:
            segments.append((current, t_start, t - 1))
            current = v
            t_start = t
    segments.append((current, t_start, T - 1))
    return segments


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
        Pre-built ``MazeOracleDataset``.  Mutually exclusive with *root*.
    state_mode:
        ``"latent"`` or ``"pixel"``.
    pixel_style:
        ``"game"`` (pygame shapes) or ``"raw"`` (coloured squares).
        Only relevant when ``state_mode="pixel"``.
    obs_cell_size:
        Pixels per grid cell in the rendered observation.  Smaller values
        produce smaller frames and a much more compact pixel cache.
        Only relevant when ``state_mode="pixel"``.
    min_group_size:
        Drop transition types with fewer than this many examples (no valid
        positive pair can be formed otherwise).
    seed:
        RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        root: str | None = None,
        maze_dataset: MazeOracleDataset | None = None,
        state_mode: str = "latent",
        pixel_style: str = "game",
        obs_cell_size: int = 8,
        max_trajs: int | None = None,
        resize_obs: tuple[int, int] | None = None,
        min_group_size: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        assert state_mode in ("latent", "pixel"), (
            f"state_mode must be 'latent' or 'pixel'; got '{state_mode}'"
        )
        assert pixel_style in ("game", "raw"), (
            f"pixel_style must be 'game' or 'raw'; got '{pixel_style}'"
        )
        self.state_mode = state_mode
        self.pixel_style = pixel_style
        self.obs_cell_size = obs_cell_size
        self.resize_obs = resize_obs  # (H_out, W_out) or None
        variant = "full" if state_mode == "pixel" else "actions_only"

        if maze_dataset is not None:
            self.ds = maze_dataset
        elif root is not None:
            self.ds = MazeOracleDataset(
                root=root, variant=variant,
                pixel_style=pixel_style, obs_cell_size=obs_cell_size,
                max_trajs=max_trajs, resize=resize_obs,
            )
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
            self.ds.prepare_pixels()
            self.ds._ensure_pixel_cache()
            H, W, C = self.ds._room_pixels_mmap[0, 0].shape
            if resize_obs is not None:
                H, W = resize_obs
            self.pixel_shape = (C, H, W)
            self.state_dim = None

        # Build index: list of (traj_idx, src_start, src_end, dst_start, dst_end)
        raw_index = self._build_index()
        type_keys = [self._transition_type(e) for e in raw_index]

        groups: dict[Hashable, list[int]] = defaultdict(list)
        for pos, key in enumerate(type_keys):
            groups[key].append(pos)

        valid_positions: set[int] = set()
        for members in groups.values():
            if len(members) >= min_group_size:
                valid_positions.update(members)

        self.index = [raw_index[i] for i in sorted(valid_positions)]
        remapped = {old: new for new, old in enumerate(sorted(valid_positions))}

        self.type_groups: dict[Hashable, list[int]] = defaultdict(list)
        for old_pos, key in enumerate(type_keys):
            if old_pos in valid_positions:
                self.type_groups[key].append(remapped[old_pos])

        self.item_type = [self._transition_type(e) for e in self.index]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_index(self) -> list[tuple]:
        """Return all valid index entries.

        Each entry is a tuple
        ``(traj_idx, src_start, src_end, dst_start, dst_end)``
        where ``src_*`` are the inclusive timestep bounds of the source
        segment and ``dst_*`` of the destination segment.
        """

    @abstractmethod
    def _transition_type(self, entry: tuple) -> Hashable:
        """Return a hashable key grouping entries of the same transition type."""

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _extract_state(self, traj_idx: int, t: int) -> torch.Tensor:
        """Extract state at timestep ``t`` of trajectory ``traj_idx``.

        ``"latent"`` — one-hot (phase | material | room) flat vector.
        ``"pixel"``  — room pixel frame (C, H, W) float32 in [0, 1].
        """
        if self.state_mode == "pixel":
            import numpy as np
            self.ds._ensure_pixel_cache()
            frame = np.array(self.ds._room_pixels_mmap[traj_idx, t])
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
        traj_a, src_lo_a, src_hi_a, dst_lo_a, dst_hi_a = self.index[idx]
        key = self.item_type[idx]

        # Sample t1, t2 uniformly within the source / destination segments
        t1_a = self._rng.randint(src_lo_a, src_hi_a)
        t2_a = self._rng.randint(dst_lo_a, dst_hi_a)

        # Sample a distinct positive entry from the same transition type
        candidates = self.type_groups[key]
        pos_idx = idx
        while pos_idx == idx and len(candidates) > 1:
            pos_idx = self._rng.choice(candidates)
        traj_b, src_lo_b, src_hi_b, dst_lo_b, dst_hi_b = self.index[pos_idx]
        t1_b = self._rng.randint(src_lo_b, src_hi_b)
        t2_b = self._rng.randint(dst_lo_b, dst_hi_b)

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
            f"group sizes min={min(sizes)} max={max(sizes)} "
            f"mean={sum(sizes)/n_types:.1f}"
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class RoomTransitionPairDataset(PairDataset):
    """Pairs spanning full room-to-room transitions.

    Segments each trajectory into room visits.  Each consecutive pair of
    visits (room A → room B) becomes one index entry.

    ``s1`` is sampled uniformly from the room-A visit;
    ``s2`` is sampled uniformly from the room-B visit.

    Transition type: ``(from_room_id, to_room_id)``.
    """

    def _build_index(self) -> list[tuple]:
        index = []
        N = self.ds.n_trajs
        for i in range(N):
            segments = _segment_by_label(self.ds.room_labels[i])
            for j in range(len(segments) - 1):
                _, src_start, src_end = segments[j]
                _, dst_start, dst_end = segments[j + 1]
                index.append((i, src_start, src_end, dst_start, dst_end))
        return index

    def _transition_type(self, entry: tuple) -> tuple:
        traj_idx, src_start, src_end, dst_start, dst_end = entry
        from_room = int(self.ds.room_labels[traj_idx, src_start])
        to_room = int(self.ds.room_labels[traj_idx, dst_start])
        return (from_room, to_room)


class PhaseTransitionPairDataset(PairDataset):
    """Pairs spanning full phase-to-phase transitions.

    Segments each trajectory by phase label.  Each consecutive
    (phase A → phase B) visit pair becomes one index entry.

    ``s1`` is sampled uniformly from the phase-A segment;
    ``s2`` is sampled uniformly from the phase-B segment.

    Transition type: ``(from_phase_id, to_phase_id)``.
    """

    def _build_index(self) -> list[tuple]:
        index = []
        N = self.ds.n_trajs
        for i in range(N):
            segments = _segment_by_label(self.ds.phase_labels[i])
            for j in range(len(segments) - 1):
                _, src_start, src_end = segments[j]
                _, dst_start, dst_end = segments[j + 1]
                index.append((i, src_start, src_end, dst_start, dst_end))
        return index

    def _transition_type(self, entry: tuple) -> tuple:
        traj_idx, src_start, src_end, dst_start, dst_end = entry
        from_phase = int(self.ds.phase_labels[traj_idx, src_start])
        to_phase = int(self.ds.phase_labels[traj_idx, dst_start])
        return (from_phase, to_phase)


class WindowPairDataset(PairDataset):
    """Pairs defined by fixed-length sliding windows.

    Each window of length ``window_len`` becomes one index entry.
    ``s1`` is always the first frame, ``s2`` the last.

    Positive pair: another window where the first frame is in the same
    ``(phase, room)`` context.

    Suitable for SimSiam / BYOL where no explicit negatives are needed.
    """

    def __init__(
        self,
        root: str | None = None,
        maze_dataset: MazeOracleDataset | None = None,
        state_mode: str = "latent",
        pixel_style: str = "game",
        obs_cell_size: int = 8,
        max_trajs: int | None = None,
        resize_obs: tuple[int, int] | None = None,
        window_len: int = 10,
        stride: int | None = None,
        min_group_size: int = 2,
        seed: int = 42,
    ):
        self._window_len = window_len
        self._stride = stride if stride is not None else window_len
        super().__init__(
            root=root, maze_dataset=maze_dataset, state_mode=state_mode,
            pixel_style=pixel_style, obs_cell_size=obs_cell_size,
            max_trajs=max_trajs, resize_obs=resize_obs,
            min_group_size=min_group_size, seed=seed,
        )

    def _build_index(self) -> list[tuple]:
        index = []
        N, T = self.ds.actions.shape
        for i in range(N):
            t = 0
            while t + self._window_len - 1 < T:
                t_end = t + self._window_len - 1
                # src and dst ranges are single points (no intra-window sampling)
                index.append((i, t, t, t_end, t_end))
                t += self._stride
        return index

    def _transition_type(self, entry: tuple) -> tuple:
        traj_idx, src_start, *_ = entry
        phase = int(self.ds.phase_labels[traj_idx, src_start])
        room = int(self.ds.room_labels[traj_idx, src_start])
        return (phase, room)

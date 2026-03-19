"""Replay buffers for DQN.

Exports
-------
ReplayBuffer            -- flat numpy arrays for symbolic/vector observations
DictReplayBuffer        -- per-key numpy arrays for dict (image) observations
FramedDictReplayBuffer  -- memory-efficient buffer for frame-stacked pixel obs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import gymnasium


# ---------------------------------------------------------------------------
# Flat replay buffer (used by train_dqn.py)
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size circular replay buffer storing (s, a, r, s', done) transitions.

    Args:
        obs_shape: Shape of a single observation (e.g. ``(obs_dim,)``).
        buffer_size: Maximum number of transitions to store.
        device: Torch device to move sampled batches onto.
        seed: Optional seed for the internal numpy RNG used during sampling.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        buffer_size: int,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device

        self._obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(buffer_size, dtype=np.int64)
        self._rewards = np.zeros(buffer_size, dtype=np.float32)
        self._dones = np.zeros(buffer_size, dtype=np.float32)

        self._pos = 0
        self._size = 0

        # Dedicated RNG so sampling is independent of global numpy state.
        self._rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._obs[self._pos] = obs
        self._next_obs[self._pos] = next_obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self._size}."
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self._obs[idx], device=self.device),
            actions=torch.as_tensor(self._actions[idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self._next_obs[idx], device=self.device),
            dones=torch.as_tensor(self._dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Dict replay buffer (kept for reference; superseded by FramedDictReplayBuffer)
# ---------------------------------------------------------------------------

@dataclass
class DictBatch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: dict[str, torch.Tensor]
    dones: torch.Tensor


class DictReplayBuffer:
    """Circular replay buffer for dict observations (e.g. image + map pairs).

    Each observation key gets its own pre-allocated numpy array, preserving
    the original dtype (typically ``uint8`` for pixel arrays, saving 4× memory
    compared to ``float32``).

    Note: for frame-stacked observations prefer ``FramedDictReplayBuffer``
    which avoids the ~8× redundancy between ``obs`` and ``next_obs``.

    Args:
        obs_space: A ``gymnasium.spaces.Dict`` describing the observation.
        buffer_size: Maximum number of transitions to store.
        device: Torch device to send sampled batches to.
        seed: Optional seed for the internal sampling RNG.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        buffer_size: int,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.device = device
        self._rng = np.random.default_rng(seed)

        self._obs: dict[str, np.ndarray] = {
            k: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for k, space in obs_space.spaces.items()
        }
        self._next_obs: dict[str, np.ndarray] = {
            k: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for k, space in obs_space.spaces.items()
        }
        self._actions = np.zeros(buffer_size, dtype=np.int64)
        self._rewards = np.zeros(buffer_size, dtype=np.float32)
        self._dones = np.zeros(buffer_size, dtype=np.float32)

        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        for k in self._obs:
            self._obs[k][self._pos] = obs[k]
            self._next_obs[k][self._pos] = next_obs[k]
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> DictBatch:
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self._size}."
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return DictBatch(
            obs={k: torch.as_tensor(v[idx], device=self.device) for k, v in self._obs.items()},
            next_obs={k: torch.as_tensor(v[idx], device=self.device) for k, v in self._next_obs.items()},
            actions=torch.as_tensor(self._actions[idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[idx], device=self.device),
            dones=torch.as_tensor(self._dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Framed dict replay buffer — memory-efficient for frame-stacked pixel obs
# ---------------------------------------------------------------------------

class FramedDictReplayBuffer:
    """Memory-efficient replay buffer for frame-stacked dict observations.

    Instead of storing a full ``n_stack``-frame stack for both ``obs`` and
    ``next_obs`` (2 × n_stack frames per transition), only the **newest raw
    frame** from ``obs`` is stored per transition.  Both stacks are
    reconstructed on-the-fly at sample time by looking back through the frame
    history, clamping at episode boundaries so frames are never mixed across
    resets.

    Memory vs ``DictReplayBuffer``::

        DictReplayBuffer:         buffer_size × 2 × n_stack × frame_bytes
        FramedDictReplayBuffer:   buffer_size × 1              × frame_bytes
        Reduction: 2 × n_stack   (8× for n_stack=4)

    Usage notes
    -----------
    - Call ``on_reset()`` immediately after each ``env.reset()`` so episode
      boundaries are recorded correctly.
    - Pass ``terminated`` and ``truncated`` separately to ``add()``; both
      trigger a boundary mark, but only ``terminated`` zeroes the Bellman
      bootstrap.
    - The most recently added position is excluded from sampling because its
      ``next_obs`` frame (at ``pos + 1``) has not been written yet.

    Args:
        single_frame_space: ``gymnasium.spaces.Dict`` with ``(H, W, 1)``
            shapes — the per-stream space *before* frame stacking.  Derive
            from the stacked obs space by dividing channel count by ``n_stack``.
        buffer_size: Maximum number of transitions.
        n_stack: Frames per stacked observation (must match ``FrameStackWrapper``).
        device: Torch device for sampled batches.
        seed: Optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        single_frame_space: gymnasium.spaces.Dict,
        buffer_size: int,
        n_stack: int,
        device: torch.device,
        seed: int | None = None,
    ) -> None:
        self.n_stack = n_stack
        self.buffer_size = buffer_size
        self.device = device
        self._rng = np.random.default_rng(seed)

        # One raw (H, W, 1) frame per stream per transition.
        self._frames: dict[str, np.ndarray] = {
            k: np.zeros((buffer_size, *space.shape), dtype=space.dtype)
            for k, space in single_frame_space.spaces.items()
        }
        self._actions = np.zeros(buffer_size, dtype=np.int64)
        self._rewards = np.zeros(buffer_size, dtype=np.float32)
        self._dones = np.zeros(buffer_size, dtype=np.float32)   # terminated only
        self._ep_start = np.zeros(buffer_size, dtype=bool)

        self._pos = 0
        self._size = 0
        self._next_is_ep_start = True

    def on_reset(self) -> None:
        """Signal that the next ``add()`` call begins a new episode."""
        self._next_is_ep_start = True

    def add(
        self,
        obs_stack: dict[str, np.ndarray],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition — only the newest frame of obs_stack is kept."""
        for k in self._frames:
            self._frames[k][self._pos] = obs_stack[k][..., -1:]  # (H, W, 1)
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(terminated)
        self._ep_start[self._pos] = self._next_is_ep_start
        self._next_is_ep_start = terminated or truncated

        self._pos = (self._pos + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def _build_stack(self, end_pos: int, key: str) -> np.ndarray:
        """Reconstruct an n_stack-frame observation ending at end_pos.

        Walks backward from end_pos, clamping at episode boundaries by
        repeating the boundary frame for all older slots.

        Returns ``(H, W, n_stack)`` uint8, oldest frame first.
        """
        frames = []
        for i in range(self.n_stack):
            pos = (end_pos - i) % self.buffer_size
            frames.append(self._frames[key][pos])
            if i < self.n_stack - 1 and self._ep_start[pos]:
                # Hit episode boundary — repeat this frame for older slots.
                frames.extend([self._frames[key][pos]] * (self.n_stack - 1 - i))
                break
        frames.reverse()                        # oldest → newest
        return np.concatenate(frames, axis=-1)  # (H, W, n_stack)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        """Sample valid indices, excluding the most recently written position."""
        if self._size < 2:
            raise ValueError("Buffer needs at least 2 transitions to sample.")
        if self._size < self.buffer_size:
            # Buffer not full: valid range is [0, _size − 2].
            return self._rng.integers(0, self._size - 1, size=batch_size)
        else:
            # Buffer full: exclude (_pos − 1), which has no written next-frame.
            # Shift the range to start from _pos (oldest valid entry).
            idx = self._rng.integers(0, self.buffer_size - 1, size=batch_size)
            return (idx + self._pos) % self.buffer_size

    def sample(self, batch_size: int) -> DictBatch:
        idx = self._sample_indices(batch_size)
        next_idx = (idx + 1) % self.buffer_size

        obs_arrays = {
            k: np.stack([self._build_stack(i, k) for i in idx])
            for k in self._frames
        }
        next_obs_arrays = {
            k: np.stack([self._build_stack(i, k) for i in next_idx])
            for k in self._frames
        }
        return DictBatch(
            obs={k: torch.as_tensor(v, device=self.device) for k, v in obs_arrays.items()},
            next_obs={k: torch.as_tensor(v, device=self.device) for k, v in next_obs_arrays.items()},
            actions=torch.as_tensor(self._actions[idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[idx], device=self.device),
            dones=torch.as_tensor(self._dones[idx], device=self.device),
        )

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

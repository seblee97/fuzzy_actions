"""Replay buffers for DQN.

Exports
-------
ReplayBuffer        -- flat numpy arrays for symbolic/vector observations
DictReplayBuffer    -- per-key numpy arrays for dict (image) observations
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
# Dict replay buffer (used by train_dqn_modular.py with image observations)
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

    Args:
        obs_space: A ``gymnasium.spaces.Dict`` describing the observation.
        buffer_size: Maximum number of transitions to store.
        device: Torch device to send sampled batches to.
        seed: Optional seed for the internal sampling RNG.

    Memory estimate (uint8 pixel obs)::

        bytes = buffer_size × sum(np.prod(s.shape) × s.dtype.itemsize for s in obs_space)
                × 2   (obs + next_obs)
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

        # Pre-allocate one array per observation key (obs and next_obs).
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
            obs={
                k: torch.as_tensor(v[idx], device=self.device)
                for k, v in self._obs.items()
            },
            next_obs={
                k: torch.as_tensor(v[idx], device=self.device)
                for k, v in self._next_obs.items()
            },
            actions=torch.as_tensor(self._actions[idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[idx], device=self.device),
            dones=torch.as_tensor(self._dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self._size

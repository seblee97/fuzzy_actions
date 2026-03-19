"""Shared utilities: seeding, schedules, environment factories."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed *all* sources of randomness for full reproducibility.

    Sources covered:
    - Python stdlib ``random``
    - ``numpy`` global RNG
    - PyTorch CPU and CUDA RNGs
    - ``PYTHONHASHSEED`` environment variable (affects dict/set ordering)
    - cuDNN deterministic / benchmark flags (optional)

    Note: Pass ``seed`` to ``env.reset(seed=seed)`` separately to seed the
    environment's internal RNG.

    Args:
        seed: Integer seed.
        deterministic_cudnn: If True, force cuDNN into deterministic mode.
            Guarantees reproducibility on GPU at a potential throughput cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Exploration schedule
# ---------------------------------------------------------------------------

def linear_schedule(
    start: float,
    end: float,
    fraction: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Linearly decay from *start* to *end* over the first *fraction* of training.

    After ``fraction * total_steps`` steps the value clamps at *end*.
    """
    duration = fraction * total_steps
    slope = (end - start) / duration
    return max(end, start + slope * current_step)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(layout_path: str | Path, seed: int, **env_kwargs):
    """Create and seed a ``GridWorldEnv`` from a layout file.

    Args:
        layout_path: Path to a ``.txt`` layout file.
        seed: Seed passed to ``env.reset()``.
        **env_kwargs: Forwarded to ``GridWorldEnv``.

    Returns:
        A seeded ``GridWorldEnv`` instance (already reset once).
    """
    from gridworld_env import GridWorldEnv

    env = GridWorldEnv(str(layout_path), **env_kwargs)
    env.reset(seed=seed)
    return env


def make_modular_env(
    n_rooms: int,
    room_h: int,
    room_w: int,
    map_cell_size: int,
    distractor: bool,
    layout_seed: int,
    local_size: tuple[int, int] = (84, 84),
    n_stack: int = 4,
    **env_kwargs,
):
    """Procedurally generate a ``ModularMazeEnv`` with Atari-style preprocessing.

    Observation pipeline applied in order:

    1. ``ModularMazeEnv`` with ``obs_mode="room_pixels"`` and
       ``global_map_mode="image"`` — produces raw RGB dict obs.
    2. ``AtariPreprocessWrapper`` — grayscale + resize local obs to
       ``local_size``; grayscale map (spatial size preserved).
       Both streams become ``(H, W, 1)`` uint8.
    3. ``FrameStackWrapper`` — concatenates the last ``n_stack`` frames along
       the channel axis → ``(H, W, n_stack)`` uint8 per stream.

    The resulting observation space is a ``gymnasium.spaces.Dict`` with:
    - ``"obs"``:       ``(*local_size, n_stack)`` uint8
    - ``"map_image"``: ``(map_H, map_W, n_stack)`` uint8

    Seeds
    -----
    ``layout_seed`` is passed to ``generate_world_grid``.  The episode seed
    (agent start position etc.) is set separately via ``env.reset(seed=...)``.

    Args:
        n_rooms: Number of rooms (1–26).
        room_h: Room height in cells (walls included).
        room_w: Room width in cells (walls included).
        map_cell_size: Pixel size of each room cell in the global map image.
        distractor: If True, 50 % of rooms get a spurious irrelevant object.
        layout_seed: RNG seed for ``generate_world_grid``.
        local_size: ``(H, W)`` to resize the local room image to (default 84×84).
        n_stack: Number of frames to stack (default 4).
        **env_kwargs: Forwarded to ``ModularMazeEnv`` (e.g. ``max_steps``).

    Returns:
        A fully wrapped ``ModularMazeEnv``, not yet reset.
    """
    from gridworld_env.procgen import generate_world_grid
    from gridworld_env.modular_maze import ModularMazeEnv
    from fuzzy_actions.wrappers import AtariPreprocessWrapper, FrameStackWrapper

    layout = generate_world_grid(
        n_rooms=n_rooms,
        room_h=room_h,
        room_w=room_w,
        distractor=distractor,
        seed=layout_seed,
    )
    env = ModularMazeEnv(
        layout,
        obs_mode="room_pixels",
        global_map_mode="image",
        map_cell_size=map_cell_size,
        **env_kwargs,
    )
    env = AtariPreprocessWrapper(env, local_size=local_size)
    env = FrameStackWrapper(env, n_stack=n_stack)
    return env

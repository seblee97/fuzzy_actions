"""Gymnasium observation wrappers for the modular maze environment."""

from __future__ import annotations

import collections

import numpy as np
import gymnasium
from gymnasium import spaces


class PixelDownscaleWrapper(gymnasium.ObservationWrapper):
    """Downscale the local ``"obs"`` pixel observation by an integer factor.

    The ``"map_image"`` key is left unchanged — it is already small.
    Downscaling uses block-average pooling (area averaging).

    Args:
        env: A ``ModularMazeEnv`` with dict obs (``"obs"`` + ``"map_image"``).
        local_scale: Integer divisor applied to both H and W of ``"obs"``.
    """

    def __init__(self, env: gymnasium.Env, local_scale: int = 4) -> None:
        super().__init__(env)
        self.local_scale = local_scale

        orig_local = env.observation_space["obs"]
        orig_h, orig_w, orig_c = orig_local.shape
        new_h, new_w = orig_h // local_scale, orig_w // local_scale

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0, high=255,
                    shape=(new_h, new_w, orig_c),
                    dtype=np.uint8,
                ),
                "map_image": env.observation_space["map_image"],
            }
        )

    def observation(self, obs: dict) -> dict:
        local = obs["obs"]
        h, w, c = local.shape
        s = self.local_scale
        new_h, new_w = h // s, w // s
        downscaled = (
            local[: new_h * s, : new_w * s]
            .reshape(new_h, s, new_w, s, c)
            .mean(axis=(1, 3))
            .astype(np.uint8)
        )
        return {"obs": downscaled, "map_image": obs["map_image"]}


class AtariPreprocessWrapper(gymnasium.ObservationWrapper):
    """Grayscale + resize preprocessing matching Atari DQN papers.

    Applied independently to both pixel streams:

    - ``"obs"`` (local room view): converted to grayscale and resized to
      ``local_size`` (default 84×84) using bilinear interpolation.
    - ``"map_image"`` (global map): converted to grayscale; spatial size is
      preserved (it is already small — one cell per room at ``map_cell_size``
      pixels).

    Both outputs are ``(H, W, 1)`` uint8 so that ``FrameStackWrapper`` can
    stack them along the channel axis to produce ``(H, W, n_stack)`` inputs.

    Requires **Pillow** (``pip install pillow``).

    Args:
        env: A ``ModularMazeEnv`` (optionally already wrapped) with dict obs.
        local_size: ``(height, width)`` to resize the local observation to.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        local_size: tuple[int, int] = (84, 84),
    ) -> None:
        super().__init__(env)
        from PIL import Image  # deferred import so the rest of the module works without pillow
        self._Image = Image
        self.local_size = local_size  # (H, W)

        map_h, map_w = env.observation_space["map_image"].shape[:2]

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0, high=255,
                    shape=(*local_size, 1),
                    dtype=np.uint8,
                ),
                "map_image": spaces.Box(
                    low=0, high=255,
                    shape=(map_h, map_w, 1),
                    dtype=np.uint8,
                ),
            }
        )

    def observation(self, obs: dict) -> dict:
        # Local: grayscale + bilinear resize to local_size
        local = self._Image.fromarray(obs["obs"]).convert("L")
        # PIL.resize takes (width, height)
        local = local.resize(
            (self.local_size[1], self.local_size[0]),
            self._Image.BILINEAR,
        )
        local_arr = np.array(local, dtype=np.uint8)[..., np.newaxis]  # (H, W, 1)

        # Map: grayscale only (preserve spatial dims)
        map_arr = np.array(
            self._Image.fromarray(obs["map_image"]).convert("L"),
            dtype=np.uint8,
        )[..., np.newaxis]  # (H, W, 1)

        return {"obs": local_arr, "map_image": map_arr}


class FrameStackWrapper(gymnasium.Wrapper):
    """Stack the last ``n_stack`` frames along the channel axis for all streams.

    On ``reset``, the buffer is filled with ``n_stack`` copies of the initial
    observation.  On each ``step``, the oldest frame is dropped and the new
    one appended.

    Input observation space must be a ``Dict`` of ``Box`` spaces with shape
    ``(H, W, C)``.  Output shape is ``(H, W, C * n_stack)`` per key.

    This is applied to *both* pixel streams:

    - ``"obs"``:       ``(84, 84, 1)`` → ``(84, 84, n_stack)``
    - ``"map_image"``: ``(H, W, 1)``  → ``(H, W, n_stack)``

    Args:
        env: Wrapped environment with dict pixel observations.
        n_stack: Number of consecutive frames to stack (default 4, matching
            the original DQN paper).
    """

    def __init__(self, env: gymnasium.Env, n_stack: int = 4) -> None:
        super().__init__(env)
        self.n_stack = n_stack
        self._frames: dict[str, collections.deque] = {}

        new_spaces = {}
        for k, space in env.observation_space.spaces.items():
            h, w, c = space.shape
            new_spaces[k] = spaces.Box(
                low=0, high=255,
                shape=(h, w, c * n_stack),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill each stream's deque with n_stack copies of the first frame.
        self._frames = {
            k: collections.deque(
                [np.copy(v) for _ in range(self.n_stack)],
                maxlen=self.n_stack,
            )
            for k, v in obs.items()
        }
        return self._stack(), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k, v in obs.items():
            self._frames[k].append(v)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> dict[str, np.ndarray]:
        return {
            k: np.concatenate(list(frames), axis=-1)
            for k, frames in self._frames.items()
        }

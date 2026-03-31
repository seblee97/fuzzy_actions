"""Oracle trajectory dataset for ModularMazeEnv.

Loads pre-generated trajectories (actions + labels + start positions) produced
by ``scripts/generate_oracle_dataset.py`` (single fixed layout) or
``scripts/generate_distribution_dataset.py`` (distribution of layout variants).
The dataset type is detected automatically from the files present in *root*.

Two variants:
    * **actions_only** — input ``x`` is a one-hot action sequence ``(T, 8)``.
    * **full** — input ``x`` is a dict of pixel tensors + one-hot actions.
      Pixels are rendered lazily from the stored action sequences and cached to
      disk as memory-mapped numpy files.  Call ``dataset.prepare_pixels()`` to
      pre-render the full cache (one-time cost), or let it happen automatically
      on first ``__getitem__`` access.

Label tuple format (matching other RPL datasets)::

    x, (seq_labels, dense_labels, cts_labels, cts_dense_labels, aux_labels)

Where:
    * ``dense_labels``  — ``(T, 3)`` int64: [phase_id, material_id, room_id]
    * ``seq_labels``    — ``(1,)`` int64: whether all safes were opened (0/1)
    * ``aux_labels``    — ``(T,)`` int64: raw action ids (useful for analysis)
    * ``cts_labels``, ``cts_dense_labels`` — empty tensors (unused)

For distribution datasets, ``__getitem__`` also returns ``variant_id`` as an
additional entry in ``aux_labels`` is NOT changed — variant ids are accessible
via ``dataset.variant_ids[idx]`` if needed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MazeOracleDataset(Dataset):
    """RPL-compatible dataset of oracle trajectories in a ModularMazeEnv.

    Supports both single-layout datasets (``generate_oracle_dataset.py``) and
    distribution datasets (``generate_distribution_dataset.py``).  The type is
    detected automatically: if ``variants.npz`` is present in *root* the dataset
    is treated as a distribution dataset.

    Parameters
    ----------
    root:
        Path to a dataset directory (e.g. ``datasets/oracle/directional``).
        Must contain ``labels.npz`` and ``metadata.json``.
    variant:
        ``"actions_only"`` or ``"full"``.  The *full* variant renders pixel
        observations lazily and caches them to ``<root>/pixel_cache/``.
    seq_len:
        If not None, crop sequences to this length (must be ≤ stored length).
    """

    PHASE_NAMES = [
        "get_key", "make_key", "get_key_location",
        "collect_reward", "goto_next_room",
    ]
    MATERIAL_NAMES = ["none", "diamond", "ruby", "sapphire"]

    def __init__(
        self,
        root: str | Path,
        variant: str = "actions_only",
        seq_len: int | None = None,
        pixel_style: str = "game",
    ):
        super().__init__()
        root = Path(root)
        assert variant in ("actions_only", "full"), (
            f"variant must be 'actions_only' or 'full', got '{variant}'"
        )
        assert pixel_style in ("game", "raw"), (
            f"pixel_style must be 'game' or 'raw'; got '{pixel_style}'"
        )
        self.root = root
        self.variant = variant
        self.pixel_style = pixel_style

        # Load metadata
        with open(root / "metadata.json") as f:
            self.metadata = json.load(f)

        # Load label arrays
        npz = np.load(root / "labels.npz")
        self.actions          = torch.tensor(npz["actions"],          dtype=torch.long)
        self.phase_labels     = torch.tensor(npz["phase_labels"],     dtype=torch.long)
        self.material_labels  = torch.tensor(npz["material_labels"],  dtype=torch.long)
        self.room_labels      = torch.tensor(npz["room_labels"],      dtype=torch.long)
        self.completion_steps = torch.tensor(npz["completion_steps"], dtype=torch.long)
        self.start_positions  = npz["start_positions"]                # (N, 2) int16

        self.n_trajs, self.stored_seq_len = self.actions.shape
        self.seq_len = seq_len or self.stored_seq_len
        assert self.seq_len <= self.stored_seq_len

        self.n_actions   = 8
        self.n_phases    = len(self.PHASE_NAMES)
        self.n_materials = len(self.MATERIAL_NAMES)

        # --- Distribution dataset detection --------------------------------
        self.is_distribution = (root / "variants.npz").exists()
        if self.is_distribution:
            self.variant_ids         = npz["variant_ids"]              # (N,) int32
            self.n_trajs_per_variant = self.metadata["n_trajs_per_variant"]
            self._variant_data: Optional[np.lib.npyio.NpzFile] = None
            self._variant_envs: Dict[int, object] = {}                 # vid -> pixel env
        else:
            self.variant_ids         = None
            self.n_trajs_per_variant = None
            self._pixel_env          = None

        # Shared pixel cache state
        self._pixel_cache_dir: Optional[Path] = None
        self._room_pixels_mmap: Optional[np.ndarray] = None
        self._map_images_mmap:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Seed helper
    # ------------------------------------------------------------------

    def _traj_seed(self, global_idx: int) -> int:
        """Return the RNG seed used when generating trajectory *global_idx*."""
        layout_seed = self.metadata["layout_seed"]
        if self.is_distribution:
            vid    = int(self.variant_ids[global_idx])
            within = global_idx - vid * self.n_trajs_per_variant
            return vid * 1_000_000 + layout_seed + 1000 + within
        else:
            return layout_seed + 1000 + global_idx

    # ------------------------------------------------------------------
    # Pixel env helpers
    # ------------------------------------------------------------------

    def _get_pixel_env(self, variant_id: int = 0):
        """Return (and lazily create) a pixel-mode env for *variant_id*.

        For single-layout datasets *variant_id* is ignored and a single env
        is reused.  For distribution datasets a separate env is cached per
        variant.
        """
        from gridworld_env.modular_maze import ModularMazeEnv
        from gridworld_env.procgen import generate_world_grid

        meta = self.metadata

        if not self.is_distribution:
            if self._pixel_env is None:
                from gridworld_env.replay import create_pixel_env
                self._pixel_env = create_pixel_env(
                    layout_seed=meta["layout_seed"],
                    n_rooms=meta["n_rooms"],
                    room_h=meta["room_h"],
                    room_w=meta["room_w"],
                    seq_len=meta["seq_len"],
                    pixel_render_style=self.pixel_style,
                )
            return self._pixel_env

        # Distribution: one env per variant, cached in _variant_envs
        if variant_id not in self._variant_envs:
            from gridworld_env.env_distribution import apply_variant, LayoutVariant

            # Load base layout
            base = generate_world_grid(
                n_rooms=meta["n_rooms"],
                room_h=meta["room_h"],
                room_w=meta["room_w"],
                seed=meta["layout_seed"],
            )

            # Load variant position data (open once, keep file handle)
            if self._variant_data is None:
                self._variant_data = np.load(self.root / "variants.npz")
            vd = self._variant_data

            layout = apply_variant(base, LayoutVariant(
                key_positions      = vd["key_positions"][variant_id],
                safe_positions     = vd["safe_positions"][variant_id],
                door_positions     = vd["door_positions"][variant_id],
                material_positions = vd["material_positions"][variant_id],
                npc_positions      = vd["npc_positions"][variant_id],
            ))

            self._variant_envs[variant_id] = ModularMazeEnv(
                layout,
                obs_mode="room_pixels",
                global_map_mode="image",
                max_steps=meta["seq_len"] + 10,
                terminate_on_all_safes_opened=False,
                pixel_render_style=self.pixel_style,
            )

        return self._variant_envs[variant_id]

    # ------------------------------------------------------------------
    # Pixel cache
    # ------------------------------------------------------------------

    @property
    def pixels_cached(self) -> bool:
        """True if the full pixel cache exists on disk."""
        cache = self.root / f"pixel_cache_{self.pixel_style}"
        return (
            (cache / "room_pixels.npy").exists()
            and (cache / "map_images.npy").exists()
        )

    def prepare_pixels(self, num_workers: int = 1) -> None:
        """Pre-render and cache all pixel observations to disk.

        This is a one-time operation.  Progress is printed to stdout.
        Trajectories that are already cached are skipped (safe to interrupt
        and resume).
        """
        from gridworld_env.replay import replay_trajectory

        cache = self.root / f"pixel_cache_{self.pixel_style}"
        cache.mkdir(parents=True, exist_ok=True)

        # Determine pixel shapes from a dummy env
        probe_env = self._get_pixel_env(variant_id=0)
        probe_env.reset(seed=0)
        obs         = probe_env._get_observation()
        local_shape = obs["obs"].shape       # (H, W, 3)
        map_shape   = obs["map_image"].shape # (mH, mW, 3)

        N, T = self.n_trajs, self.stored_seq_len

        # Create or open memmaps
        rp_path = cache / "room_pixels.npy"
        mi_path = cache / "map_images.npy"

        if rp_path.exists():
            room_pixels = np.load(str(rp_path), mmap_mode="r+")
            map_images  = np.load(str(mi_path), mmap_mode="r+")
        else:
            room_pixels = np.lib.format.open_memmap(
                str(rp_path), mode="w+", dtype=np.uint8,
                shape=(N, T, *local_shape),
            )
            map_images = np.lib.format.open_memmap(
                str(mi_path), mode="w+", dtype=np.uint8,
                shape=(N, T, *map_shape),
            )

        # Sidecar tracking file
        done_path = cache / "rendered.npy"
        rendered  = np.load(str(done_path)) if done_path.exists() else np.zeros(N, dtype=bool)

        actions_np = self.actions.numpy()
        t0     = time.time()
        n_todo = int((~rendered).sum())
        n_done = 0

        print(f"Rendering pixels: {n_todo} trajectories to go "
              f"({int(rendered.sum())} already cached)")

        for i in range(N):
            if rendered[i]:
                continue

            traj_seed  = self._traj_seed(i)
            vid        = int(self.variant_ids[i]) if self.is_distribution else 0
            env        = self._get_pixel_env(variant_id=vid)

            rp, mi = replay_trajectory(
                env, traj_seed, self.start_positions[i], actions_np[i],
            )
            room_pixels[i] = rp
            map_images[i]  = mi
            rendered[i]    = True
            n_done        += 1

            if n_done % max(1, n_todo // 20) == 0:
                elapsed = time.time() - t0
                rate    = n_done / elapsed
                print(f"  [{n_done:>6}/{n_todo}]  {rate:.1f} traj/s")

        np.save(str(done_path), rendered)
        room_pixels.flush()
        map_images.flush()

        elapsed = time.time() - t0
        print(f"Pixel cache complete in {elapsed:.1f}s  →  {cache}")

    def _ensure_pixel_cache(self) -> None:
        """Open (and build if needed) the pixel cache memmaps."""
        if self._room_pixels_mmap is not None:
            return

        if not self.pixels_cached:
            print("Pixel cache not found — rendering now (one-time cost)...",
                  file=sys.stderr)
            self.prepare_pixels()

        cache = self.root / f"pixel_cache_{self.pixel_style}"
        self._pixel_cache_dir  = cache
        self._room_pixels_mmap = np.load(str(cache / "room_pixels.npy"), mmap_mode="r")
        self._map_images_mmap  = np.load(str(cache / "map_images.npy"),  mmap_mode="r")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_trajs

    def __getitem__(self, idx: int):
        T       = self.seq_len
        actions = self.actions[idx, :T]   # (T,)

        if self.variant == "actions_only":
            x = F.one_hot(actions, num_classes=self.n_actions).float().T  # (8, T)
        else:
            self._ensure_pixel_cache()

            room_px = torch.tensor(
                np.array(self._room_pixels_mmap[idx, :T]), dtype=torch.float32,
            )
            room_px = room_px.permute(0, 3, 1, 2) / 255.0   # (T, C, H, W)

            map_img = torch.tensor(
                np.array(self._map_images_mmap[idx, :T]), dtype=torch.float32,
            )
            map_img = map_img.permute(0, 3, 1, 2) / 255.0

            action_oh = F.one_hot(actions, num_classes=self.n_actions).float()
            x = {"room_pixels": room_px, "map_image": map_img, "actions": action_oh}

        # -- Labels -------------------------------------------------------
        seq_labels = (self.completion_steps[idx] > 0).long().unsqueeze(0)  # (1,)

        dense_labels = torch.stack([
            self.phase_labels[idx, :T],
            self.material_labels[idx, :T],
            self.room_labels[idx, :T],
        ], dim=-1)  # (T, 3)

        aux_labels       = actions
        cts_labels       = torch.tensor([])
        cts_dense_labels = torch.tensor([])

        return x, (seq_labels, dense_labels, cts_labels, cts_dense_labels, aux_labels)

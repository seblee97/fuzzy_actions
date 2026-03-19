"""DQN training on a single procedurally generated ModularMazeEnv task.

Two-stream network
------------------
The agent receives two pixel inputs at each step:

  local_obs   (H_room, W_room, 3) uint8
      Partial observability: only the agent's current room is visible.
      Processed by ``LocalCNN`` (Nature DQN convolutions + adaptive pool).

  map_image   (H_map, W_map, 3) uint8
      Global map showing all rooms; the agent's current room is highlighted.
      Processed by ``MapNet`` (lightweight 2-layer CNN + adaptive pool).

Their embeddings are concatenated and fed through a shared MLP head to
produce Q-values for the 8 discrete actions.

Procgen
-------
One maze is generated at startup using ``generate_world_grid`` seeded by
``--seed``.  The same layout is used for every episode of training and
evaluation; only the agent's starting position (and any other stochastic
episode initialisation) varies via the per-episode reset seed.

Reproducibility
---------------
A single ``--seed`` controls every source of randomness:
  - Python random, numpy, torch (CPU + CUDA), cuDNN flags
  - Layout generation:          layout_seed = seed
  - Per-episode env reset:      env_seed    = seed + ep_count
  - Replay buffer sampling RNG: seed + 1
  - Evaluation episode resets:  seed + 50_000 + eval_ep

Memory
------
Pixel observations are stored as uint8 in the replay buffer.  Normalisation
to float32 happens inside the network forward pass.
With the defaults (local_size=84×84, n_stack=4, n_rooms=4):
  - local obs: 84×84×4 = 28 KB per transition
  - map image: 16×16×4 = 1 KB per transition (4 rooms in 2×2 grid)
  - buffer_size=50_000 → ~2.9 GB RAM (obs + next_obs)

Usage
-----
    python train_dqn_modular.py --seed 42

    python train_dqn_modular.py \\
        --n-rooms 9 --room-h 9 --room-w 11 \\
        --total-timesteps 2000000 --seed 0

    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium

from fuzzy_actions import (
    FramedDictReplayBuffer,
    TwoStreamQNetwork,
    linear_schedule,
    make_modular_env,
    set_seeds,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- procgen ---
    n_rooms: int = 4          # number of rooms per episode layout
    room_h: int = 9           # room height in cells (includes walls)
    room_w: int = 11          # room width in cells (includes walls)
    map_cell_size: int = 8    # pixels per room cell in the global map image
    distractor: bool = False          # add spurious objects to some rooms
    local_size: tuple[int, int] = (84, 84)  # resize local obs to this (H, W)
    n_stack: int = 4                  # frames to stack along channel axis

    # --- training ---
    total_timesteps: int = 1_000_000
    seed: int = 42

    # --- DQN hyperparameters ---
    lr: float = 1e-4
    buffer_size: int = 50_000
    gamma: float = 0.99
    batch_size: int = 32
    learning_starts: int = 20_000
    train_frequency: int = 4
    target_update_frequency: int = 2_000

    # --- epsilon-greedy exploration ---
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.2

    # --- network ---
    local_embed_dim: int = 256
    map_embed_dim: int = 64
    hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (256, 128))

    # --- environment ---
    max_steps: int = 500
    step_reward: float = -0.01
    collision_reward: float = -0.1
    terminate_on_all_safes_opened: bool = True

    # --- evaluation ---
    eval_frequency: int = 20_000
    eval_episodes: int = 10   # layouts per eval (each gets a fresh seed)

    # --- logging / saving ---
    run_name: str = ""
    save_frequency: int = 100_000  # 0 = save best + final only
    runs_dir: str = "runs"


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Two-stream DQN on procedurally generated ModularMazeEnv"
    )

    # procgen
    p.add_argument("--n-rooms", type=int, default=cfg.n_rooms)
    p.add_argument("--room-h", type=int, default=cfg.room_h)
    p.add_argument("--room-w", type=int, default=cfg.room_w)
    p.add_argument("--map-cell-size", type=int, default=cfg.map_cell_size)
    p.add_argument("--distractor", action="store_true", default=cfg.distractor)
    p.add_argument("--local-size", type=int, nargs=2, default=list(cfg.local_size), metavar=("H", "W"))
    p.add_argument("--n-stack", type=int, default=cfg.n_stack)

    # training
    p.add_argument("--total-timesteps", type=int, default=cfg.total_timesteps)
    p.add_argument("--seed", type=int, default=cfg.seed)

    # DQN
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--buffer-size", type=int, default=cfg.buffer_size)
    p.add_argument("--gamma", type=float, default=cfg.gamma)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--learning-starts", type=int, default=cfg.learning_starts)
    p.add_argument("--train-frequency", type=int, default=cfg.train_frequency)
    p.add_argument("--target-update-frequency", type=int, default=cfg.target_update_frequency)

    # exploration
    p.add_argument("--start-epsilon", type=float, default=cfg.start_epsilon)
    p.add_argument("--end-epsilon", type=float, default=cfg.end_epsilon)
    p.add_argument("--exploration-fraction", type=float, default=cfg.exploration_fraction)

    # network
    p.add_argument("--local-embed-dim", type=int, default=cfg.local_embed_dim)
    p.add_argument("--map-embed-dim", type=int, default=cfg.map_embed_dim)
    p.add_argument(
        "--hidden-sizes", type=int, nargs="+",
        default=list(cfg.hidden_sizes), metavar="N",
    )

    # environment
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--step-reward", type=float, default=cfg.step_reward)
    p.add_argument("--collision-reward", type=float, default=cfg.collision_reward)
    p.add_argument(
        "--no-terminate-on-clear", dest="terminate_on_all_safes_opened",
        action="store_false", default=cfg.terminate_on_all_safes_opened,
    )

    # evaluation
    p.add_argument("--eval-frequency", type=int, default=cfg.eval_frequency)
    p.add_argument("--eval-episodes", type=int, default=cfg.eval_episodes)

    # logging
    p.add_argument("--run-name", default=cfg.run_name)
    p.add_argument("--save-frequency", type=int, default=cfg.save_frequency)
    p.add_argument("--runs-dir", default=cfg.runs_dir)

    args = p.parse_args()
    cfg.n_rooms = args.n_rooms
    cfg.room_h = args.room_h
    cfg.room_w = args.room_w
    cfg.map_cell_size = args.map_cell_size
    cfg.distractor = args.distractor
    cfg.local_size = tuple(args.local_size)
    cfg.n_stack = args.n_stack
    cfg.total_timesteps = args.total_timesteps
    cfg.seed = args.seed
    cfg.lr = args.lr
    cfg.buffer_size = args.buffer_size
    cfg.gamma = args.gamma
    cfg.batch_size = args.batch_size
    cfg.learning_starts = args.learning_starts
    cfg.train_frequency = args.train_frequency
    cfg.target_update_frequency = args.target_update_frequency
    cfg.start_epsilon = args.start_epsilon
    cfg.end_epsilon = args.end_epsilon
    cfg.exploration_fraction = args.exploration_fraction
    cfg.local_embed_dim = args.local_embed_dim
    cfg.map_embed_dim = args.map_embed_dim
    cfg.hidden_sizes = tuple(args.hidden_sizes)
    cfg.max_steps = args.max_steps
    cfg.step_reward = args.step_reward
    cfg.collision_reward = args.collision_reward
    cfg.terminate_on_all_safes_opened = args.terminate_on_all_safes_opened
    cfg.eval_frequency = args.eval_frequency
    cfg.eval_episodes = args.eval_episodes
    cfg.run_name = args.run_name
    cfg.save_frequency = args.save_frequency
    cfg.runs_dir = args.runs_dir
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_kwargs(cfg: Config) -> dict:
    """Shared kwargs forwarded to make_modular_env."""
    return dict(
        n_rooms=cfg.n_rooms,
        room_h=cfg.room_h,
        room_w=cfg.room_w,
        map_cell_size=cfg.map_cell_size,
        distractor=cfg.distractor,
        local_size=cfg.local_size,
        n_stack=cfg.n_stack,
        max_steps=cfg.max_steps,
        step_reward=cfg.step_reward,
        collision_reward=cfg.collision_reward,
        terminate_on_all_safes_opened=cfg.terminate_on_all_safes_opened,
    )


def _forward(network: TwoStreamQNetwork, obs: dict, device: torch.device) -> torch.Tensor:
    """Forward pass from a *single* dict observation (adds batch dim)."""
    local_obs = torch.as_tensor(obs["obs"], device=device).unsqueeze(0)
    map_obs = torch.as_tensor(obs["map_image"], device=device).unsqueeze(0)
    return network(local_obs, map_obs)


def _forward_batch(
    network: TwoStreamQNetwork,
    obs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Forward pass from a *batched* dict of tensors (already on device)."""
    return network(obs["obs"], obs["map_image"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    cfg: Config,
    eval_env,
    q_network: TwoStreamQNetwork,
    device: torch.device,
) -> dict[str, float]:
    """Run greedy evaluation for ``cfg.eval_episodes`` episodes on *eval_env*.

    The same fixed layout is used for every episode (matching training).
    Each episode is reset with a distinct seed so the agent's start position
    varies, giving a less noisy return estimate.

    Returns:
        Dict with ``mean_return``, ``std_return``, ``mean_length``,
        ``mean_safes_opened``.
    """
    q_network.eval()
    returns, lengths, safes = [], [], []

    with torch.no_grad():
        for ep in range(cfg.eval_episodes):
            obs, _ = eval_env.reset(seed=cfg.seed + 50_000 + ep)
            ep_return, ep_length = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = _forward(q_network, obs, device).argmax(dim=1).item()
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_return += reward
                ep_length += 1
            returns.append(ep_return)
            lengths.append(ep_length)
            safes.append(info.get("safes_opened", 0))

    q_network.train()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_safes_opened": float(np.mean(safes)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    # ------------------------------------------------------------------
    # Seeding — must happen before *anything* random
    # ------------------------------------------------------------------
    set_seeds(cfg.seed)

    # ------------------------------------------------------------------
    # Run name & directories
    # ------------------------------------------------------------------
    if not cfg.run_name:
        cfg.run_name = (
            f"dqn_modular_r{cfg.n_rooms}_seed{cfg.seed}_{int(time.time())}"
        )
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(cfg, run_dir / "config.txt")

    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"Run: {cfg.run_name}")
    print(f"Log dir: {run_dir}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Environment — generated once from cfg.seed, reused every episode
    # ------------------------------------------------------------------
    env = make_modular_env(layout_seed=cfg.seed, **_env_kwargs(cfg))
    eval_env = make_modular_env(layout_seed=cfg.seed, **_env_kwargs(cfg))
    obs_space = env.observation_space
    n_actions = env.action_space.n
    local_obs_shape: tuple = obs_space["obs"].shape       # (H, W, 3)
    map_obs_shape: tuple = obs_space["map_image"].shape   # (H, W, 3)

    print(
        f"n_actions={n_actions}  "
        f"local_obs={local_obs_shape}  "
        f"map_obs={map_obs_shape}"
    )

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    q_network = TwoStreamQNetwork(
        local_obs_shape=local_obs_shape,
        map_obs_shape=map_obs_shape,
        n_actions=n_actions,
        local_embed_dim=cfg.local_embed_dim,
        map_embed_dim=cfg.map_embed_dim,
        hidden_sizes=cfg.hidden_sizes,
    ).to(device)

    target_network = TwoStreamQNetwork(
        local_obs_shape=local_obs_shape,
        map_obs_shape=map_obs_shape,
        n_actions=n_actions,
        local_embed_dim=cfg.local_embed_dim,
        map_embed_dim=cfg.map_embed_dim,
        hidden_sizes=cfg.hidden_sizes,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=cfg.lr)

    n_params = sum(p.numel() for p in q_network.parameters())
    print(f"Network parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    # Derive the single-frame space (H, W, 1) by dividing channel count by
    # n_stack — FramedDictReplayBuffer stores one frame per transition instead
    # of full stacks for both obs and next_obs (8× smaller for n_stack=4).
    single_frame_space = gymnasium.spaces.Dict({
        k: gymnasium.spaces.Box(
            low=0, high=255,
            shape=(*space.shape[:2], space.shape[2] // cfg.n_stack),
            dtype=space.dtype,
        )
        for k, space in obs_space.spaces.items()
    })
    replay_buffer = FramedDictReplayBuffer(
        single_frame_space=single_frame_space,
        buffer_size=cfg.buffer_size,
        n_stack=cfg.n_stack,
        device=device,
        seed=cfg.seed + 1,
    )
    _print_memory_estimate(cfg, single_frame_space)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    ep_count = 0
    best_eval_return = -float("inf")
    start_time = time.time()

    print(f"\nStarting training for {cfg.total_timesteps:,} steps...\n")

    while global_step < cfg.total_timesteps:
        # ---- Episode initialisation ----------------------------------
        # Same layout every episode; vary the reset seed so the agent's
        # starting position (and any other stochastic init) differs.
        obs, _ = env.reset(seed=cfg.seed + ep_count)
        replay_buffer.on_reset()
        ep_count += 1
        ep_return, ep_length = 0.0, 0
        terminated = truncated = False

        # ---- Collect one episode -------------------------------------
        while not (terminated or truncated) and global_step < cfg.total_timesteps:
            global_step += 1

            # Epsilon-greedy action selection
            epsilon = linear_schedule(
                cfg.start_epsilon,
                cfg.end_epsilon,
                cfg.exploration_fraction,
                global_step,
                cfg.total_timesteps,
            )
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = _forward(q_network, obs, device).argmax(dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1

            # Store transition (note: use terminated, not done, for the
            # bootstrap mask — truncation does NOT mean the episode ended
            # in a terminal state).
            replay_buffer.add(obs, action, reward, terminated, truncated)
            obs = next_obs

            # ---- Gradient update -------------------------------------
            if (
                global_step >= cfg.learning_starts
                and global_step % cfg.train_frequency == 0
                and len(replay_buffer) >= cfg.batch_size
            ):
                batch = replay_buffer.sample(cfg.batch_size)

                with torch.no_grad():
                    # Double DQN: online net selects action, target net values it.
                    next_actions = _forward_batch(q_network, batch.next_obs).argmax(dim=1, keepdim=True)
                    target_q = _forward_batch(target_network, batch.next_obs).gather(1, next_actions).squeeze(1)
                    td_target = batch.rewards + cfg.gamma * target_q * (1.0 - batch.dones)

                current_q = _forward_batch(q_network, batch.obs).gather(
                    1, batch.actions.unsqueeze(1)
                ).squeeze(1)
                loss = F.huber_loss(current_q, td_target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
                optimizer.step()

                writer.add_scalar("train/td_loss", loss.item(), global_step)
                writer.add_scalar("train/q_values_mean", current_q.mean().item(), global_step)

            # ---- Hard target network update --------------------------
            if global_step % cfg.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            # ---- Evaluation ------------------------------------------
            if global_step % cfg.eval_frequency == 0:
                eval_stats = evaluate(cfg, eval_env, q_network, device)
                for k, v in eval_stats.items():
                    writer.add_scalar(f"eval/{k}", v, global_step)
                print(
                    f"  [EVAL] step={global_step:>9,}  "
                    f"return={eval_stats['mean_return']:+.3f}±{eval_stats['std_return']:.2f}  "
                    f"safes={eval_stats['mean_safes_opened']:.2f}  "
                    f"len={eval_stats['mean_length']:.0f}"
                )
                if eval_stats["mean_return"] > best_eval_return:
                    best_eval_return = eval_stats["mean_return"]
                    _save_checkpoint(q_network, optimizer, global_step, cfg, run_dir / "best.pt")
                    print(f"  [BEST] {best_eval_return:+.3f}")

            # ---- Periodic checkpoint ---------------------------------
            if cfg.save_frequency > 0 and global_step % cfg.save_frequency == 0:
                _save_checkpoint(
                    q_network, optimizer, global_step, cfg,
                    run_dir / f"checkpoint_{global_step}.pt",
                )

        # ---- Episode bookkeeping -------------------------------------
        writer.add_scalar("train/episode_return", ep_return, global_step)
        writer.add_scalar("train/episode_length", ep_length, global_step)
        writer.add_scalar("train/epsilon", epsilon, global_step)
        writer.add_scalar("train/safes_opened", info.get("safes_opened", 0), global_step)
        writer.add_scalar("train/buffer_size", len(replay_buffer), global_step)

        if ep_count % 20 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(
                f"step={global_step:>9,}  ep={ep_count:>5}  "
                f"return={ep_return:+.3f}  len={ep_length:>4}  "
                f"safes={info.get('safes_opened', 0)}/{info.get('safes_total', '?')}  "
                f"eps={epsilon:.3f}  buf={len(replay_buffer):>6,}  sps={sps}"
            )

    # ------------------------------------------------------------------
    # Final save & cleanup
    # ------------------------------------------------------------------
    _save_checkpoint(q_network, optimizer, cfg.total_timesteps, cfg, run_dir / "final.pt")
    writer.close()
    env.close()
    eval_env.close()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s ({cfg.total_timesteps / elapsed:.0f} sps)")
    print(f"Best eval return: {best_eval_return:+.3f}")
    print(f"Artifacts: {run_dir}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_checkpoint(
    q_network: TwoStreamQNetwork,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "q_network_state_dict": q_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )


def _save_config(cfg: Config, path: Path) -> None:
    with open(path, "w") as f:
        for k, v in cfg.__dict__.items():
            f.write(f"{k} = {v!r}\n")


def _print_memory_estimate(cfg: Config, single_frame_space) -> None:
    """Print replay buffer memory estimate (FramedDictReplayBuffer stores 1 frame/transition)."""
    bytes_per_transition = sum(
        np.prod(s.shape) * np.dtype(s.dtype).itemsize
        for s in single_frame_space.spaces.values()
    )  # 1 frame per stream, no obs+next_obs duplication
    total_gb = cfg.buffer_size * bytes_per_transition / 1e9
    print(
        f"Replay buffer memory estimate: "
        f"{cfg.buffer_size:,} × {bytes_per_transition / 1024:.1f} KB "
        f"= {total_gb:.2f} GB  (FramedDictReplayBuffer, {cfg.n_stack}× smaller than naive)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

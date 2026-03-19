"""DQN training loop for gridworld_env tasks.

Usage
-----
    python train_dqn.py --layout layouts/simple_nav.txt --seed 42

    # Override any hyperparameter via CLI:
    python train_dqn.py --layout layouts/key_door.txt \\
        --total-timesteps 1000000 --lr 3e-4 --seed 0

    # Watch with TensorBoard:
    tensorboard --logdir runs/

Reproducibility
---------------
All sources of randomness are seeded through a single ``--seed`` argument:
  - Python ``random``, ``numpy``, ``torch`` (CPU + CUDA), cuDNN flags
  - Environment RNG via ``env.reset(seed=...)``
  - Replay buffer sampling RNG (derived seed: seed + 1)
  - Evaluation environment (derived seed: seed + 2)
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

from fuzzy_actions import QNetwork, ReplayBuffer, linear_schedule, make_env, set_seeds


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- environment ---
    layout: str = "layouts/simple_nav.txt"
    max_steps: int = 200          # Max steps per episode (applied to env)
    step_reward: float = -0.01    # Per-step time-pressure reward
    collision_reward: float = -0.1

    # --- training ---
    total_timesteps: int = 500_000
    seed: int = 42

    # --- DQN hyperparameters ---
    lr: float = 1e-4              # Adam learning rate
    buffer_size: int = 100_000    # Replay buffer capacity
    gamma: float = 0.99           # Discount factor
    batch_size: int = 64          # Minibatch size
    learning_starts: int = 10_000 # Steps before first gradient update
    train_frequency: int = 4      # Update every N environment steps
    target_update_frequency: int = 1_000  # Hard-copy Q -> Q_target every N steps

    # --- epsilon-greedy exploration ---
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.15  # Fraction of training over which eps decays

    # --- network ---
    hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (128, 128))

    # --- evaluation ---
    eval_frequency: int = 10_000  # Evaluate every N steps
    eval_episodes: int = 20       # Number of episodes per evaluation run

    # --- logging / saving ---
    run_name: str = ""            # Auto-generated if empty
    save_frequency: int = 50_000  # Save checkpoint every N steps (0 = disabled)
    runs_dir: str = "runs"


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="DQN training on gridworld_env tasks")

    # environment
    p.add_argument("--layout", default=cfg.layout)
    p.add_argument("--max-steps", type=int, default=cfg.max_steps)
    p.add_argument("--step-reward", type=float, default=cfg.step_reward)
    p.add_argument("--collision-reward", type=float, default=cfg.collision_reward)

    # training
    p.add_argument("--total-timesteps", type=int, default=cfg.total_timesteps)
    p.add_argument("--seed", type=int, default=cfg.seed)

    # DQN hyperparameters
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
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=list(cfg.hidden_sizes),
        metavar="N",
    )

    # evaluation
    p.add_argument("--eval-frequency", type=int, default=cfg.eval_frequency)
    p.add_argument("--eval-episodes", type=int, default=cfg.eval_episodes)

    # logging
    p.add_argument("--run-name", default=cfg.run_name)
    p.add_argument("--save-frequency", type=int, default=cfg.save_frequency)
    p.add_argument("--runs-dir", default=cfg.runs_dir)

    args = p.parse_args()

    cfg.layout = args.layout
    cfg.max_steps = args.max_steps
    cfg.step_reward = args.step_reward
    cfg.collision_reward = args.collision_reward
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
    cfg.hidden_sizes = tuple(args.hidden_sizes)
    cfg.eval_frequency = args.eval_frequency
    cfg.eval_episodes = args.eval_episodes
    cfg.run_name = args.run_name
    cfg.save_frequency = args.save_frequency
    cfg.runs_dir = args.runs_dir

    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    env,
    q_network: QNetwork,
    n_episodes: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Run *n_episodes* greedy episodes and return summary statistics.

    The evaluation environment is reset with a seed derived from *seed* so that
    evaluation episodes are consistent across checkpoints.

    Args:
        env: A pre-created evaluation ``GridWorldEnv``.
        q_network: The current Q-network (used greedily, no exploration).
        n_episodes: Number of episodes to evaluate.
        device: Device for inference.
        seed: Base seed; episode i uses ``seed + i`` to vary initial states
              while remaining reproducible.

    Returns:
        Dict with keys ``mean_return``, ``std_return``, ``mean_length``.
    """
    q_network.eval()
    returns = []
    lengths = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = q_network(obs_t).argmax(dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                ep_length += 1

            returns.append(ep_return)
            lengths.append(ep_length)

    q_network.train()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
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
    # Run name & logging
    # ------------------------------------------------------------------
    layout_stem = Path(cfg.layout).stem
    if not cfg.run_name:
        cfg.run_name = (
            f"dqn_{layout_stem}_seed{cfg.seed}_{int(time.time())}"
        )

    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"Run: {cfg.run_name}")
    print(f"Log dir: {run_dir}")

    # Save config alongside the run
    _save_config(cfg, run_dir / "config.txt")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    env_kwargs = dict(
        max_steps=cfg.max_steps,
        step_reward=cfg.step_reward,
        collision_reward=cfg.collision_reward,
        obs_mode="symbolic",
        flatten_obs=True,
    )

    # Training environment — seeded once; individual episodes vary via the
    # episode counter (see reset call inside the loop).
    env = make_env(cfg.layout, seed=cfg.seed, **env_kwargs)

    # Evaluation environment — uses a separate derived seed so eval episodes
    # are independent from training.
    eval_env = make_env(cfg.layout, seed=cfg.seed + 2, **env_kwargs)

    obs_dim: int = env.observation_space.shape[0]
    n_actions: int = env.action_space.n
    print(f"obs_dim={obs_dim}  n_actions={n_actions}")

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    q_network = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(device)
    target_network = QNetwork(obs_dim, n_actions, cfg.hidden_sizes).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    # Derived seed (seed+1) keeps buffer sampling independent from env/torch.
    replay_buffer = ReplayBuffer(
        obs_shape=(obs_dim,),
        buffer_size=cfg.buffer_size,
        device=device,
        seed=cfg.seed + 1,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    obs, _ = env.reset(seed=cfg.seed)  # seed training env explicitly
    ep_return = 0.0
    ep_length = 0
    ep_count = 0
    start_time = time.time()
    best_eval_return = -float("inf")

    print(f"\nStarting training for {cfg.total_timesteps:,} steps...\n")

    for global_step in range(1, cfg.total_timesteps + 1):

        # --- Epsilon-greedy action selection ---
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
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = q_network(obs_t).argmax(dim=1).item()

        # --- Environment step ---
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_return += reward
        ep_length += 1

        # Store transition.
        # When the episode ends due to truncation (time limit), the next_obs
        # from the env is the *true* next observation (not a reset obs), so
        # we can store it directly.
        replay_buffer.add(obs, action, reward, next_obs, terminated)

        obs = next_obs

        # --- Episode end ---
        if done:
            ep_count += 1
            writer.add_scalar("train/episode_return", ep_return, global_step)
            writer.add_scalar("train/episode_length", ep_length, global_step)
            writer.add_scalar("train/epsilon", epsilon, global_step)

            if ep_count % 50 == 0:
                sps = int(global_step / (time.time() - start_time))
                print(
                    f"step={global_step:>8,}  ep={ep_count:>5}  "
                    f"return={ep_return:+.3f}  len={ep_length:>4}  "
                    f"eps={epsilon:.3f}  buf={len(replay_buffer):>7,}  sps={sps}"
                )

            ep_return = 0.0
            ep_length = 0
            # Re-seed each episode from its index for full reproducibility of
            # the training trajectory.
            obs, _ = env.reset(seed=cfg.seed + 1000 + ep_count)

        # --- Gradient update ---
        if global_step >= cfg.learning_starts and global_step % cfg.train_frequency == 0:
            batch = replay_buffer.sample(cfg.batch_size)

            with torch.no_grad():
                # Double-DQN: use online net to select action, target net to evaluate
                next_actions = q_network(batch.next_obs).argmax(dim=1, keepdim=True)
                target_q = target_network(batch.next_obs).gather(1, next_actions).squeeze(1)
                td_target = batch.rewards + cfg.gamma * target_q * (1.0 - batch.dones)

            current_q = q_network(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            loss = F.huber_loss(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability (common in DQN)
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()

            writer.add_scalar("train/td_loss", loss.item(), global_step)
            writer.add_scalar("train/q_values_mean", current_q.mean().item(), global_step)

        # --- Hard target network update ---
        if global_step % cfg.target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        # --- Evaluation ---
        if global_step % cfg.eval_frequency == 0:
            eval_stats = evaluate(
                eval_env,
                q_network,
                cfg.eval_episodes,
                device,
                seed=cfg.seed + 2,  # reproducible eval
            )
            writer.add_scalar("eval/mean_return", eval_stats["mean_return"], global_step)
            writer.add_scalar("eval/std_return", eval_stats["std_return"], global_step)
            writer.add_scalar("eval/mean_length", eval_stats["mean_length"], global_step)

            print(
                f"  [EVAL] step={global_step:>8,}  "
                f"mean_return={eval_stats['mean_return']:+.3f}  "
                f"std={eval_stats['std_return']:.3f}  "
                f"mean_len={eval_stats['mean_length']:.1f}"
            )

            # Save best model
            if eval_stats["mean_return"] > best_eval_return:
                best_eval_return = eval_stats["mean_return"]
                _save_checkpoint(q_network, optimizer, global_step, cfg, run_dir / "best.pt")
                print(f"  [BEST] New best eval return: {best_eval_return:+.3f}")

        # --- Periodic checkpoint ---
        if cfg.save_frequency > 0 and global_step % cfg.save_frequency == 0:
            _save_checkpoint(
                q_network, optimizer, global_step, cfg,
                run_dir / f"checkpoint_{global_step}.pt",
            )

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    _save_checkpoint(q_network, optimizer, cfg.total_timesteps, cfg, run_dir / "final.pt")
    writer.close()
    env.close()
    eval_env.close()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s  ({cfg.total_timesteps / total_time:.0f} sps)")
    print(f"Best eval return: {best_eval_return:+.3f}")
    print(f"Artifacts saved to: {run_dir}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    q_network: QNetwork,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

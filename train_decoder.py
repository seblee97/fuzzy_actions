"""Training of the sequence decoder (component 3).

Loads a frozen encoder + inverse model from a ``train_hierarchical.py``
checkpoint and trains a GRU-based SequenceDecoder to reconstruct the
primitive action sequence from the latent action z.

Architecture
------------
Encoder and InverseModel are frozen.  Only SequenceDecoder is trained::

    s1, s2 ──► encoder (frozen) ──► enc_s1, enc_s2
                                         │
                              InverseModel (frozen)
                                         │
                                         z
                                         │
                              SequenceDecoder (trained)
                                         │
                                  action logits (B, T, n_actions)

Loss
----
Cross-entropy between predicted logits and ground-truth action sequence,
using teacher forcing (BOS-shifted ground-truth as decoder input).

Batch format
------------
The DataLoader must yield dicts with keys::

    s1      — start states  (B, *)
    s2      — end states    (B, *)
    actions — action sequence (B, T) int64

Dataset
-------
Same dynamic import mechanism as ``train_hierarchical.py``.

Usage
-----
    python train_decoder.py \\
        --checkpoint runs/hierarchical_infonce_latent_seed42/final.pt \\
        --data-path datasets/oracle/directional \\
        --n-actions 8 --seq-len 64

    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hierarchical import InverseModel, SequenceDecoder, StateEncoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- checkpoint from train_hierarchical.py ---
    checkpoint: str = ""

    # --- dataset ---
    data_path: str = ""
    dataset_class: str = "maze_dataset.MazeOracleDataset"
    dataset_kwargs: str = "{}"
    num_workers: int = 4

    # --- decoder ---
    n_actions: int = 8
    seq_len: int = 64
    hidden_dim: int = 256
    n_layers: int = 2
    use_enc_s1_context: bool = True     # feed enc(s1) as decoder context

    # --- optimiser ---
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0

    # --- training ---
    epochs: int = 100
    batch_size: int = 256
    seed: int = 42

    # --- logging / saving ---
    run_name: str = ""
    runs_dir: str = "runs"
    save_frequency: int = 10
    log_frequency: int = 50


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train SequenceDecoder with frozen encoder+inverse")

    p.add_argument("--checkpoint", required=True, help="Path to train_hierarchical.py checkpoint")
    p.add_argument("--data-path", default=cfg.data_path)
    p.add_argument("--dataset-class", default=cfg.dataset_class)
    p.add_argument("--dataset-kwargs", default=cfg.dataset_kwargs)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)

    p.add_argument("--n-actions", type=int, default=cfg.n_actions)
    p.add_argument("--seq-len", type=int, default=cfg.seq_len)
    p.add_argument("--hidden-dim", type=int, default=cfg.hidden_dim)
    p.add_argument("--n-layers", type=int, default=cfg.n_layers)
    p.add_argument("--no-enc-s1-context", dest="use_enc_s1_context",
                   action="store_false", default=cfg.use_enc_s1_context)

    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)

    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--seed", type=int, default=cfg.seed)

    p.add_argument("--run-name", default=cfg.run_name)
    p.add_argument("--runs-dir", default=cfg.runs_dir)
    p.add_argument("--save-frequency", type=int, default=cfg.save_frequency)
    p.add_argument("--log-frequency", type=int, default=cfg.log_frequency)

    args = p.parse_args()
    cfg.checkpoint = args.checkpoint
    cfg.data_path = args.data_path
    cfg.dataset_class = args.dataset_class
    cfg.dataset_kwargs = args.dataset_kwargs
    cfg.num_workers = args.num_workers
    cfg.n_actions = args.n_actions
    cfg.seq_len = args.seq_len
    cfg.hidden_dim = args.hidden_dim
    cfg.n_layers = args.n_layers
    cfg.use_enc_s1_context = args.use_enc_s1_context
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.run_name = args.run_name
    cfg.runs_dir = args.runs_dir
    cfg.save_frequency = args.save_frequency
    cfg.log_frequency = args.log_frequency
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_frozen_models(
    checkpoint_path: str, device: torch.device
) -> tuple[StateEncoder, InverseModel, dict]:
    """Load and freeze encoder + inverse from a train_hierarchical checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    enc_cfg = ckpt["config"]

    # Re-instantiate encoder from saved config
    encoder_kwargs: dict = {
        "mode": enc_cfg["encoder_mode"],
        "embed_dim": enc_cfg["embed_dim"],
    }
    if enc_cfg["encoder_mode"] == "latent":
        encoder_kwargs["state_dim"] = enc_cfg["state_dim"]
        encoder_kwargs["hidden_sizes"] = list(enc_cfg["encoder_hidden_sizes"])
    elif enc_cfg["encoder_mode"] == "pixel":
        encoder_kwargs["in_channels"] = enc_cfg["pixel_channels"]
    else:
        encoder_kwargs["input_dim"] = enc_cfg["input_dim"]

    encoder = StateEncoder(**encoder_kwargs)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Re-instantiate inverse model from saved config
    inverse = InverseModel(
        embed_dim=enc_cfg["embed_dim"],
        z_dim=enc_cfg["z_dim"],
        hidden_sizes=list(enc_cfg["inverse_hidden_sizes"]),
        proj_hidden_dim=enc_cfg["proj_hidden_dim"],
        proj_dim=enc_cfg["proj_dim"],
    )
    inverse.load_state_dict(ckpt["inverse_state_dict"])
    inverse.to(device)
    inverse.eval()
    for p in inverse.parameters():
        p.requires_grad_(False)

    return encoder, inverse, enc_cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(cfg: Config):
    module_path, class_name = cfg.dataset_class.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    DatasetCls = getattr(mod, class_name)
    extra_kwargs = json.loads(cfg.dataset_kwargs)
    return DatasetCls(root=cfg.data_path, **extra_kwargs)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    decoder: SequenceDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    enc_cfg: dict,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "encoder_config": enc_cfg,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)

    if not cfg.run_name:
        cfg.run_name = (
            f"decoder_seed{cfg.seed}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        )
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.txt", "w") as f:
        for k, v in cfg.__dict__.items():
            f.write(f"{k} = {v!r}\n")

    writer = SummaryWriter(log_dir=str(run_dir))
    metrics_path = run_dir / "metrics.jsonl"

    def _log(event: str, step: int, data: dict) -> None:
        for k, v in data.items():
            writer.add_scalar(f"{event}/{k}", v, step)
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"event": event, "step": step, **data}) + "\n")

    print(f"Run: {cfg.run_name}")
    print(f"Log dir: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Frozen models
    encoder, inverse, enc_cfg = load_frozen_models(cfg.checkpoint, device)
    z_dim: int = enc_cfg["z_dim"]
    embed_dim: int = enc_cfg["embed_dim"]
    print(f"Loaded checkpoint: {cfg.checkpoint}")
    print(f"  encoder_mode={enc_cfg['encoder_mode']}  embed_dim={embed_dim}  z_dim={z_dim}")

    # Dataset
    dataset = load_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} samples  |  {len(loader):,} batches/epoch")

    # Decoder
    decoder = SequenceDecoder(
        z_dim=z_dim,
        n_actions=cfg.n_actions,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        embed_dim=embed_dim if cfg.use_enc_s1_context else None,
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Training loop
    global_step = 0
    batch_keys = {"s1", "s2", "actions"}
    keys_validated = False

    print(f"\nStarting decoder training for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        decoder.train()
        epoch_loss = epoch_acc = 0.0
        t0 = time.time()

        for batch in loader:
            global_step += 1

            if not keys_validated:
                missing = batch_keys - set(batch.keys())
                if missing:
                    raise KeyError(
                        f"Batch is missing required keys: {missing}.  "
                        f"Got: {set(batch.keys())}.  "
                        "Ensure your dataset/collate_fn produces "
                        "{'s1', 's2', 'actions'} dicts."
                    )
                keys_validated = True

            s1 = batch["s1"].to(device, non_blocking=True)
            s2 = batch["s2"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)  # (B, T)

            # Frozen forward pass
            with torch.no_grad():
                enc_s1 = encoder(s1)
                enc_s2 = encoder(s2)
                z, _ = inverse(enc_s1, enc_s2)

            # Decoder: teacher forcing, predicts each action in the sequence
            ctx = enc_s1 if cfg.use_enc_s1_context else None
            logits = decoder(z, actions, ctx)          # (B, T, n_actions)
            B, T, A = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, A), actions.reshape(B * T)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == actions).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc

            if global_step % cfg.log_frequency == 0:
                _log("train", global_step, {"loss": loss.item(), "accuracy": acc})

        n = len(loader)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch + 1:>4}/{cfg.epochs}  "
            f"loss={epoch_loss / n:.4f}  "
            f"acc={epoch_acc / n:.4f}  "
            f"({elapsed:.1f}s)"
        )
        _log("epoch", epoch + 1, {
            "loss": epoch_loss / n,
            "accuracy": epoch_acc / n,
        })

        if cfg.save_frequency > 0 and (epoch + 1) % cfg.save_frequency == 0:
            save_checkpoint(
                decoder, optimizer, epoch + 1, cfg, enc_cfg,
                run_dir / f"checkpoint_epoch{epoch + 1:04d}.pt",
            )

    save_checkpoint(
        decoder, optimizer, cfg.epochs, cfg, enc_cfg, run_dir / "final.pt"
    )
    writer.close()
    print(f"\nTraining complete.  Artifacts: {run_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

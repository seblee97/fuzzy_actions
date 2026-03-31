"""Joint training of the inverse model and forward model (components 1 + 2).

Architecture
------------
Three jointly-trained modules:

  StateEncoder    — projects states to a shared embedding space.
  InverseModel    — estimates latent action z from (enc_s1, enc_s2).
  ForwardModel    — predicts enc_s2 from enc_s1 and z.

An optional Predictor MLP sits on the online branch for SimSiam / BYOL.

Loss
----
total_loss = contrastive_weight · L_contrastive + forward_weight · L_forward

Contrastive options (--loss-type):
  infonce  — NT-Xent with in-batch negatives (default)
  simsiam  — symmetric stop-gradient cosine similarity
  byol     — EMA target + predictor regression

Batch format
------------
The DataLoader must yield dicts with keys::

    s1_a, s2_a  — anchor transition states      (B, *)
    s1_b, s2_b  — positive-view transition states (B, *)

Both views should represent transitions of the same semantic type (e.g.
same room-to-room jump).  Pair construction is the caller's responsibility.
This script validates the keys on the first batch and raises a descriptive
error if they are missing.

Dataset
-------
The dataset class is specified at the command line via --dataset-class
(dotted import path, e.g. ``maze_dataset.MazeOracleDataset``) and loaded
dynamically.  Extra constructor kwargs can be passed as a JSON string with
--dataset-kwargs.

Usage
-----
    python train_hierarchical.py \\
        --data-path datasets/oracle/directional \\
        --encoder-mode latent --state-dim 64 \\
        --loss-type infonce \\
        --epochs 100

    python train_hierarchical.py \\
        --data-path datasets/oracle/directional \\
        --encoder-mode pixel --pixel-channels 3 \\
        --loss-type byol --ema-decay 0.996

    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hierarchical import (
    BYOLLoss,
    EMAUpdater,
    ForwardLoss,
    ForwardModel,
    InfoNCELoss,
    InverseModel,
    Predictor,
    SimSiamLoss,
    StateEncoder,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- dataset ---
    data_path: str = ""
    dataset_class: str = "maze_dataset.MazeOracleDataset"
    dataset_kwargs: str = "{}"          # JSON string of extra constructor kwargs
    num_workers: int = 4

    # --- encoder ---
    encoder_mode: str = "latent"        # "latent" | "pixel" | "embedding"
    state_dim: int = 64                 # required for encoder_mode="latent"
    encoder_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (256,))
    pixel_channels: int = 3             # required for encoder_mode="pixel"
    input_dim: int = 64                 # required for encoder_mode="embedding"
    embed_dim: int = 256

    # --- inverse model ---
    z_dim: int = 128
    proj_dim: int = 128
    inverse_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (512, 256))
    proj_hidden_dim: int = 256

    # --- forward model ---
    forward_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (512, 256))
    forward_loss_mode: str = "mse"      # "mse" | "cosine"
    forward_weight: float = 1.0

    # --- contrastive loss ---
    loss_type: str = "infonce"          # "infonce" | "simsiam" | "byol"
    contrastive_weight: float = 1.0
    temperature: float = 0.07           # InfoNCE only
    ema_decay: float = 0.996            # BYOL only
    predictor_hidden_dim: int = 512     # SimSiam / BYOL predictor

    # --- optimiser ---
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0

    # --- training ---
    epochs: int = 100
    batch_size: int = 256
    seed: int = 42

    # --- data fraction ---
    data_fraction: float = 1.0          # use this fraction of the dataset (0 < f ≤ 1)

    # --- logging / saving ---
    run_name: str = ""
    runs_dir: str = "runs"
    save_frequency: int = 10            # save checkpoint every N epochs (0 = end only)
    log_frequency: int = 50             # log every N batches


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Joint training of inverse model + forward model"
    )

    # dataset
    p.add_argument("--data-path", default=cfg.data_path)
    p.add_argument("--dataset-class", default=cfg.dataset_class)
    p.add_argument("--dataset-kwargs", default=cfg.dataset_kwargs)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)

    # encoder
    p.add_argument("--encoder-mode", default=cfg.encoder_mode,
                   choices=["latent", "pixel", "embedding"])
    p.add_argument("--state-dim", type=int, default=cfg.state_dim)
    p.add_argument("--encoder-hidden-sizes", type=int, nargs="+",
                   default=list(cfg.encoder_hidden_sizes), metavar="N")
    p.add_argument("--pixel-channels", type=int, default=cfg.pixel_channels)
    p.add_argument("--input-dim", type=int, default=cfg.input_dim)
    p.add_argument("--embed-dim", type=int, default=cfg.embed_dim)

    # inverse
    p.add_argument("--z-dim", type=int, default=cfg.z_dim)
    p.add_argument("--proj-dim", type=int, default=cfg.proj_dim)
    p.add_argument("--inverse-hidden-sizes", type=int, nargs="+",
                   default=list(cfg.inverse_hidden_sizes), metavar="N")
    p.add_argument("--proj-hidden-dim", type=int, default=cfg.proj_hidden_dim)

    # forward
    p.add_argument("--forward-hidden-sizes", type=int, nargs="+",
                   default=list(cfg.forward_hidden_sizes), metavar="N")
    p.add_argument("--forward-loss-mode", default=cfg.forward_loss_mode,
                   choices=["mse", "cosine"])
    p.add_argument("--forward-weight", type=float, default=cfg.forward_weight)

    # contrastive
    p.add_argument("--loss-type", default=cfg.loss_type,
                   choices=["infonce", "simsiam", "byol"])
    p.add_argument("--contrastive-weight", type=float, default=cfg.contrastive_weight)
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--ema-decay", type=float, default=cfg.ema_decay)
    p.add_argument("--predictor-hidden-dim", type=int, default=cfg.predictor_hidden_dim)

    # optimiser
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)

    # training
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--seed", type=int, default=cfg.seed)

    # data fraction
    p.add_argument("--data-fraction", type=float, default=cfg.data_fraction,
                   metavar="F", help="fraction of dataset to use, 0 < F ≤ 1")

    # logging
    p.add_argument("--run-name", default=cfg.run_name)
    p.add_argument("--runs-dir", default=cfg.runs_dir)
    p.add_argument("--save-frequency", type=int, default=cfg.save_frequency)
    p.add_argument("--log-frequency", type=int, default=cfg.log_frequency)

    args = p.parse_args()
    cfg.data_path = args.data_path
    cfg.dataset_class = args.dataset_class
    cfg.dataset_kwargs = args.dataset_kwargs
    cfg.num_workers = args.num_workers
    cfg.encoder_mode = args.encoder_mode
    cfg.state_dim = args.state_dim
    cfg.encoder_hidden_sizes = tuple(args.encoder_hidden_sizes)
    cfg.pixel_channels = args.pixel_channels
    cfg.input_dim = args.input_dim
    cfg.embed_dim = args.embed_dim
    cfg.z_dim = args.z_dim
    cfg.proj_dim = args.proj_dim
    cfg.inverse_hidden_sizes = tuple(args.inverse_hidden_sizes)
    cfg.proj_hidden_dim = args.proj_hidden_dim
    cfg.forward_hidden_sizes = tuple(args.forward_hidden_sizes)
    cfg.forward_loss_mode = args.forward_loss_mode
    cfg.forward_weight = args.forward_weight
    cfg.loss_type = args.loss_type
    cfg.contrastive_weight = args.contrastive_weight
    cfg.temperature = args.temperature
    cfg.ema_decay = args.ema_decay
    cfg.predictor_hidden_dim = args.predictor_hidden_dim
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.data_fraction = args.data_fraction
    cfg.run_name = args.run_name
    cfg.runs_dir = args.runs_dir
    cfg.save_frequency = args.save_frequency
    cfg.log_frequency = args.log_frequency
    return cfg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_encoder(cfg: Config) -> StateEncoder:
    kwargs: dict = {"mode": cfg.encoder_mode, "embed_dim": cfg.embed_dim}
    if cfg.encoder_mode == "latent":
        kwargs["state_dim"] = cfg.state_dim
        kwargs["hidden_sizes"] = list(cfg.encoder_hidden_sizes)
    elif cfg.encoder_mode == "pixel":
        kwargs["in_channels"] = cfg.pixel_channels
    else:
        kwargs["input_dim"] = cfg.input_dim
    return StateEncoder(**kwargs)


def build_models(cfg: Config) -> tuple[
    StateEncoder, InverseModel, ForwardModel, Predictor | None
]:
    encoder = build_encoder(cfg)
    inverse = InverseModel(
        embed_dim=cfg.embed_dim,
        z_dim=cfg.z_dim,
        hidden_sizes=list(cfg.inverse_hidden_sizes),
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_dim=cfg.proj_dim,
    )
    forward_model = ForwardModel(
        embed_dim=cfg.embed_dim,
        z_dim=cfg.z_dim,
        hidden_sizes=list(cfg.forward_hidden_sizes),
    )
    predictor = (
        Predictor(z_dim=cfg.proj_dim, hidden_dim=cfg.predictor_hidden_dim)
        if cfg.loss_type in ("simsiam", "byol")
        else None
    )
    return encoder, inverse, forward_model, predictor


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
# Sample frame saving
# ---------------------------------------------------------------------------

def save_sample_frames(
    dataset,
    run_dir: Path,
    n_pairs: int = 8,
    seed: int = 0,
) -> None:
    """Save a contact sheet of sample (s1, s2) pairs to *run_dir/sample_frames.png*.

    Only runs when the dataset produces pixel states (tensors with 3 dims).
    Silently skips for latent/flat states since there is nothing to display.

    Each row shows one pair: s1_a on the left, s2_a on the right, with a
    narrow gap between them so the transition direction is clear.
    """
    import random as _random
    from PIL import Image as _Image

    rng = _random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_pairs, len(dataset)))

    frames = []
    for idx in indices:
        item = dataset[idx]
        s1, s2 = item["s1_a"], item["s2_a"]
        if s1.ndim != 3:
            # Latent / flat state — nothing to render
            print("sample_frames: skipping (state is not a pixel tensor)")
            return
        frames.append((s1, s2))

    if not frames:
        return

    # Convert (C, H, W) float [0,1] → (H, W, C) uint8
    def to_uint8(t):
        return (t.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy())

    gap = 4  # pixel gap between s1 and s2 within a pair
    row_gap = 2  # pixel gap between rows
    sample_h, sample_w = to_uint8(frames[0][0]).shape[:2]
    sheet_w = sample_w * 2 + gap
    sheet_h = len(frames) * sample_h + (len(frames) - 1) * row_gap

    sheet = (
        255 * __import__("numpy").ones((sheet_h, sheet_w, 3), dtype="uint8")
    )
    for row, (s1, s2) in enumerate(frames):
        y = row * (sample_h + row_gap)
        sheet[y : y + sample_h, :sample_w] = to_uint8(s1)
        sheet[y : y + sample_h, sample_w + gap :] = to_uint8(s2)

    out_path = run_dir / "sample_frames.png"
    _Image.fromarray(sheet).save(out_path)
    print(f"Sample frames saved → {out_path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    encoder: StateEncoder,
    inverse: InverseModel,
    forward_model: ForwardModel,
    predictor: Predictor | None,
    target_encoder: StateEncoder | None,
    target_inverse: InverseModel | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "inverse_state_dict": inverse.state_dict(),
            "forward_state_dict": forward_model.state_dict(),
            "predictor_state_dict": predictor.state_dict() if predictor else None,
            "target_encoder_state_dict": target_encoder.state_dict() if target_encoder else None,
            "target_inverse_state_dict": target_inverse.state_dict() if target_inverse else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    # Seeding
    torch.manual_seed(cfg.seed)

    # Run directory
    if not cfg.run_name:
        cfg.run_name = (
            f"hierarchical_{cfg.loss_type}_{cfg.encoder_mode}"
            f"_seed{cfg.seed}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
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

    # Dataset
    base_dataset = load_dataset(cfg)

    if cfg.data_fraction < 1.0:
        import random as _random
        from torch.utils.data import Subset
        n_full = len(base_dataset)
        n_keep = max(1, int(n_full * cfg.data_fraction))
        indices = _random.Random(cfg.seed).sample(range(n_full), n_keep)
        dataset = Subset(base_dataset, indices)
        print(f"Using {n_keep:,} / {n_full:,} samples ({cfg.data_fraction:.1%})")
    else:
        dataset = base_dataset

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} samples  |  {len(loader):,} batches/epoch")
    save_sample_frames(base_dataset, run_dir, seed=cfg.seed)

    # Auto-detect input shape from dataset
    if cfg.encoder_mode == "latent" and hasattr(base_dataset, "state_dim") and base_dataset.state_dim is not None:
        if cfg.state_dim != base_dataset.state_dim:
            print(f"Auto-setting state_dim={base_dataset.state_dim} from dataset (was {cfg.state_dim})")
            cfg.state_dim = base_dataset.state_dim
    elif cfg.encoder_mode == "pixel" and hasattr(base_dataset, "pixel_shape") and base_dataset.pixel_shape is not None:
        C, *_ = base_dataset.pixel_shape
        if cfg.pixel_channels != C:
            print(f"Auto-setting pixel_channels={C} from dataset (was {cfg.pixel_channels})")
            cfg.pixel_channels = C

    # Models
    encoder, inverse, forward_model, predictor = build_models(cfg)
    encoder = encoder.to(device)
    inverse = inverse.to(device)
    forward_model = forward_model.to(device)
    if predictor is not None:
        predictor = predictor.to(device)

    n_params = sum(
        p.numel() for m in [encoder, inverse, forward_model]
        for p in m.parameters()
    )
    print(f"Trainable parameters: {n_params:,}")

    # EMA target (BYOL only) — must be created BEFORE optimizer so its params
    # are never included in optimizer.param_groups
    ema: EMAUpdater | None = None
    target_encoder: StateEncoder | None = None
    target_inverse: InverseModel | None = None
    if cfg.loss_type == "byol":
        ema_enc = EMAUpdater.from_online(encoder, decay=cfg.ema_decay)
        ema_inv = EMAUpdater.from_online(inverse, decay=cfg.ema_decay)
        target_encoder = ema_enc.target.to(device)
        target_inverse = ema_inv.target.to(device)
        # Wrap both updaters under a single step() call
        class _DualEMA:
            def __init__(self, a: EMAUpdater, b: EMAUpdater):
                self._a, self._b = a, b
            def step(self):
                self._a.step(); self._b.step()
        ema = _DualEMA(ema_enc, ema_inv)  # type: ignore[assignment]

    # Losses
    forward_loss_fn = ForwardLoss(mode=cfg.forward_loss_mode)

    if cfg.loss_type == "infonce":
        contrastive_loss_fn = InfoNCELoss(temperature=cfg.temperature)
    elif cfg.loss_type == "simsiam":
        contrastive_loss_fn = SimSiamLoss()
    else:
        contrastive_loss_fn = BYOLLoss()

    # Optimiser — explicit param lists to exclude EMA targets
    opt_params = (
        list(encoder.parameters())
        + list(inverse.parameters())
        + list(forward_model.parameters())
        + (list(predictor.parameters()) if predictor else [])
    )
    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training loop
    global_step = 0
    batch_keys = {"s1_a", "s2_a", "s1_b", "s2_b"}
    keys_validated = False

    print(f"\nStarting training for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        encoder.train()
        inverse.train()
        forward_model.train()
        if predictor is not None:
            predictor.train()

        epoch_c_loss = epoch_f_loss = 0.0
        t0 = time.time()

        for batch in loader:
            global_step += 1

            # Validate batch format once
            if not keys_validated:
                missing = batch_keys - set(batch.keys())
                if missing:
                    raise KeyError(
                        f"Batch is missing required keys: {missing}.  "
                        f"Got: {set(batch.keys())}.  "
                        "Ensure your dataset/collate_fn produces "
                        "{'s1_a', 's2_a', 's1_b', 's2_b'} dicts."
                    )
                keys_validated = True

            s1_a = batch["s1_a"].to(device, non_blocking=True)
            s2_a = batch["s2_a"].to(device, non_blocking=True)
            s1_b = batch["s1_b"].to(device, non_blocking=True)
            s2_b = batch["s2_b"].to(device, non_blocking=True)

            # Encode anchor view
            enc_s1_a = encoder(s1_a)
            enc_s2_a = encoder(s2_a)
            z_a, z_proj_a = inverse(enc_s1_a, enc_s2_a)

            # Encode positive view (target or online depending on loss type)
            if cfg.loss_type == "byol":
                with torch.no_grad():
                    enc_s1_b = target_encoder(s1_b)
                    enc_s2_b = target_encoder(s2_b)
                    _, z_proj_b = target_inverse(enc_s1_b, enc_s2_b)
            else:
                enc_s1_b = encoder(s1_b)
                enc_s2_b = encoder(s2_b)
                _, z_proj_b = inverse(enc_s1_b, enc_s2_b)

            # Contrastive loss
            if cfg.loss_type == "infonce":
                c_loss = contrastive_loss_fn(z_proj_a, z_proj_b)
            elif cfg.loss_type == "simsiam":
                p_a = predictor(z_proj_a)
                p_b = predictor(z_proj_b)
                c_loss = 0.5 * (
                    contrastive_loss_fn(p_a, z_proj_b)
                    + contrastive_loss_fn(p_b, z_proj_a)
                )
            else:  # byol
                p_a = predictor(z_proj_a)
                c_loss = contrastive_loss_fn(p_a, z_proj_b)

            # Forward loss
            s2_pred = forward_model(enc_s1_a, z_a)
            f_loss = forward_loss_fn(s2_pred, enc_s2_a)

            total_loss = (
                cfg.contrastive_weight * c_loss + cfg.forward_weight * f_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(opt_params, cfg.grad_clip)
            optimizer.step()

            if cfg.loss_type == "byol":
                ema.step()

            epoch_c_loss += c_loss.item()
            epoch_f_loss += f_loss.item()

            if global_step % cfg.log_frequency == 0:
                _log("train", global_step, {
                    "contrastive_loss": c_loss.item(),
                    "forward_loss": f_loss.item(),
                    "total_loss": total_loss.item(),
                })

        # End-of-epoch logging
        n = len(loader)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch + 1:>4}/{cfg.epochs}  "
            f"c_loss={epoch_c_loss / n:.4f}  "
            f"f_loss={epoch_f_loss / n:.4f}  "
            f"({elapsed:.1f}s)"
        )
        _log("epoch", epoch + 1, {
            "contrastive_loss": epoch_c_loss / n,
            "forward_loss": epoch_f_loss / n,
        })

        # Checkpoint
        if cfg.save_frequency > 0 and (epoch + 1) % cfg.save_frequency == 0:
            save_checkpoint(
                encoder, inverse, forward_model, predictor,
                target_encoder, target_inverse,
                optimizer, epoch + 1, cfg,
                run_dir / f"checkpoint_epoch{epoch + 1:04d}.pt",
            )

    # Final checkpoint
    save_checkpoint(
        encoder, inverse, forward_model, predictor,
        target_encoder, target_inverse,
        optimizer, cfg.epochs, cfg,
        run_dir / "final.pt",
    )
    writer.close()
    print(f"\nTraining complete.  Artifacts: {run_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

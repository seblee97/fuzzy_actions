"""Joint training of encoder, inverse model, forward model, and prior.

Architecture
------------
Four modules trained jointly:

  StateEncoder  — s → x  (L2-normalised embedding)
  InverseModel  — (x1, x2) → z
  ForwardModel  — (x1, z) → x2_pred
  Prior         — x1 → z_prior  (optional; only built when prior_loss_type != "null")

Loss
----
  L = w_fwd  · L_forward(x2_pred, x2)
    + w_inv  · L_inverse(z, z_pos?)
    + w_prior· L_prior(z_prior, z)
    + w_reg  · L_reg(z)

Each slot is independently configurable via --*-loss-type flags.  Loss
implementations may carry their own trainable parameters (e.g. the InfoNCE
projection head); these are collected and included in the optimiser
automatically.

Loss types
----------
  inverse  : "null" | "infonce"
  forward  : "null" | "mse"
  prior    : "null" | "mse"
  reg      : "null" | "l2"

Datasets
--------
Non-contrastive losses (inverse_loss_type="null") need only {s1, s2} per
sample — use PhaseTransitionDataset / RoomTransitionDataset.

Contrastive inverse losses (inverse_loss_type="infonce") additionally need
a positive pair {s1_b, s2_b} — use PhaseTransitionPairDataset or similar.

Usage
-----
    # Default: MSE forward + MSE prior (JEPA-style, no contrastive)
    python train_hierarchical.py \\
        --data-path datasets/oracle_distribution/directional \\
        --dataset-kwargs '{"state_mode":"pixel","pixel_style":"game","obs_cell_size":32,"resize_obs":[84,84],"max_trajs":1000}' \\
        --encoder-mode pixel

    # Add InfoNCE on z
    python train_hierarchical.py \\
        --data-path datasets/oracle_distribution/directional \\
        --dataset-class pair_datasets.PhaseTransitionPairDataset \\
        --dataset-kwargs '{"state_mode":"pixel","pixel_style":"game","obs_cell_size":32,"resize_obs":[84,84],"max_trajs":1000}' \\
        --encoder-mode pixel --inverse-loss-type infonce

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hierarchical import (
    AbstractEncRegLoss,
    EMAUpdater,
    ForwardModel,
    NConditionedForwardModel,
    NullEncRegLoss,
    VICRegVarLoss,
    InfoNCEInverseLoss,
    InverseModel,
    L2RegLoss,
    MSEForwardLoss,
    MSEPriorLoss,
    NullForwardLoss,
    NullInverseLoss,
    NullPriorLoss,
    NullRegLoss,
    Prior,
    StateEncoder,
    AbstractInverseLoss,
    AbstractForwardLoss,
    AbstractPriorLoss,
    AbstractRegLoss,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- dataset ---
    data_path: str = ""
    dataset_class: str = "pair_datasets.PhaseTransitionDataset"
    dataset_kwargs: str = "{}"
    num_workers: int = 4

    # --- encoder ---
    encoder_mode: str = "latent"        # "latent" | "pixel" | "embedding"
    state_dim: int = 64
    encoder_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (256,))
    pixel_channels: int = 3
    input_dim: int = 64
    embed_dim: int = 256

    # --- inverse model ---
    z_dim: int = 128
    inverse_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (512, 256))

    # --- forward model ---
    forward_hidden_sizes: tuple[int, ...] = field(default_factory=lambda: (512, 256))
    forward_model_type: str = "base"    # "base" | "n_conditioned"
    n_embed_dim: int = 16

    # --- prior model ---
    prior_hidden_dim: int = 256

    # --- loss types ---
    inverse_loss_type: str = "null"     # "null" | "infonce"
    forward_loss_type: str = "mse"      # "null" | "mse"
    prior_loss_type: str = "mse"        # "null" | "mse"
    reg_loss_type: str = "l2"           # "null" | "l2"
    enc_reg_loss_type: str = "null"     # "null" | "vicreg_var"

    # --- loss weights ---
    w_forward: float = 1.0
    w_inverse: float = 1.0
    w_prior: float = 1.0
    w_reg: float = 1.0
    w_enc_reg: float = 1.0

    # --- loss-specific params ---
    # enc_reg / vicreg_var
    vicreg_target_std: float = 0.1   # target per-dim std for VICReg variance hinge

    # noise injected into z before the forward model (0 = disabled)
    z_noise_std: float = 0.0
    # forward/mse
    forward_use_predictor: bool = True
    forward_pred_hidden_dim: int = 256
    # inverse/infonce
    inv_proj_dim: int = 128
    inv_proj_hidden_dim: int = 256
    temperature: float = 0.07

    # --- optimiser ---
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0

    # --- training ---
    epochs: int = 100
    batch_size: int = 256
    seed: int = 42

    # --- data fraction ---
    data_fraction: float = 1.0

    # --- gradient flow ---
    # JSON string controlling per-loss gradient flow; empty string = defaults.
    # Format: {"forward": {"updates": [...], "stop_grad": [...]}, ...}
    # Valid updates names : encoder, inverse, forward, prior, inverse_loss, forward_loss
    # Valid stop_grad names: x1, x2, z, z_pos, x2_pred, z_prior
    grad_config: str = ""

    # --- logging / saving ---
    run_name: str = ""
    runs_dir: str = "runs"
    save_frequency: int = 10
    log_frequency: int = 50
    shuffle_test_frequency: int = 200  # steps between shuffle tests (0 = disabled)


# ---------------------------------------------------------------------------
# Gradient flow configuration
# ---------------------------------------------------------------------------

@dataclass
class LossSlotConfig:
    """Gradient flow spec for one loss slot."""
    updates: list   # module names whose optimizers are stepped by this loss
    stop_grad: list = field(default_factory=list)  # tensor names to detach


@dataclass
class GradFlowConfig:
    """Gradient flow for all four loss slots.

    Defaults reproduce the previous hard-coded behaviour:
      - forward  : target x2 is detached; updates encoder + inverse + forward
      - inverse  : no stop-grad; updates inverse (+ inverse_loss projector)
      - prior    : z detached so inverse is not updated; updates prior only
      - reg      : no stop-grad; updates inverse
    """
    forward: LossSlotConfig = field(default_factory=lambda: LossSlotConfig(
        updates=["encoder", "inverse", "forward", "forward_loss"],
        stop_grad=["x2"],
    ))
    inverse: LossSlotConfig = field(default_factory=lambda: LossSlotConfig(
        updates=["encoder", "inverse", "inverse_loss"],
        stop_grad=[],
    ))
    prior: LossSlotConfig = field(default_factory=lambda: LossSlotConfig(
        updates=["prior"],
        stop_grad=["z"],
    ))
    reg: LossSlotConfig = field(default_factory=lambda: LossSlotConfig(
        updates=["inverse"],
        stop_grad=[],
    ))
    enc_reg: LossSlotConfig = field(default_factory=lambda: LossSlotConfig(
        updates=["encoder"],
        stop_grad=[],
    ))


def _parse_grad_config(json_str: str) -> GradFlowConfig:
    cfg = GradFlowConfig()
    if not json_str:
        return cfg
    data = json.loads(json_str)
    for slot in ("forward", "inverse", "prior", "reg", "enc_reg"):
        if slot not in data:
            continue
        d = data[slot]
        default = getattr(cfg, slot)
        setattr(cfg, slot, LossSlotConfig(
            updates=d.get("updates", default.updates),
            stop_grad=d.get("stop_grad", default.stop_grad),
        ))
    return cfg


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Hierarchical representation learning"
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

    # inverse model
    p.add_argument("--z-dim", type=int, default=cfg.z_dim)
    p.add_argument("--inverse-hidden-sizes", type=int, nargs="+",
                   default=list(cfg.inverse_hidden_sizes), metavar="N")

    # forward model
    p.add_argument("--forward-hidden-sizes", type=int, nargs="+",
                   default=list(cfg.forward_hidden_sizes), metavar="N")
    p.add_argument("--forward-model-type", default=cfg.forward_model_type,
                   choices=["base", "n_conditioned"])
    p.add_argument("--n-embed-dim", type=int, default=cfg.n_embed_dim)

    # prior model
    p.add_argument("--prior-hidden-dim", type=int, default=cfg.prior_hidden_dim)

    # loss types
    p.add_argument("--inverse-loss-type", default=cfg.inverse_loss_type,
                   choices=["null", "infonce"])
    p.add_argument("--forward-loss-type", default=cfg.forward_loss_type,
                   choices=["null", "mse"])
    p.add_argument("--prior-loss-type", default=cfg.prior_loss_type,
                   choices=["null", "mse"])
    p.add_argument("--reg-loss-type", default=cfg.reg_loss_type,
                   choices=["null", "l2"])
    p.add_argument("--enc-reg-loss-type", default=cfg.enc_reg_loss_type,
                   choices=["null", "vicreg_var"])

    # loss weights
    p.add_argument("--w-forward", type=float, default=cfg.w_forward)
    p.add_argument("--w-inverse", type=float, default=cfg.w_inverse)
    p.add_argument("--w-prior", type=float, default=cfg.w_prior)
    p.add_argument("--w-reg", type=float, default=cfg.w_reg)
    p.add_argument("--w-enc-reg", type=float, default=cfg.w_enc_reg)

    # loss-specific
    p.add_argument("--forward-use-predictor", action=argparse.BooleanOptionalAction,
                   default=cfg.forward_use_predictor)
    p.add_argument("--forward-pred-hidden-dim", type=int, default=cfg.forward_pred_hidden_dim)
    p.add_argument("--inv-proj-dim", type=int, default=cfg.inv_proj_dim)
    p.add_argument("--inv-proj-hidden-dim", type=int, default=cfg.inv_proj_hidden_dim)
    p.add_argument("--temperature", type=float, default=cfg.temperature)
    p.add_argument("--vicreg-target-std", type=float, default=cfg.vicreg_target_std,
                   help="target per-dim std for VICReg variance hinge (enc_reg)")
    p.add_argument("--z-noise-std", type=float, default=cfg.z_noise_std,
                   help="std of Gaussian noise added to z before the forward model (0=off)")

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
    p.add_argument("--shuffle-test-frequency", type=int, default=cfg.shuffle_test_frequency,
                   help="steps between shuffle tests (0 = disabled)")
    p.add_argument("--grad-config", default=cfg.grad_config, metavar="JSON",
                   help="JSON controlling gradient flow per loss slot")

    args = p.parse_args()
    cfg.data_path             = args.data_path
    cfg.dataset_class         = args.dataset_class
    cfg.dataset_kwargs        = args.dataset_kwargs
    cfg.num_workers           = args.num_workers
    cfg.encoder_mode          = args.encoder_mode
    cfg.state_dim             = args.state_dim
    cfg.encoder_hidden_sizes  = tuple(args.encoder_hidden_sizes)
    cfg.pixel_channels        = args.pixel_channels
    cfg.input_dim             = args.input_dim
    cfg.embed_dim             = args.embed_dim
    cfg.z_dim                 = args.z_dim
    cfg.inverse_hidden_sizes  = tuple(args.inverse_hidden_sizes)
    cfg.forward_hidden_sizes  = tuple(args.forward_hidden_sizes)
    cfg.forward_model_type    = args.forward_model_type
    cfg.n_embed_dim           = args.n_embed_dim
    cfg.prior_hidden_dim      = args.prior_hidden_dim
    cfg.inverse_loss_type     = args.inverse_loss_type
    cfg.forward_loss_type     = args.forward_loss_type
    cfg.prior_loss_type       = args.prior_loss_type
    cfg.reg_loss_type         = args.reg_loss_type
    cfg.enc_reg_loss_type     = args.enc_reg_loss_type
    cfg.w_forward             = args.w_forward
    cfg.w_inverse             = args.w_inverse
    cfg.w_prior               = args.w_prior
    cfg.w_reg                 = args.w_reg
    cfg.w_enc_reg             = args.w_enc_reg
    cfg.vicreg_target_std     = args.vicreg_target_std
    cfg.forward_use_predictor = args.forward_use_predictor
    cfg.forward_pred_hidden_dim = args.forward_pred_hidden_dim
    cfg.inv_proj_dim          = args.inv_proj_dim
    cfg.inv_proj_hidden_dim   = args.inv_proj_hidden_dim
    cfg.temperature           = args.temperature
    cfg.z_noise_std           = args.z_noise_std
    cfg.lr                    = args.lr
    cfg.weight_decay          = args.weight_decay
    cfg.grad_clip             = args.grad_clip
    cfg.epochs                = args.epochs
    cfg.batch_size            = args.batch_size
    cfg.seed                  = args.seed
    cfg.data_fraction         = args.data_fraction
    cfg.run_name              = args.run_name
    cfg.runs_dir              = args.runs_dir
    cfg.save_frequency        = args.save_frequency
    cfg.log_frequency         = args.log_frequency
    cfg.shuffle_test_frequency = args.shuffle_test_frequency
    cfg.grad_config           = args.grad_config
    return cfg


# ---------------------------------------------------------------------------
# Model and loss construction
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


def build_losses(cfg: Config) -> tuple[
    AbstractInverseLoss,
    AbstractForwardLoss,
    AbstractPriorLoss,
    AbstractRegLoss,
]:
    if cfg.inverse_loss_type == "infonce":
        inverse_loss = InfoNCEInverseLoss(
            z_dim=cfg.z_dim,
            proj_dim=cfg.inv_proj_dim,
            proj_hidden_dim=cfg.inv_proj_hidden_dim,
            temperature=cfg.temperature,
        )
    else:
        inverse_loss = NullInverseLoss()

    if cfg.forward_loss_type == "mse":
        forward_loss = MSEForwardLoss(
            embed_dim=cfg.embed_dim,
            use_predictor=cfg.forward_use_predictor,
            hidden_dim=cfg.forward_pred_hidden_dim,
        )
    else:
        forward_loss = NullForwardLoss()

    prior_loss   = MSEPriorLoss()  if cfg.prior_loss_type   == "mse"        else NullPriorLoss()
    reg_loss     = L2RegLoss()     if cfg.reg_loss_type     == "l2"         else NullRegLoss()
    enc_reg_loss = VICRegVarLoss(target_std=cfg.vicreg_target_std) \
                   if cfg.enc_reg_loss_type == "vicreg_var" else NullEncRegLoss()

    return inverse_loss, forward_loss, prior_loss, reg_loss, enc_reg_loss


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
    """Save a contact sheet of sample (s1, s2) pairs to run_dir/sample_frames.png."""
    import random as _random
    from PIL import Image as _Image

    rng = _random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_pairs, len(dataset)))

    frames = []
    for idx in indices:
        item = dataset[idx]
        s1 = item.get("s1", item.get("s1_a"))
        s2 = item.get("s2", item.get("s2_a"))
        if s1 is None or s1.ndim != 3:
            print("sample_frames: skipping (state_mode is not 'pixel')")
            return
        frames.append((s1, s2))

    if not frames:
        return

    def to_uint8(t):
        return t.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()

    import numpy as np
    gap, row_gap = 4, 2
    sample_h, sample_w = to_uint8(frames[0][0]).shape[:2]
    sheet = np.full(
        (len(frames) * sample_h + (len(frames) - 1) * row_gap, sample_w * 2 + gap, 3),
        255, dtype=np.uint8,
    )
    for row, (s1, s2) in enumerate(frames):
        y = row * (sample_h + row_gap)
        sheet[y:y + sample_h, :sample_w] = to_uint8(s1)
        sheet[y:y + sample_h, sample_w + gap:] = to_uint8(s2)

    out_path = run_dir / "sample_frames.png"
    _Image.fromarray(sheet).save(out_path)
    print(f"Sample frames saved → {out_path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    modules_dict: dict,
    optimizers: dict,
    epoch: int,
    cfg: Config,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "module_state_dicts":    {k: m.state_dict() for k, m in modules_dict.items()},
            "optimizer_state_dicts": {k: o.state_dict() for k, o in optimizers.items()},
            "config":                cfg.__dict__,
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
            f"hierarchical_inv{cfg.inverse_loss_type}_fwd{cfg.forward_loss_type}"
            f"_prior{cfg.prior_loss_type}_{cfg.encoder_mode}"
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
            print(f"Auto-setting state_dim={base_dataset.state_dim} (was {cfg.state_dim})")
            cfg.state_dim = base_dataset.state_dim
    elif cfg.encoder_mode == "pixel" and hasattr(base_dataset, "pixel_shape") and base_dataset.pixel_shape is not None:
        C, *_ = base_dataset.pixel_shape
        if cfg.pixel_channels != C:
            print(f"Auto-setting pixel_channels={C} (was {cfg.pixel_channels})")
            cfg.pixel_channels = C

    # Build models
    encoder      = build_encoder(cfg).to(device)
    inverse      = InverseModel(
        embed_dim=cfg.embed_dim,
        z_dim=cfg.z_dim,
        hidden_sizes=list(cfg.inverse_hidden_sizes),
    ).to(device)
    if cfg.forward_model_type == "n_conditioned":
        forward_model = NConditionedForwardModel(
            embed_dim=cfg.embed_dim,
            z_dim=cfg.z_dim,
            n_embed_dim=cfg.n_embed_dim,
            hidden_sizes=list(cfg.forward_hidden_sizes),
        ).to(device)
    else:
        forward_model = ForwardModel(
            embed_dim=cfg.embed_dim,
            z_dim=cfg.z_dim,
            hidden_sizes=list(cfg.forward_hidden_sizes),
        ).to(device)
    prior = (
        Prior(cfg.embed_dim, cfg.z_dim, cfg.prior_hidden_dim).to(device)
        if cfg.prior_loss_type != "null" else None
    )

    # Build losses (may contain trainable parameters)
    inverse_loss_fn, forward_loss_fn, prior_loss_fn, reg_loss_fn, enc_reg_loss_fn = build_losses(cfg)
    inverse_loss_fn  = inverse_loss_fn.to(device)
    forward_loss_fn  = forward_loss_fn.to(device)
    enc_reg_loss_fn  = enc_reg_loss_fn.to(device)

    # Named module registry — keys are the valid names for `updates` in grad config
    modules_dict: dict[str, nn.Module] = {
        "encoder":      encoder,
        "inverse":      inverse,
        "forward":      forward_model,
        "inverse_loss": inverse_loss_fn,
        "forward_loss": forward_loss_fn,
    }
    if prior is not None:
        modules_dict["prior"] = prior

    n_params = sum(p.numel() for m in modules_dict.values() for p in m.parameters())
    print(f"Trainable parameters (total): {n_params:,}")

    # Per-module optimizers (only for modules that have parameters)
    optimizers: dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam(list(m.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for name, m in modules_dict.items()
        if any(True for _ in m.parameters())
    }

    # Gradient flow config
    grad_cfg = _parse_grad_config(cfg.grad_config)

    # Loss slots: (short_name, weight, fn, slot_grad_config)
    # Null losses are skipped (no graph, no point doing backward)
    _NULL_TYPES = (NullForwardLoss, NullInverseLoss, NullPriorLoss, NullRegLoss, NullEncRegLoss)
    loss_slots = [
        ("fwd",     cfg.w_forward,  forward_loss_fn,  grad_cfg.forward),
        ("inv",     cfg.w_inverse,  inverse_loss_fn,  grad_cfg.inverse),
        ("prior",   cfg.w_prior,    prior_loss_fn,    grad_cfg.prior),
        ("reg",     cfg.w_reg,      reg_loss_fn,      grad_cfg.reg),
        ("enc_reg", cfg.w_enc_reg,  enc_reg_loss_fn,  grad_cfg.enc_reg),
    ]
    active_slots = [
        (name, w, fn, sc) for name, w, fn, sc in loss_slots
        if w != 0.0 and not isinstance(fn, _NULL_TYPES)
    ]

    # Determine batch key names from first batch
    s1_key = s2_key = None

    print(f"\nLoss: fwd={cfg.forward_loss_type}(w={cfg.w_forward})  "
          f"inv={cfg.inverse_loss_type}(w={cfg.w_inverse})  "
          f"prior={cfg.prior_loss_type}(w={cfg.w_prior})  "
          f"reg={cfg.reg_loss_type}(w={cfg.w_reg})")
    print(f"Starting training for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        for m in modules_dict.values():
            m.train()

        epoch_losses = {"fwd": 0.0, "inv": 0.0, "prior": 0.0, "reg": 0.0, "enc_reg": 0.0}
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            # Detect key names once
            if s1_key is None:
                keys = set(batch.keys())
                if "s1" in keys:
                    s1_key, s2_key = "s1", "s2"
                elif "s1_a" in keys:
                    s1_key, s2_key = "s1_a", "s2_a"
                else:
                    raise KeyError(f"Batch must have 's1'/'s2' or 's1_a'/'s2_a'. Got: {keys}")

            s1 = batch[s1_key].to(device, non_blocking=True)
            s2 = batch[s2_key].to(device, non_blocking=True)

            # Pre-move pair branch and N to device (if present)
            s1_b = batch["s1_b"].to(device, non_blocking=True) if "s1_b" in batch else None
            s2_b = batch["s2_b"].to(device, non_blocking=True) if "s2_b" in batch else None
            n_context = batch.get("n_a", batch.get("n"))
            if n_context is not None:
                n_context = n_context.to(device, non_blocking=True)

            # --- Per-loss backward passes (fresh forward pass each time) ---
            # A fresh forward pass avoids the retain_graph in-place version
            # conflict that arises when optimizer.step() modifies parameters
            # between backward calls on the same graph.
            batch_losses: dict[str, float] = {}
            # Track peak grad norm per module across all backward passes this batch
            peak_grad_norms: dict[str, float] = {mn: 0.0 for mn in modules_dict}

            for name, w, fn, slot_cfg in active_slots:
                for opt in optimizers.values():
                    opt.zero_grad()

                x1 = encoder(s1)
                x2 = encoder(s2)
                z  = inverse(x1, x2)
                z_fwd = z + torch.randn_like(z) * cfg.z_noise_std if cfg.z_noise_std > 0 else z
                x2_pred = forward_model(x1, z_fwd, context=n_context)
                z_prior = prior(x1) if prior is not None else None

                z_pos = None
                if inverse_loss_fn.needs_pairs and s1_b is not None:
                    z_pos = inverse(encoder(s1_b), encoder(s2_b))

                # Apply external stop_grad to tensor pool
                sg = slot_cfg.stop_grad
                pool = {
                    "x1": x1, "x2": x2, "z": z,
                    "x2_pred": x2_pred, "z_prior": z_prior, "z_pos": z_pos,
                }
                t = {k: (v.detach() if k in sg and v is not None else v)
                     for k, v in pool.items()}

                if name == "fwd":
                    l = fn(t["x2_pred"], t["x2"], stop_grad=sg)
                elif name == "inv":
                    l = fn(t["z"], t["z_pos"], stop_grad=sg)
                elif name == "prior":
                    l = fn(t["z_prior"], t["z"], stop_grad=sg) if t["z_prior"] is not None else x1.new_tensor(0.0)
                elif name == "reg":
                    l = fn(t["z"], stop_grad=sg)
                else:  # enc_reg
                    l = fn(t["x1"], t["x2"], stop_grad=sg)

                batch_losses[name] = l.item()

                (w * l).backward()

                # Capture grad norms before clip/step, take max across slots
                for mn, m in modules_dict.items():
                    norms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
                    if norms:
                        peak_grad_norms[mn] = max(peak_grad_norms[mn], sum(norms) / len(norms))

                update_params = [
                    p for mn in slot_cfg.updates if mn in modules_dict
                    for p in modules_dict[mn].parameters()
                ]
                if update_params:
                    nn.utils.clip_grad_norm_(update_params, cfg.grad_clip)
                for mn in slot_cfg.updates:
                    if mn in optimizers:
                        optimizers[mn].step()

            epoch_losses["fwd"]     += batch_losses.get("fwd",     0.0)
            epoch_losses["inv"]     += batch_losses.get("inv",     0.0)
            epoch_losses["prior"]   += batch_losses.get("prior",   0.0)
            epoch_losses["reg"]     += batch_losses.get("reg",     0.0)
            epoch_losses["enc_reg"] += batch_losses.get("enc_reg", 0.0)

            global_step = epoch * len(loader) + batch_idx

            # --- Diagnostics (use last forward pass's tensors) ---
            if global_step % cfg.log_frequency == 0:
                with torch.no_grad():
                    # z stats
                    z_std_per_dim = z.std(dim=0)                          # (z_dim,)
                    z_var = z_std_per_dim.pow(2).mean().item()
                    z_std = z_std_per_dim.mean().item()
                    z_dead_frac = (z_std_per_dim < 0.01).float().mean().item()

                    # z effective dimensionality (participation ratio)
                    z_centered = z - z.mean(dim=0, keepdim=True)
                    cov = z_centered.T @ z_centered / (z.shape[0] - 1)   # (D, D)
                    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
                    z_eff_dim = eigvals.sum().pow(2) / (eigvals.pow(2).sum() + 1e-8)

                    # Encoder health: x1-x2 cosine similarity
                    x1_x2_cos = torch.nn.functional.cosine_similarity(x1, x2, dim=-1).mean().item()

                    # x2_pred quality: cosine similarity with x2
                    x2_pred_cos = torch.nn.functional.cosine_similarity(x2_pred, x2, dim=-1).mean().item()

                    # z same-type vs different-type cosine similarity (requires z_pos)
                    if z_pos is not None:
                        z_same_cos = torch.nn.functional.cosine_similarity(z, z_pos, dim=-1).mean().item()
                        z_diff_cos = torch.nn.functional.cosine_similarity(
                            z, z_pos[torch.randperm(z_pos.shape[0], device=device)], dim=-1
                        ).mean().item()
                    else:
                        z_same_cos = z_diff_cos = float("nan")

                    # Prior quality: cosine similarity between z_prior and z
                    if z_prior is not None:
                        prior_z_cos = torch.nn.functional.cosine_similarity(z_prior, z, dim=-1).mean().item()
                    else:
                        prior_z_cos = float("nan")

                    # Per-module peak gradient norms across all backward passes this batch
                    grad_norms = {f"grad_norm/{mn}": v for mn, v in peak_grad_norms.items()}

                total_val = sum(
                    w * batch_losses.get(n, 0.0)
                    for n, w, _, _ in loss_slots
                )
                _log("train", global_step, {
                    "loss_fwd":      batch_losses.get("fwd",     0.0),
                    "loss_inv":      batch_losses.get("inv",     0.0),
                    "loss_prior":    batch_losses.get("prior",   0.0),
                    "loss_reg":      batch_losses.get("reg",     0.0),
                    "loss_enc_reg":  batch_losses.get("enc_reg", 0.0),
                    "total_loss":    total_val,
                    # z
                    "z_var":         z_var,
                    "z_std":         z_std,
                    "z_dead_frac":   z_dead_frac,
                    "z_eff_dim":     z_eff_dim.item(),
                    "z_same_cos":    z_same_cos,
                    "z_diff_cos":    z_diff_cos,
                    # encoder
                    "x1_x2_cos":     x1_x2_cos,
                    # forward model
                    "x2_pred_cos":   x2_pred_cos,
                    # prior
                    "prior_z_cos":   prior_z_cos,
                    **grad_norms,
                })

            if (cfg.shuffle_test_frequency > 0
                    and global_step % cfg.shuffle_test_frequency == 0
                    and not isinstance(forward_loss_fn, NullForwardLoss)):
                with torch.no_grad():
                    x1_d = encoder(s1)
                    x2_d = encoder(s2)
                    z_clean = inverse(x1_d, x2_d)
                    z_shuf  = z_clean[torch.randperm(z_clean.shape[0], device=device)]
                    fwd_slot = grad_cfg.forward
                    loss_clean = forward_loss_fn(
                        forward_model(x1_d, z_clean, context=n_context),
                        x2_d, stop_grad=fwd_slot.stop_grad,
                    ).item()
                    loss_shuf = forward_loss_fn(
                        forward_model(x1_d, z_shuf, context=n_context),
                        x2_d, stop_grad=fwd_slot.stop_grad,
                    ).item()
                    z_sensitivity = (loss_shuf - loss_clean) / (loss_clean + 1e-8)
                _log("diag", global_step, {
                    "shuffle_fwd_clean": loss_clean,
                    "shuffle_fwd_shuf":  loss_shuf,
                    "z_sensitivity":     z_sensitivity,
                })
                print(
                    f"  [shuffle test] step={global_step}  "
                    f"fwd_clean={loss_clean:.4f}  fwd_shuf={loss_shuf:.4f}  "
                    f"sensitivity={z_sensitivity:+.3f}"
                )

        n = len(loader)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch + 1:>4}/{cfg.epochs}  "
            f"fwd={epoch_losses['fwd']/n:.4f}  "
            f"inv={epoch_losses['inv']/n:.4f}  "
            f"prior={epoch_losses['prior']/n:.4f}  "
            f"reg={epoch_losses['reg']/n:.4f}  "
            f"enc_reg={epoch_losses['enc_reg']/n:.4f}  "
            f"({elapsed:.1f}s)"
        )
        _log("epoch", epoch + 1, {k: v / n for k, v in epoch_losses.items()})

        if cfg.save_frequency > 0 and (epoch + 1) % cfg.save_frequency == 0:
            save_checkpoint(
                modules_dict, optimizers, epoch + 1, cfg,
                run_dir / f"checkpoint_epoch{epoch + 1:04d}.pt",
            )

    save_checkpoint(
        modules_dict, optimizers, cfg.epochs, cfg,
        run_dir / "final.pt",
    )
    writer.close()
    print(f"\nTraining complete.  Artifacts: {run_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

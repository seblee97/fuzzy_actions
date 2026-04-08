"""Generate sweep configs and a SLURM array job.

Usage:
    python scripts/generate_sweep.py           # print summary
    python scripts/generate_sweep.py --write   # write configs + slurm script
    python scripts/generate_sweep.py --submit  # write configs + slurm script + sbatch
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Sweep grid — edit this section to define your sweep
# ---------------------------------------------------------------------------

FIXED = {
    "data_path": "$DATA",
    "dataset_class": "pair_datasets.PhaseTransitionPairDataset",
    "dataset_kwargs": '{"state_mode":"pixel","pixel_style":"game","obs_cell_size":32,"resize_obs":[84,84]}',
    "encoder_mode": "pixel",
    "inverse_loss_type": "infonce",
    "forward_loss_type": "mse",
    "prior_loss_type": "mse",
    "forward_model_type": "n_conditioned",
    "no_forward_use_predictor": True,
    "enc_reg_loss_type": "vicreg_var",
    "vicreg_target_std": 0.1,
    "z_noise_std": 0.1,
    "epochs": 100,
    "batch_size": 256,
    "num_workers": 8,
    "log_frequency": 50,
    "save_frequency": 10,
    "shuffle_test_frequency": 500,
}

GRID = {
    # Force forward model to use z by stop-gradding x1, or keep baseline
    "grad_config": [
        # baseline: forward loss doesn't update encoder, keys detached in InfoNCE
        '{"forward": {"updates": ["inverse", "forward", "forward_loss"], "stop_grad": ["x2"]}, "inverse": {"stop_grad": ["keys"]}}',
        # also stop-grad x1 in forward loss: model must use z to predict x2
        '{"forward": {"updates": ["inverse", "forward", "forward_loss"], "stop_grad": ["x1", "x2"]}, "inverse": {"stop_grad": ["keys"]}}',
    ],
    "w_enc_reg": [10.0, 25.0, 50.0],
    "w_forward": [0.1, 0.5, 1.0],
    "z_dim": [32, 128],
}

# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _grid_configs() -> list[dict]:
    keys = list(GRID.keys())
    configs = []
    for values in product(*[GRID[k] for k in keys]):
        cfg = dict(FIXED)
        cfg.update(dict(zip(keys, values)))
        configs.append(cfg)
    return configs


def _config_to_args(cfg: dict) -> str:
    """Convert a config dict to a train_hierarchical.py argument string."""
    parts = []
    bool_flags = {"no_forward_use_predictor"}
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if k in bool_flags:
            if v:
                parts.append(flag)
        elif isinstance(v, str) and (" " in v or "{" in v):
            parts.append(f"{flag} '{v}'")
        else:
            parts.append(f"{flag} {v}")
    return " \\\n    ".join(parts)


def _config_name(cfg: dict, idx: int) -> str:
    grad = "sgx1x2" if "x1" in cfg.get("grad_config", "") else "sgx2"
    return (
        f"sweep_{idx:03d}"
        f"_encr{cfg['w_enc_reg']}"
        f"_wfwd{cfg['w_forward']}"
        f"_zdim{cfg['z_dim']}"
        f"_{grad}"
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SLURM_HEADER = """\
#!/bin/bash
#SBATCH --job-name=hier_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-{last_idx}

set -e

VENV="/mnt/home/slee1/venvs/fuzzy_actions/bin/activate"
REPO="/mnt/home/slee1/projects/fuzzy_actions"
DATA="/mnt/home/slee1/ceph/fuzzy_data/oracle_distribution/directional"
RESULTS="/mnt/home/slee1/ceph/fuzzy_actions"
SWEEP_DIR="$REPO/scripts/sweep_configs"

source "$VENV"
export SDL_AUDIODRIVER=dummy

mkdir -p "$REPO/logs"
mkdir -p "$RESULTS"

cd "$REPO"

# Read the config file for this array task
CONFIG_FILE="$SWEEP_DIR/config_$(printf '%03d' $SLURM_ARRAY_TASK_ID).json"
RUN_NAME=$(python -c "import json; c=json.load(open('$CONFIG_FILE')); print(c['run_name'])")_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

python train_hierarchical.py \\
    $(python scripts/generate_sweep.py --args-for $SLURM_ARRAY_TASK_ID) \\
    --run-name "$RUN_NAME" \\
    --runs-dir "$RESULTS/runs"
"""


def write_configs(configs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, cfg in enumerate(configs):
        name = _config_name(cfg, i)
        record = dict(cfg)
        record["run_name"] = name
        path = out_dir / f"config_{i:03d}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
    print(f"Wrote {len(configs)} configs to {out_dir}/")


def write_slurm(configs: list[dict], out_path: Path) -> None:
    header = SLURM_HEADER.format(last_idx=len(configs) - 1)
    out_path.write_text(header)
    print(f"Wrote SLURM array script to {out_path}")
    print(f"Submit with: sbatch {out_path}")


def print_args_for(configs: list[dict], idx: int) -> None:
    """Print CLI args for a given config index (used by the slurm script)."""
    cfg = dict(configs[idx])
    cfg.pop("run_name", None)
    # Replace $DATA placeholder — the slurm script handles this via shell expansion
    for k, v in cfg.items():
        if isinstance(v, str):
            cfg[k] = v.replace("$DATA", "${DATA}")
    print(_config_to_args(cfg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write", action="store_true", help="write configs and slurm script"
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="write configs + slurm script, then sbatch",
    )
    p.add_argument(
        "--args-for",
        type=int,
        default=None,
        metavar="IDX",
        help="print CLI args for config IDX (used internally by slurm)",
    )
    args = p.parse_args()

    configs = _grid_configs()
    scripts_dir = Path(__file__).parent

    if args.args_for is not None:
        print_args_for(configs, args.args_for)
    elif args.write or args.submit:
        slurm_path = scripts_dir / "sweep.slurm"
        write_configs(configs, scripts_dir / "sweep_configs")
        write_slurm(configs, slurm_path)
        if args.submit:
            result = subprocess.run(
                ["sbatch", str(slurm_path)], capture_output=True, text=True
            )
            print(result.stdout.strip())
            if result.returncode != 0:
                print(result.stderr.strip())
    else:
        print(f"Total configs: {len(configs)}")
        print(f"Grid axes:")
        for k, v in GRID.items():
            print(f"  {k}: {v}")
        print(f"\nFirst config args:")
        print(_config_to_args(dict(FIXED) | {k: GRID[k][0] for k in GRID}))

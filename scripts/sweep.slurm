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
#SBATCH --array=0-35

set -e

VENV="/mnt/home/slee1/venvs/fuzzy_actions/bin/activate"
REPO="/mnt/home/slee1/projects/fuzzy_actions"
DATA="/mnt/home/slee1/ceph/fuzzy_data/oracle_distribution/directional"
RESULTS="/mnt/home/slee1/ceph/fuzzy_actions"
SWEEP_DIR="$REPO/scripts/sweep_configs"

source "$VENV"
export SDL_AUDIODRIVER=dummy

mkdir -p "$REPO/logs"
mkdir -p "$RESULTS/runs"

cd "$REPO"

CONFIG_FILE="$SWEEP_DIR/config_$(printf '%03d' $SLURM_ARRAY_TASK_ID).json"

# Convert JSON config to CLI args, substituting $DATA for the data path
ARGS=$(python - <<EOF
import json, sys
cfg = json.load(open("$CONFIG_FILE"))
parts = []
bool_flags = {"no_forward_use_predictor"}
skip = {"run_name", "data_path"}
for k, v in cfg.items():
    if k in skip:
        continue
    flag = "--" + k.replace("_", "-")
    if k in bool_flags:
        if v:
            parts.append(flag)
    elif isinstance(v, str) and ("{" in v or "[" in v):
        parts.append(f"{flag} '{v}'")
    else:
        parts.append(f"{flag} {v}")
print(" ".join(parts))
EOF
)

RUN_NAME=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['run_name'])")
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

eval python train_hierarchical.py \
    --data-path "$DATA" \
    --runs-dir "$RESULTS/runs" \
    --run-name "$RUN_NAME" \
    $ARGS

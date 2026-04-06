#!/bin/bash
#SBATCH --job-name=hier_baseline
#SBATCH --output=logs/hier_baseline_%j.out
#SBATCH --error=logs/hier_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

VENV="/mnt/home/slee1/venvs/fuzzy_actions/bin/activate"
REPO="/mnt/home/slee1/projects/fuzzy_actions"
DATA="/mnt/home/slee1/ceph/fuzzy_data/oracle_distribution/directional"
RESULTS="/mnt/home/slee1/ceph/fuzzy_actions"

source "$VENV"

export SDL_AUDIODRIVER=dummy

mkdir -p "$REPO/logs"
mkdir -p "$RESULTS"

cd "$REPO"

python train_hierarchical.py \
    --data-path "$DATA" \
    --dataset-class pair_datasets.PhaseTransitionPairDataset \
    --dataset-kwargs '{"state_mode":"pixel","pixel_style":"game","obs_cell_size":32,"resize_obs":[84,84]}' \
    --encoder-mode pixel \
    --inverse-loss-type infonce \
    --forward-loss-type mse \
    --prior-loss-type mse \
    --forward-model-type n_conditioned \
    --z-noise-std 0.1 \
    --no-forward-use-predictor \
    --enc-reg-loss-type vicreg_var \
    --vicreg-target-std 0.1 \
    --grad-config '{"inverse": {"stop_grad": ["keys"]}}' \
    --epochs 100 \
    --batch-size 256 \
    --num-workers 8 \
    --run-name "baseline_full_${SLURM_JOB_ID}" \
    --runs-dir "$RESULTS/runs" \
    --log-frequency 50 \
    --save-frequency 10 \
    --shuffle-test-frequency 500

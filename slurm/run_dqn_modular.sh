#!/bin/bash
#SBATCH --job-name=dqn_modular
#SBATCH --partition=ccn
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/home/slee1/projects/fuzzy_actions/slurm/logs/%j.out
#SBATCH --error=/mnt/home/slee1/projects/fuzzy_actions/slurm/logs/%j.err

source /mnt/home/slee1/venvs/fuzzy_actions/bin/activate

cd /mnt/home/slee1/projects/fuzzy_actions

python train_dqn_modular.py \
    --seed 42 \
    --n-rooms 4 \
    --total-timesteps 1000000 \
    --runs-dir /mnt/home/slee1/ceph/fuzzy/runs

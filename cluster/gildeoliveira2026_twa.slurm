#!/bin/bash
# SLURM array job: 100 jobs × 5.5e6 trajectories = 5.5e8 (the paper's statistics).
# Adapt partition/GPU type to the cluster. Submit with: sbatch cluster/gildeoliveira2026_twa.slurm
#SBATCH --job-name=gildeoliveira2026_twa
#SBATCH --array=1-100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/twa_%A_%a.out
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
julia --project=cluster cluster/gildeoliveira2026_twa.jl results/gildeoliveira2026 "$SLURM_ARRAY_TASK_ID" 5500000 10000 Float64

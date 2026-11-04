#!/usr/bin/env bash
#SBATCH --job-name=smimic-steptest
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Velocity step test (App. G), RL vs distilled velocity interface.
# Usage: sbatch slurm/step_test.slurm <out.json> <ckpt_rl.pt> <ckpt_distill.pt>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
$SMIMIC_PY code/scripts/step_test.py data/processed/lafan1_50hz.npz walk,run,sprint "$1" "$2" "$3"

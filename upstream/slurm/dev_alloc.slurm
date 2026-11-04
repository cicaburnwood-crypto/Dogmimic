#!/usr/bin/env bash
#SBATCH --job-name=smimic-dev
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.err
# Development allocation: holds one GPU so test commands can be run with
#   srun --jobid=<id> --overlap bash -lc '...'
set -euo pipefail
nvidia-smi
sleep 43000

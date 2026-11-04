#!/usr/bin/env bash
#SBATCH --job-name=smimic-bench
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Fig. 4a: world-model vs MuJoCo Warp control steps per second on one GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name --format=csv
$SMIMIC_PY code/scripts/bench_rollout.py --no-dr --sizes 32768,65536 --out results/bench_rollout_a6000_large.json

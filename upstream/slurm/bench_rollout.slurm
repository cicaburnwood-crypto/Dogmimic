#!/usr/bin/env bash
#SBATCH --job-name=smimic-bench
#SBATCH --account=allcs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Fig. 4a: world-model vs MuJoCo Warp control steps per second on one GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name --format=csv
$SMIMIC_PY code/scripts/bench_rollout.py --out results/bench_rollout_a6000.json

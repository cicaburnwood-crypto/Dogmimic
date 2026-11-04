#!/usr/bin/env bash
#SBATCH --job-name=ppo-sim
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=30:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Usage: sbatch slurm/train_ppo.slurm --out runs/<name> [ppo.py flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
ARGS=("$@")
OUT=""
for ((i=0; i<${#ARGS[@]}; i++)); do [ "${ARGS[$i]}" = "--out" ] && OUT="${ARGS[$((i+1))]}"; done
RESUME=()
if [ -n "$OUT" ] && [ -f "$OUT/last.pt" ]; then RESUME=(--resume "$OUT/last.pt"); fi
$SMIMIC_PY -m supermimic.ppo "${ARGS[@]}" "${RESUME[@]}"

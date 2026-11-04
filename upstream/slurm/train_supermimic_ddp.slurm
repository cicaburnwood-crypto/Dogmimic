#!/usr/bin/env bash
#SBATCH --job-name=smimic-ddp
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Data-parallel SuperMimic on 2 GPUs (paper: gradients averaged, lr sqrt(2) x 2e-4).
# Usage: sbatch slurm/train_supermimic_ddp.slurm --out runs/<name> [train.py flags]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l)
ARGS=("$@")
OUT=""
for ((i=0; i<${#ARGS[@]}; i++)); do [ "${ARGS[$i]}" = "--out" ] && OUT="${ARGS[$((i+1))]}"; done
RESUME=()
if [ -n "$OUT" ] && [ -f "$OUT/last.pt" ]; then RESUME=(--resume "$OUT/last.pt"); fi
$(dirname $SMIMIC_PY)/torchrun --standalone --nproc_per_node=$NGPU -m supermimic.train "${ARGS[@]}" "${RESUME[@]}"

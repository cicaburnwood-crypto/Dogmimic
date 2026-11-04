#!/usr/bin/env bash
#SBATCH --job-name=bones-dl
#SBATCH --account=pci
#SBATCH --partition=pci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%j-%x.out
#SBATCH --error=slurm/logs/%j-%x.out
# Downloads BONES-SEED g1.tar.gz (23.5 GB, gated: needs HF_TOKEN of an account that
# accepted the license), extracts it on node-local disk, converts to a 50 Hz motion
# library and copies only the single .npz back to shared storage.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the repository root
source env.sh
source setup_local_env.sh
: "${HF_TOKEN:?set HF_TOKEN (or run huggingface-cli login) first}"
export WORK=$SMIMIC_LOCAL/bones
mkdir -p $WORK
$SMIMIC_LOCAL/bin/uv pip install --python $SMIMIC_PY huggingface_hub
$SMIMIC_PY - << 'PY'
import os
from huggingface_hub import hf_hub_download
for f in ["g1.tar.gz", "metadata/seed_metadata_v004.parquet"]:
    p = hf_hub_download("bones-studio/seed", f, repo_type="dataset", local_dir=os.environ["WORK"], token=os.environ["HF_TOKEN"])
    print(p, os.path.getsize(p))
PY
cd $WORK && tar -xzf g1.tar.gz && ls g1 | head
cd "$SMIMIC_ROOT"
$SMIMIC_PY code/scripts/prepare_motions.py --format bones --fps-in 120 --input $WORK/g1/csv \
  --output $WORK/bones_seed_raw_50hz.npz --workers 16
mkdir -p data/processed data/raw/bones_meta
cp $WORK/bones_seed_raw_50hz.npz data/processed/
cp $WORK/metadata/seed_metadata_v004.parquet data/raw/bones_meta/

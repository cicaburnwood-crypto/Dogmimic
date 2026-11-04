#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source deployment/activate.sh
exec "$SMIMIC_PY" experiments/jump06/train_jump.py \
  --motions data/processed/g1_standing_jump_060m.npz \
  --out runs/g1_jump060_v1 --iterations 16000 --wall-hours 1.25 \
  --eval-every 1000 --ckpt-every 1000 --log-every 100 "$@"

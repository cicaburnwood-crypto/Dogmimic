#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$SMIMIC_ROOT/experiments/three_skills/engine:$SMIMIC_ROOT/experiments/three_skills:$PYTHONPATH"
exec "$SMIMIC_PY" -u experiments/three_skills/train.py \
  --motions "experiments/three_skills/targets/$SM_TASK.npz" \
  --num-envs 2048 --wm-batch 1024 --pi-batch 2048 --wm-batches 4 --critic-epochs 2 \
  --buffer-steps 768 --horizon 32 --iterations 50000 --wall-hours 3 \
  --log-every 100 --ckpt-every 2000 --eval-every 2000 --lr-decay-iters 60000 "$@"

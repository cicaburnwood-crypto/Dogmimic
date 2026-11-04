#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" SM_ROBOT=b2 SM_TASK=b2_stable_full_cycle_v2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$SMIMIC_ROOT/experiments/three_skills/engine:$SMIMIC_ROOT/experiments/b2_cycle:$SMIMIC_ROOT/experiments/three_skills:$PYTHONPATH"
exec "$SMIMIC_PY" -u experiments/b2_cycle/train_cycle.py \
  --motions experiments/b2_cycle/b2_full_cycle.npz \
  --num-envs 2048 --wm-batch 1024 --pi-batch 2048 --wm-batches 4 --critic-epochs 2 \
  --buffer-steps 1000 --horizon 32 --iterations 50000 --wall-hours 3 --lr .0001 --action-noise .05 \
  --log-every 100 --ckpt-every 2000 --eval-every 2000 --lr-decay-iters 60000 "$@"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" SM_ROBOT=b2 SM_TASK=b2_payload_recovery_v4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$SMIMIC_ROOT/experiments/three_skills/engine:$SMIMIC_ROOT/experiments/b2_payload_recovery:$SMIMIC_ROOT/experiments/b2_payload:$SMIMIC_ROOT/experiments/b2_cycle:$SMIMIC_ROOT/experiments/three_skills:$PYTHONPATH"
exec "$SMIMIC_PY" -u "experiments/b2_payload_recovery/${B2_TRAIN_ENTRY:-train_recovery.py}" \
  --motions experiments/b2_cycle/b2_full_cycle.npz \
  --num-envs 2048 --wm-batch 1024 --pi-batch 2048 --wm-batches 4 --critic-epochs 1 \
  --buffer-steps 1000 --horizon 8 --iterations 30000 --wall-hours 3 --lr .0001 --action-noise .015 \
  --warmup-steps 128 --log-every 100 --ckpt-every 500 --eval-every 500 --lr-decay-iters 40000 --use-critic false "$@"

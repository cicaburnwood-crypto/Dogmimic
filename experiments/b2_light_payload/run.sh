#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES=6 SM_ROBOT=b2 SM_TASK=b2_light_payload
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
unset B2_WM_SOURCE
export B2_WM_PREFIT="${B2_WM_PREFIT:-500}"
: "${B2_POLICY_SOURCE:?Set an explicitly verified own-policy checkpoint before running}"
export B2_POLICY_SOURCE
export B2_INIT_CHECKPOINT="$PWD/runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt"
source deployment/activate.sh
export PYTHONPATH="$PWD/experiments/b2_light_payload:$PWD/experiments/three_skills/engine:$PWD/experiments/b2_payload_recovery:$PWD/experiments/b2_payload:$PWD/experiments/b2_cycle:$PWD/experiments/three_skills:$PYTHONPATH"
exec "$SMIMIC_PY" -u experiments/b2_light_payload/train_repair.py --motions experiments/b2_cycle/b2_full_cycle.npz --num-envs 1024 --wm-batch 512 --pi-batch 512 --wm-batches 4 --buffer-steps 512 --horizon 32 --iterations 20000 --wall-hours 6 --lr .0001 --action-noise .015 --warmup-steps 128 --log-every 100 --ckpt-every 1000 --eval-every 1000 --use-critic false "$@"

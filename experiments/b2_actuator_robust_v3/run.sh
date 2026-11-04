#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES=6 SM_ROBOT=b2 SM_TASK=b2_actuator_robust_v3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export B2_WM_SOURCE="$PWD/runs/b2_actuator_robust_v2_20261101_190925/last.pt"
export B2_POLICY_SOURCE="$PWD/runs/b2_actuator_robust_v2_20261101_190925/best.pt"
export B2_INIT_CHECKPOINT="$PWD/runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt"
source deployment/activate.sh
export PYTHONPATH="$PWD/experiments/b2_actuator_robust_v3:$PWD/experiments/three_skills/engine:$PWD/experiments/b2_payload_recovery:$PWD/experiments/b2_payload:$PWD/experiments/b2_cycle:$PWD/experiments/three_skills:$PYTHONPATH"
exec "$SMIMIC_PY" -u experiments/b2_actuator_robust_v3/train_guarded.py --motions experiments/b2_cycle/b2_full_cycle.npz --num-envs 1024 --wm-batch 512 --pi-batch 512 --wm-batches 4 --buffer-steps 512 --horizon 16 --iterations 40000 --wall-hours 6 --lr .0001 --action-noise .015 --warmup-steps 128 --log-every 100 --ckpt-every 2000 --eval-every 2000 --use-critic false "$@"

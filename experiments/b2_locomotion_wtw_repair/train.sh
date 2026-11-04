#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export SM_ROBOT=b2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$PWD/experiments/b2_locomotion_wtw_repair/engine:$PWD/experiments/b2_locomotion_wtw:$PWD/experiments/three_skills:$PYTHONPATH"
export TORCHINDUCTOR_COMPILE_THREADS=2
IFS=, read -ra GPUs <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPUs[@]} -gt 1 ]]; then
  exec python -u -m torch.distributed.run --standalone --nproc_per_node="${#GPUs[@]}" experiments/b2_locomotion_wtw_repair/train.py "$@"
else
  exec python -u experiments/b2_locomotion_wtw_repair/train.py "$@"
fi

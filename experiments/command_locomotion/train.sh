#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
source deployment/activate.sh
export TORCHINDUCTOR_COMPILE_THREADS=2
if [[ -n "${COMMAND_COLD_CACHE:-}" ]]; then
  export WARP_CACHE_PATH="$COMMAND_COLD_CACHE/warp"
  export TORCHINDUCTOR_CACHE_DIR="$COMMAND_COLD_CACHE/inductor"
  export TRITON_CACHE_DIR="$COMMAND_COLD_CACHE/triton"
fi
IFS=, read -ra GPUs <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#GPUs[@]} -gt 1 ]]; then
  exec python -u -m torch.distributed.run --standalone --nproc_per_node="${#GPUs[@]}" experiments/command_locomotion/train.py "$@"
else
  exec python -u experiments/command_locomotion/train.py "$@"
fi

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
source deployment/activate.sh
if [[ -n "${SPRINT_COLD_CACHE:-}" ]]; then
  export XDG_CACHE_HOME="$SPRINT_COLD_CACHE"
  export WARP_CACHE_PATH="$SPRINT_COLD_CACHE/warp"
  export TORCHINDUCTOR_CACHE_DIR="$SPRINT_COLD_CACHE/torchinductor"
  export TRITON_CACHE_DIR="$SPRINT_COLD_CACHE/triton"
  export CUDA_CACHE_PATH="$SPRINT_COLD_CACHE/cuda"
  mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
fi
ngpu="$($SMIMIC_PY -c 'import torch; print(torch.cuda.device_count())')"
exec "$SMIMIC_PY" -m torch.distributed.run --standalone --nproc_per_node="$ngpu" \
  experiments/sprint/train_sprint.py --motions data/processed/g1_sprint_ladder.npz \
  --lr-sqrt-world false --eval-every 0 "$@"

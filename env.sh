# Source from the repository root before running anything: `source env.sh`.
# Puts the package on PYTHONPATH and keeps every cache inside the repository (.cache/),
# away from small home-directory quotas.
export SMIMIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Interpreter used by the scripts; setup_local_env.sh overrides it inside Slurm jobs.
export SMIMIC_PY="${SMIMIC_PY:-python3}"
export PYTHONPATH="$SMIMIC_ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export XDG_CACHE_HOME=$SMIMIC_ROOT/.cache
export WARP_CACHE_PATH=$SMIMIC_ROOT/.cache/warp
export TORCH_HOME=$SMIMIC_ROOT/.cache/torch
export TRITON_CACHE_DIR=$SMIMIC_ROOT/.cache/triton
export HF_HOME=$SMIMIC_ROOT/.cache/huggingface
export MPLCONFIGDIR=$SMIMIC_ROOT/.cache/matplotlib
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR"

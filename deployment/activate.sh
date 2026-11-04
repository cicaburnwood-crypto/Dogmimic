# Usage: source deployment/activate.sh (from any directory).
export SMIMIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SMIMIC_ROOT/.venv/bin/activate"
export SMIMIC_PY="$SMIMIC_ROOT/.venv/bin/python"
source "$SMIMIC_ROOT/env.sh"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCHINDUCTOR_CACHE_DIR="$SMIMIC_ROOT/.cache/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

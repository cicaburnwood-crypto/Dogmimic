#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export SM_ROBOT=b2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$PWD/experiments/b2_locomotion_support/engine:$PWD/experiments/b2_locomotion_support:$PWD/experiments/three_skills:$PYTHONPATH"
exec python -u "$@"

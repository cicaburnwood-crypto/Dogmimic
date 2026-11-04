#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export SM_ROBOT=b2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source deployment/activate.sh
export PYTHONPATH="$PWD/experiments/three_skills/engine:$PWD/experiments/b2_payload_recovery:$PWD/experiments/b2_cycle:$PWD/experiments/three_skills:$PYTHONPATH"
exec .venv/bin/python -u experiments/b2_cycle/evaluate_payloads.py "$@"

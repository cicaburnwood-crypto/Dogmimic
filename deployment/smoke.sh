#!/usr/bin/env bash
# Full training batch sizes, shortened duration; this does not establish convergence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source deployment/activate.sh
mkdir -p logs data/processed
"$SMIMIC_PY" deployment/verify_runtime.py > logs/runtime.log 2>&1
"$SMIMIC_PY" code/scripts/prepare_motions.py --format lafan --fps-in 30 --input data/raw/lafan1_g1 --output data/processed/lafan1_50hz.npz > logs/prepare_motions.log 2>&1
./deployment/train.sh --out runs/smoke_5090 --iterations 100 --eval-every 0 --ckpt-every 100 --log-every 10 > logs/smoke_train.log 2>&1
"$SMIMIC_PY" deployment/verify_training.py runs/smoke_5090 > logs/training-verification.log 2>&1
"$SMIMIC_PY" -m supermimic.evaluate --run runs/smoke_5090 --out smoke_eval.jsonl > logs/smoke_eval.log 2>&1

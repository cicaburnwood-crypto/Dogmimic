#!/usr/bin/env bash
# Runs the formal SuperMimic LAFAN1 1-GPU experiment inside the dev allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
source setup_local_env.sh
OUT=runs/smimic_lafan1_1gpu
RESUME=()
if [ -f "$OUT/last.pt" ]; then RESUME=(--resume "$OUT/last.pt"); fi
exec $SMIMIC_PY -m supermimic.train --out $OUT --wall-hours 8.5 --eval-every 4000 --ckpt-every 4000 "${RESUME[@]}"

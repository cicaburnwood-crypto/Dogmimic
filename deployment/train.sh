#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 ./deployment/train.sh --out runs/lafan1_5090 --wall-hours 3
# CUDA_VISIBLE_DEVICES=0,1 ./deployment/train.sh --out runs/lafan1_2gpu --wall-hours 3
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source deployment/activate.sh
args=("$@")
out=""
explicit_resume=false
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--out" ]]; then out="${args[$((i+1))]}"; fi
  if [[ "${args[$i]}" == "--out="* ]]; then out="${args[$i]#--out=}"; fi
  if [[ "${args[$i]}" == "--resume" || "${args[$i]}" == "--resume="* ]]; then explicit_resume=true; fi
done
if [[ -z "$out" ]]; then
  echo 'An explicit --out runs/<name> is required.' >&2
  exit 2
fi
if [[ "$explicit_resume" == false && -f "$out/last.pt" ]]; then
  args+=(--resume "$out/last.pt")
fi
ngpu="$($SMIMIC_PY -c 'import torch; print(torch.cuda.device_count())')"
if ((ngpu == 0)); then echo 'No visible CUDA GPU.' >&2; exit 1; fi
if ((ngpu == 1)); then
  exec "$SMIMIC_PY" -m supermimic.train "${args[@]}"
else
  exec "$SMIMIC_PY" -m torch.distributed.run --standalone --nproc_per_node="$ngpu" -m supermimic.train "${args[@]}"
fi

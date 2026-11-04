#!/usr/bin/env bash
# Fig. 3b/c: imagination horizon H x {critic, no critic} on LAFAN1 (H=32+critic is the main run).
# Each run trains for 6 h of wall-clock training time; metric = training return (paper-style).
set -euo pipefail
cd "$(dirname "$0")/.."
for H in 8 16 32 64; do
  for C in true false; do
    if [ "$H" = 32 ] && [ "$C" = true ]; then continue; fi
    name=abl_H${H}_critic_${C}
    sbatch --job-name=abl-H${H}-${C} slurm/train_supermimic.slurm \
      --out runs/$name --horizon $H --use-critic $C --wall-hours 6 --eval-every 8000 --ckpt-every 8000
  done
done

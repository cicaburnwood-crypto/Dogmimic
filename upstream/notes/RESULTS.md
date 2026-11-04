# SuperMimic reproduction — results log

Hardware: NVIDIA RTX A6000 (48 GB) on Princeton Ionic `pci`; the paper used H100 80 GB SXM5.
All hours below are training wall-clock on one A6000 (evaluation excluded, as in the paper;
the first ~6 min include one-time torch.compile).

## Status (2026-10-21, evening)

| Experiment (paper ref.) | Run | Status |
| --- | --- | --- |
| LAFAN1 controlled comparison, SuperMimic 1 GPU (Fig. 3a) | `runs/smimic_lafan1_1gpu` | running (12 h budget) |
| LAFAN1 PPO-Sim 1 GPU (Fig. 3a) | `runs/ppo_lafan1_1gpu` | running (18 h budget) |
| Horizon × critic ablation (Fig. 3b,c) | `runs/abl_H*_critic_*` | H32/no-critic running; H8, H16 ×2, H64 queued |
| Rollout speed (Fig. 4a) | `results/bench_rollout_a6000.json` | done up to 16k envs |
| PPO-WM vs BPTT (Fig. 4b) | `code/supermimic/ppo_wm.py` | implemented, smoke-tested |
| BONES-SEED 129,514 motions vs SONIC (Fig. 2a, Table A.4) | — | **blocked: dataset is license-gated; needs the user's HF token** |
| Dataset-length sweep (Fig. 2b,c) | — | blocked on BONES-SEED |

## Early numbers

Benchmark normalized return on LAFAN1 (40 clips, each from frame 0, no DR/noise):

| | ours (A6000) | paper (H100) |
| --- | --- | --- |
| SuperMimic, first eval | 0.512 @ 0.42 h (it 4000), 13/40 failed | 0.524 @ 0.078 h (~2.7k it) |
| PPO-Sim, first eval | 0.240 @ 0.50 h, 25/40 failed | 0.210 @ 0.48 h |
| SuperMimic, it 8000 | 0.678 @ 0.73 h, 10/40 failed | 0.667 @ 0.23 h |
| SuperMimic, it 12000 | 0.756 @ 1.03 h, 7/40 failed | 0.810 @ 1.04 h |
| **SuperMimic, it 16000** | **0.833 @ 1.34 h, 1/40 failed (target 0.82 reached)** | 0.833 @ 1.19 h (0.82 within 1.2 h) |
| PPO-Sim @ 1.5 h | 0.333, 23/40 failed | ~0.30 @ 1.44 h |
| SuperMimic, it 40000 | **0.910 @ 3.19 h, 0/40 failed** | 0.866 @ 2.99 h (final) |
| PPO-Sim @ 3.5 h | 0.385, 20/40 failed | ~0.42 @ 3.4 h |

Rollout speed (control steps/s, one 50 Hz step = 4 MuJoCo Warp substeps):

| envs | MuJoCo Warp | world model TF32 | ratio | paper ratio (H100) |
| --- | --- | --- | --- | --- |
| 512 | 5.07e4 | 5.53e5 | 10.9× | 9.3× |
| 2048 | 1.09e5 | 1.68e6 | 15.5× | 11× |
| 4096 | 1.26e5 | 1.84e6 | 14.6× | 12.7× |
| 8192 | 1.16e5 | 2.04e6 | 17.6× | 17× |
| 16384 | 9.24e4 | 1.99e6 | 21.5× | 26× |

Note on hardware: MuJoCo Warp on the A6000 is about as fast as the paper's H100 numbers at
4096 envs, while the MLP-heavy SuperMimic update is ~2.7× slower per iteration (0.28 s vs
0.105 s). Wall-clock comparisons on A6000 are therefore tilted toward PPO-Sim.


## Horizon × critic ablation (Fig. 3b,c), LAFAN1 — in progress

Benchmark eval (normalized return, failed clips of 40) at matching iterations; training return
is logged too but, unlike the paper's, stays below eval return (our training episodes start
from BeyondMimic's randomized resets and the hard-bin-weighted sampler).

| H | critic | eval @ ~16k it | best eval so far | paper final training return |
| --- | --- | --- | --- | --- |
| 8 | yes | 0.483 (17 failed) @ 1.00 h | 0.483 | 0.392 (0.82 not reached) |
| 16 | yes | 0.769 (4 failed) @ 1.06 h | 0.769 | 0.843 |
| 16 | no | 0.028 (40 failed) @ 0.99 h | 0.030 | 0.25 (fails) |
| 32 | yes (main run) | 0.833 (1 failed) @ 1.34 h | 0.878 @ 1.96 h | 0.866 |
| 32 | no | 0.576 (21 failed) @ 1.11 h | 0.629 @ 1.67 h | 0.84 |
| 64 | yes | 0.633 (11 failed) @ 1.19 h (8k it) | 0.633 | 0.863 |

So far: without a critic, H = 16 fails completely while H = 16 with a critic learns; the critic
also cuts failures sharply at H = 32; H = 64 costs ~2× more wall-clock per iteration.

Update 2026-10-21 23:56Z: H=8 with critic collapsed (best 0.483, then -0.010, 40/40 failed) and
H=16 without critic never learned (best 0.035, 40/40 failed); both stopped at ~2.3 h to free
shared GPUs. H=16 with critic 0.835 (3 failed) @ 2.47 h; H=32 without critic 0.718 (17 failed)
@ 2.78 h; H=64 with critic 0.799 (3 failed) @ 2.25 h. Audit dashboard: notes/AUDIT_DASHBOARD.md.

Update 2026-10-22 01:28Z: SuperMimic main run best 0.926 (0/40 failed) @ 4.72 h; PPO-Sim 0.440
(17 failed) @ 5.0 h. H=16 critic 0.891 (0 failed) @ 3.88 h; H=32 no critic best 0.899 (3 failed)
@ 3.88 h, i.e. it reaches the critic run's nominal return later (as the paper says, the critic
matters most after perturbations: push test pending); H=64 critic 0.844 @ 3.31 h. Fig. 4b
SuperMimic on walk/run/sprint: 0.902 (1/18 failed) @ 1.43 h; PPO-WM queued behind H=64.

## Push recovery (Sec. 5.3) — 2026-10-22 02:33Z, results/push_test.json

Static T-pose reference (synthetic, not in LAFAN1), horizontal root-velocity push at t = 2 s,
8 directions per magnitude; "settled" = root speed < 0.1 m/s for 0.5 s (our criterion).
Matched benchmark return: critic run ckpt 48000 (0.900, 2 failed) vs no-critic ckpt 72000
(0.898, 3 failed).

| push (m/s) | critic: falls / median settle | no critic: falls / median settle |
| --- | --- | --- |
| 0.5 | 0 / 1.2 s | 0 / 1.3 s |
| 1.0 | 0 / 2.8 s | 0 / 1.5 s |
| 1.5 | 0 / 3.3 s | 0 / 1.8 s |
| 2.0 | 0 / 2.6 s (2 never settled) | 0 / 2.6 s |
| 2.5 | 0 / 2.7 s (2 never settled) | 1 / 3.2 s |
| 3.0 | 0 / 5.9 s (1 never settled) | 3 / 3.8 s |
| 3.5 | 2 (5 never settled) | 7 |
| 4.0 | 4 (4 never settled) | 8 (all) |
| total falls | 6 / 64 | 19 / 64 |

Partly reproduced: the critic policy is far more robust to large pushes and the critic-free one
falls at 4 m/s, as in the paper; but our critic-free policy does settle after small pushes (the
paper says it keeps stumbling even at 0.5 m/s) and our critic policy settles slower than the
paper's 0.5-2.4 s.

Update 2026-10-22 03:18Z: H=32 without critic finished (6 h): final 0.823 (10 failed), best 0.899.
Velocity study started (code/supermimic/velocity.py, code/scripts/step_test.py): runs/vel_rl
(job 31941569) and runs/vel_distill (31941570), 100k iterations each on the LAFAN1
walk/run/sprint clips only (paper: LAFAN1 + BONES-SEED walk/run). Dev allocation released.

Update 2026-10-22 04:52Z: SuperMimic main run best 0.941 (0/40) @ 7.77 h (100k iterations);
PPO-Sim 0.470 (14 failed) @ 8.0 h. H=64 critic final 0.914 (0 failed), H=16 critic final 0.901
(1 failed). Velocity runs learning: velocity-env fall share 0.99 -> ~0.06-0.18 by 1.5 h; motion
interface 0.81 on the 18 walk/run/sprint clips. PPO-WM 16k x 32: training return 0.26 after
7.2e8 imagined steps (SuperMimic on the same clips passed 0.82 training return at 2.38 h).
Dashboard now has the Fig. 4b chart with the paper's curves from its figure data.

Update 2026-10-22 06:55Z: main run 0.944 best (0/40) @ 9.30 h; PPO-Sim 0.482 best @ 9 h.
Fig. 4b SuperMimic (walk/run/sprint) finished: 0.948 (0/18) @ 5 h. PPO-WM 16k x 32 @ 3.4 h:
eval 0.492 (5/18 failed); best smoothed training return 0.286 at 1.04e9 imagined steps, which
SuperMimic reached after 8.4e7 -> 12x fewer so far (paper: 64x at the end of PPO-WM's run).
Velocity runs @ 3.5 h: motion interface eval 0.922 (RL run) / 0.896 (distill run).

## Fig. 4b (PPO-WM vs BPTT), LAFAN1 walk/run/sprint — final 2026-10-22 08:31Z

PPO-WM 16k x 32 (the paper's best PPO-WM batch), 5 h: eval 0.523 (4/18 failed); SuperMimic on the
same clips, 5 h: eval 0.948 (0/18). Best smoothed training return of PPO-WM: 0.333 at 2.7e9 imagined
steps; SuperMimic reached 0.333 after 1.0e8 -> 27x fewer imagined steps. At that same return the
paper's own curves (paper/src/figures/data/fig5b) give 12.7x; the paper's 64x is measured at
return 0.84, which its PPO-WM reached after ~6e10 imagined steps (~100 h of our PPO-WM run, not
attempted). Status: partly reproduced (direction and matched-return gap reproduced, the 64x
endpoint not run).

## Main LAFAN1 run finished — 2026-10-22 09:30Z

SuperMimic, 1 x A6000, 12 h of training (155,680 iterations): final benchmark 0.946, 0/40 clips
failed; best 0.947 @ 11.4 h. Target 0.82 first reached @ 1.34 h (paper: within 1.2 h on an H100);
0.866 (the paper's 3 h value) passed @ 1.65 h. PPO-Sim at the same 12 h: 0.492 (12 failed),
still running to 18 h (paper: 0.60 after 18 h).

Update 2026-10-22 11:42Z: rollout benchmark at 32k/64k envs could not be measured: constructing the
mjlab/MuJoCo Warp simulation for >=32,768 worlds exceeded 120 GB of host RAM and thrashed (two
attempts, with and without per-world DR fields); cancelled. Claim C7 rests on 2k-16k (15-22x).
2-GPU data-parallel run (runs/smimic_lafan1_2gpu): 0.323 s/iteration (1 GPU: 0.28 s), eval 0.895
(1/40 failed) @ 1.93 h vs 0.878 @ 1.96 h on one GPU. H=64 without critic: 0.806 (7 failed) @ 1.89 h.
H=128 with critic runs with expandable segments, 1.15 s/iteration, training return 0.17 @ 1.94 h.
Velocity runs @ 8.1 h: motion interface 0.95 on the walk/run/sprint clips; step test at the end.
PPO-Sim @ 14.5 h: 0.522 (9 failed).

## Velocity step test (App. G) — smoke run 2026-10-22 14:18Z, results/step_test_smoke.json

672 held commands (15 x 15 grids on the (vx,vy), (vx,wz), (vy,wz) planes, standing excluded) x 4
trials from standing LAFAN1 frames, 5 s each; relative steady-state error over the last 2 s, median
over non-fallen trials. RL velocity interface at ckpt 88000 (not yet final) vs distilled interface at
100000. Both runs trained on LAFAN1 walk/run/sprint only (paper: LAFAN1 + BONES-SEED walk/run).

| | ours (smoke) | paper |
| --- | --- | --- |
| RL better / distill better (of 672) | 659 / 13 | 650 / 15 |
| median rel. error RL / distill | 0.043 / 0.223 | 0.16 / 0.49 |
| commands where distill fell in all trials | 0 | 7 |

Per plane (median rel. error, RL / distill): vx-vy 0.051 / 0.344, vx-wz 0.039 / 0.154,
vy-wz 0.040 / 0.237. Direction reproduced (RL velocity interface tracks far better); our absolute
errors are lower than the paper's for both, and our distilled interface never falls in all trials.
Final run queued as job 31946447 (after the RL run reaches 100k iterations).

# SuperMimic reproduction — implementation notes

Paper: *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*
(Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath; arXiv 2610.28480, CoRL 2026). No code
was released; everything here is re-implemented from the paper's text and its TeX source.

## Layout

| Path | What |
| --- | --- |
| `code/supermimic/` | implementation (package) |
| `code/scripts/` | data prep, evaluation tests (push, step), benchmarks, status, audit export |
| `code/scripts/dev/` | sanity checks and profilers used while building it |
| `slurm/` | batch scripts (logs go to `slurm/logs/`) |
| `results/` | small result files (rollout benchmark, push/step tests, audit tables) |
| `env.sh`, `setup_local_env.sh`, `requirements.txt` | environment (node-local uv venv in jobs) |
| `paper/` (not in git) | PDF + TeX source, read-only reference |
| `data/raw/lafan1_g1/` (not in git) | 40 LAFAN1 G1 CSVs (lvhaidong/LAFAN1_Retargeting_Dataset) |
| `data/processed/lafan1_50hz.npz` (not in git) | motion library: 40 clips, 441,120 frames, 2.451 h (paper: 40 clips, 2.45 h) |
| `runs/<name>/` (not in git) | `log.jsonl` (training), `eval.jsonl` (benchmark protocol), checkpoints |

## What follows the paper exactly

- Simulator: mjlab 1.2 / MuJoCo Warp 3.5 G1 (29 DoF), 5 ms physics, 4 substeps, 50 Hz control,
  BeyondMimic actuators (low PD gains), action `q_sp = alpha * a + q0`.
- Observation (Table A.1): q - q0 (+ encoder bias), qd, IMU gyro, projected gravity, previous
  action; noise U(±0.01, ±0.5, ±0.2, ±0.05); history of 10.
- Reward (Table A.2): BeyondMimic tracking terms (std 0.3/0.4/1.0/3.14, root 0.5 weights),
  action-rate -0.1, joint-limit -2, no self-collision term, all × dt = 0.02.
- Termination: |root z error| > 0.25 m or tilt error > 0.8; no end-effector termination.
- Sim context xi (Table A.5): foot friction U[0.3, 1.2], joint offsets U[-0.01, 0.01], torso COM
  offset; no push perturbations.
- Networks (Table A.3): dynamics MLP (2048, 2048, 2048); motion encoder (2048, 1024, 512, 256) on
  K = 10 future reference frames -> 32-d tanh latent; decoder (2048, 1024, 512, 256) on latent +
  10-frame obs history; critic (256, 128).
- Training (Table A.4): k = 8 autoregressive windows, 10 × 2048 windows per world-model update,
  H = 32, 4096 imagined rollouts, Adam lr 2e-4 (policy betas 0.9/0.95), clip 1.0 then 0.1
  element-wise, gamma 0.99, lambda 0.95, critic 5 epochs × 4 minibatches, 4096 envs, action noise
  N(0, 0.1²), 1000-step replay per env, UTD 0.4, cosine LR to 0.03× over 250k iterations.
- Return-based reset sampling over 1 s bins: 50% uniform, 40% softmax(-25 G_b / G_max), 10% past
  softmax weight.
- Benchmark protocol (App. F): each clip once from frame 0, friction 1.0, no DR / noise,
  deterministic; fail = training termination; normalized return = reward sum / max possible.

## Choices the paper leaves open (ours)

- **State** = qpos/qvel of the floating base + joints (71 numbers). Reward, observation and
  termination heads are *analytic* (paper allows "learned or computed analytically"), computed
  with our differentiable forward kinematics (`kinematics.py`, matches MuJoCo to 1e-6). So the
  world-model loss reduces to the state term of Eq. 1.
- **World-model parameterization**: inputs are heading-frame features (root height, tilt, local
  velocities, joints, 29 body positions/velocities from FK) + action + xi; the MLP predicts the
  normalized one-step change (root displacement in heading frame, rotation vector, velocity
  changes, joint deltas), integrated analytically. Loss = per-dimension MSE normalized by the
  running std of one-step deltas, summed over the 8 window steps.
- **Termination in imagination**: soft sigmoid of the two analytic thresholds (temperatures
  0.02 m and 0.04), giving the differentiable survival weight.
- **Goal features** per future frame: reference root position and rotation relative to the
  robot root (position needs state estimation, like BeyondMimic's default variant), reference
  joint positions/velocities, reference root velocities in the reference frame (73 × 10).
- **Clip ends** inside imagination are treated as the end of the task (no bootstrap); the critic
  gets the remaining clip time as an input.
- **Iteration / UTD**: one iteration = one simulator step of all 4096 envs; on average 0.4
  updates per iteration (2 every 5), each update = 10 world-model gradient steps + 1 policy step +
  critic fit. (The paper's wording admits other readings; this one matches its 0.105 s/iteration
  on an H100.)
- **Reference/robot anchor** = pelvis (paper text: "root (pelvis)"; mjlab's BeyondMimic uses torso).
- **set_const** after COM randomization is batched every 50 steps (it costs 0.1 s per call for
  4096 worlds); the COM shift itself acts immediately.
- **Precision**: A6000 instead of H100; MLP matmuls in bf16 autocast (paper: TF32) for ~2× speed;
  kinematics, integration and losses in fp32.

## Baseline PPO-Sim

`code/supermimic/ppo.py`: same env/obs/reward/termination/DR/network sizes/eval; BeyondMimic PPO
hyperparameters (lr 1e-3 adaptive KL 0.01, clip 0.2, entropy 0.005, 5 epochs × 4 minibatches,
24 steps/env), 8192 envs per GPU, Gaussian policy with learned per-joint std (init 1.0).

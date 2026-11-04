# Three skill experiments on super_5090

GPU 4: G1 Funky Chicken inspired original choreography.
GPU 5: G1 vertical jump and 360 degree turn, including landing recovery.
GPU 6: Unitree B2 transition from four feet to a rear-leg stand at 82 degrees, then hold.

These experiments test the existing community SuperMimic reproduction, based on
`dhmbb2/supermimic_vibe` commit `072f0910dadc9d95c720e625b546120e9ce917c9`.
They use its deterministic learned world model, analytic reward and termination,
imagined differentiable rollouts and BPTT policy update. No PPO substitution.
All policy and world-model parameters start randomly, without previous checkpoints.

The references are newly authored kinematic targets with explicit provenance in
`targets/*.json`. They are not recordings, extractions or exact reconstructions of
the official SuperMimic demonstration. The official browser demo is restricted to
browser evaluation, and BONES-SEED requires access approval; neither source was
used to obtain training trajectories. The user's clarification authorizes training
these skills to test our reproduction. This is reference-based motion tracking,
distinct from the earlier reference-free start/stop locomotion experiment.

The B2 model comes from Unitree's BSD-3-Clause `unitree_mujoco`, commit
`1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`, `unitree_robots/b2/b2.xml`.
The original license is retained under `vendor/unitree_mujoco/LICENSE`.
Derived training XML adds a floor and position PD actuators (kp 250, kd 8), retaining
the original per-actuator torque bounds and joint limits. Sensor and home-keyframe
blocks are omitted in the derived scene. The underlying robot is B2, not Go2 or B2-W.

An isolated copy under `engine/supermimic` adapts state/action dimensions to B2's
12 joints, and uses the lateral body axis for heading to avoid the vertical forward
axis singularity in rear standing. G1 remains 29 joints. The deployed base code and
previous experiments are not modified.

Each job uses one GPU, 2,048 physics environments, a three-layer 2,048-unit world
model, a 512/256/128 actor, horizon 32, 4 WM batches per update and 0.4 updates per
physics iteration. First-stage limit: 50,000 iterations or three hours total wall time.
Checkpoint and independent 32-trial evaluation every 2,000 iterations. Save best
and last separately. Stop early only after two evaluations with at least 90% success
including the nominal initial state. A successful launch is not a successful skill.

Training resets mix 40% clip-start states with sampled reference phases. Evaluation
always starts at frame zero, at rest, with a nominal trial and 31 perturbed trials.
Evaluation criteria:

- Dance: whole clip without a fall and joint RMS tracking error below 0.22 rad.
- Jump: full clip, accumulated yaw within 30 degrees of 360, both feet airborne
  above 9 cm for over 0.2 s, and settled root speed below 0.3 m/s at the end.
- B2: full clip; during the last five seconds, at least 95% of frames above 70
  degrees, front feet above 0.3 m and both hind feet below 0.09 m.

`audit.py` checks joint/actuator order, FK versus MuJoCo, state flattening,
finite real physics transitions, and gradient flow through WM into the fresh actor.
`latest_launch.json` records PIDs, physical GPU IDs, paths, hashes and launch times.
Training metadata and final timing distinguish total wall time from update time.

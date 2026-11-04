Frozen B2 C1 actor 2000 sim2sim diagnostics (2026-11-03)

No policy updates, reward edits, robot connections or firmware-protection changes.
Selected export/checkpoint hashes are recorded in inputs/hindstand/policy.json.
Inputs/runtime.py and inputs/pd_adapter.py are frozen copies of the live deployment
used during the failed trial. The controller is imported without DDS or a server.

Execution environment: SSH super_5090, project /home/liudiwen/projects/supermimic_vibe.
Existing .venv: MuJoCo 3.5.0, MuJoCo Warp 3.5.0, mjlab 1.2.0, Python 3.11.
PyBullet 3.2.7 is isolated under experiments/b2_sim2sim/deps; shared .venv unchanged.
The frozen XML files currently reference the existing remote Unitree mesh directory.

Preflight:
.venv/bin/python experiments/b2_sim2sim/audit.py --out <new-output>/audit.json

Each independent backend, with a fresh output directory:
.venv/bin/python experiments/b2_sim2sim/run.py --backend mujoco --out <new-output> --trials 4
CUDA_VISIBLE_DEVICES=6 .venv/bin/python experiments/b2_sim2sim/run.py --backend mjlab --out <new-output> --trials 4
.venv/bin/python experiments/b2_sim2sim/run.py --backend bullet --out <new-output> --trials 4

This is a diagnostic subset: 2 loads x 7 conditions x 4 fixed trials per backend.
Trial 0 is fixed before execution; trials 1-3 add fixed 0.006-rad joint perturbations.
Neither successful videos nor this reduced matrix constitute full acceptance.
Original success criteria and 6 rad/s / 1501 W experimental acceptance are retained.

Valid recorded outputs:
- diagnostics/b2_sim2sim_20261103/paired_v1/{mujoco,mjlab}
- diagnostics/b2_sim2sim_20261103/bullet_com_corrected/bullet
- diagnostics/b2_sim2sim_20261103/sensitivity (separate 5-ms / no-armature controls)

Do not use smoke, smoke_v2 or paired_v1/bullet as final evidence. They are preserved
development/preliminary outputs. Frozen source hashes accompany the valid runs.

Important comparability limits:
MuJoCo CPU and mjlab/Warp are two numerical implementations in one physics family.
Bullet uses transferred rigid inertias/geometry but lacks generalized armature;
contact compliance and implicit vs explicit PD integration are not equivalent.
Rigid joint mass-matrix parity after removing armature was verified independently.
The original actuator order differs from policy joint order; adapters map both
control inputs and force outputs explicitly, verified by the audit.

Measured initial q/dq/IMU are from attempt 1793687162374260969. Root height is floor
aligned and root linear velocity is zero because trustworthy measurements were
unavailable. Simulation supplies ideal proprioception and root position. Unknown
manufacturer bit_flag 0x20 logic, driver faults and electrical transients are not
modelled. Passing simulation would not explain or disable that protection.

Render saved trajectories without stepping physics:
CUDA_VISIBLE_DEVICES=6 MUJOCO_EGL_DEVICE_ID=6 MUJOCO_GL=egl .venv/bin/python experiments/b2_sim2sim/render.py --folder diagnostics/b2_sim2sim_20261103/paired_v1 --bullet-folder diagnostics/b2_sim2sim_20261103/bullet_com_corrected

The common MuJoCo renderer only displays each backend's own saved physical states.
Videos include complete trial 0 trajectories, including failures, with final scores.

Primary documentation:
https://github.com/mujocolab/mjlab
https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
https://github.com/bulletphysics/bullet3

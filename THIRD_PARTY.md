# Third-party sources

This repository is a snapshot of a local robotics research workspace. Existing copyright
notices and license texts are preserved. The Dogmimic name does not imply authorship of
third-party code or endorsement by the original projects.

| Component | Source / pinned revision | Location and license |
| --- | --- | --- |
| SuperMimic-vibe | https://github.com/dhmbb2/supermimic_vibe — `072f0910dadc9d95c720e625b546120e9ce917c9` | `upstream/`, root `code/`, and task-local `engine/` copies. No top-level LICENSE was present in the pinned upstream checkout; this snapshot does not invent or replace its licensing terms. Task-specific edits are described by existing provenance files. |
| NJU SuperMimic-Repro-Sim2Real | https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real — `f8f60d7bbdba68468c050ef100af02d752ca238e` | `vendor/nju_supermimic/`; MIT and the additional notices in its `LICENSES/` directory. Historical comparison backend. |
| Unitree SDK2 Python | https://github.com/unitreerobotics/unitree_sdk2_python — `814556d15970dd2ecf1c9984e845ca02ab07e206` | `deployments/*/vendor/unitree_sdk2_python-*`; BSD 3-Clause. Binary libraries and recorded audio are omitted. |
| Unitree MuJoCo robot descriptions | https://github.com/unitreerobotics/unitree_mujoco | `experiments/three_skills/vendor/unitree_mujoco/`; BSD 3-Clause. |
| Walk These Ways | https://github.com/Improbable-AI/walk-these-ways | `experiments/b2_locomotion_wtw/vendor/` and `experiments/b2_locomotion_wtw_repair/vendor/`; MIT with additional bundled notices. |

Training datasets, downloaded motion CSVs, learned checkpoint weights, videos, raw robot
telemetry, and local authentication data are not redistributed in this code snapshot.
No repository-wide open-source license is added by this upload.

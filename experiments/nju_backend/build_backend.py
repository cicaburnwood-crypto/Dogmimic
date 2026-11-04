"""Materialize robot-dimension adapters over a pinned, untouched NJU checkout.

No optimizer, replay, rollout objective, lambda return or reward formula changes.
G1 remains the default 29-joint implementation. B2 selects 12 real joints.
"""
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = ROOT / 'vendor/nju_supermimic/src/supermimic'
if not UPSTREAM.exists():
    UPSTREAM = Path('/home/liudiwen/projects/supermimic_nju/src/supermimic')
DEST = HERE / 'backend_src/supermimic'
PIN = 'f8f60d7bbdba68468c050ef100af02d752ca238e'


def main():
    shutil.copytree(UPSTREAM, DEST, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    changed = {}
    def edit(rel, pairs, imports=''):
        p = DEST / rel
        old = p.read_text(); s = old
        for a, b in pairs:
            assert a in s, (rel, a)
            s = s.replace(a, b)
        if imports:
            s = s.replace('from __future__ import annotations', 'from __future__ import annotations\n' + imports)
        ast.parse(s); p.write_text(s)
        changed[rel] = ''.join(difflib.unified_diff(old.splitlines(True), s.splitlines(True), fromfile=rel, tofile=rel))
    imp = 'from supermimic.robot_layout import NJ, NQ, NV, SD, DD, GF, OD, XD, ROOT_NAME'
    (DEST/'robot_layout.py').write_text('''"""Task dimensions only; selected before importing the NJU training backend."""
import os
ROBOT = os.environ.get("SM_NJU_ROBOT", "G1")
assert ROBOT in ("B2", "G1")
NJ = 12 if ROBOT == "B2" else 29
NQ, NV = 7 + NJ, 6 + NJ
SD, DD, GF, OD = 13 + 2 * NJ, 12 + 2 * NJ, 10 + 2 * NJ, 6 + 3 * NJ
XD = NJ + 4 + (2 if ROBOT == "B2" else 0)
ROOT_NAME = "base_link" if ROBOT == "B2" else "pelvis"
''')
    state_slices = [('7:36', '7:NQ'), ('36:39', 'NQ:NQ+3'), ('39:42', 'NQ+3:NQ+6'), ('42:71', 'NQ+6:SD')]
    edit('world_model/state_spec.py', [
        ('STATE_DIM = 71', 'STATE_DIM = SD'), ('FEAT_DIM = 71', 'FEAT_DIM = SD'), ('DELTA_DIM = 70', 'DELTA_DIM = DD'), ('NJ = 29', '# NJ imported from task layout'),
        ('(0.75,) + (0.0,) * 29 + (0.0,) * 3', '(0.75,) + (0.0,) * NJ + (0.0,) * 3 + ((0.5, 1.0) if NJ == 12 else ())'),
        ('(0.45,) + (0.01,) * 29 + (0.05,) * 3', '(0.45,) + (0.01,) * NJ + (0.05,) * 3 + ((0.5, 0.1) if NJ == 12 else ())'),
        *state_slices, ('6:35', '6:6+NJ'), ('35:38', '6+NJ:9+NJ'), ('38:41', '9+NJ:12+NJ'), ('41:70', '12+NJ:DD'),
    ], imp)
    edit('world_model/dynamics.py', [('feat_dim: int = 71', 'feat_dim: int = SD'), ('act_dim: int = 29', 'act_dim: int = NJ'), ('xi_dim: int = 33', 'xi_dim: int = XD'), ('out_dim: int = 70', 'out_dim: int = DD')], imp)
    edit('world_model/heads.py', [('OBS_DIM = 93', 'OBS_DIM = OD'), ('"joint_pos", 29','"joint_pos", NJ'), ('"joint_vel", 29','"joint_vel", NJ'), ('"actions", 29','"actions", NJ'), *state_slices, ('1:30', '1:1+NJ'), ('s[..., :36], s[..., 36:]', 's[..., :NQ], s[..., NQ:]')], imp)
    edit('policy/nets.py', [('GOAL_DIM = 680','GOAL_DIM = 10 * GF'), ('ACT_DIM = 29','ACT_DIM = NJ'), ('sizes = (29, 29, 3, 3, 3, 1)','sizes = (NJ, NJ, 3, 3, 3, 1)'), ('divmod(goal_dim, 68)','divmod(goal_dim, GF)'), ('obs_dim: int = 930','obs_dim: int = 10 * OD'), ('feat_dim: int = 71','feat_dim: int = SD'), ('xi_dim: int = 33','xi_dim: int = XD')], imp)
    edit('envs/mdp/motion_library.py', [('GOAL_FEAT_DIM = 68','GOAL_FEAT_DIM = GF'), ('REF_DIM = 71','REF_DIM = SD'), ('names.index("pelvis")','names.index(ROOT_NAME)'), *state_slices, ('42 + j','NQ + 6 + j')], imp)
    edit('envs/state_io.py', [('STATE_DIM = 36 + 35','STATE_DIM = SD'), ('NQ = 36\nNV = 35','# NQ and NV imported from task layout')], imp)
    edit('envs/sim_context.py', [('XI_DIM = 33','XI_DIM = XD'), ('!= 29', '!= NJ'), ('xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)', 'xi = torch.cat([friction.reshape(n, 1), bias, com], dim=-1)\n        if NJ == 12:\n            xi = torch.cat([xi, self.env.b2_load[:, None] / 5., self.env.b2_mass_scale[:, None]], dim=-1)')], imp)
    edit('data/recorder.py', [('STATE_DIM, 29, XI_DIM','STATE_DIM, NJ, XI_DIM'), ('env_ids.numel(), 29','env_ids.numel(), NJ')], imp)
    edit('imagination/rollout.py', [('* 29','* NJ'), ('noise_half[29 + j]', 'noise_half[NJ + j]'), ('39:42', 'NQ+3:NQ+6')], imp)
    edit('envs/lock_waist.py', [('if not is_locked_entity(robot):','if len(robot.joint_names) == 12 or not is_locked_entity(robot):')])
    edit('training/loop.py', [('cfg.env.future_frames * 68','cfg.env.future_frames * GF'), ('fk = G1Kinematics(load_g1_model(), device=self.device)','fk = G1Kinematics(load_task_model(), device=self.device)')], imp + '\nfrom task_adapter import load_task_model')
    edit('envs/tracking_env_cfg.py', [
        ('    # ---------------------------------------------------------------- observations','    # ---------------------------------------------------------------- observations'),
        ('    return cfg\n\n\ndef robot_cfg', '    from task_adapter import adapt_env_cfg\n    return adapt_env_cfg(cfg, a)\n\n\ndef robot_cfg'),
        ('FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"', 'FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"\nif ROOT_NAME == "base_link":\n    TRACKED_BODIES = ("base_link", "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf")\n    ANCHOR_BODY = ROOT_BODY = ROOT_NAME\n    FOOT_GEOM_REGEX = r"^(FL|FR|RL|RR)_foot_collision$"'),
    ], imp)
    # This hook only exposes graceful STOP and wall-time management, never changes an update.
    edit('training/loop.py', [('        while self.iteration < sch.total_iterations:', '        while self.iteration < sch.total_iterations:\n            if (self.log_dir / "STOP").exists():\n                if self.is_main:\n                    print("[INFO] requested STOP", flush=True)\n                break')])
    # Learning equations and reward formulas must remain byte-identical.
    unchanged = ['world_model/world_model.py', 'world_model/wm_trainer.py', 'imagination/policy_trainer.py', 'imagination/objective.py', 'imagination/compiled.py', 'data/replay_buffer.py', 'envs/mdp/rewards.py', 'envs/mdp/commands.py', 'kinematics/g1_fk.py', 'training/schedule.py']
    hashes = {}
    for f in unchanged:
        a = (UPSTREAM/f).read_bytes(); b = (DEST/f).read_bytes()
        assert a == b, f
        hashes[f] = hashlib.sha256(a).hexdigest()
    (HERE/'backend_adaptation.diff').write_text('\n'.join(''.join(difflib.unified_diff((UPSTREAM/f).read_text().splitlines(True), (DEST/f).read_text().splitlines(True), fromfile=f, tofile=f)) for f in changed))
    (HERE/'backend_provenance.json').write_text(json.dumps(dict(upstream_commit=PIN, unchanged_learning_files=hashes, modified_files=list(changed), initialization='random; no checkpoint or teacher'), indent=2)+'\n')
    print(json.dumps({'backend':str(DEST), 'unchanged_learning_files':len(hashes), 'adapted_files':len(changed)}))

if __name__ == '__main__':
    main()

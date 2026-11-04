"""Convert the existing qpos/qvel trajectories without re-timing or editing them."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from task_adapter import HERE, LEGACY, NJ, load_task_model, prepare_b2_asset
from supermimic.kinematics.g1_fk import G1Kinematics


def main():
    robot = 'B2' if NJ==12 else 'G1'
    if NJ==12: prepare_b2_asset()
    source = LEGACY/('experiments/b2_cycle/b2_full_cycle.npz' if NJ==12 else 'data/processed/g1_standing_jump_060m.npz')
    expected = '2cd20993005a089c0f4397c44a735e3abe94309042a882620470abb913a9aad0' if NJ==12 else 'ff50976a079d42b9f2419646de0f9893c63f12436516fb2026b1d8f1ab08a004'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); assert digest==expected
    d = np.load(source); m = load_task_model(); fk = G1Kinematics(m)
    assert d['fps']==50 and d['qpos'].shape[1]==m.nq and d['qvel'].shape[1]==m.nv
    q = torch.from_numpy(d['qpos']); v = torch.from_numpy(d['qvel'])
    out = fk(q,v)
    # Check the generic torch FK against MuJoCo, including link-frame velocities.
    md = mujoco.MjData(m); max_errors = np.zeros(4)
    for i in np.linspace(0,len(q)-1,25,dtype=int):
        md.qpos[:]=d['qpos'][i]; md.qvel[:]=d['qvel'][i]; mujoco.mj_forward(m,md)
        vel = np.zeros((m.nbody-1,6))
        for j in range(1,m.nbody): mujoco.mj_objectVelocity(m,md,mujoco.mjtObj.mjOBJ_XBODY,j,vel[j-1],0)
        errs = [np.max(abs(out['pos'][i].numpy()-md.xpos[1:])),
                np.max(1-np.abs((out['quat'][i].numpy()*md.xquat[1:]).sum(-1))),
                np.max(abs(out['lin_vel'][i].numpy()-vel[:,3:])), np.max(abs(out['ang_vel'][i].numpy()-vel[:,:3]))]
        max_errors = np.maximum(max_errors,errs)
    assert max_errors.max()<2e-4, max_errors
    dest=HERE/'data'/robot;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'motion.npz', fps=np.array(50.), joint_pos=d['qpos'][:,7:],joint_vel=d['qvel'][:,6:],
        body_names=np.array(fk.body_names),joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]),
        body_pos_w=out['pos'].numpy(),body_quat_w=out['quat'].numpy(),body_lin_vel_w=out['lin_vel'].numpy(),body_ang_vel_w=out['ang_vel'].numpy())
    info=dict(robot=robot,source=str(source),source_sha256=digest,frames=len(q),seconds=(len(q)-1)/50,unchanged_qpos_qvel=True,
              fk_max_errors=max_errors.tolist(),converted_sha256=hashlib.sha256((dest/'motion.npz').read_bytes()).hexdigest())
    (dest/'provenance.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))

if __name__=='__main__':main()

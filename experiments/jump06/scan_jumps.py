import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.asset_zoo.robots import get_g1_robot_cfg

spec=get_g1_robot_cfg().spec_fn()
m=spec.compile(); d=mujoco.MjData(m)
assert m.nq==36
feet=[[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot') and 'collision' in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)] for side in ('left','right')]
print('feet',[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),int(m.geom_type[i]),m.geom_size[i].tolist()) for i in feet[0]])
raw=np.load('data/processed/lafan1_50hz.npz')
records=[]
for ci,name in enumerate(raw['names']):
    if not str(name).startswith('jumps'): continue
    start=int(raw['starts'][ci]); n=int(raw['lengths'][ci]); q=raw['qpos'][start:start+n]; v=raw['qvel'][start:start+n]
    h=np.zeros((n,2)); xy=np.zeros((n,2,2))
    for t in range(n):
        d.qpos[:]=q[t]; mujoco.mj_kinematics(m,d)
        for side,ids in enumerate(feet):
            # Exact capsule-to-horizontal-plane clearance.
            assert all(m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE for i in ids)
            h[t,side]=min(d.geom_xpos[i,2]-m.geom_size[i,0]-m.geom_size[i,1]*abs(d.geom_xmat[i].reshape(3,3)[2,2]) for i in ids)
            xy[t,side]=d.geom_xpos[ids,:2].mean(axis=0)
    air=(h.min(axis=1)>.015)
    edges=np.diff(np.r_[False,air,False].astype(int)); starts=np.where(edges==1)[0]; ends=np.where(edges==-1)[0]
    for a,b in zip(starts,ends):
        if b-a<4 or a<50 or b+50>=n: continue
        take=max(0,a-1); land=min(n-1,b)
        disp=xy[land].mean(axis=0)-xy[take].mean(axis=0)
        speed=np.linalg.norm(v[max(0,a-50):a-20,:2],axis=1)
        pre_shift=np.linalg.norm(xy[a-20].mean(axis=0)-xy[a-50].mean(axis=0))
        rec=dict(clip=str(name),clip_id=ci,takeoff=int(a),landing=int(b),airtime=(b-a)/50,foot_distance=float(np.linalg.norm(disp)),direction=disp.tolist(),root_distance=float(np.linalg.norm(q[land,:2]-q[take,:2])),pre_speed=float(np.mean(speed)),pre_foot_shift=float(pre_shift),height=float(h[a:b].min(axis=1).max()),end_speed=float(np.linalg.norm(v[b+40:b+50,:2],axis=1).mean()))
        records.append(rec)
    np.savez(f'experiments/jump06/{Path(str(name)).stem}_scan.npz',qpos=q,qvel=v,foot_height=h,foot_xy=xy)
Path('experiments/jump06/candidates.json').write_text(json.dumps(records,indent=2)+'\n')
for r in sorted(records,key=lambda r:abs(r['foot_distance']-.6)+r['pre_foot_shift']*2)[:25]: print(json.dumps(r))

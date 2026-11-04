"""Periodic straight sprint references with an explicit cadence/speed ladder."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from mjlab.asset_zoo.robots import get_g1_robot_cfg

raw=np.load('data/processed/lafan1_50hz.npz')
name='sprint1_subject2.csv';a=5964;period_frames=28
i=list(raw['names']).index(name);start=int(raw['starts'][i])+a
q=raw['qpos'][start:start+period_frames+1].astype(float).copy()
delta=q[-1,:2]-q[0,:2];heading=np.arctan2(delta[1],delta[0])
align=Rotation.from_euler('z',-heading)
q[:,:3]=align.apply(q[:,:3]-np.r_[q[0,:2],0.])
rot=align*Rotation.from_quat(q[:,[4,5,6,3]])
u=np.linspace(0,1,len(q));stride=q[-1,0]-q[0,0]
features=np.c_[q[:,:3]-u[:,None]*np.r_[q[-1,:2]-q[0,:2],0.],rot.as_rotvec(),q[:,7:]]
features[[0,-1]]=(features[0]+features[-1])/2
spline=CubicSpline(u,features,bc_type='periodic')
speeds=[2.,3.,4.,5.,6.,7.,8.]
t=np.arange(601)/50
m=get_g1_robot_cfg().spec_fn().compile();d=mujoco.MjData(m)
feet=[[g for g in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith(side+'_foot') and 'collision' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')] for side in ('left','right')]
poses=[];vels=[];meta=[]
for speed in speeds:
    phase=(t*speed/stride)%1
    f=spline(phase);r=Rotation.from_rotvec(f[:,3:6])
    pos=f[:,:3].copy();pos[:,0]+=speed*t
    qp=np.c_[pos,r.as_quat()[:,[3,0,1,2]],f[:,6:]]
    heights=[]
    for row in qp:
        d.qpos[:]=row;mujoco.mj_kinematics(m,d)
        heights.append([min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2]) for g in ids) for ids in feet])
    heights=np.array(heights)
    shift=max(0.,-.002-float(heights.min()));qp[:,2]+=shift
    vel=np.zeros((len(t),35));vel[:,:3]=np.gradient(qp[:,:3],.02,axis=0)
    vel[1:-1,3:6]=(r[2:]*r[:-2].inv()).as_rotvec()/.04
    vel[0,3:6]=vel[1,3:6];vel[-1,3:6]=vel[-2,3:6]
    vel[:,6:]=np.gradient(qp[:,7:],.02,axis=0)
    poses.append(qp);vels.append(vel)
    meta.append(dict(speed_m_s=speed,stride_period_s=stride/speed,root_height_range_m=[float(qp[:,2].min()),float(qp[:,2].max())],max_joint_speed_rad_s=float(np.max(np.abs(vel[:,6:]))),max_both_foot_clearance_m=float((heights+shift).min(1).max()),floor_shift_m=shift))
out=Path('data/processed/g1_sprint_ladder.npz')
np.savez(out,qpos=np.concatenate(poses).astype(np.float32),qvel=np.concatenate(vels).astype(np.float32),starts=np.arange(len(speeds))*len(t),lengths=np.full(len(speeds),len(t)),names=np.array([f'sprint_{s:.1f}mps' for s in speeds]),fps=np.array(50.))
record=dict(source=f'lvhaidong/LAFAN1_Retargeting_Dataset:g1/{name}',source_frames_50hz=[a,a+period_frames],stride_length_m=float(stride),duration_s=12.,speeds=meta,method='Periodic cubic spline of one aligned sprint cycle; cadence scales with speed; model and actuator limits unchanged.')
Path('experiments/sprint/reference.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

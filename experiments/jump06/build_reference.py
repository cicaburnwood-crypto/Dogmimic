"""One standing broad jump, adapted from a public LAFAN1 G1 motion.
Foot-center forward displacement is exactly 0.6 m; stance feet are fixed by IK.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from mjlab.asset_zoo.robots import get_g1_robot_cfg

m=get_g1_robot_cfg().spec_fn().compile(); d=mujoco.MjData(m)
raw=np.load('experiments/jump06/jumps1_subject1_scan.npz')
a,b=353,368
first,last=a-35,b+6
q=raw['qpos'][first:last+1].astype(float).copy()
foot_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
geom_ids=[[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(s+'_foot') and 'collision' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '')] for s in ('left','right')]
leg_joints=[[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,s+'_'+j+'_joint') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] for s in ('left','right')]
leg_adr=[m.jnt_qposadr[js] for js in leg_joints]

def fk(qp):
    d.qpos[:]=qp; mujoco.mj_kinematics(m,d)
    return d.xpos[foot_ids].copy(),d.xmat[foot_ids].reshape(2,3,3).copy()
def foot_centers(qp):
    fk(qp); return np.array([d.geom_xpos[g,:2].mean(axis=0) for g in geom_ids])
def clearance(qp):
    fk(qp);return np.array([min(d.geom_xpos[i,2]-m.geom_size[i,0]-m.geom_size[i,1]*abs(d.geom_xmat[i].reshape(3,3)[2,2]) for i in g) for g in geom_ids])

# Align the first-to-last foot displacement with +x and place initial feet around x=y=0.
origin=foot_centers(q[0]).mean(axis=0)
direction=foot_centers(q[-1]).mean(axis=0)-origin
yaw=np.arctan2(direction[1],direction[0]); rot=Rotation.from_euler('z',-yaw)
q[:,:3]=(rot.apply(q[:,:3]-np.r_[origin,0]))
q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
# Scale absolute forward trajectories and solve the legs to preserve foot targets.
centers0=foot_centers(q[0]).mean(axis=0); centers1=foot_centers(q[-1]).mean(axis=0)
scale=.6/(centers1[0]-centers0[0])
footP=[];footR=[]
for qp in q:
    p,r=fk(qp);footP.append(p);footR.append(r)
footP=np.array(footP);footR=np.array(footR)
q[:,0]*=scale;footP[:,:,0]*=scale
# Leave knee flexion margin after leveling the soles.
q[:,2]=np.maximum(q[:,2]-.035,.68)
# Use level, parallel feet throughout the planted phases; keep recorded orientation in air.
identity=np.eye(3)
startP=footP[0].copy(); endP=footP[-1].copy()
# A 0.6 m translation of the initial stance gives identical foot separation and final balance.
endP=startP+np.array([.6,0,0])
# Calculate exact ankle height required by a level collision sole.
for side in range(2):
    # Geometry positions in ankle frame are fixed; sole bottom determines the ankle height.
    lowest=min(m.geom_pos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix()[2,2]) for g in geom_ids[side])
    # Feet capsules lie horizontally in the level ankle frame.
    startP[side,2]=endP[side,2]=-lowest-.001

# Contact boundaries from the source (toe-off can precede both-foot clearance by a few frames).
take=a-first-2;land=b-first+1
for t in range(len(q)):
    if t<=take:
        footP[t]=startP;footR[t]=identity
    elif t>=land:
        footP[t]=endP;footR[t]=identity
    else:
        u=(t-take)/(land-take)
        # Smooth swing from fixed departure stance to fixed landing stance.
        u3=u*u*(3-2*u)
        footP[t]=(1-u3)*startP+u3*endP
        footP[t,:,2]+=0.07*np.sin(np.pi*u)
        footR[t]=identity

# Replace the post-landing continuation by a balanced recovery and 1.2 s hold.
qstart=q[0].copy(); qend=qstart.copy(); qend[0]+=.6
qend[7:]=qstart[7:]
recover=45
last_q=q[-1].copy()
slerp=Slerp([0,1],Rotation.from_quat(np.stack([last_q[[4,5,6,3]],qend[[4,5,6,3]]])))
extra=[]
for k in range(1,recover+1):
    u=k/recover;w=u*u*(3-2*u)
    qp=(1-w)*last_q+w*qend;qp[3:7]=slerp([w]).as_quat()[0,[3,0,1,2]];extra.append(qp)
q=np.r_[q,np.array(extra),np.repeat(qend[None],60,axis=0)]
footP=np.concatenate([footP,np.repeat(endP[None],recover+60,axis=0)])
footR=np.concatenate([footR,np.repeat(np.tile(identity,(2,1,1))[None],recover+60,axis=0)])
max_error=0.
for t in range(len(q)):
    for side in range(2):
        adr=leg_adr[side];js=leg_joints[side]
        targetP=footP[t,side];targetR=footR[t,side]
        def residual(x):
            qp=q[t].copy();qp[adr]=x
            p,r=fk(qp)
            return np.r_[(p[side]-targetP)*8,Rotation.from_matrix(targetR.T@r[side]).as_rotvec()]
        result=least_squares(residual,np.clip(q[t,adr],m.jnt_range[js,0]+1e-5,m.jnt_range[js,1]-1e-5),bounds=(m.jnt_range[js,0]+1e-5,m.jnt_range[js,1]-1e-5),max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q[t,adr]=result.x
        max_error=max(max_error,float(np.linalg.norm(residual(result.x)[:3])/8))
# Hold the actual solved initial pose, with zero initial velocity.
q=np.r_[np.repeat(q[:1],25,axis=0),q]
v=np.zeros((len(q),35));v[:,:3]=np.gradient(q[:,:3],.02,axis=0)
R=Rotation.from_quat(q[:,[4,5,6,3]])
v[1:-1,3:6]=(R[2:]*R[:-2].inv()).as_rotvec()/.04
v[0,3:6]=v[1,3:6];v[-1,3:6]=v[-2,3:6]
v[:,6:]=np.gradient(q[:,7:],.02,axis=0)
heights=np.array([clearance(qp) for qp in q]);centers=np.array([foot_centers(qp) for qp in q])
assert max_error<.01, max_error
worst_t,worst_side=np.unravel_index(np.argmin(heights),heights.shape)
print('worst',worst_t,worst_side,'qroot',q[worst_t,:7].tolist(),'heights',heights[worst_t].tolist(),'leg',q[worst_t,leg_adr[worst_side]].tolist(),'limits',m.jnt_range[leg_joints[worst_side]].tolist(),'ankle',fk(q[worst_t])[0].tolist(),'rotvec',Rotation.from_matrix(fk(q[worst_t])[1]).as_rotvec().tolist(),flush=True)
assert heights.min()>-.006, float(heights.min())
assert np.linalg.norm(v[0])<1e-8
forward=float((centers[-1]-centers[0]).mean(axis=0)[0])
assert abs(forward-.6)<.005,forward
assert (heights.min(axis=1)>.025).sum()>=4
out=Path('data/processed/g1_standing_jump_060m.npz')
np.savez(out,qpos=q.astype(np.float32),qvel=v.astype(np.float32),starts=np.array([0]),lengths=np.array([len(q)]),names=np.array(['g1_standing_jump_060m']),fps=np.array(50.))
meta=dict(source='lvhaidong/LAFAN1_Retargeting_Dataset:g1/jumps1_subject1.csv',processed_source_frames=[first,last],target_distance_m=.6,reference_foot_displacement_m=forward,forward_scale=float(scale),frames=len(q),duration_s=(len(q)-1)/50,initial_hold_s=.5,recovery_s=.9,final_hold_s=1.2,max_ik_position_error_m=max_error,minimum_foot_clearance_m=float(heights.min()),max_both_foot_clearance_m=float(heights.min(axis=1).max()),airtime_above_025_s=float((heights.min(axis=1)>.025).sum()/50),takeoff_reference_frame=int(take+25),landing_reference_frame=int(land+25))
Path('experiments/jump06/reference.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))

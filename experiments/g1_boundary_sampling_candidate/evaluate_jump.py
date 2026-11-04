import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.rotations import quat_from_euler_xyz,quat_mul,mat_to_quat
from supermimic.motion_lib import MotionLibrary
from supermimic.evaluate import load_supermimic_policy
from jump_task import make_env

def contact_bounds(heights,clearance_start,clearance_end):
    """Locate separation/landing independently of the 2 cm flight quality gate.

    A 3 mm sole threshold accommodates the discrete 50 Hz samples. Motion
    after separation must not be counted as a standing-start violation.
    """
    take=clearance_start
    while take>0 and heights[take-1].min()>.003:
        take-=1
    both=np.where(heights[clearance_end:].max(-1)<.003)[0]
    land=int(clearance_end+both[0]) if len(both) else -1
    return int(take),land

@torch.no_grad()
def geometry(env,states):
    P,R=env.kin.forward(states['pos'],states['rot'],states['q'])
    feet=env.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link'))
    m=env.mj_model
    ids=[[g for g in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').split('/')[-1].startswith(side+'_foot') and 'collision' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')] for side in ('left','right')]
    from scipy.spatial.transform import Rotation
    loc=torch.tensor(m.geom_pos[ids],dtype=torch.float32,device=env.device)
    q=m.geom_quat[ids];axes=Rotation.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).as_matrix()[:,:,2].reshape(2,7,3)
    axes=torch.tensor(axes,dtype=torch.float32,device=env.device)
    size=torch.tensor(m.geom_size[ids],dtype=torch.float32,device=env.device)
    Rp=R[...,feet,:,:];pos=P[...,feet,:]
    gp=pos[...,None,:]+torch.einsum('...sij,sgj->...sgi',Rp,loc)
    ax=torch.einsum('...sij,sgj->...sgi',Rp,axes)
    h=(gp[...,2]-size[...,0]-size[...,1]*ax[...,2].abs()).min(-1).values
    key=env.kin.body_index(('pelvis','torso_link','left_knee_link','right_knee_link','left_wrist_yaw_link','right_wrist_yaw_link'))
    return gp[...,:2].mean(-2).cpu().numpy(),h.cpu().numpy(),P[...,key,2].min(-1).values.cpu().numpy()

@torch.no_grad()
def run_trials(policy,lib,env,seed=20261027):
    torch.manual_seed(seed)
    n=env.num_envs;dev=env.device;steps=int(lib.lengths[0])-1
    env.reset()
    ids=torch.arange(n,device=dev);env.reset_envs(ids,frames=torch.zeros(n,dtype=torch.long,device=dev))
    ref=lib.frame(torch.zeros(n,dtype=torch.long,device=dev))
    pos,quat,q=ref['pos'].clone(),ref['quat'].clone(),ref['q'].clone()
    # Trial 0 is exactly nominal; others have small initial pose errors, zero momentum.
    pos[1:,:2]+=(torch.rand(n-1,2,device=dev)*2-1)*.003
    q[1:]+=(torch.rand(n-1,29,device=dev)*2-1)*.008
    angles=(torch.rand(n-1,3,device=dev)*2-1)*.008
    quat[1:]=quat_mul(quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2]),quat[1:])
    pos,quat,q=env.prepare_eval(pos,quat,q)
    env.write_state(ids,pos,quat,q,torch.zeros(n,3,device=dev),torch.zeros(n,3,device=dev),torch.zeros(n,29,device=dev))
    env._finish_resets(ids,force=True)
    active=torch.ones(n,dtype=torch.bool,device=dev);terminated=torch.zeros_like(active)
    completed=torch.zeros_like(active);ret=torch.zeros(n,device=dev);count=torch.zeros(n,dtype=torch.long,device=dev)
    records=[{k:v.clone() for k,v in env.state.items()}]
    for t in range(steps):
        o,g=env.policy_inputs();state,info=env.step(policy(o,g))
        records.append({k:torch.where(active.reshape((n,)+(1,)*(v.ndim-1)),v,records[-1][k]) for k,v in state.items()})
        ret+=info['reward']*active;count+=active.long()
        terminated|=info['terminated']&active
        completed|=info['truncated']&active&(t==steps-1)
        active&=~info['done']
        if not active.any():break
    while len(records)<steps+1:records.append(records[-1])
    states={k:torch.stack([r[k] for r in records]) for k in records[0]}
    xy,h,body_z=geometry(env,states)
    root=states['pos'].cpu().numpy();v=states['v'].cpu().numpy();tilt=torch.acos(states['rot'][...,2,2].clamp(-1,1)).cpu().numpy()
    complete=completed.cpu().numpy();failed=terminated.cpu().numpy();returns=(ret/(steps*env.tcfg.max_step_reward)).cpu().numpy()
    trials=[]
    for i in range(n):
        mask=h[:,i].min(-1)>.02
        edges=np.diff(np.r_[False,mask,False].astype(int));aa=np.where(edges==1)[0];bb=np.where(edges==-1)[0]
        flights=[(a,b) for a,b in zip(aa,bb) if b-a>=4 and a>=15]
        take,land=-1,-1;airtime=0.;distance=0.;pre_shift=float('inf');slide=float('inf')
        if flights:
            a,b=max(flights,key=lambda ab:ab[1]-ab[0]);airtime=(b-a)/50
            take,land=contact_bounds(h[:,i],a,b)
            pre_shift=float(np.linalg.norm(xy[:take,i]-xy[0,i],axis=-1).max())
        center=xy[:,i].mean(axis=1)-xy[0,i].mean(axis=0)
        if land>=0:
            distance=float(center[land,0]);slide=float(np.linalg.norm(center[-1]-center[land]))
        end_speed=float(np.linalg.norm(v[-50:,i,:2],axis=-1).max())
        settled=bool(complete[i] and end_speed<.25 and tilt[-50:,i].max()<np.deg2rad(25) and root[-50:,i,2].min()>.5)
        success=bool(complete[i] and not failed[i] and land>=0 and airtime>=.08 and abs(distance-.6)<=.10 and abs(center[land,1])<.10 and pre_shift<.08 and slide<.10 and abs(center[-1,0]-.6)<.10 and settled and body_z[:,i].min()>.07)
        # Use null for unavailable measures so JSON remains portable.
        trials.append(dict(trial=i,success=success,complete=bool(complete[i]),terminated=bool(failed[i]),airtime_s=float(airtime),takeoff_frame=take,landing_frame=land,landing_distance_m=distance,final_distance_m=float(center[-1,0]),lateral_m=float(center[-1,1]),pre_takeoff_foot_shift_m=pre_shift if np.isfinite(pre_shift) else None,post_landing_shift_m=slide if np.isfinite(slide) else None,final_max_speed_m_s=end_speed,settled=settled,norm_return=float(returns[i])))
    result=dict(metric_version=2,success_rate=float(np.mean([r['success'] for r in trials])),nominal_success=trials[0]['success'],completion_rate=float(complete.mean()),mean_norm_return=float(returns.mean()),nominal=trials[0],trials=trials)
    qpos=torch.cat([states['pos'],mat_to_quat(states['rot']),states['q']],dim=-1).cpu().numpy()
    trajectory=dict(qpos=qpos,foot_xy=xy,foot_height=h,root_velocity=v,returns=returns,success=np.array([r['success'] for r in trials]),fps=np.array(50))
    return result,trajectory

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--trials',type=int,default=64);p.add_argument('--seed',type=int,default=20261028);args=p.parse_args()
    from supermimic import networks
    networks.AMP['enabled']=True
    torch.backends.cuda.matmul.allow_tf32=True
    lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cuda')
    env=make_env(lib,'cuda',args.trials,evaluate=True)
    policy,ck=load_supermimic_policy(args.checkpoint,'cuda')
    result,traj=run_trials(policy,lib,env,args.seed)
    result.update(checkpoint=args.checkpoint,it=ck['it'],seed=args.seed,num_trials=args.trials)
    Path(args.out).mkdir(parents=True,exist_ok=True)
    Path(args.out,'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(Path(args.out,'rollout.npz'),**traj)
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))
if __name__=='__main__':main()

"""Full-cycle physical evaluation with unfiltered, world-frame XYZ acceleration."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.task import state_to_flat

CRITERIA=dict(com_acc_rms_max=1.5,center_acc_rms_max=2.,com_acc_peak_max=8.,center_acc_peak_max=10.,
              com_speed_rms_max=.10,center_speed_rms_max=.12,angular_speed_rms_max=.25,
              com_position_std_norm_max=.015,body_inclination_std_deg_max=1.5,foot_xy_drift_max=.04,
              hold_pose_fraction_min=.95,return_pose_fraction_min=.95)

def metrics_for_window(arr,sel):
    def rms(x):return np.sqrt(np.mean(np.sum(x[sel]**2,axis=-1),axis=0))
    def peak(x):return np.max(np.linalg.norm(x[sel],axis=-1),axis=0)
    def per_axis(x):return np.sqrt(np.mean(x[sel]**2,axis=0))
    return dict(com_acc_rms=rms(arr['com_acc']),center_acc_rms=rms(arr['center_acc']),
                com_acc_peak=peak(arr['com_acc']),center_acc_peak=peak(arr['center_acc']),
                com_acc_rms_xyz=per_axis(arr['com_acc']),center_acc_rms_xyz=per_axis(arr['center_acc']),
                com_speed_rms=rms(arr['com_v']),center_speed_rms=rms(arr['state'][:,:,19:22]),
                angular_speed_rms=rms(arr['state'][:,:,22:25]),
                com_position_std_norm=np.linalg.norm(np.std(arr['com_pos'][sel],axis=0),axis=-1),
                body_inclination_std_deg=np.std(arr['inclination'][sel],axis=0))

def stable_mask(m):
    keys=('com_acc_rms','center_acc_rms','com_acc_peak','center_acc_peak','com_speed_rms','center_speed_rms','angular_speed_rms','com_position_std_norm','body_inclination_std_deg')
    passed=np.ones(len(m['com_acc_rms']),dtype=bool)
    for k in keys:passed&=m[k]<=CRITERIA[k+'_max']
    return passed

@torch.no_grad()
def evaluate(policy,env,seed=20261101):
    torch.manual_seed(seed);policy.eval();n=env.num_envs;dev=env.device
    ids=torch.arange(n,device=dev);env.reset();env.reset_envs(ids,torch.zeros_like(ids));env._finish_resets(ids,force=True)
    env.sim.data.qpos[1:,env.q_adr]+=.006*torch.randn(n-1,12,device=dev);env._finish_resets(ids,force=True)
    live=torch.ones(n,device=dev,dtype=torch.bool);ret=torch.zeros(n,device=dev)
    rec={k:[] for k in ('state','feet','alive','com_pos','com_v','com_acc','center_acc')}
    T=int(env.lib.lengths[0])-1
    for t in range(T):
        a=policy(*env.policy_inputs());s,info=env.step(a)
        ret+=live*info['reward'];live&=~info['terminated']
        values=dict(state=state_to_flat(s),feet=env.task.foot_positions(env.task.body_kin(s)),alive=live,
                    com_pos=info['com_pos'],com_v=info['com_velocity'],com_acc=info['com_acceleration'],center_acc=info['center_acceleration'])
        for k,v in values.items():rec[k].append(v.cpu().numpy().copy())
    arr={k:np.asarray(v) for k,v in rec.items()};arr['time_s']=(np.arange(T)+1)*env.tcfg.dt
    w,x,y,z=np.moveaxis(arr['state'][:,:,3:7],-1,0)
    up=2*(x*z-w*y);rzz=1-2*(x*x+y*y)
    arr['inclination']=np.degrees(np.arcsin(np.clip(up,-1,1)))
    stand=(arr['time_s']>=4.)&(arr['time_s']<=9.3)
    final=arr['time_s']>=13.
    sm=metrics_for_window(arr,stand);fm=metrics_for_window(arr,final)
    feet=arr['feet'];alive=arr['alive'];complete=alive.all(0)
    hind_contact=((feet[:,:,2:,2]>=.020)&(feet[:,:,2:,2]<=.045)).all(-1)
    all_contact=((feet[:,:,:,2]>=.020)&(feet[:,:,:,2]<=.045)).all(-1)
    hold_pose=(up>np.sin(np.deg2rad(75)))&(feet[:,:,:2,2].min(-1)>.35)&hind_contact&alive
    return_pose=(rzz>np.cos(np.deg2rad(8)))&all_contact&alive
    hold_fraction=hold_pose[stand].mean(0);return_fraction=return_pose[final].mean(0)
    drift=np.linalg.norm(feet[stand,:,2:,:2]-feet[np.flatnonzero(stand)[0],:,2:,:2],axis=-1).max(axis=(0,2))
    cycle=complete&(hold_fraction>=CRITERIA['hold_pose_fraction_min'])&(return_fraction>=CRITERIA['return_pose_fraction_min'])
    stable_stand=stable_mask(sm)&(drift<=CRITERIA['foot_xy_drift_max'])
    stable_return=stable_mask(fm)
    passed=cycle&stable_stand&stable_return
    def aggregate(m):
        return {k:(np.mean(v[complete],axis=0).tolist() if complete.any() else None) for k,v in m.items()}
    result=dict(trials=n,success_rate=float(passed.mean()),nominal_success=bool(passed[0]),completion_rate=float(complete.mean()),cycle_success_rate=float(cycle.mean()),
                stable_hindstand_rate=float((complete&stable_stand).mean()),stable_return_rate=float((complete&stable_return).mean()),
                mean_norm_return=float((ret/(T*env.tcfg.max_step_reward)).mean()),
                stand_metrics_completed_trials=aggregate(sm),return_metrics_completed_trials=aggregate(fm),criteria=CRITERIA,
                trials_detail=[dict(trial=i,completed=bool(complete[i]),cycle=bool(cycle[i]),success=bool(passed[i]),stand_pose_fraction=float(hold_fraction[i]),return_pose_fraction=float(return_fraction[i]),hind_foot_xy_drift_m=float(drift[i]),
                                    stand={k:v[i].tolist() for k,v in sm.items()},returned={k:v[i].tolist() for k,v in fm.items()}) for i in range(n)])
    policy.train()
    return result,arr

"""Actor-only MuJoCo: segment-wise locomotion + tilt tests, no learned WM."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from bootstrap import HERE
from common import write_json,sha256
from contract import *
from cases import make_cases,CRITERIA,DURATION
from task import TiltLocomotionEnv
from policy import make_policy
from contacts import ContactRecorder

@torch.no_grad()
def batch(policy,cases,trials,seed,out):
    torch.manual_seed(seed);n=len(cases)*trials;dev='cuda'
    env=TiltLocomotionEnv(dev,n,evaluate=True,randomize=True,compile=False,autoreset=False)
    fixed={k:torch.empty(n,device=dev) for k in ('friction','mass_scale','kp_scale','kd_scale','delay')}
    ranges=dict(friction=(.4,1.3,1.),mass_scale=(.9,1.1,1.),kp_scale=(.85,1.15,1.),kd_scale=(.8,1.2,1.))
    for k,v in fixed.items():
        if k=='delay':v[:]=torch.randint(0,3,(n,),device=dev);v[::trials]=0
        else:
            lo,hi,nominal=ranges[k];v[:]=lo+(hi-lo)*torch.rand(n,device=dev);v[::trials]=nominal
        for i,c in enumerate(cases):
            if k in c['domain']:v[i*trials:(i+1)*trials]=c['domain'][k]
    fixed['delay']=fixed['delay'].long()
    fixed['enc_bias']=.01*torch.rand(n,12,device=dev)-.005;fixed['enc_bias'][::trials]=0
    fixed['com_offset']=.02*torch.rand(n,3,device=dev)-.01;fixed['com_offset'][::trials]=0
    env.fixed_domain=fixed;env.nominal_stride=trials;env.reset();env.contact_recorder=ContactRecorder(env)
    target=torch.tensor([r['target'] for r in cases],device=dev).repeat_interleave(trials,0)
    reverse=torch.tensor([r['reverse'] for r in cases],device=dev).repeat_interleave(trials,0)
    zero=torch.zeros_like(target)
    failed=torch.zeros(n,dtype=torch.bool,device=dev);segment_ok=~failed
    z=torch.zeros(n,device=dev)
    metrics={k:z.clone() for k in ('velocity_rmse_m_s','yaw_rmse_rad_s','attitude_rmse_deg','attitude_peak_deg','dynamic_attitude_rmse_deg',
        'stop_speed_rms_m_s','stop_yaw_rms_rad_s','stop_drift_m','contact_slip_rms_m_s','nonfoot_contact_fraction',
        'peak_planar_speed_m_s','peak_lateral_speed_m_s','peak_joint_speed_rad_s','peak_mechanical_power_w','peak_joint_limit_error_rad')}
    metrics['min_body_height_m']=torch.full_like(z,10.)
    sums=torch.zeros(n,3,4,device=dev);counts=torch.zeros(3,device=dev)
    slip=z.clone();contact_count=z.clone();nonfoot=z.clone();samples=0;dynamic=z.clone();dynamic_count=0
    stop_start=None;curves={k:[] for k in ('time','velocity','actual_rp','actual_yaw_rate','desired','requested')}
    for step in range(round(DURATION/DT)):
        t=step*DT
        requested=target if 2<=t<12 else reverse if 12<=t<22 else zero
        env.set_target(requested)
        s,info=env.step(policy(*env.policy_inputs()));failed|=info['terminated']
        v=heading_velocity(s['rot'],s['v']);rp=angles(s['rot']);rates=euler_rates(s['rot'],s['w'])
        fp,fv=env.task.foot_state(env.task.body_kin(s))
        if t>=.5:
            metrics['peak_joint_speed_rad_s']=torch.maximum(metrics['peak_joint_speed_rad_s'],info['peak_speed'])
            metrics['peak_mechanical_power_w']=torch.maximum(metrics['peak_mechanical_power_w'],info['peak_power'])
            metrics['peak_planar_speed_m_s']=torch.maximum(metrics['peak_planar_speed_m_s'],v[:,:2].norm(dim=-1))
            metrics['peak_lateral_speed_m_s']=torch.maximum(metrics['peak_lateral_speed_m_s'],v[:,1].abs())
            metrics['min_body_height_m']=torch.minimum(metrics['min_body_height_m'],s['pos'][:,2])
            limit=(torch.relu(env.task.qmin-s['q'])+torch.relu(s['q']-env.task.qmax)).amax(-1)
            metrics['peak_joint_limit_error_rad']=torch.maximum(metrics['peak_joint_limit_error_rad'],limit)
        if t>=2:
            dynamic+=angle_error(rp,env.context[:,3:5]).square().sum(-1)*(180/math.pi)**2;dynamic_count+=1
            force,clearance,vz,nf=env.last_contacts
            loaded=force[-1]>5.
            slip+=(loaded*fv[:,:,:2].square().sum(-1)).sum(-1);contact_count+=loaded.sum(-1)
            nonfoot+=(nf>5.).sum(0);samples+=4
        segment=0 if 9<=t<12 else 1 if 19<=t<22 else 2 if 27<=t<30 else None
        if segment is not None:
            ve=(v[:,:2]-requested[:,:2]).square().sum(-1)
            ye=(rates[:,2]-requested[:,2]).square()
            ae=angle_error(rp,requested[:,3:5])*180/math.pi
            sums[:,segment,0]+=ve;sums[:,segment,1]+=ye;sums[:,segment,2]+=ae.square().sum(-1)
            sums[:,segment,3]=torch.maximum(sums[:,segment,3],ae.abs().amax(-1));counts[segment]+=1
            if segment==2:
                if stop_start is None:stop_start=s['pos'][:,:2].clone()
                metrics['stop_drift_m']=torch.maximum(metrics['stop_drift_m'],(s['pos'][:,:2]-stop_start).norm(dim=-1))
        if step%10==0:
            curves['time'].append(t+DT)
            for k,value in [('velocity',v),('actual_rp',rp),('actual_yaw_rate',rates[:,2]),('desired',env.context[:,:7]),('requested',requested)]:curves[k].append(value[::trials].cpu().numpy())
        if step%200==0:write_json(out/'eval_progress.json',dict(step=step,total_steps=round(DURATION/DT),batch_cases=[r['name'] for r in cases],updated=time.time()))
    rms=torch.sqrt(sums[:,:,:3]/counts[None,:,None].clamp_min(1))
    metrics['velocity_rmse_m_s']=rms[:,:2,0].amax(-1)
    metrics['yaw_rmse_rad_s']=rms[:,:2,1].amax(-1)
    metrics['attitude_rmse_deg']=rms[:,:,2].amax(-1)
    metrics['attitude_peak_deg']=sums[:,:,3].amax(-1)
    metrics['stop_speed_rms_m_s']=rms[:,2,0];metrics['stop_yaw_rms_rad_s']=rms[:,2,1]
    metrics['contact_slip_rms_m_s']=torch.sqrt(slip/contact_count.clamp_min(1))
    metrics['nonfoot_contact_fraction']=nonfoot/max(samples,1)
    metrics['dynamic_attitude_rmse_deg']=torch.sqrt(dynamic/max(dynamic_count,1))
    for i,raw in enumerate((target,reverse)):
        vtol=(CRITERIA['velocity_rmse_relative']*raw[:,:2].norm(dim=-1)).clamp_min(CRITERIA['velocity_rmse_absolute_m_s'])
        ytol=(CRITERIA['yaw_rmse_relative']*raw[:,2].abs()).clamp_min(CRITERIA['yaw_rmse_absolute_rad_s'])
        segment_ok&=(rms[:,i,0]<=vtol)&(rms[:,i,1]<=ytol)
    checks=dict(completed=~failed,all_motion_segments=segment_ok,
        attitude=metrics['attitude_rmse_deg']<=2.,attitude_peak=metrics['attitude_peak_deg']<=5.,
        dynamic_attitude=metrics['dynamic_attitude_rmse_deg']<=5.,
        stop_speed=metrics['stop_speed_rms_m_s']<=.10,stop_yaw=metrics['stop_yaw_rms_rad_s']<=.15,
        stop_drift=metrics['stop_drift_m']<=.10,stance_slip=metrics['contact_slip_rms_m_s']<=.20,
        nonfoot=metrics['nonfoot_contact_fraction']<=.01,
        planar_limit=metrics['peak_planar_speed_m_s']<=4.+1e-4,lateral_limit=metrics['peak_lateral_speed_m_s']<=1.+1e-4,
        joint_limits=metrics['peak_joint_limit_error_rad']<=.02,body_height=metrics['min_body_height_m']>=.30)
    passed=torch.stack(list(checks.values())).all(0);rows=[]
    for i,c in enumerate(cases):
        ids=slice(i*trials,(i+1)*trials)
        row=dict(**c,trials=trials,passed=int(passed[ids].sum()),completed=int((~failed[ids]).sum()),nominal_passed=bool(passed[i*trials]),failures={k:int((~v[ids]).sum()) for k,v in checks.items()})
        for k,v in metrics.items():row[k+'_mean']=float(v[ids].mean());row[k+'_max']=float(v[ids].max())
        row['trial_metrics']={k:v[ids].cpu().tolist() for k,v in metrics.items()};rows.append(row)
    return rows,{k:np.asarray(v) for k,v in curves.items()}

@torch.no_grad()
def evaluate(checkpoint,out,trials=32,full=False,seed=73129,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    from supermimic import networks
    networks.AMP['enabled']=False
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['task_kind']=='b2_tilt_locomotion_v1' and ck['backend']=='vibe'
    policy=make_policy().cuda();policy.load_state_dict(ck['policy']);policy.eval();it=ck['it'];del ck
    cases=make_cases(full);cases=cases[:2] if smoke else cases;rows=[]
    # Bounded GPU allocation, so the final full cross-product does not OOM.
    for start in range(0,len(cases),24):
        result,curves=batch(policy,cases[start:start+24],trials,seed+start,out);rows+=result
        np.savez_compressed(out/f'nominal_curves_{start:04d}.npz',**curves)
        write_json(out/'partial_results.json',rows);torch.cuda.empty_cache()
    result=dict(task='b2_tilt_locomotion_v1',backend='vibe',it=it,full=full,seed=seed,
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),criteria=CRITERIA,cases=rows,
        passed=sum(r['passed'] for r in rows),total=len(rows)*trials,completed=sum(r['completed'] for r in rows),
        all_passed=all(r['passed']>=math.ceil(.95*trials) and r['nominal_passed'] for r in rows),
        physical_acceptance_scope='new-task simulator acceptance; not real motor-rating or hardware qualification',
        hardware_validated=False,wall_seconds=None)
    write_json(out/'result.json',result);write_json(out/'eval_progress.json',dict(done=True,cases=len(rows),updated=time.time()))
    print(json.dumps({k:result[k] for k in ('it','passed','total','completed','all_passed')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--trials',type=int,default=32);p.add_argument('--seed',type=int,default=73129);p.add_argument('--full',action='store_true');p.add_argument('--smoke',action='store_true')
    evaluate(**vars(p.parse_args()))

"""Standalone true-physics evaluation. Loads ONLY the Actor, never the WM."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from bootstrap import HERE
from common import sha256,write_json
from contract import DT,SIM_DT,angles,angle_error
from cases import make_cases,CRITERIA,DURATION
from policy import make_policy
from task import AttitudeEnv
from contacts import ContactRecorder

@torch.no_grad()
def evaluate(checkpoint,out,trials=32,full=False,seed=73129,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);torch.set_num_threads(2)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['task_kind']=='b2_attitude_v1' and ck['backend']=='vibe'
    policy=make_policy().to('cuda');policy.load_state_dict(ck['policy']);policy.eval()
    iteration=ck['it'];del ck
    cases=make_cases(full,73129) # Case angles are fixed; seeds change randomized physical trials.
    if smoke:
        names=('neutral','roll_+30','pitch_-30','cross_axis','queued_return_roll_pitch')
        cases=[next(c for c in cases if c['name']==name) for name in names]
    count=len(cases);n=count*trials
    env=AttitudeEnv('cuda',n,evaluate=True,randomize=True,compile=False,autoreset=False)
    fixed={name:torch.zeros(n,device='cuda') for name in ('friction','mass_scale','kp_scale','kd_scale','delay')}
    for i,case in enumerate(cases):
        for key,values in fixed.items():
            # Trial zero is the nominal requested domain; others retain training randomization.
            if key=='friction':values[i*trials:(i+1)*trials]=.4+.9*torch.rand(trials,device='cuda');default=1.
            elif key=='mass_scale':values[i*trials:(i+1)*trials]=.9+.2*torch.rand(trials,device='cuda');default=1.
            elif key=='kp_scale':values[i*trials:(i+1)*trials]=.85+.3*torch.rand(trials,device='cuda');default=1.
            elif key=='kd_scale':values[i*trials:(i+1)*trials]=.8+.4*torch.rand(trials,device='cuda');default=1.
            else:values[i*trials:(i+1)*trials]=torch.randint(0,3,(trials,),device='cuda');default=0.
            if key in case['domain']:values[i*trials:(i+1)*trials]=case['domain'][key]
            else:values[i*trials]=default
    fixed['delay']=fixed['delay'].long()
    fixed['enc_bias']=.01*torch.rand(n,12,device='cuda')-.005
    fixed['com_offset']=.02*torch.rand(n,3,device='cuda')-.01
    fixed['enc_bias'][::trials]=0;fixed['com_offset'][::trials]=0
    env.fixed_domain=fixed;env.nominal_stride=trials;env.reset()
    env.contact_recorder=ContactRecorder(env)
    start_feet=env.sim.data.geom_xpos[:,env.foot_geoms,:2].clone()
    target=torch.zeros(n,2,device='cuda');last_change=torch.zeros(n,device='cuda')
    hold_count=torch.zeros(n,device='cuda');hold_squared=hold_count.clone();hold_peak=hold_count.clone();rate_squared=hold_count.clone()
    neutral_count=hold_count.clone();neutral_peak=hold_count.clone();neutral_speed=hold_count.clone()
    tracking_squared=hold_count.clone();tracking_count=hold_count.clone();drift=hold_count.clone();body_drift=hold_count.clone()
    peak_speed=hold_count.clone();peak_power=hold_count.clone();support4=hold_count.clone();support3=hold_count.clone();samples=hold_count.clone();nonfoot=hold_count.clone()
    failed=torch.zeros(n,device='cuda',dtype=torch.bool);initial_ok=torch.ones(n,device='cuda',dtype=torch.bool)
    curves={k:[] for k in ('time','actual','desired','target')}
    # Require EVERY command hold, including each recovery, not just an average across easy segments.
    segment_bad=torch.zeros(n,device='cuda',dtype=torch.bool)
    segment_peak=torch.zeros(n,device='cuda');segment_samples=segment_peak.clone()
    segment_squared=segment_peak.clone();worst_segment_rmse=segment_peak.clone()
    segment_tolerance=torch.full_like(segment_peak,CRITERIA['hold_rmse_deg'])
    worst_scaled_rmse=segment_peak.clone();overshoot=segment_peak.clone()
    lower_bound=target.clone();upper_bound=target.clone()
    for step in range(round(DURATION/DT)):
        t=step*DT
        for i,case in enumerate(cases):
            for evt in case['events']:
                if step==round(evt['time']/DT):
                    ids=slice(i*trials,(i+1)*trials)
                    if evt['time']>2:
                        segment_bad[ids]|=(segment_samples[ids]==0)|(segment_peak[ids]>CRITERIA['hold_peak_error_deg'])
                        rmse=torch.sqrt(segment_squared[ids]/segment_samples[ids].clamp_min(1))
                        worst_segment_rmse[ids]=torch.maximum(worst_segment_rmse[ids],rmse)
                        segment_bad[ids]|=rmse>segment_tolerance[ids]
                        worst_scaled_rmse[ids]=torch.maximum(worst_scaled_rmse[ids],rmse/segment_tolerance[ids])
                    segment_peak[ids]=0;segment_samples[ids]=0
                    segment_squared[ids]=0
                    new_target=torch.tensor(evt['target_degrees'],device='cuda')*math.pi/180
                    lower_bound[ids]=torch.minimum(target[ids],new_target);upper_bound[ids]=torch.maximum(target[ids],new_target)
                    target[ids]=new_target
                    magnitude=max(abs(x) for x in evt['target_degrees'])
                    segment_tolerance[ids]=min(2.,max(.25,.1*magnitude)) if magnitude>0 else 2.
                    last_change[ids]=t
        env.set_target(target)
        obs,goal=env.policy_inputs();action=policy(obs,goal)
        state,info=env.step(action)
        actual=angles(state['rot']);err=angle_error(actual,target)*180/math.pi
        desired_err=angle_error(actual,env.context[:,:2])*180/math.pi
        failed|=info['terminated']
        if t>=.5:
            peak_speed=torch.maximum(peak_speed,info['peak_speed']);peak_power=torch.maximum(peak_power,info['peak_power'])
        force,clearance,vz,nonfoot_force=env.last_contacts
        support=(force>5.)
        if 1.5<=t<2.:
            stable=(actual.abs().amax(-1)<=math.radians(2))&(state['w'].norm(dim=-1)<.15)&(state['v'].norm(dim=-1)<.08)&support.all(-1).all(0)
            initial_ok&=stable
            start_feet=env.sim.data.geom_xpos[:,env.foot_geoms,:2].clone()
        hold=(t-last_change>=CRITERIA['settling_window_starts_after_command_s'])&(t>=2.)
        neutral=hold&(target.abs().amax(-1)<1e-6)
        hold_count+=hold;hold_squared+=hold*err.square().sum(-1);hold_peak=torch.maximum(hold_peak,err.abs().amax(-1)*hold)
        rate_squared+=hold*state['w'].square().sum(-1)
        segment_peak=torch.maximum(segment_peak,err.abs().amax(-1)*hold);segment_samples+=hold
        segment_squared+=hold*err.square().sum(-1)
        neutral_count+=neutral;neutral_peak=torch.maximum(neutral_peak,err.abs().amax(-1)*neutral)
        neutral_speed+=neutral*state['v'].square().sum(-1)
        if t>=2.:
            tracking_squared+=desired_err.square().sum(-1);tracking_count+=1
            excess=torch.maximum(torch.relu(actual-upper_bound),torch.relu(lower_bound-actual)).amax(-1)*180/math.pi
            overshoot=torch.maximum(overshoot,excess)
            feet=env.sim.data.geom_xpos[:,env.foot_geoms,:2]
            drift=torch.maximum(drift,(feet-start_feet).norm(dim=-1).amax(-1))
            body_drift=torch.maximum(body_drift,state['pos'][:,:2].norm(dim=-1))
            support3+=(support.sum(-1)>=3).sum(0);samples+=4
            nonfoot+=(nonfoot_force>5.).sum(0)
            support4+=support.all(-1).sum(0)*hold
        if step%5==0:
            curves['time'].append(t+DT)
            for key,values in [('actual',actual),('desired',env.context[:,:2]),('target',target)]:curves[key].append(values[::trials].cpu().numpy())
        if step%100==0:
            write_json(out/'eval_progress.json',dict(step=step,total_steps=round(DURATION/DT),cases=count,trials=trials,updated=time.time()))
    segment_bad|=(segment_samples==0)|(segment_peak>CRITERIA['hold_peak_error_deg'])
    final_segment_rmse=torch.sqrt(segment_squared/segment_samples.clamp_min(1))
    worst_segment_rmse=torch.maximum(worst_segment_rmse,final_segment_rmse)
    segment_bad|=final_segment_rmse>segment_tolerance
    worst_scaled_rmse=torch.maximum(worst_scaled_rmse,final_segment_rmse/segment_tolerance)
    metrics=dict(hold_rmse_deg=worst_segment_rmse,pooled_hold_rmse_deg=torch.sqrt(hold_squared/hold_count.clamp_min(1)),hold_peak_deg=hold_peak,
        tracking_rmse_deg=torch.sqrt(tracking_squared/tracking_count.clamp_min(1)),neutral_peak_deg=neutral_peak,
        hold_angular_rate_rms=torch.sqrt(rate_squared/hold_count.clamp_min(1)),neutral_speed_rms=torch.sqrt(neutral_speed/neutral_count.clamp_min(1)),
        foot_drift_m=drift,body_drift_m=body_drift,four_support_fraction=support4/(4*hold_count).clamp_min(1),
        three_support_fraction=support3/samples.clamp_min(1),nonfoot_contact_fraction=nonfoot/samples.clamp_min(1),peak_speed=peak_speed,peak_power=peak_power,
        worst_segment_tolerance_ratio=worst_scaled_rmse,overshoot_deg=overshoot)
    checks=dict(completed=~failed,initial_stable=initial_ok,all_segments=~segment_bad,
        hold=metrics['hold_rmse_deg']<=CRITERIA['hold_rmse_deg'],hold_peak=hold_peak<=CRITERIA['hold_peak_error_deg'],
        tracking=metrics['tracking_rmse_deg']<=CRITERIA['tracking_rmse_deg'],neutral=(neutral_peak<=CRITERIA['neutral_peak_deg'])&(neutral_count>0),
        angular_rate=metrics['hold_angular_rate_rms']<=CRITERIA['hold_angular_rate_rms'],neutral_speed=metrics['neutral_speed_rms']<=CRITERIA['neutral_speed_rms'],
        foot_drift=drift<=CRITERIA['max_foot_drift_m'],body_drift=body_drift<=CRITERIA['max_body_drift_m'],
        four_support=metrics['four_support_fraction']>=CRITERIA['min_four_support_fraction'],three_support=metrics['three_support_fraction']>=CRITERIA['min_three_support_fraction'],
        nonfoot=metrics['nonfoot_contact_fraction']<=CRITERIA['max_nonfoot_contact_fraction'],
        speed=peak_speed<=CRITERIA['joint_speed_rad_s'],power=peak_power<=CRITERIA['mechanical_power_w'],overshoot=overshoot<=CRITERIA['max_overshoot_deg'])
    passed=torch.stack(list(checks.values())).all(0)
    motion=torch.stack([v for k,v in checks.items() if k not in ('speed','power')]).all(0)
    rows=[]
    for i,case in enumerate(cases):
        ids=slice(i*trials,(i+1)*trials)
        row={**case,'trials':trials,'passed':int(passed[ids].sum()),'motion_passed':int(motion[ids].sum()),'completed':int((~failed[ids]).sum()),'nominal_passed':bool(passed[i*trials])}
        for k,v in metrics.items():row[k+'_mean']=float(v[ids].mean());row[k+'_max']=float(v[ids].max())
        row['failures']={k:int((~v[ids]).sum()) for k,v in checks.items()}
        row['trial_metrics']={k:v[ids].cpu().tolist() for k,v in metrics.items()}
        rows.append(row)
    result=dict(task='b2_attitude_v1',backend='vibe',it=iteration,full=full,seed=seed,
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),criteria=CRITERIA,cases=rows,passed=int(passed.sum()),total=n,
        completed=int((~failed).sum()),all_passed=all(r['passed']>=math.ceil(.95*trials) and r['nominal_passed'] for r in rows),
        note='Independent MuJoCo physics; Actor only. IK feasibility is not a trained-policy result.')
    np.savez_compressed(out/'nominal_curves.npz',**{k:np.asarray(v) for k,v in curves.items()})
    write_json(out/'result.json',result);write_json(out/'eval_progress.json',dict(done=True,cases=count,trials=trials,updated=time.time()))
    print(json.dumps({k:result[k] for k in ('it','passed','total','completed','all_passed')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--trials',type=int,default=32);p.add_argument('--seed',type=int,default=73129);p.add_argument('--full',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();evaluate(**vars(a))

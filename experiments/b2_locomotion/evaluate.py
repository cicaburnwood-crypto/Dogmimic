"""Held-out full-cycle actual physics tests. Rewards never determine pass/fail."""
import argparse,json,math,os,time,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.rotations import heading_mat,mtv
from supermimic.task import state_to_flat
from task import CommandEnv
from policy import make_policy
from contract import DT

def cases():
    result=[dict(name='stand',command=[0.,0.,0.],level=.5,core=True)]
    for speed in (.25,.5,1.,1.5,2.):
        lateral=min(.6,speed*.6);forward=math.sqrt(speed**2-lateral**2)
        for direction in (-1,1):
            result.append(dict(name=f'x{direction*speed:+g}',command=[direction*speed,0.,0.],level=max(.5,speed),core=True))
            for side in (-1,1):
                result.append(dict(name=f'diagonal{direction:+d}_{side:+d}_{speed:g}',
                                   command=[direction*forward,side*lateral,0.],level=max(.5,speed),core=True))
    for y in (-.6,-.3,.3,.6):result.append(dict(name=f'y{y:+g}',command=[0.,y,0.],level=.5 if abs(y)<=.3 else 1.,core=True))
    for yaw in (-3.,-2.,-1.,-.5,.5,1.,2.,3.):
        result.append(dict(name=f'spin{yaw:+g}',command=[0.,0.,yaw],level=.5 if abs(yaw)<=.5 else (1. if abs(yaw)<=1 else 2.),core=abs(yaw)<=2))
    for vx in (1.,1.9):
        for wz in (-1.,1.):
            result.append(dict(name=f'combined_{vx:g}_{wz:+g}',command=[vx,.6,wz],level=1.5 if vx==1 else 2.,core=True))
    return result

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

@torch.no_grad()
def evaluate(policy,env,trial_count=16,seed=20261121):
    torch.manual_seed(seed);env.nominal_stride=trial_count;s=env.reset();n=env.num_envs;dev=env.device
    assert all(torch.count_nonzero(s[k])==0 for k in ('v','w','qd'))
    groups=cases();targets=torch.tensor([r['command'] for r in groups],device=dev).repeat_interleave(trial_count,0)
    live=torch.ones(n,dtype=torch.bool,device=dev);states=[];vel=[];ang=[];tilt=[];feet=[];footspeed=[];commands=[];alive=[];actions=[]
    for t in range(850):
        tm=t*DT
        raw=targets if 1.5<=tm<5.5 else (-targets if 8.5<=tm<12.5 else torch.zeros_like(targets))
        env.set_command(raw);cmd=env.commands.clone()
        a=policy(*env.policy_inputs());s,info=env.step(a);live&=~info['done']
        fp,fv=env.task.foot_state(env.task.body_kin(s))
        states.append(state_to_flat(s));vel.append(mtv(heading_mat(s['rot']),s['v'])[:,:2]);ang.append(mtv(s['rot'],s['w'])[:,2])
        tilt.append(torch.acos(s['rot'][:,2,2].clamp(-1,1)));feet.append(fp[:,:,2]);footspeed.append(fv.norm(dim=-1).mean(-1))
        commands.append(cmd);alive.append(live.clone());actions.append(a)
    state=torch.stack(states).cpu().numpy();v=torch.stack(vel).cpu().numpy();w=torch.stack(ang).cpu().numpy()
    tilt=torch.stack(tilt).cpu().numpy();fz=torch.stack(feet).cpu().numpy();fv=torch.stack(footspeed).cpu().numpy()
    cmd=torch.stack(commands).cpu().numpy();alive=torch.stack(alive).cpu().numpy();a=torch.stack(actions).cpu().numpy()
    move=np.r_[np.arange(175,275),np.arange(525,625)]
    stop=np.r_[np.arange(350,425),np.arange(725,850)]
    speed=np.linalg.norm(v,axis=-1);target=np.linalg.norm(targets[:,:2].cpu().numpy(),axis=-1)
    track=np.sqrt(((v[move]-cmd[move,:,:2])**2).sum(-1).mean(0))
    yaw=np.sqrt(((w[move]-cmd[move,:,2])**2).mean(0))
    stopmean=speed[stop].mean(0);stoppeak=speed[stop].max(0);stopw=np.abs(w[stop]).mean(0)
    drift=np.maximum(np.linalg.norm(state[424,:,:2]-state[350,:,:2],axis=-1),np.linalg.norm(state[849,:,:2]-state[725,:,:2],axis=-1))
    stand_q=np.sqrt(((state[stop,:,7:19]-env.q0.cpu().numpy())**2).mean((0,2)))
    grounded=(fz[stop].max(-1)<.065).mean(0);stopfeet=fv[stop].mean(0)
    complete=alive.all(0);upright=(tilt.max(0)<.55)&(state[:,:,2].min(0)>.4)
    tracking=(track<=np.maximum(.12,.2*target))&(yaw<=np.maximum(.15,.2*np.abs(targets[:,2].cpu().numpy())))
    settled=(stopmean<.10)&(stoppeak<.22)&(stopw<.15)&(drift<.10)&(stand_q<.20)&(grounded>=.9)&(stopfeet<.10)
    success=complete&upright&tracking&settled
    rows=[]
    for i in range(n):
        rows.append(dict(trial=i%trial_count,success=bool(success[i]),complete=bool(complete[i]),upright=bool(upright[i]),
                         tracking=bool(tracking[i]),settled=bool(settled[i]),velocity_rmse_m_s=float(track[i]),yaw_rmse_rad_s=float(yaw[i]),
                         stop_mean_m_s=float(stopmean[i]),stop_peak_m_s=float(stoppeak[i]),stop_yaw_rad_s=float(stopw[i]),
                         stop_drift_m=float(drift[i]),standing_joint_rmse_rad=float(stand_q[i]),four_feet_contact_fraction=float(grounded[i]),
                         stopped_feet_speed_m_s=float(stopfeet[i]),max_tilt_rad=float(tilt[:,i].max())))
    summaries=[]
    for j,g in enumerate(groups):
        rr=rows[j*trial_count:(j+1)*trial_count]
        summaries.append(dict(**g,passed=sum(r['success'] for r in rr),completed=sum(r['complete'] for r in rr),
                              nominal=rr[0],velocity_rmse_m_s=float(np.mean([r['velocity_rmse_m_s'] for r in rr])),
                              yaw_rmse_rad_s=float(np.mean([r['yaw_rmse_rad_s'] for r in rr])),
                              stop_mean_m_s=float(np.mean([r['stop_mean_m_s'] for r in rr])),trials=rr))
    core=[g for g in summaries if g['core']]
    passed_core=sum(g['passed'] for g in core);total_core=len(core)*trial_count
    result=dict(trials_per_case=trial_count,cases=summaries,initial_velocity_exactly_zero=True,initial_pose='standard standing',
                seed=seed,duration_seconds=17.,reference_motion=False,official_policy=False,physics='MuJoCo Warp',
                sequence=[['stand',0,1.5],['command',1.5,5.5],['stop',5.5,8.5],['opposite_command',8.5,12.5],['stop',12.5,17]],
                core_passed=passed_core,core_total=total_core,completion_count=int(complete.sum()),trial_total=n,
                all_core_95=all(g['passed']/trial_count>=.95 and g['nominal']['success'] for g in core),
                quality=1000*passed_core/total_core+100*complete.mean()-10*float(track.mean())-float(stopmean.mean()))
    # Keep fixed nominal examples plus the first failing trial of every case.
    chosen=[]
    for j in range(len(groups)):
        chosen.append(j*trial_count)
        failed=np.where(~success[j*trial_count:(j+1)*trial_count])[0]
        if len(failed) and failed[0]!=0:chosen.append(j*trial_count+int(failed[0]))
    trajectory=dict(state=state[:,chosen],commands=cmd[:,chosen],alive=alive[:,chosen],actions=a[:,chosen],
                    case_index=np.array(chosen)//trial_count,trial_index=np.array(chosen)%trial_count,success=success[chosen],dt=DT)
    return result,trajectory

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--checkpoint');p.add_argument('--trials',type=int,default=16);args=p.parse_args()
    torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)))
    networks.AMP['enabled']=False
    run=Path(args.run);out=run/'eval';out.mkdir(exist_ok=True)
    env=CommandEnv('cuda',len(cases())*args.trials,evaluate=True,randomize=True)
    seen=set();best=-float('inf');stage=.5;stage_streak=0;success_streak=0
    while True:
        paths=sorted(run.glob('ckpt_*.pt')) if args.watch else [Path(args.checkpoint)]
        if args.watch and (run/'DONE').exists():paths += [run/'last.pt']
        pending=[x for x in paths if str(x) not in seen and x.exists()]
        for path in pending:
            start=time.time();ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['task_kind']=='b2_proprioceptive_locomotion_v1' and ck['source_checkpoint'] is None
            policy=make_policy().cuda();policy.load_state_dict(ck['policy']);policy.eval()
            result,traj=evaluate(policy,env,args.trials)
            result.update(it=ck['it'],checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          train_seconds=ck['train_time'],launch_wall_seconds=ck['launch_wall_seconds'],evaluated_unix=time.time(),eval_seconds=time.time()-start)
            atomic(out/f"it_{ck['it']:07d}.json",result);atomic(run/'latest_eval.json',result)
            with (run/'eval.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in result.items() if k!='cases'})+'\n')
            if result['quality']>best:
                best=result['quality'];shutil.copy2(path,run/'best.pt');atomic(run/'best_eval.json',result)
                np.savez_compressed(run/'best_rollout.npz',**traj)
                # Pure CPU TorchScript export shares normalization and action bounds.
                from export import export_checkpoint
                export_checkpoint(run/'best.pt',run/'deploy',result)
            current=[g for g in result['cases'] if g['core'] and g['level']<=stage]
            stage_ok=all(g['passed']/args.trials>=.875 and g['nominal']['success'] for g in current)
            stage_streak=stage_streak+1 if stage_ok else 0
            if stage_streak>=2 and stage<2.:
                stage=min(2.,stage+.5);stage_streak=0
                atomic(run/'curriculum.json',dict(speed=stage,yaw_std=min(1.5,.35+.77*(stage-.5)),trigger_iteration=ck['it'],reason='two physical evaluations passed current command region'))
            success_streak=success_streak+1 if result['all_core_95'] else 0
            if success_streak>=2:atomic(run/'TARGET_MET.json',dict(it=ck['it'],reason='two held-out evaluations with every core case >=95%'));(run/'STOP').touch()
            print(json.dumps({k:v for k,v in result.items() if k!='cases'})+' stage='+str(stage),flush=True)
            seen.add(str(path));del ck,policy
        finished=(run/'DONE').exists() or (run/'launch_timing.json').exists()
        if not args.watch or (finished and not pending):break
        time.sleep(5)
    atomic(run/'EVAL_DONE.json',dict(time=time.time(),best_quality=best))

if __name__=='__main__':main()

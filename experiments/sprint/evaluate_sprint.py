"""Independent sustained-speed tests; initial motion and measured window explicit."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.motion_lib import MotionLibrary
from supermimic.evaluate import load_supermimic_policy
from supermimic.rotations import mat_to_quat,quat_from_euler_xyz,quat_mul
from sprint_task import make_env
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jump06'))
from evaluate_jump import geometry

@torch.no_grad()
def run_trials(policy,lib,env,trials_per_speed=8,seed=20261027,zero_velocity=False,min_success_rate=.875):
    torch.manual_seed(seed);n=env.num_envs;dev=env.device
    frames=lib.starts.repeat_interleave(trials_per_speed)
    ids=torch.arange(n,device=dev)
    env.reset();env.reset_envs(ids,frames=frames)
    ref=lib.frame(frames)
    pos,quat,q=ref['pos'].clone(),ref['quat'].clone(),ref['q'].clone()
    perturbed=(ids%trials_per_speed)!=0;k=int(perturbed.sum())
    pos[perturbed,:2]+=(torch.rand(k,2,device=dev)*2-1)*.005
    q[perturbed]+=(torch.rand(k,29,device=dev)*2-1)*.01
    angles=(torch.rand(k,3,device=dev)*2-1)*.01
    quat[perturbed]=quat_mul(quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2]),quat[perturbed])
    v,w,qd=ref['v'].clone(),ref['w'].clone(),ref['qd'].clone()
    if zero_velocity:v.zero_();w.zero_();qd.zero_()
    env.write_state(ids,pos,quat,q,v,w,qd);env._finish_resets(ids,force=True)
    active=torch.ones(n,dtype=torch.bool,device=dev);failed=torch.zeros_like(active);complete=torch.zeros_like(active)
    ret=torch.zeros(n,device=dev);alive_steps=torch.zeros(n,device=dev)
    records=[{k:v.clone() for k,v in env.state.items()}]
    steps=int(lib.lengths[0])-1
    for t in range(steps):
        o,g=env.policy_inputs();state,info=env.step(policy(o,g))
        records.append({k:torch.where(active.reshape((n,)+(1,)*(v.ndim-1)),v,records[-1][k]) for k,v in state.items()})
        ret+=info['reward']*active;alive_steps+=active
        failed|=info['terminated']&active;complete|=info['truncated']&active&(t==steps-1)
        active&=~info['done']
        if not active.any():break
    while len(records)<steps+1:records.append(records[-1])
    s={k:torch.stack([r[k] for r in records]) for k in records[0]}
    xy,h,body_z=geometry(env,s)
    p=s['pos'].cpu().numpy();vel=s['v'].cpu().numpy()
    tilt=torch.acos(s['rot'][...,2,2].clamp(-1,1)).cpu().numpy()
    completed=complete.cpu().numpy();terminated=failed.cpu().numpy()
    returns=(ret/(steps*env.tcfg.max_step_reward)).cpu().numpy()
    survival=alive_steps.cpu().numpy()/50
    speeds=np.repeat([float(name.split('_')[1][:-3]) for name in lib.names],trials_per_speed)
    warm=100;seconds=(steps-warm)/50
    trials=[]
    for i,target in enumerate(speeds):
        measured=(p[-1,i,0]-p[warm,i,0])/seconds
        lateral=float(np.max(np.abs(p[warm:,i,1]-p[warm,i,1])))
        rolling=(p[warm+50:,i,0]-p[warm:-50,i,0])
        air=h[warm:,i].min(-1)>.01
        flights=int(np.sum(np.diff(np.r_[False,air].astype(int))==1))
        left=int(np.sum((h[warm:,i,0]<.01)&(h[warm:,i,1]>.025)))
        right=int(np.sum((h[warm:,i,1]<.01)&(h[warm:,i,0]>.025)))
        upright=bool(p[warm:,i,2].min()>.45 and tilt[warm:,i].max()<.75 and body_z[:,i].min()>.07)
        success=bool(completed[i] and not terminated[i] and .85*target<=measured<=1.15*target and rolling.min()>.7*target and lateral<.5 and upright and flights>=5 and left>=10 and right>=10)
        trials.append(dict(trial=int(i%trials_per_speed),target_m_s=float(target),success=success,complete=bool(completed[i]),terminated=bool(terminated[i]),survival_s=float(survival[i]),mean_speed_m_s=float(measured),min_1s_speed_m_s=float(rolling.min()),max_1s_speed_m_s=float(rolling.max()),max_lateral_m=lateral,min_root_height_m=float(p[warm:,i,2].min()),max_tilt_rad=float(tilt[warm:,i].max()),min_body_height_m=float(body_z[:,i].min()),flight_events=flights,left_support_frames=left,right_support_frames=right,upright=upright,norm_return=float(returns[i])))
    groups=[]
    for c,name in enumerate(lib.names):
        rows=trials[c*trials_per_speed:(c+1)*trials_per_speed];ok=[r for r in rows if r['success']]
        groups.append(dict(name=name,target_m_s=rows[0]['target_m_s'],success_rate=len(ok)/len(rows),completion_rate=float(np.mean([r['complete'] for r in rows])),mean_speed_m_s=float(np.mean([r['mean_speed_m_s'] for r in rows])),successful_mean_speed_m_s=float(np.mean([r['mean_speed_m_s'] for r in ok])) if ok else 0.,nominal=rows[0]))
    stable=[g for g in groups if g['success_rate']>=min_success_rate and g['nominal']['success']]
    result=dict(seed=seed,trials_per_speed=trials_per_speed,min_success_rate=min_success_rate,initialization='zero velocity in reference pose' if zero_velocity else 'reference running pose and velocities',duration_s=steps/50,measured_window_s=[2.,steps/50],max_stable_target_m_s=max([g['target_m_s'] for g in stable],default=0.),max_stable_measured_m_s=max([g['successful_mean_speed_m_s'] for g in stable],default=0.),mean_norm_return=float(returns.mean()),groups=groups,trials=trials)
    traj=dict(qpos=torch.cat([s['pos'],mat_to_quat(s['rot']),s['q']],dim=-1).cpu().numpy(),root_velocity=vel,foot_xy=xy,foot_height=h,targets=speeds,success=np.array([r['success'] for r in trials]),fps=np.array(50))
    return result,traj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--out',required=True);ap.add_argument('--run');ap.add_argument('--runs',nargs='+');ap.add_argument('--watch',action='store_true');ap.add_argument('--trials-per-speed',type=int,default=8);ap.add_argument('--seed',type=int,default=20261027);ap.add_argument('--min-success-rate',type=float,default=.875);ap.add_argument('--zero-velocity',action='store_true');ap.add_argument('--stop-at-target',type=float);args=ap.parse_args()
    if args.stop_at_target is not None and (not args.watch or not args.run or args.runs):
        ap.error('--stop-at-target requires --watch --run for one training run')
    networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True
    lib=MotionLibrary('data/processed/g1_sprint_ladder.npz','cuda');env=make_env(lib,'cuda',lib.num_clips*args.trials_per_speed,evaluate=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);seen=set();best={}
    run_dirs=[Path(r) for r in (args.runs or [args.run])] if args.watch else []
    while True:
        candidates=sorted([p for run in run_dirs for p in run.glob('ckpt_*.pt')],key=lambda p:p.stat().st_mtime) if args.watch else [Path(args.checkpoint)]
        todo=[c for c in candidates if str(c) not in seen]
        for path in todo:
            evaluation_start=time.time()
            policy,ck=load_supermimic_policy(str(path),'cuda')
            result,traj=run_trials(policy,lib,env,args.trials_per_speed,args.seed,args.zero_velocity,args.min_success_rate)
            result.update(checkpoint=str(path),it=ck['it'],train_seconds=ck['train_time'],world=ck.get('world',1),checkpoint_wall_seconds=ck.get('wall_seconds'),launch_wall_seconds=ck.get('launch_wall_seconds'),evaluated_unix=time.time(),evaluation_wall_seconds=time.time()-evaluation_start)
            run_out=out/path.parent.name if args.runs else out;run_out.mkdir(parents=True,exist_ok=True)
            dest=run_out/f'it_{ck["it"]:07d}' if args.watch else run_out;dest.mkdir(parents=True,exist_ok=True)
            (dest/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
            np.savez_compressed(dest/'rollout.npz',**traj)
            with (run_out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('trials','groups')})+' groups='+str([(g['target_m_s'],g['success_rate'],round(g['mean_speed_m_s'],2)) for g in result['groups']]),flush=True)
            score=result['max_stable_measured_m_s']+.001*result['mean_norm_return']
            if args.watch and score>best.get(str(run_out),-1.):
                best[str(run_out)]=score;shutil.copy2(path,run_out/'selected.pt');(run_out/'selected_eval.json').write_text(json.dumps(result,indent=2)+'\n')
            if args.stop_at_target is not None and result['max_stable_target_m_s']>=args.stop_at_target and not (run_out/'accepted.pt').exists():
                shutil.copy2(path,run_out/'accepted.pt')
                shutil.copy2(dest/'rollout.npz',run_out/'accepted_rollout.npz')
                (run_out/'accepted_eval.json').write_text(json.dumps(result,indent=2)+'\n')
                Path(args.run,'STOP').touch()
                print('Acceptance criterion met; requested training stop.',flush=True)
            seen.add(str(path));del policy,ck
        if not args.watch or (all((r/'DONE').exists() for r in run_dirs) and not todo):break
        time.sleep(5)

if __name__=='__main__':main()

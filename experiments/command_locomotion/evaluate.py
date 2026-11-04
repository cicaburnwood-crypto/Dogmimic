"""Held-out zero-velocity start/stop/restart tests in actual MuJoCo physics."""
import argparse,json,time,shutil,os
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.rotations import mat_to_quat
from train import make_policy
from task import CommandEnv

@torch.no_grad()
def evaluate(policy,env,speeds,trials,seed):
    torch.manual_seed(seed);n=env.num_envs;dev=env.device
    ids=torch.arange(n,device=dev)
    s=env.reset()
    q=s['q'].clone();q[ids%trials!=0]+=(torch.rand_like(q[ids%trials!=0])*2-1)*.015
    env.write_state(ids,s['pos'],mat_to_quat(s['rot']),q,torch.zeros_like(s['v']),torch.zeros_like(s['w']),torch.zeros_like(s['qd']))
    env._finish_resets(ids,force=True)
    initial_max_velocity=max(float(env.state[k].abs().max()) for k in ('v','w','qd'))
    targets=torch.tensor(speeds,device=dev).repeat_interleave(trials)
    active=torch.ones(n,dtype=torch.bool,device=dev);failed=~active
    states=[];commands=[];alive=[]
    for t in range(800):
        cmd=torch.zeros(n,3,device=dev)
        if 50<=t<250 or 400<=t<600:cmd[:,0]=targets
        env.set_command(cmd)
        obs,goal=env.policy_inputs();s,info=env.step(policy(obs,goal))
        states.append({k:v.clone() for k,v in s.items()});commands.append(cmd.clone());alive.append(active.clone())
        failed|=info['terminated']&active;active&=~info['done']
    s={k:torch.stack([x[k] for x in states]) for k in states[0]}
    p=s['pos'].cpu().numpy();v=s['v'].cpu().numpy();rot=s['rot'].cpu().numpy()
    vel=np.linalg.norm(v[...,:2],axis=-1)
    tilt=np.arccos(np.clip(rot[...,2,2],-1,1))
    failed=failed.cpu().numpy();rows=[]
    for i,target in enumerate(np.repeat(speeds,trials)):
        moving=[float(v[a:b,i,0].mean()) for a,b in ((150,250),(500,600))]
        stop_mean=[float(vel[a:b,i].mean()) for a,b in ((350,400),(700,800))]
        stop_peak=[float(vel[a:b,i].max()) for a,b in ((350,400),(700,800))]
        drift=[float(np.linalg.norm(p[b-1,i,:2]-p[a,i,:2])) for a,b in ((350,400),(700,800))]
        stable=not bool(failed[i]) and p[:,i,2].min()>.5 and tilt[:,i].max()<.6
        track=all(abs(x-target)<=max(.12,.2*target) for x in moving)
        stop=all(x<.15 for x in stop_mean) and all(x<.25 for x in stop_peak) and max(drift)<.2
        lateral=float(np.abs(p[:,i,1]).max())
        ok=stable and track and stop and lateral<.7
        rows.append(dict(target_m_s=float(target),trial=i%trials,success=bool(ok),survived=not bool(failed[i]),stable=bool(stable),
                         movement_means_m_s=moving,stop_mean_m_s=stop_mean,stop_peak_m_s=stop_peak,stop_drift_m=drift,
                         lateral_max_m=lateral,min_root_z_m=float(p[:,i,2].min()),max_tilt_rad=float(tilt[:,i].max())))
    groups=[]
    for j,target in enumerate(speeds):
        rr=rows[j*trials:(j+1)*trials]
        groups.append(dict(target_m_s=target,success_rate=float(np.mean([r['success'] for r in rr])),
                           survival_rate=float(np.mean([r['survived'] for r in rr])),
                           moving_m_s=np.mean([r['movement_means_m_s'] for r in rr],0).tolist(),
                           stopped_m_s=np.mean([r['stop_mean_m_s'] for r in rr],0).tolist(),nominal=rr[0]))
    passed=[g['target_m_s'] for g in groups if g['success_rate']>=.9 and g['nominal']['success']]
    result=dict(seed=seed,initial_max_velocity=initial_max_velocity,reference_motion=False,trials_per_speed=trials,
                duration_s=16,sequence=[['stop',0,1],['go',1,5],['stop',5,8],['go',8,12],['stop',12,16]],
                max_passed_speed=max(passed,default=0),groups=groups,trials=rows)
    traj=dict(qpos=torch.cat([s['pos'],mat_to_quat(s['rot']),s['q']],-1).cpu().numpy(),root_velocity=v,
              commands=torch.stack(commands).cpu().numpy(),active=torch.stack(alive).cpu().numpy(),
              targets=np.repeat(speeds,trials),success=np.array([r['success'] for r in rows]),fps=np.array(50))
    return result,traj

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--run');p.add_argument('--out',required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--trials',type=int,default=16)
    p.add_argument('--speeds',type=float,nargs='+',default=[.3,.5,1.,1.5,2.,3.]);p.add_argument('--seed',type=int,default=20261101)
    p.add_argument('--stop-at-speed',type=float,default=0.);args=p.parse_args()
    os.sched_setaffinity(0,set(range(24,28)));torch.set_num_threads(2)
    networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True
    env=CommandEnv('cuda',len(args.speeds)*args.trials,evaluate=True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);seen=set();best=-1
    while True:
        paths=sorted(Path(args.run).glob('ckpt_*.pt')) if args.watch else [Path(args.checkpoint)]
        pending=[path for path in paths if str(path) not in seen]
        for path in pending:
            start=time.time();ck=torch.load(path,map_location='cuda',weights_only=False)
            assert ck['task_kind']=='reference_free_command_v1'
            policy=make_policy().to('cuda');policy.load_state_dict(ck['policy']);policy.eval()
            res,traj=evaluate(policy,env,args.speeds,args.trials,args.seed)
            res.update(checkpoint=str(path),it=ck['it'],train_seconds=ck['train_time'],world=ck['world'],
                       launch_wall_seconds=ck['launch_wall_seconds'],evaluated_unix=time.time(),eval_seconds=time.time()-start)
            dest=out/f'it_{ck["it"]:07d}';dest.mkdir(parents=True,exist_ok=True)
            (dest/'evaluation.json').write_text(json.dumps(res,indent=2));np.savez_compressed(dest/'rollout.npz',**traj)
            with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(res)+'\n')
            print(json.dumps({k:v for k,v in res.items() if k not in ('groups','trials')})+' groups='+str([(g['target_m_s'],g['success_rate'],g['survival_rate'],[round(v,3) for v in g['moving_m_s']],[round(v,3) for v in g['stopped_m_s']]) for g in res['groups']]),flush=True)
            score=res['max_passed_speed']+sum(g['success_rate']*.001 for g in res['groups'])
            if score>best:
                best=score;shutil.copy2(path,out/'selected.pt');(out/'selected_eval.json').write_text(json.dumps(res,indent=2))
            if args.stop_at_speed and res['max_passed_speed']>=args.stop_at_speed:
                Path(args.run,'STOP').touch()
            seen.add(str(path));del ck,policy
        if not args.watch or (Path(args.run,'DONE').exists() and not pending):break
        time.sleep(5)

if __name__=='__main__':main()

"""Independent frozen physical acceptance, with a clean-policy input adapter.

Legacy code is evaluation-only. No training process imports these branches.
Training-baseline physics and conservative hardware-informed stress physics are
explicitly different test domains, and are never pooled into one metric.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path
from common import HERE,ROOT,sha256,write_json

CRITERIA={
    'B2':('experiments/b2_cycle/evaluate_cycle.py','5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'),
    'G1':('experiments/g1_contact_repair/evaluate_jump.py','333379431a2dd84a49a786517b379a5ff9157c36f1ba379c26afce54357db2a6'),
}

def run(robot, checkpoint, out, full=False, pack='DEV-A', only_case=None, trials=None):
    import torch
    import numpy as np
    torch.backends.cuda.matmul.allow_tf32=True;torch.set_num_threads(2)
    cp=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    digest=sha256(cp)
    extra={'metadata.json':''}
    policy=torch.jit.load(str(cp),map_location='cuda',_extra_files=extra).eval()
    meta=json.loads(extra['metadata.json'])
    assert meta['backend']=='NJU' and meta['robot']==robot
    ck={'it':meta['iteration']}
    q0=torch.tensor(meta['q0'],device='cuda')
    scale=torch.tensor(meta['scale'],device='cuda')
    goals=torch.tensor(meta['goal_feat'],device='cuda')
    path,expected=CRITERIA[robot]
    assert sha256(ROOT/path)==expected
    manifest=json.loads((HERE/'evaluation_manifest.json').read_text())
    if robot=='B2':
        os.environ['SM_ROBOT']='b2'
        paths=['b2_light_payload','three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills']
        sys.path[:0]=[str(ROOT/'experiments'/p) for p in paths]
        from robust_env import make_env
        from cycle_env import CycleLibrary
        from evaluate_cycle import evaluate as run_trials
        lib=CycleLibrary(ROOT/'experiments/b2_cycle/b2_full_cycle.npz','cuda')
        env=make_env(lib,'cuda',trials or 64,evaluate=True,kg=0.)
        cases=manifest['cases']['B2'] if full else manifest['cases']['B2'][:2]
    else:
        os.environ.pop('SM_ROBOT',None)
        sys.path[:0]=[str(ROOT/'experiments/g1_boundary_repair'),str(ROOT/'experiments/g1_boundary_repair/engine')]
        from robust_env import RobustEnv
        from supermimic.motion_lib import MotionLibrary
        from evaluate_jump import run_trials
        lib=MotionLibrary(ROOT/'data/processed/g1_standing_jump_060m.npz','cuda')
        env=RobustEnv(lib,'cuda',trials or 32,evaluate=True)
        cases=manifest['cases']['G1']
    from supermimic import networks as old_networks
    import inspect
    assert sha256(inspect.getfile(inspect.unwrap(run_trials)))==expected,'Loaded acceptance differs from the frozen evaluator'
    old_networks.AMP['enabled']=True
    class Adapter:
        def __init__(self):self.hist=None;self.last_ep=None;self.prev=None
        def eval(self):policy.eval();return self
        def train(self):return self
        def reset(self):self.hist=None;self.last_ep=None;self.prev=None
        def __call__(self,*ignored):
            s=env.state;n=env.num_envs;nj=len(q0)
            if self.prev is None:self.prev=torch.zeros(n,nj,device=env.device)
            reset=torch.ones(n,dtype=torch.bool,device=env.device) if self.last_ep is None else env.ep_id!=self.last_ep
            self.prev[reset]=0.
            # IMU body-frame gyro/gravity, joint encoder and last clean NJU action only.
            gyro=torch.einsum('nji,nj->ni',s['rot'],s['w'])
            gravity=-s['rot'][:,2,:]
            obs=torch.cat([s['q']+env.enc_bias-q0,s['qd'],gyro,gravity,self.prev],-1)
            if self.hist is None:self.hist=obs[:,None,:].repeat(1,10,1)
            else:self.hist=torch.cat([self.hist[:,1:],obs[:,None]],1)
            self.hist[reset]=obs[reset,None,:]
            chunks=[];st=0
            for width in (nj,nj,3,3,nj):
                chunks.append(self.hist[:,:,st:st+width].reshape(n,-1));st+=width
            idx=(env.frame[:,None]+torch.arange(10,device=env.device)[None]).clamp(max=len(goals)-1)
            a=policy(torch.cat(chunks,-1),goals[idx].flatten(1))
            self.prev=a.clone();self.last_ep=env.ep_id.clone()
            # Preserve the NJU absolute setpoint under the frozen evaluator's affine action map.
            return (q0+a*scale-env.q0)/env.action_scale

    adapted=Adapter();rows=[];started=time.time()
    if only_case:cases=[c for c in cases if c['case_id']==only_case]
    for case in cases:
        seed=manifest['batch_seeds'][robot][pack][case['case_id']]
        adapted.reset()
        with torch.no_grad():
            if robot=='B2':
                env.fixed_loads.fill_(case['nominal_load_kg'])
                env.fixed_mass=case['mass_multiplier'];env.fixed_friction=case['friction']
                env.asymmetric_eval=case['asymmetric'];env.max_speed.zero_();env.max_power.zero_()
                result,trajectory=run_trials(adapted,env,seed=seed)
                result.update(passed=round(result['success_rate']*env.num_envs),completed=round(result['completion_rate']*env.num_envs),
                              cycle_passed=round(result['cycle_success_rate']*env.num_envs),
                              peak_speed=float(env.max_speed.max()),peak_mechanical_power=float(env.max_power.max()),
                              trial_peak_speed=env.max_speed.cpu().tolist(),trial_peak_power=env.max_power.cpu().tolist())
                result['actuator_pass']=result['peak_speed']<=6. and result['peak_mechanical_power']<=1501.
            else:
                env.case=case['parameters']
                result,trajectory=run_trials(adapted,lib,env,seed=seed)
                result.update(passed=sum(t['success'] for t in result['trials']),completed=sum(t['complete'] for t in result['trials']))
        result.update(case=case,case_id=case['case_id'],batch_seed=seed)
        rows.append(result)
        np.savez_compressed(out/(case['case_id']+'_rollout.npz'),**trajectory)
        write_json(out/'eval_progress.json',dict(cases_done=len(rows),cases_total=len(cases),case_id=case['case_id'],updated=time.time(),it=ck['it']))
        print('PHYSICAL_CASE',robot,case['case_id'],result['passed'],env.num_envs,flush=True)
    full_count=48 if robot=='B2' else 21
    complete_coverage=len(rows)==full_count and env.num_envs==(64 if robot=='B2' else 32)
    if robot=='B2':
        ok=all(r['passed']>=61 and r['nominal_success'] and r['actuator_pass'] for r in rows)
    else:ok=all(r['passed']>=31 for r in rows) and any(r['case_id']=='nominal' and r['nominal_success'] for r in rows)
    assert sha256(cp)==digest
    report=dict(robot=robot,it=ck['it'],pack=pack,domain='unchanged_strict_physics',full=full,
                checkpoint=str(cp),checkpoint_sha256=digest,complete_coverage=complete_coverage,
                all_passed=bool(ok and complete_coverage),passed=sum(r['passed'] for r in rows),
                total=len(rows)*env.num_envs,completed=sum(r['completed'] for r in rows),conditions=rows,
                elapsed_seconds=time.time()-started,finished=time.time(),evaluator_sha256=expected,backend='NJU',source_checkpoint_sha256=meta['checkpoint_sha256'])
    write_json(out/'result.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--robot',choices=['B2','G1'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--pack',default='DEV-A')
    p.add_argument('--case');p.add_argument('--trials',type=int)
    a=p.parse_args();run(a.robot,a.checkpoint,a.out,a.full,a.pack,a.case,a.trials)

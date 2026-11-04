"""Immutable-checkpoint wrapper around unchanged physical success functions."""
import argparse
import json
import time
from pathlib import Path
from common import ROOT, DESIGN, setup, read_json, write_json, sha256, isolated_rng


def evaluate(robot, checkpoint, output, pack='DEV-A', full=True, only_cases=None):
    setup(robot)
    import torch
    import numpy as np
    from supermimic import networks
    from supermimic.motion_lib import MotionLibrary
    torch.set_num_threads(2)
    networks.AMP['enabled']=True
    torch.backends.cuda.matmul.allow_tf32=True
    expected=read_json(DESIGN/'experiment_matrix.json')['robots'][robot]['acceptance']
    assert sha256(ROOT/expected['file'])==expected['sha256']
    manifest=read_json(DESIGN/'evaluation_manifest.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cp=Path(checkpoint);digest=sha256(cp)
    ck=torch.load(cp,map_location='cuda',weights_only=False)
    if robot=='B2':
        from stage_policy import from_checkpoint
        from robust_env import make_env
        from evaluate_cycle import evaluate as run_trials
        from cycle_env import CycleLibrary
        lib=CycleLibrary('experiments/b2_cycle/b2_full_cycle.npz','cuda')
        env=make_env(lib,'cuda',64,evaluate=True,kg=0.)
        policy=from_checkpoint(ck,'cuda')
        cases=manifest['cases'][robot] if full else manifest['cases'][robot][:2]
    else:
        from supermimic.evaluate import load_supermimic_policy
        from robust_env import RobustEnv
        from evaluate_jump import run_trials
        lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cuda')
        env=RobustEnv(lib,'cuda',32,evaluate=True)
        policy,_=load_supermimic_policy(str(cp),'cuda')
        cases=manifest['cases'][robot]
    if only_cases:
        cases=[c for c in cases if c['case_id'] in only_cases]
    rows=[];start=time.time()
    for case in cases:
        seed=manifest['batch_seeds'][robot][pack][case['case_id']]
        policy.eval()
        with torch.no_grad(),isolated_rng('cuda'):
            if robot=='B2':
                env.fixed_loads.fill_(case['nominal_load_kg'])
                env.fixed_mass=case['mass_multiplier'];env.fixed_friction=case['friction']
                env.asymmetric_eval=case['asymmetric'];env.max_speed.zero_();env.max_power.zero_()
                result,trajectory=run_trials(policy,env,seed=seed)
                result.update(passed=round(result['success_rate']*64),
                              completed=round(result['completion_rate']*64),
                              cycle_passed=round(result['cycle_success_rate']*64),
                              peak_speed=float(env.max_speed.max()),
                              peak_mechanical_power=float(env.max_power.max()),
                              trial_peak_speed=env.max_speed.cpu().tolist(),
                              trial_peak_power=env.max_power.cpu().tolist())
                result['actuator_pass']=result['peak_speed']<=6 and result['peak_mechanical_power']<=1501
            else:
                env.case=case['parameters']
                result,trajectory=run_trials(policy,lib,env,seed=seed)
                result.update(passed=sum(t['success'] for t in result['trials']),
                              completed=sum(t['complete'] for t in result['trials']))
        result.update(case=case,case_id=case['case_id'],batch_seed=seed)
        np.savez_compressed(output/(case['case_id']+'_rollout.npz'),**trajectory)
        rows.append(result)
        write_json(output/'eval_progress.json',dict(robot=robot,cases_done=len(rows),cases_total=len(cases),
                   case_id=case['case_id'],checkpoint_sha256=digest,updated=time.time()))
        print(json.dumps(dict(robot=robot,case=case['case_id'],passed=result['passed'],
                              completed=result['completed'],cases_done=len(rows))),flush=True)
    assert sha256(cp)==digest,'Evaluated checkpoint changed'
    counts=[r['passed'] for r in rows]
    complete_coverage=len(rows)==(48 if robot=='B2' else 21)
    if robot=='B2':
        passed=all(r['passed']>=61 and r['nominal_success'] and r['actuator_pass'] for r in rows)
    else:
        passed=all(r['passed']>=31 for r in rows) and any(r['case_id']=='nominal' and r['nominal_success'] for r in rows)
    report=dict(robot=robot,pack=pack,it=int(ck.get('it',0)),actor_updates=ck.get('actor_updates',0),
                source_iteration=ck.get('source_iteration'),checkpoint=str(cp),checkpoint_sha256=digest,
                complete_coverage=complete_coverage,all_passed=bool(passed and complete_coverage),
                passed=sum(counts),total=len(rows)*(64 if robot=='B2' else 32),
                completed=sum(r['completed'] for r in rows),conditions=rows,
                elapsed_seconds=time.time()-start,finished=time.time(),evaluator_sha256=expected['sha256'])
    write_json(output/'result.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--robot',required=True,choices=['B2','G1'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pack',default='DEV-A');p.add_argument('--basic',action='store_true')
    p.add_argument('--case',action='append')
    a=p.parse_args()
    evaluate(a.robot,a.checkpoint,a.out,a.pack,not a.basic,a.case)

"""Supplementary crossed-condition evaluation; never changes active training.

Uses the existing full-cycle success function and actuator thresholds unchanged.
Invoke only when evaluation GPU capacity is available. --plan needs no GPU.
"""
import argparse,hashlib,itertools,json,os,sys,time
from pathlib import Path

def cases():
    return [dict(kg=kg,friction=fr,mass_scale=m,asymmetric=asym)
            for kg,fr,m,asym in itertools.product([0,5,10,15],[.4,.7,1.,1.3],[.9,1.,1.1],[False,True])]

def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',action='store_true');a.add_argument('--check-load',action='store_true');a.add_argument('--environment',choices=['b2_actuator_robust_v2','b2_effective_action_candidate','b2_balance_repair','b2_power_repair']);a.add_argument('--checkpoint');a.add_argument('--out');args=a.parse_args()
    grid=cases()
    if args.plan:
        print(json.dumps(dict(cases=len(grid),trials_per_case=64,total_trials=len(grid)*64,conditions=grid),indent=2));return
    if not args.checkpoint or (not args.out and not args.check_load):a.error('--checkpoint and --out are required for physical evaluation')
    if not args.environment:a.error('--environment is required: select the checkpoint training environment explicitly')
    root=Path(__file__).resolve().parents[2];os.chdir(root);os.environ['SM_ROBOT']='b2'
    sys.path[:0]=[str(root/'experiments'/n) for n in [args.environment,'three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills']]
    import torch
    from supermimic import networks
    from cycle_env import CycleLibrary as MotionLibrary
    from stage_policy import from_checkpoint
    import robust_env
    from robust_env import make_env,SPEED,POWER
    env_file=Path(robust_env.__file__).resolve()
    assert env_file == root/'experiments'/args.environment/'robust_env.py', env_file
    provenance=dict(environment=args.environment,environment_file=str(env_file),environment_sha256=hashlib.sha256(env_file.read_bytes()).hexdigest())
    from evaluate_cycle import evaluate,CRITERIA
    torch.set_num_threads(2);networks.AMP['enabled']=not args.check_load
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if args.check_load:
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);p=from_checkpoint(ck,'cpu').eval()
        n=p.teacher.obs_norm.mean.numel();g=p.teacher.goal_norm.mean.numel()
        with torch.no_grad():out=p(torch.zeros(2,n),torch.zeros(2,g+1))
        assert out.shape==(2,12) and torch.isfinite(out).all()
        print(json.dumps(dict(**provenance,passed=True,iteration=ck['it'],policy_output_shape=list(out.shape),cases=len(grid),unchanged_criteria=CRITERIA,speed_threshold=SPEED,power_threshold=POWER+1)));return
    cp=Path(args.checkpoint).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    sha=hashlib.sha256(cp.read_bytes()).hexdigest()
    ck=torch.load(cp,map_location='cuda',weights_only=False);policy=from_checkpoint(ck,'cuda').eval()
    lib=MotionLibrary('experiments/b2_cycle/b2_full_cycle.npz','cuda');env=make_env(lib,'cuda',64,evaluate=True,kg=0.)
    start=time.time();rows=[]
    for case in grid:
        env.fixed_loads.fill_(case['kg']);env.fixed_mass=case['mass_scale'];env.fixed_friction=case['friction'];env.asymmetric_eval=case['asymmetric'];env.max_speed.zero_();env.max_power.zero_()
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):result,_=evaluate(policy,env,seed=20261103)
        result.update(case,peak_speed=float(env.max_speed.max()),peak_mechanical_power=float(env.max_power.max()))
        result['actuator_pass']=result['peak_speed']<=SPEED and result['peak_mechanical_power']<=POWER+1
        result['passed']=result['success_rate']>=.95 and result['nominal_success'] and result['actuator_pass'];rows.append(result)
        report=dict(**provenance,checkpoint=str(cp),checkpoint_sha256=sha,it=ck['it'],criteria=CRITERIA,cases_done=len(rows),cases_total=len(grid),complete=len(rows)==len(grid),all_passed=len(rows)==len(grid) and all(x['passed'] for x in rows),elapsed_s=time.time()-start,conditions=rows)
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'report.json')
        print(json.dumps(dict(case=case,success_rate=result['success_rate'],actuator_pass=result['actuator_pass'])),flush=True)
if __name__=='__main__':main()

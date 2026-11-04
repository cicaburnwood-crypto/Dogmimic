import json,time,hashlib,subprocess,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
for device in ['cpu','cuda']:
 d=json.loads((root/f'runs/b2_balance_repair_{device}_smoke/report.json').read_text());assert d['passed'] and d['worst_joint_gradient_verified'] and d['physical_torque_identical']
assert json.loads((root/'runs/b2_balance_physics_audit/report.json').read_text())['passed_implementation_check']
old=root/'runs/b2_effective_continue_20261101_213330'
assert (old/'EVAL_DONE.json').exists()
assert not Path('/proc/4058356').exists()
assert hashlib.sha256((root/'experiments/b2_cycle/evaluate_cycle.py').read_bytes()).hexdigest()=='5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
actor=root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt'
raw=actor.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='24f489e37949788dcef81f1af0f68de6556886685b6b9ca17803c2a430b37a85'
out=root/'runs'/time.strftime('b2_balance_repair_%Y%m%d_%H%M%S');out.mkdir();(out/'source.pt').write_bytes(raw)
for n in ['best.pt','best_eval.json','latest_eval.json','EVAL_DONE.json']:
 shutil.copy2(old/n,out/('previous_'+n))
shutil.copytree(p,out/'source_code',ignore=shutil.ignore_patterns('__pycache__','latest_launch.json','monitor_state.json'))
env=os.environ.copy();env.update(B2_POLICY_SOURCE=str(out/'source.pt'),B2_WM_PREFIT='2000')
cmd=['bash',str(p/'run.sh'),'--out',str(out)]
with (out/'console.log').open('ab',buffering=0) as f:
 proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(out=str(out),pid=proc.pid,gpu=6,started=time.time(),iterations=20000,num_envs=1024,wm_prefit=2000,horizon=32,source_sha256=sha,initialization='own v3 actor rollback; fresh WM/replay/optimizers; frozen own actor anchor',repairs=['remove artificial absolute target slew that suppresses balancing','worst joint endpoint speed penalty','enveloped torque power reward','preserve worst loaded full cycle in model selection'],acceptance_speed=6,acceptance_power=1501,command_power=1200,loads='25% zero; remaining continuous5-15kg',push_n=[8,16],command=cmd)
(out/'launch.json').write_text(json.dumps(d,indent=2));(p/'latest_launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

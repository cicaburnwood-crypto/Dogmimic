import json,time,hashlib,subprocess,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
for device in ['cpu','cuda']:
 d=json.loads((root/f'runs/b2_power_repair_{device}_smoke/report.json').read_text());assert d['passed'] and d['worst_joint_gradient_verified'] and d['physical_torque_identical'] and d['power_penalty_nonzero_gradient']
old=root/'runs/b2_balance_repair_20261101_230646'
assert (old/'EVAL_DONE.json').exists()
assert not Path('/proc/4117482').exists()
assert hashlib.sha256((root/'experiments/b2_cycle/evaluate_cycle.py').read_bytes()).hexdigest()=='5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
actor=root/'runs/b2_balance_repair_20261101_230646/best.pt'
raw=actor.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='5f99e59307296f9e234cb16c60a2175b80346b7c274d0f934a31b0607f9aa643'
out=root/'runs'/time.strftime('b2_power_repair_%Y%m%d_%H%M%S');out.mkdir();(out/'source.pt').write_bytes(raw)
for n in ['best.pt','best_eval.json','latest_eval.json','EVAL_DONE.json']:
 shutil.copy2(old/n,out/('previous_'+n))
shutil.copytree(p,out/'source_code',ignore=shutil.ignore_patterns('__pycache__','latest_launch.json','monitor_state.json'))
env=os.environ.copy();env.update(B2_POLICY_SOURCE=str(out/'source.pt'),B2_WM_PREFIT='500')
cmd=['bash',str(p/'run.sh'),'--out',str(out)]
with (out/'console.log').open('ab',buffering=0) as f:
 proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(out=str(out),pid=proc.pid,gpu=6,started=time.time(),iterations=20000,num_envs=1024,wm_prefit=500,horizon=32,source_sha256=sha,initialization='own balance-repair best Actor and WM continued; fresh replay/optimizers; frozen own anchor',repairs=['replace identically-zero capped-power penalty with uncapped demand plus pre-torque/post-speed power'],acceptance_speed=6,acceptance_power=1501,command_power=1200,loads='25% zero; remaining continuous5-15kg',push_n=[8,16],command=cmd)
(out/'launch.json').write_text(json.dumps(d,indent=2));(p/'latest_launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

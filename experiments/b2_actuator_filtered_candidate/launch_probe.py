import json,time,hashlib,subprocess,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
assert json.loads((p/'trainer_cuda_smoke_h16.json').read_text())['passed']
assert hashlib.sha256((root/'experiments/b2_cycle/evaluate_cycle.py').read_bytes()).hexdigest()=='5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
out=root/'runs'/time.strftime('b2_filtered_training_probe_%Y%m%d_%H%M%S');out.mkdir()
source=root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt';raw=source.read_bytes();(out/'source.pt').write_bytes(raw)
shutil.copytree(p,out/'source_code',ignore=shutil.ignore_patterns('__pycache__','latest_probe.json'))
env=os.environ.copy();env['B2_POLICY_SOURCE']=str(out/'source.pt');env['B2_WM_PREFIT']='2000'
cmd=['bash',str(p/'run.sh'),'--out',str(out),'--iterations','6000','--wall-hours','1','--num-envs','256','--wm-batch','256','--pi-batch','256','--ckpt-every','1000','--eval-every','1000']
with (out/'console.log').open('ab',buffering=0) as f:proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(out=str(out),pid=proc.pid,gpu=6,started=time.time(),iterations=6000,wm_prefit=2000,source_sha256=hashlib.sha256(raw).hexdigest(),role='bounded adaptation probe, not replacement or deployment',command=cmd)
(out/'launch.json').write_text(json.dumps(d,indent=2));(p/'latest_probe.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

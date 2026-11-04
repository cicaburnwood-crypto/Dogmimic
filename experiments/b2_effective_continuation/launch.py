import json,time,hashlib,subprocess,os,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
assert json.loads((root/'runs/b2_effective_continue_cuda_smoke/report.json').read_text())['passed']
old=root/'runs/b2_effective_training_probe_20261101_205738'
assert (old/'EVAL_DONE.json').exists()
assert hashlib.sha256((root/'experiments/b2_cycle/evaluate_cycle.py').read_bytes()).hexdigest()=='5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
raw=(old/'best.pt').read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='f5aae854c90fe26f15ccc484505b7a303c43f630466164bfb9d25fb42d19d4ad'
out=root/'runs'/time.strftime('b2_effective_continue_%Y%m%d_%H%M%S');out.mkdir();(out/'source.pt').write_bytes(raw)
anchor=root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt'
assert hashlib.sha256(anchor.read_bytes()).hexdigest()=='24f489e37949788dcef81f1af0f68de6556886685b6b9ca17803c2a430b37a85'
shutil.copytree(p,out/'source_code',ignore=shutil.ignore_patterns('__pycache__','latest_launch.json'))
shutil.copytree(root/'experiments/b2_effective_action_candidate',out/'parent_source',ignore=shutil.ignore_patterns('__pycache__','monitor_state.json'))
env=os.environ.copy();env.update(B2_POLICY_SOURCE=str(anchor),B2_CONTINUE_CHECKPOINT=str(out/'source.pt'),B2_WM_PREFIT='500')
cmd=['bash',str(p/'run.sh'),'--out',str(out)]
with (out/'console.log').open('ab',buffering=0) as f:proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(out=str(out),pid=proc.pid,gpu=6,started=time.time(),iterations=20000,num_envs=1024,wm_prefit=500,source_sha256=sha,anchor_source_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),role='own Actor and WM continuation; fresh replay/optimizers; same reward physics and acceptance',command=cmd)
(out/'launch.json').write_text(json.dumps(d,indent=2));(p/'latest_launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

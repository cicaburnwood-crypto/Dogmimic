import json,time,subprocess,hashlib,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
assert json.loads((p/'audit.json').read_text())['passed']
assert json.loads((root/'runs/b2_actuator_robust_smoke4/smoke_audit.json').read_text())['passed']
expected='5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
assert hashlib.sha256((root/'experiments/b2_cycle/evaluate_cycle.py').read_bytes()).hexdigest()==expected
out=root/'runs'/time.strftime('b2_actuator_robust_v1_%Y%m%d_%H%M%S');out.mkdir();shutil.copytree(p,out/'source',ignore=shutil.ignore_patterns('__pycache__','latest_launch.json'))
with (out/'console.log').open('ab',buffering=0) as f:
 proc=subprocess.Popen(['bash',str(p/'run.sh'),'--out',str(out)],cwd=root,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
d=dict(out=str(out),pid=proc.pid,gpu=6,launch_unix=time.time(),detached=True,iterations=40000,wall_hours=6,source_checkpoint='runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt',source_sha256='14c5f8ba3eac277e8aeb865c5b5deee7e93d669162de49c5e719892e84a78208',original_evaluator_sha256=expected,full_eval_conditions=28,trials_per_condition=64,required_consecutive_full_passes=2,deployment_allowed_only_after_full_validation=True,experimental_actuator_budget=dict(speed_soft=6,mechanical_power_abs_w=1500,torque_nm=[160,160,240]*4),code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.suffix in ['.py','.sh','.json']})
(out/'launch.json').write_text(json.dumps(d,indent=2));(p/'latest_launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))

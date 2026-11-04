import json,os,time,subprocess,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
audit=json.loads((HERE/'audit.json').read_text())
assert audit['real_imagined_reward_parity'] and audit['acceleration_gradient_verified']
source=ROOT/'runs/b2_hindstand_20261030_211659/best.pt'
assert source.exists()
out=ROOT/'runs'/time.strftime('b2_stable_cycle_v2_%Y%m%d_%H%M%S');out.mkdir(exist_ok=False)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',B2_INIT_CHECKPOINT=str(source),PYTHONUNBUFFERED='1')
cmd=['bash','experiments/b2_cycle/train.sh','--out',str(out),'--seed','20261102']
with (out/'console.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(gpu=6,pid=proc.pid,out=str(out),launch_unix=time.time(),source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         initialized_from_own_policy=True,reference_sha256=hashlib.sha256((HERE/'b2_full_cycle.npz').read_bytes()).hexdigest(),
         command=cmd,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
(out/'launch.json').write_text(json.dumps(rec,indent=2)+'\n');(HERE/'latest_launch.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec,indent=2))

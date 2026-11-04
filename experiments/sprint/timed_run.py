"""Fresh-model, fresh-compiler-cache wall-clock measurement. Refuses existing runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--gpus',required=True)
args,extra=p.parse_known_args()
if '--resume' in extra:
    p.error('The timed run must start with fresh parameters; resume is forbidden.')
root=Path(__file__).resolve().parents[2]
out=(root/args.out).resolve()
out.mkdir(parents=True,exist_ok=False)
start=time.time()
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=args.gpus,SPRINT_LAUNCH_UNIX=str(start),SPRINT_COLD_CACHE=str(out/'cold_cache'))
command=['bash','experiments/sprint/train.sh','--out',str(out),*extra]
sources=list((root/'experiments/sprint').glob('*.py'))+[root/'experiments/sprint/train.sh',root/'data/processed/g1_sprint_ladder.npz']
record=dict(start_unix=start,gpus=args.gpus,world=len(args.gpus.split(',')),command=command,
    fresh_model=True,fresh_optimizer=True,fresh_replay=True,fresh_compiler_caches=True,
    source_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'console.log').open('w') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['pid']=process.pid
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    code=process.wait()
record.update(end_unix=time.time(),total_wall_seconds=time.time()-start,returncode=code)
(out/'launch_timing.json').write_text(json.dumps(record,indent=2)+'\n')
raise SystemExit(code)

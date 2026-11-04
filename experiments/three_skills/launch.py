"""Launch authorized independent jobs detached, without claiming learning success."""
import os,json,time,subprocess,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
stamp=time.strftime('%Y%m%d_%H%M%S')
jobs=[(4,'g1_funky_chicken','g1'),(5,'g1_jump_full_turn','g1'),(6,'b2_hindstand','b2')]
result=[]
for gpu,task,robot in jobs:
    audit=json.loads((HERE/f'audit_{task}.json').read_text())
    assert audit['gradient_path_verified'] and audit['invalid_physics']==0
    out=ROOT/'runs'/f'{task}_{stamp}'
    out.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),SM_TASK=task,SM_ROBOT=robot,PYTHONUNBUFFERED='1')
    cmd=['bash','experiments/three_skills/train.sh','--out',str(out),'--seed',str(20261030+gpu)]
    log=open(out/'console.log','ab',buffering=0)
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rec=dict(task=task,robot=robot,gpu=gpu,pid=process.pid,out=str(out),launch_unix=time.time(),reference_sha256=hashlib.sha256((HERE/'targets'/f'{task}.npz').read_bytes()).hexdigest(),cmd=cmd)
    (out/'launch.json').write_text(json.dumps(rec,indent=2)+'\n');result.append(rec);log.close()
(HERE/'latest_launch.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

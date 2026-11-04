"""One bounded, detached training run. No scheduler or automatic restart."""
import datetime
import json
import os
import subprocess
from pathlib import Path
from bootstrap import HERE,ROOT
from common import write_json,sha256

def main():
    validation=ROOT/'runs/b2_attitude_validation_20261103'
    check=json.loads((validation/'VALIDATED.json').read_text())
    assert check['real_updates']>=3 and check['production_batch'] and check['engine_learning_files_unchanged']
    assert json.loads((HERE/'feasibility.json').read_text())['all_reachable']
    probe=json.loads((validation/'interface_probe/result.json').read_text())
    assert probe['total']>0 and len(probe['cases'])==5
    assert json.loads((validation/'export_check.json').read_text())['max_error']<1e-6
    compiled=json.loads((validation/'compiled_check.json').read_text())
    assert compiled['iterations']==10 and compiled['updates']>=4 and compiled['finite']
    assert compiled['num_envs']==1024 and compiled['compile_wm'] and compiled['export_max_error']<1e-6
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    out=ROOT/'runs'/('b2_attitude_vibe_'+now.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SM_','B2_','G1_','TORCHELASTIC_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='2')
    cmd=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'train.py'),'--out',str(out)]
    write_json(out/'frozen_contract.json',dict(evaluation=json.loads((HERE/'evaluation_manifest.json').read_text()),
        sources={str(p.relative_to(HERE)):sha256(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)},
        model_sha256=sha256(HERE/'b2_attitude.xml'),reference_policy=False,random_initialization=True))
    with (out/'train.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(task='b2_attitude_v1',backend='vibe',out=str(out),pid=proc.pid,session_id=proc.pid,cmd=cmd,cwd=str(ROOT),gpus=[6],
        started=now.isoformat(),budget=20000,wall_hours=6,initialization='random Actor/WM/critic',old_tasks_restarted=False,
        periodic_automation=False,hardware_connected=False,validation=str(validation/'VALIDATED.json'),
        compiled_validation=str(validation/'compiled_check.json'))
    write_json(out/'launch.json',launch);write_json(HERE/'latest_launch.json',launch)
    write_json(ROOT/'experiments/active_training_backend.json',dict(backend='vibe',task='b2_attitude_v1',launch=str(HERE/'latest_launch.json'),status_script=str(HERE/'status.py'),periodic_automation=False))
    print(json.dumps(launch,ensure_ascii=False,indent=2))

if __name__=='__main__':main()


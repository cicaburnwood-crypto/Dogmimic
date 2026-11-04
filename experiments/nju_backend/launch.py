"""Start the authorized bounded NJU experiment after finite integration checks."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from common import HERE,ROOT,write_json,sha256


def main():
    validation=json.loads((HERE/'validation_launch.json').read_text())
    for robot,v in validation.items():
        run=Path(v['out']);checks=json.loads((run/'VALIDATED.json').read_text())
        assert checks['updates']>=3 and checks['horizon']==32 and checks['wm_window']==8 and checks['critic']
        probe=json.loads((run/'interface_probe/result.json').read_text())
        assert probe['backend']=='NJU' and probe['robot']==robot and probe['total']>0
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    py='/home/liudiwen/projects/supermimic_nju/.venv/bin/python'
    meta=dict(backend='NJU',upstream_commit='f8f60d7bbdba68468c050ef100af02d752ca238e',
              started=datetime.datetime.now(datetime.timezone.utc).isoformat(),periodic_automation=False,
              initialization='random',validation=validation,runs={})
    for robot,gpus in [('B2','6'),('G1','4,5')]:
        out=ROOT/'runs'/f'nju_{robot.lower()}_test_{stamp}';out.mkdir()
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('SM_', 'TORCHELASTIC_')) or key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT'):
                env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='4')
        cmd=[py]
        if robot=='G1':cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
        cmd+=[str(HERE/'train.py'),'--robot',robot,'--out',str(out),'--envs','1024','--batch','512',
              '--iterations','20000','--max-hours','6']
        with (out/'train.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        job=dict(out=str(out),pid=proc.pid,session_id=proc.pid,gpus=gpus,cmd=cmd,cwd=str(ROOT),budget=20000,
                 validation_file=str(Path(validation[robot]['out'])/'VALIDATED.json'))
        meta['runs'][robot]=job
        write_json(out/'launch.json',job)
    write_json(HERE/'latest_launch.json',meta)
    write_json(ROOT/'experiments/active_training_backend.json',dict(backend='NJU',launch=str(HERE/'latest_launch.json'),status_script=str(HERE/'status.py'),periodic_automation=False))
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

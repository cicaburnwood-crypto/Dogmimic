"""Detached supervisor: fresh DDP training and independent physical evaluation."""
import argparse,json,os,time,subprocess,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpus',default='4,5')
    p.add_argument('--eval-gpu',default='6');p.add_argument('--trials',type=int,default=16);p.add_argument('--supervise',action='store_true')
    a,extra=p.parse_known_args()
    if '--resume' in extra:raise ValueError('Fresh initialization only')
    root=Path(__file__).resolve().parents[2];out=(root/a.out).resolve();here=Path(__file__).parent
    if not a.supervise:
        out.mkdir(parents=True,exist_ok=False)
        start=time.time()
        frozen_json={'model.json','upstream_reward_config.json','adaptation_manifest.json','SOURCE.json'}
        sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in here.rglob('*') if f.is_file() and (f.suffix in ('.py','.sh','.xml') or f.name in frozen_json) and '__pycache__' not in str(f)}
        record=dict(out=str(out),start_unix=start,train_gpus=a.gpus,eval_gpu=a.eval_gpu,random_policy=True,random_world_model=True,
                    source_checkpoint=None,reference_motion=False,official_policy=False,public_contact_gait=True,
                    reward_source='MIT Walk These Ways',reward_source_commit='0e7236bdc81ce855cbe3d70345a7899452bdeb1c',
                    training_algorithm='SuperMimic world-model + BPTT',train_args=extra,sha256=sources)
        (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        (out/'curriculum.json').write_text(json.dumps(dict(speed=.5,yaw_std=.35,reason='fresh initial command curriculum'))+'\n')
        cmd=[str(root/'.venv/bin/python'),str(__file__),'--supervise','--out',a.out,'--gpus',a.gpus,'--eval-gpu',a.eval_gpu,'--trials',str(a.trials),*extra]
        with (out/'supervisor.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=root,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        record['supervisor_pid']=proc.pid;(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        (here/'latest_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return
    record=json.loads((out/'launch.json').read_text());env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpus,COMMAND_LAUNCH_UNIX=str(record['start_unix']))
    log=(out/'console.log').open('w')
    train=subprocess.Popen(['bash','experiments/b2_locomotion_wtw/train.sh','--out',str(out),*extra],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    ee=env.copy();ee['CUDA_VISIBLE_DEVICES']=a.eval_gpu
    elog=(out/'evaluation.log').open('w')
    evaluation=subprocess.Popen(['bash','experiments/b2_locomotion_wtw/run_python.sh','experiments/b2_locomotion_wtw/evaluate.py','--run',str(out),'--watch','--trials',str(a.trials)],cwd=root,env=ee,stdin=subprocess.DEVNULL,stdout=elog,stderr=subprocess.STDOUT)
    record.update(train_pid=train.pid,eval_pid=evaluation.pid,supervisor_pid=os.getpid())
    (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');(here/'latest_launch.json').write_text(json.dumps(record,indent=2)+'\n')
    while train.poll() is None:
        if evaluation.poll() is not None and not (out/'DONE').exists():
            (out/'GUARD_STOP.json').write_text(json.dumps(dict(reason='independent evaluator exited before training finished',eval_returncode=evaluation.returncode,time=time.time()))+'\n')
            (out/'STOP').touch()
        time.sleep(3)
    code=train.returncode;record.update(train_returncode=code,train_end_unix=time.time(),total_wall_seconds=time.time()-record['start_unix'])
    (out/'launch_timing.json').write_text(json.dumps(record,indent=2)+'\n')
    ecode=evaluation.wait();record.update(eval_returncode=ecode,evaluation_end_unix=time.time())
    (out/'supervisor_timing.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()

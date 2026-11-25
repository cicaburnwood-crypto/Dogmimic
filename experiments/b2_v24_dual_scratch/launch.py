"""Launch an isolated V24 run, recording the exact process and weight source."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--out',required=True);p.add_argument('--wm-donor')
    p.add_argument('--detach',action='store_true');p.add_argument('--no-promote',action='store_true',help='Isolated finite comparison; keep primary latest_launch unchanged');opts,extra=p.parse_known_args()
    here=Path(__file__).resolve().parent;root=here.parent.parent
    gpu_uuid=None
    if opts.device=='cuda':
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        selected=[row.split(', ') for row in rows if int(row.split(',')[0])==opts.gpu][0]
        gpu_uuid=selected[1]
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if gpu_uuid in apps or float(selected[2])>1024 or float(selected[3])>10:
            raise RuntimeError(f'GPU {opts.gpu} is busy; no competing job will be stopped: {selected}')
    out=root/opts.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'launch.json').exists():raise RuntimeError('Refusing to overwrite an existing run identity')
    cmd=[sys.executable,'-u',str(here/'train.py'),'--device',opts.device,'--out',opts.out,
         '--workspace',str(here/'workspace_refined.json'),*extra]
    if opts.wm_donor:
        cmd.extend(['--wm-donor',opts.wm_donor])
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=gpu_uuid if opts.device=='cuda' else ''
    with (out/'stdout.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=opts.detach)
    metadata=dict(run=opts.out,pid=child.pid,supervisor_pid=os.getpid(),session_id=os.getsid(child.pid),
        gpu=opts.gpu if opts.device=='cuda' else None,gpu_uuid=gpu_uuid,device=opts.device,cwd=str(root),command=cmd,started_hk=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        started_unix=time.time(),detached=opts.detach,wm_donor=opts.wm_donor,
        actor_initialization='random' if '--resume' not in extra else 'same_fresh_actor_continuation',
        official_actor_critic=True,robot_io=False,accepted=False)
    for target in ([out/'launch.json'] if opts.no_promote else [out/'launch.json',here/'latest_launch.json']):
        target.write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)
    if not opts.detach:
        code=child.wait();(out/'launcher_exit.json').write_text(json.dumps(dict(returncode=code,finished_unix=time.time())))
        sys.exit(code)
if __name__=='__main__':main()

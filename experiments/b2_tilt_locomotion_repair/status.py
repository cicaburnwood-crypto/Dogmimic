import json
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return None

def status():
    launch=read(HERE/'latest_launch.json')
    if launch is None:return dict(launched=False)
    run=Path(launch['out']);proc=Path('/proc')/str(launch['pid'])
    try:
        argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x];cwd=str((proc/'cwd').resolve())
    except OSError:argv=[];cwd=None
    result=dict(checked=time.time(),launch=launch,process_present=bool(argv),matches_run=str(run) in argv,cwd=cwd,
        status=read(run/'status.json'),flags={x:(run/x).exists() for x in ('STOP','DONE','EVAL_DONE.json','GUARD_STOP.json','TARGET_MET.json','accepted.pt')},error=read(run/'ERROR.json'),
        sampling=read(run/'curriculum.json'),guard=read(run/'regression_guard.json'),timing=read(run/'timing.json'))
    for f in ('initial_eval.json','latest_eval.json','best_eval.json'):
        d=read(run/f)
        result[f]={k:d.get(k) for k in ('it','passed','total','completed','full','all_passed','best_sha256')} if d else None
    if (run/'train.log').exists():
        p=run/'train.log';result['log_age_s']=time.time()-p.stat().st_mtime
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));result['log_tail']=f.read().decode(errors='replace').splitlines()[-7:]
    return result

if __name__=='__main__':print(json.dumps(status(),ensure_ascii=False,indent=2))


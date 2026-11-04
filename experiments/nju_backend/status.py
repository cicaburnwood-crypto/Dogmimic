"""One-shot read-only status; does not schedule, restart, train or evaluate."""
import json
import re
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def compact(d):
    if not d:return None
    keys=('it','passed','total','completed','full','all_passed','selection_rank','source_checkpoint_sha256','best_sha256')
    out={k:d[k] for k in keys if k in d}
    out['cases']=[{k:r[k] for k in ('case_id','passed','completed','peak_speed','peak_mechanical_power','actuator_pass') if k in r} for r in d.get('conditions',[])]
    return out


def status():
    launch=read(HERE/'latest_launch.json')
    if launch is None:return {'error':'No NJU training launch yet'}
    out={'checked_unix':time.time(),'backend':'NJU','runs':{}}
    for robot,job in launch['runs'].items():
        run=Path(job['out']);proc=Path('/proc')/str(job['pid'])
        try:
            cmd=(proc/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in cmd if x]
            cwd=str((proc/'cwd').resolve())
        except (FileNotFoundError,PermissionError):argv=[];cwd=None
        log=run/'train.log'
        row={'out':str(run),'pid':job['pid'],'process_present':bool(argv),'argv':argv,'cwd':cwd,
             'matches_run':str(run) in argv,'train':read(run/'status.json'),
             'latest':compact(read(run/'latest_eval.json')),'best':compact(read(run/'best_eval.json')),
             'initial':compact(read(run/'initial_eval.json')),'timing':read(run/'timing.json'),
             'flags':{f:(run/f).exists() for f in ('STOP','DONE','EVAL_DONE.json','accepted.pt')},
             'errors':{p.name:read(p) for p in run.glob('ERROR_rank*.json')}}
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-6000));tail=f.read().decode(errors='replace')
            row['log_tail']=tail.splitlines()[-8:];row['log_age_seconds']=time.time()-log.stat().st_mtime
            matches=re.findall(r'it (\d+) \|.*?updates (\d+) \|',tail)
            if matches:row['last_logged_progress']={'iteration':int(matches[-1][0]),'updates':int(matches[-1][1])}
        out['runs'][robot]=row
    return out

if __name__=='__main__':print(json.dumps(status(),ensure_ascii=False,indent=2))

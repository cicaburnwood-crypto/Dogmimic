import json,time,subprocess,shlex
from pathlib import Path

here=Path(__file__).resolve().parent
launch=json.loads((here/'latest_launch.json').read_text());run=Path(launch['out'])
def read(name):
    p=run/name
    return json.loads(p.read_text()) if p.exists() else None
def process(pid):
    path=Path(f'/proc/{pid}/cmdline')
    if not path.exists():return dict(pid=pid,alive=False,identity_match=False)
    argv=path.read_bytes().rstrip(b'\x00').split(b'\x00');argv=[a.decode(errors='replace') for a in argv]
    cwd=Path(f'/proc/{pid}/cwd').resolve();matched=False
    for flag in ('--out','--run'):
        if flag in argv and argv.index(flag)+1<len(argv):
            target=Path(argv[argv.index(flag)+1]);target=target if target.is_absolute() else cwd/target
            matched|=target.resolve()==run.resolve()
    return dict(pid=pid,alive=matched,identity_match=matched,cmdline=shlex.join(argv),cwd=str(cwd))
log=run/'log.jsonl';rows=log.read_text().splitlines() if log.exists() else []
last=json.loads(rows[-1]) if rows else None
def compact(result):
    if not result:return None
    small={k:v for k,v in result.items() if k!='cases'}
    small['cases']=[{k:v for k,v in g.items() if k not in ('trials','nominal')}|{'nominal_pass':g['nominal']['success']} for g in result['cases']]
    return small
result=dict(checked_unix=time.time(),run=str(run),elapsed_seconds=time.time()-launch['start_unix'],
            train=process(launch.get('train_pid',-1)),evaluation=process(launch.get('eval_pid',-1)),supervisor=process(launch.get('supervisor_pid',-1)),
            gpus=dict(train=launch['train_gpus'],evaluation=launch['eval_gpu']),done=(run/'DONE').exists(),
            guard_stop=read('GUARD_STOP.json'),target_met=read('TARGET_MET.json'),last_log=last,
            curriculum=read('curriculum.json'),latest=compact(read('latest_eval.json')),best=compact(read('best_eval.json')),
            timing=read('timing.json'),launch_timing=read('launch_timing.json'),
            eval_done=read('EVAL_DONE.json'),supervisor_timing=read('supervisor_timing.json'))
if last:
    args=launch.get('train_args',[])
    def option(name,default):return float(args[args.index(name)+1]) if name in args else default
    rate=last['iter_s'];limit=option('--iterations',80000);wall=option('--wall-hours',6)*3600
    result['budget_iterations']=int(limit)
    result['budget_eta_seconds']=0. if result['done'] else max(0.,min((limit-last['it'])*rate,wall-result['elapsed_seconds']))
print(json.dumps(result,indent=2))

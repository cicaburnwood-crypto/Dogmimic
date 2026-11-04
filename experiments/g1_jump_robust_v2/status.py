import json,time,os
from pathlib import Path
b=Path(__file__).resolve().parent;a=json.loads((b/'latest_launch.json').read_text());p=Path(a['out']);r=dict(launch=a,elapsed_seconds=time.time()-a['started'])
for name in ['train','eval']:
 pid=a[name+'_pid']
 try:r[name+'_cmdline']=Path('/proc/%s/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:r[name+'_cmdline']=None
for name in ['latest_eval','best_eval','initial_eval','eval_progress','EVAL_DONE','TRAIN_FAILED']:
 f=p/(name+'.json')
 if f.exists():r[name]={k:v for k,v in json.loads(f.read_text()).items() if k!='cases'}
f=p/'log.jsonl'
if f.exists():
 lines=f.read_text().splitlines();r['last_train']=json.loads(lines[-1]) if lines else None
r['done']=(p/'DONE').exists();print(json.dumps(r,indent=2))

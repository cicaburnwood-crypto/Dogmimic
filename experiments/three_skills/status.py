import json,time,os,datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
rows=[]
for rec in json.loads((HERE/'latest_launch.json').read_text()):
    out=Path(rec['out']);pid=rec['pid']
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        running=stat[2]!='Z'
    except FileNotFoundError:running=False
    logs=(out/'log.jsonl').read_text().splitlines() if (out/'log.jsonl').exists() else []
    last=json.loads(logs[-1]) if logs else {}
    console=(out/'console.log').read_text(errors='replace')
    timing=json.loads((out/'timing.json').read_text()) if (out/'timing.json').exists() else {}
    elapsed=(timing.get('end_unix',time.time())-rec['launch_unix'])
    row={**rec,'running':running,'done':(out/'DONE').exists(),'wall_seconds':elapsed,
         'last':last,'best_eval':json.loads((out/'best_eval.json').read_text()) if (out/'best_eval.json').exists() else None,
         'traceback':'Traceback (most recent call last)' in console}
    if not running and not row['done']:row['console_tail']=console.splitlines()[-14:]
    rows.append(row)
print(json.dumps({'checked_at':datetime.datetime.now().astimezone().isoformat(),'jobs':rows},indent=2))

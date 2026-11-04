import datetime,json,time
from pathlib import Path
from zoneinfo import ZoneInfo

here=Path(__file__).resolve().parent
r=json.loads((here/'latest_launch.json').read_text());out=Path(r['out'])
try:running=Path(f'/proc/{r["pid"]}/stat').read_text().split()[2]!='Z'
except FileNotFoundError:running=False
lines=(out/'log.jsonl').read_text().splitlines() if (out/'log.jsonl').exists() else []
evals=[json.loads(x) for x in (out/'eval.jsonl').read_text().splitlines()] if (out/'eval.jsonl').exists() else []
console=(out/'console.log').read_text(errors='replace')
timing=json.loads((out/'timing.json').read_text()) if (out/'timing.json').exists() else {}
result=dict(checked_at=datetime.datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),gpu=r['gpu'],pid=r['pid'],out=str(out),
            running=running,done=(out/'DONE').exists(),wall_seconds=timing.get('end_unix',time.time())-r['launch_unix'],
            last=json.loads(lines[-1]) if lines else {},
            evaluations=[{k:v for k,v in e.items() if k!='trials_detail'} for e in evals[-2:]],
            traceback='Traceback (most recent call last)' in console,console_tail=console.splitlines()[-3:])
print(json.dumps(result,indent=2))

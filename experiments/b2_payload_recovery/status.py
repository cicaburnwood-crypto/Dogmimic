"""Read-only status for the most recently launched guarded B2 continuation."""
import datetime,json,os,subprocess,time
from pathlib import Path

here=Path(__file__).resolve().parent
launch=json.loads((here/'latest_launch.json').read_text())
out=Path(launch['out']);pid=launch['pid']
proc=Path('/proc')/str(pid)
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
running=bool(cmd and str(out) in cmd)
logs=out/'log.jsonl'
rows=logs.read_text().splitlines() if logs.exists() else []
last=json.loads(rows[-1]) if rows else None
evaluation=json.loads((out/'latest_eval.json').read_text()) if (out/'latest_eval.json').exists() else None
report=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),variant=launch['variant'],pid=pid,gpu=launch['gpu'],
            running=running,done=(out/'DONE').exists(),guard_stop=(out/'GUARD_STOP').exists(),out=str(out),
            elapsed_seconds=time.time()-launch['launch_unix'],source_checkpoint=launch['source_checkpoint'],
            payload_range_kg=launch['payload_range_kg'],force_range_n=launch['force_range_n'],last_training=last)
if running:
    report.update(session_id=os.getsid(pid),stdin=os.readlink(proc/'fd/0'))
if evaluation:
    report['latest_eval']={k:evaluation[k] for k in ('it','accepted','guard_reasons','quality','guard_rollbacks')}
    report['latest_eval']['loads']=[dict(kg=r['payload_kg'],trials=r['trials'],
        strict=round(r['trials']*r['success_rate']),complete=round(r['trials']*r['completion_rate']),
        stable_hindstand=round(r['trials']*r['stable_hindstand_rate']),stable_return=round(r['trials']*r['stable_return_rate']),
        return_com_std_m=r['return_metrics_completed_trials']['com_position_std_norm'],failure_reasons=r['failure_reason_counts'])
        for r in evaluation['conditions']]
gpu=subprocess.run(['nvidia-smi','-i',str(launch['gpu']),'--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
report['gpu_util_memory']=gpu.stdout.strip()
print(json.dumps(report,indent=2))

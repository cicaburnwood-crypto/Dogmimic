"""Collect measured pilot results without estimating missing milestones."""
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[2]
names=['sprint_bench_v2_1gpu','sprint_bench_v2_2gpu','sprint_bench_v2_4gpu',
       'sprint_fast32','sprint_fast16','sprint_baseline','sprint_fast16_2gpu','sprint_focus4']
rows=[]
for name in names:
    run=root/'runs'/name
    meta=json.loads((run/'run_meta.json').read_text())
    logs=[json.loads(x) for x in (run/'log.jsonl').read_text().splitlines()]
    steady=[r['iter_s'] for r in logs if r['it']>=500] or [r['iter_s'] for r in logs if r['it']>=200]
    ep=root/'runs/sprint_evaluations'/name/'evaluations.jsonl'
    ev=[json.loads(x) for x in ep.read_text().splitlines()] if ep.exists() else []
    milestones={}
    for speed in [2,3,4,5,6,7,8]:
        match=next((r for r in ev if any(g['target_m_s']==speed and g['success_rate']==1 for g in r['groups'])),None)
        milestones[str(speed)]={k:match[k] for k in ['it','train_seconds','checkpoint_wall_seconds']} if match else None
    timing=json.loads((run/'timing.json').read_text()) if (run/'timing.json').exists() else None
    rows.append(dict(name=name,meta=meta,timing=timing,done=(run/'DONE').exists(),last_iteration=logs[-1]['it'],
        median_steady_iteration_seconds=statistics.median(steady),first_8_of_8_pass=milestones,
        evaluations=[{k:r[k] for k in ['it','checkpoint_wall_seconds','max_stable_target_m_s','max_stable_measured_m_s']} for r in ev]))
print(json.dumps(rows,indent=2))

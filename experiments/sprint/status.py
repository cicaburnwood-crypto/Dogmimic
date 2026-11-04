import json,time
from pathlib import Path

run=Path('runs/sprint_final_fresh')
launch=json.loads((run/'launch.json').read_text())
result={'elapsed_s':round(time.time()-launch['start_unix'],1),'done':(run/'DONE').exists()}
p=run/'log.jsonl'
if p.exists() and p.stat().st_size:
    r=json.loads(p.read_text().splitlines()[-1])
    result['training']={k:r[k] for k in ['it','iter_s','ep_norm_return','env_steps']}
p=Path('runs/sprint_final_evaluation/evaluations.jsonl')
if p.exists():
    records=[json.loads(x) for x in p.read_text().splitlines()]
    r=records[-1]
    result['latest_evaluation']={'it':r['it'],'groups':[(g['target_m_s'],g['success_rate'],round(g['mean_speed_m_s'],3)) for g in r['groups']]}
    result['first_stable_by_target']={}
    for speed in [2,3,4,5,6,7,8]:
        for r in records:
            g=next(g for g in r['groups'] if g['target_m_s']==speed)
            if g['success_rate']>=.9 and g['nominal']['success']:
                result['first_stable_by_target'][speed]=dict(it=r['it'],launch_wall_s=round(r['launch_wall_seconds'],1),acceptance_wall_s=round(r['evaluated_unix']-launch['start_unix'],1),success_rate=g['success_rate'],actual_m_s=round(g['successful_mean_speed_m_s'],3))
                break
print(json.dumps(result))

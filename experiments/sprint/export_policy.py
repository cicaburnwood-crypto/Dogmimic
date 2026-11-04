"""Export independently validated simulation weights and measured timing evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch

def sha256(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

p=argparse.ArgumentParser()
for name in ['checkpoint','evaluation','run','out']:p.add_argument('--'+name,required=True)
args=p.parse_args()
ck=torch.load(args.checkpoint,map_location='cpu')
finite={name:all(bool(torch.isfinite(v).all()) for v in ck[name].values()) for name in ['policy','wm','critic']}
assert all(finite.values()), 'Non-finite learned model state.'
result=json.loads(Path(args.evaluation).read_text())
assert result['it']==ck['it']
valid=[g for g in result['groups'] if g['success_rate']>=.9 and g['nominal']['success']]
assert valid, 'No speed passed the independent validation criterion.'
fastest=max(valid,key=lambda g:g['successful_mean_speed_m_s'])
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
keys=['policy','cfg','it','updates','env_steps','train_time','task_cfg','world','wall_seconds','launch_wall_seconds']
torch.save({k:ck[k] for k in keys if k in ck},out/'policy.pt')
run=Path(args.run);launch=json.loads((run/'launch.json').read_text())
timing=json.loads((run/'launch_timing.json').read_text()) if (run/'launch_timing.json').exists() else None
manifest=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint),
    policy_sha256=sha256(out/'policy.pt'),iterations=ck['it'],updates=ck['updates'],env_steps=ck['env_steps'],
    train_cfg=ck['cfg'],task_cfg=ck['task_cfg'],world=ck['world'],finite_learned_state=finite,
    model_creation_wall_seconds=ck.get('launch_wall_seconds'),training_loop_seconds=ck['train_time'],
    independent_acceptance_wall_seconds=result['evaluated_unix']-launch['start_unix'],
    export_completed_wall_seconds=time.time()-launch['start_unix'],
    training_process_timing=timing,launch=launch,
    invalid_physics_transitions_rank0=ck.get('invalid_physics_transitions'),
    fastest_validated_group=fastest,evaluation={k:v for k,v in result.items() if k!='trials'},
    packages={name:importlib.metadata.version(name) for name in ['torch','mujoco','mujoco-warp','mjlab','warp-lang','numpy']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(policy=str(out/'policy.pt'),iteration=ck['it'],measured_m_s=fastest['successful_mean_speed_m_s'],wall_seconds=manifest['model_creation_wall_seconds']),indent=2))

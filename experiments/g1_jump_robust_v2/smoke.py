import sys,json
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b/'engine'))
import torch
from supermimic.motion_lib import MotionLibrary
from robust_env import RobustEnv
from supermimic.evaluate import load_supermimic_policy
lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cuda');env=RobustEnv(lib,'cuda',32,evaluate=True);env.case={'friction':.2,'asym':.3,'mass':1.2,'delay_ms':20,'force':[20,0,0]};env.reset()
print('XI',env.xi.shape,'friction',env.sim.model.geom_friction.shape,'mass',env.sim.model.body_mass.shape,flush=True)
p,ck=load_supermimic_policy('deployments/g1_actions/sources/jump_60cm.pt','cuda')
from evaluate_jump import run_trials
r,_=run_trials(p,lib,env);print('SMOKE',r['success_rate'],r['completion_rate'],flush=True)
from mjlab.managers.event_manager import RecomputeLevel
print('CONTACT_FIELDS', [x for x in dir(env.sim.data) if 'contact' in x],flush=True)
(b/'smoke.json').write_text(json.dumps(dict(success=r['success_rate'],xi=list(env.xi.shape),finite=bool(torch.isfinite(env.xi).all()),case=env.case),indent=2))

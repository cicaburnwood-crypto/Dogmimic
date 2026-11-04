import json
from pathlib import Path
import torch
from supermimic import networks
from supermimic.networks import Policy
from cycle_env import CycleLibrary
from recovery_env import make_recovery_env
from phase_policy import PhasePolicy

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
torch.set_num_threads(2);networks.AMP['enabled']=True
lib=CycleLibrary(str(ROOT/'experiments/b2_cycle/b2_full_cycle.npz'),'cuda')
env=make_recovery_env(lib,'cuda',8,evaluate=True,fixed_loads=10.,pushes=False);env.reset()
ids=torch.arange(8,device='cuda')
ck=torch.load(ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt',map_location='cuda',weights_only=False)
p=PhasePolicy(210,195).cuda();p.load_state_dict(ck['policy'])
old=Policy(210,195,enc_hidden=(512,256,128),dec_hidden=(512,256,128)).cuda();old.load_state_dict(ck['policy'])
checks=[]
for frame,limit in ((0,.04),(200,.008),(650,.04)):
    env.reset_envs(ids,torch.full_like(ids,frame));env._finish_resets(ids,force=True)
    obs,goal=env.policy_inputs();p.load_state_dict(ck['policy'])
    with torch.no_grad():
        initial=float((p(obs,goal)-old(obs,goal)).abs().max());assert initial==0.
        p.hind_head.bias.fill_(100.);p.quad_head.bias.fill_(100.)
        delta=float(((p(obs,goal)-old(obs,goal))*.7).abs().max());assert delta<limit+1e-6 and delta>limit-1e-5
    checks.append(dict(frame=frame,initial_error=initial,max_delta_rad=delta))
p.load_state_dict(ck['policy'])
env.reset_envs(ids,torch.zeros_like(ids));env._finish_resets(ids,force=True);obs,goal=env.policy_inputs()
with torch.no_grad():
    p.quad_head.bias.fill_(.0001)
    small=float(((p(obs,goal)-old(obs,goal))*.7).abs().mean());assert 5e-5<small<9e-5
p.load_state_dict(ck['policy'])
env.reset_envs(ids,torch.full_like(ids,200));env._finish_resets(ids,force=True)
p(*env.policy_inputs()).sum().backward()
assert p.hind_head.weight.grad.abs().max()>0 and p.quad_head.weight.grad.abs().max()<1e-6
assert all(x.grad is None for x in p.teacher.parameters())
env.reset_envs(ids,torch.full_like(ids,100));env._finish_resets(ids,force=True)
_,still=p.gates(env.policy_inputs()[1]);assert still.max()<.1
result=dict(passed=True,phase_limits_and_identity=checks,submilliradian_correction_verified_rad=small,
            frozen_teacher_features=True,independent_head_gradients_verified=True,moving_reference_correction_attenuation=float(still.max()))
(HERE/'audit_phase.json').write_text(json.dumps(result,indent=2)+'\n');print('PHASE_AUDIT '+json.dumps(result),flush=True)

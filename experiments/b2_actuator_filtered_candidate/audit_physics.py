"""Small real simulator integration check, not full task acceptance."""
import argparse,json
from pathlib import Path
import torch
import train_guarded
from cycle_env import CycleLibrary as MotionLibrary
from robust_env import make_env
from actuator_filter import advance_context
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2)
lib=MotionLibrary('experiments/b2_cycle/b2_full_cycle.npz',a.device)
e=make_env(lib,a.device,4,evaluate=True,kg=0.)
e.reset();ids=torch.arange(4,device=a.device);e.reset_envs(ids,torch.zeros(4,dtype=torch.long,device=a.device));e._finish_resets(ids,force=True)
assert torch.allclose(e.filtered_target,e.sim.data.qpos[:,e.q_adr])
for i in range(12):
 before=e.xi.clone();action=torch.zeros(4,12,device=a.device)
 expected=advance_context(before,action,e.q0,e.action_scale,e.task.qmin,e.task.qmax)
 _,info=e.step(action);alive=~info['done']
 assert torch.allclose(e.xi[alive,31:],expected[alive,31:])
 assert torch.isfinite(e.sim.data.qpos).all()
 assert torch.isfinite(info['reward']).all()
e.sim.data.qpos[0,e.q_adr]+=.006
e._finish_resets(ids[:1],force=True)
assert torch.allclose(e.filtered_target[:1],e.sim.data.qpos[:1,e.q_adr])
keep=e.filtered_target[1:].clone();e.reset_envs(ids[:1],torch.zeros(1,dtype=torch.long,device=a.device));e._finish_resets(ids[:1],force=True)
assert torch.equal(e.filtered_target[1:],keep)
assert torch.allclose(e.filtered_target[:1],e.sim.data.qpos[:1,e.q_adr])
print(json.dumps(dict(passed=True,device=a.device,real_simulator=True,transitions=12,environments=4,partial_reset=True,scope='Integration only; no policy quality or full acceptance claim')))

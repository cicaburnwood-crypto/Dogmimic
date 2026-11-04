"""Read-only code-path diagnosis using an independent CPU simulator."""
import json,torch
import train_guarded
from cycle_env import CycleLibrary
from robust_env import make_env
torch.set_num_threads(2)
e=make_env(CycleLibrary('experiments/b2_cycle/b2_full_cycle.npz','cpu'),'cpu',4,evaluate=True,kg=0.)
e.reset();ids=torch.arange(4);e.reset_envs(ids,torch.zeros(4,dtype=torch.long));e._finish_resets(ids,force=True)
# Enter the first naturally sampled push window; do not change sampled force.
e.frame=e.push_start[:,0].clone();expected,remaining=e._force_context();record=[];realstep=e.sim.step
assert (expected.norm(dim=-1)>=8-1e-5).all() and (expected.norm(dim=-1)<=16+1e-5).all()
def capture():
 record.append(e.sim.data.xfrc_applied[:,1,:3].clone());realstep()
e.sim.step=capture
e.step(torch.zeros(4,12))
assert len(record)==4 and all(torch.equal(x,expected) for x in record)
r=dict(passed=True,physical_force_applied_all_four_substeps=True,force_norms_N=expected.norm(dim=-1).tolist(),reported_pushed_transitions=e.pushed_transitions,counter_is_stale=e.pushed_transitions==0,scope='CPU execution of current candidate step; counter not updated by overridden step, physical force application verified; does not quantify live-run robustness')
print(json.dumps(r))

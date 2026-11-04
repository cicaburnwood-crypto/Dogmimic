"""Exercise the no-reference and zero-velocity contracts before long training."""
import json,os
import torch
from supermimic.motion_lib import MotionLibrary
def forbidden(*a,**kw):raise AssertionError('MotionLibrary must never be initialized')
MotionLibrary.__init__=forbidden
from task import CommandEnv
from train import make_policy
torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)))
torch.manual_seed(20261031)
env=CommandEnv('cuda',16,evaluate=True);s=env.reset()
assert env.lib is None
assert all(torch.count_nonzero(s[k]).item()==0 for k in ('v','w','qd'))
assert env.policy_inputs()[0].shape==(16,480)
policy=make_policy().to('cuda')
a=policy(*env.policy_inputs());assert a.shape==(16,29) and torch.isfinite(a).all()
zero_command=env.commands.clone();go_command=zero_command.clone();go_command[:,0]=.5
r0=env.task.command_fast(s,zero_command,a,a,env.enc_bias)['r']
r1=env.task.command_fast(s,go_command,a,a,env.enc_bias)['r']
assert not torch.equal(r0,r1)
an=env.task.command_fast(s,go_command,a,a,env.enc_bias)
an['r'].mean().backward()
fails=0
for _ in range(100):
    s,info=env.step(torch.zeros_like(a));fails+=int(info['terminated'].sum())
result=dict(motion_library_instantiation_forbidden=True,initial_velocity_exactly_zero=True,
            observation_dim=480,command_dim=3,random_parameters=True,finite_actions=True,
            reward_responds_to_command=True,zero_action_stand_2s_falls=fails,
            final_root_z=float(s['pos'][:,2].mean()),final_speed=float(s['v'][:,:2].norm(dim=-1).mean()))
print('AUDIT '+json.dumps(result),flush=True)

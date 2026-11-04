import json,math,os
from pathlib import Path
import torch
from supermimic.motion_lib import MotionLibrary
def forbidden(*a,**k):raise AssertionError('No reference MotionLibrary is permitted')
MotionLibrary.__init__=forbidden
from contract import *
from task import CommandEnv,XI_DIM
from policy import make_policy
from supermimic.networks import WorldModel
from supermimic.task import state_to_flat,flat_to_state
from replay import CommandReplay

torch.set_num_threads(2);torch.manual_seed(20261031)
env=CommandEnv('cuda',32,evaluate=True,randomize=True)
s=env.reset()
assert env.lib is None
assert all(torch.count_nonzero(s[k])==0 for k in ('v','w','qd'))
assert torch.equal(s['q'],env.q0.expand(32,-1))
assert env.policy_inputs()[0].shape==(32,420)
assert env.xi.shape==(32,XI_DIM)
samples=torch.randn(100000,3)*8
targets=samples.clone();targets[:,1].clamp_(-.6,.6)
targets[:,:2]*=(2/targets[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1.)
assert targets[:,:2].norm(dim=-1).max()<2.000001 and targets[:,1].abs().max()<=.600001
assert torch.equal(targets[:,2],samples[:,2])
env.set_command([20.,20.,20.])
for _ in range(400):env.set_command([20.,20.,20.])
assert torch.allclose(env.commands[:,2],torch.full((32,),20.,device='cuda'),atol=1e-4)
assert env.commands[:,:2].norm(dim=-1).max()<=2.00001
assert constrain_command(0,2,123)==[0.,.6,123]
assert math.isclose(math.hypot(*constrain_command(2,.6,0)[:2]),2.)
env.reset();policy=make_policy().cuda();buf=CommandReplay(32,64,XI_DIM,'cuda')
fails=0
for t in range(64):
    with torch.no_grad():
        a=.03*torch.randn(32,12,device='cuda')
        flat=state_to_flat(env.state);xi=env.xi.clone();ep=env.ep_id.clone();cmd=env.commands.clone()
        nxt,info=env.step(a);buf.add(flat,a,state_to_flat(nxt),xi,cmd,ep)
        fails+=int(info['terminated'].sum())
assert torch.isfinite(buf.s).all() and torch.isfinite(buf.s1).all()
s0,A,S1,X,valid=buf.sample_context_windows(16,8)
assert torch.allclose(X[:,1:,20:],A[:,:-1],atol=1e-6)
feat=env.task.features(flat_to_state(s0))
wm=WorldModel(env.task,feat.shape[-1],XI_DIM,hidden=(64,64)).cuda()
st,xi,cmd,hist,prev=buf.sample_starts(16,HISTORY,env.task)
act=policy(hist.flatten(1),cmd);pred=wm.step(flat_to_state(st),act,xi)
loss=-env.task.command_fast(pred,cmd,act,prev,xi[:,1:13])['r'].mean();loss.backward()
grad=sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None)
assert math.isfinite(grad) and grad>0
out=dict(no_motion_library=True,fresh_policy_and_world_model=True,standard_pose=True,initial_velocity_exactly_zero=True,
         actor_observation='joint q/dq, body gyro, projected gravity, previous action, 10 frames; command xyz',
         actor_obs_dim=420,ground_truth_velocity_actor_input=False,policy_gradient_through_world_model=grad,
         command_samples=100000,xy_norm_max=2,abs_y_max=.6,yaw_command_not_clipped=True,
         per_step_delay_context_verified=True,physical_mass_scale=[float(env.mass_scale.min()),float(env.mass_scale.max())],
         control_delay_ms=[float(env.delay.min())*5,float(env.delay.max())*5],baseline_1p28s_terminations=fails,
         baseline_final_height_m=float(env.state['pos'][:,2].mean()),model=env.model_meta)
Path(__file__).with_name('audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('AUDIT '+json.dumps(out),flush=True)

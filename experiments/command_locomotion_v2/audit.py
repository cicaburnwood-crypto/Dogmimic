"""Verify migration equivalence and world-frame reward/observation contracts."""
import argparse,json,os
import torch
from supermimic.motion_lib import MotionLibrary
def forbidden(*args,**kwargs):raise AssertionError('Motion data is forbidden')
MotionLibrary.__init__=forbidden
from supermimic.networks import Policy,Critic
from supermimic.rotations import yaw_mat
from task import CommandEnv
from train import make_policy
from migrate import expand_policy,expand_critic
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)));torch.manual_seed(123)
ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
old=Policy(480,3,enc_hidden=(128,128),dec_hidden=(512,256,128)).cuda();old.load_state_dict(ck['policy'])
new=make_policy().cuda();expand_policy(ck['policy'],new)
obs=torch.randn(32,480,device='cuda');cmd=torch.randn(32,3,device='cuda')
goal=torch.cat([cmd,torch.randn_like(cmd)],-1)
with torch.no_grad():
    error=float((old(obs,cmd)-new(obs,goal)).abs().max())
assert error<1e-5,error
old_dim=ck['critic']['norm.mean'].numel();feat=old_dim-3-33-1
v0=Critic(old_dim).cuda();v0.load_state_dict(ck['critic'])
v1=Critic(old_dim+3).cuda();expand_critic(ck['critic'],v1,feat)
x=torch.randn(32,old_dim,device='cuda');x1=torch.cat([x[:,:feat+3],torch.randn(32,3,device='cuda'),x[:,feat+3:]],-1)
with torch.no_grad():ve=float((v0(x)-v1(x1)).abs().max())
assert ve<1e-4,ve
env=CommandEnv('cuda',16,evaluate=True,max_speed=2.)
s=env.reset();assert all(torch.count_nonzero(s[k])==0 for k in ('v','w','qd'))
cmd=torch.zeros(16,3,device='cuda');cmd[:,0]=1.
action=torch.zeros(16,29,device='cuda')
s['v'][:,0]=1.
r0=env.task.command_fast(s,cmd,action,action,env.enc_bias)
s['rot']=yaw_mat(torch.full((16,),.3,device='cuda'));s['pos'][:,1]=.4
r1=env.task.command_fast(s,cmd,action,action,env.enc_bias)
assert torch.allclose(r0['terms']['velocity'],r1['terms']['velocity'])
assert (r1['terms']['heading']<0).all() and (r1['terms']['cross_track']<0).all()
assert r1['goal'].shape==(16,6) and torch.allclose(r1['goal'][:,5],s['pos'][:,1])
print('AUDIT '+json.dumps(dict(policy_max_error=error,critic_max_error=ve,zero_initial_velocity=True,
    motion_library_forbidden=True,world_velocity_reward_verified=True,heading_and_lateral_feedback=True)),flush=True)

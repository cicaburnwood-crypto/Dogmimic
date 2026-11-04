"""Run actual environment control/reset methods with a CPU simulator test double."""
import ast,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from actuator_filter import filtered_target,advance_context
class Parent:
    @property
    def xi(self):return self.base_context
    def reset_envs(self,ids,frames=None):
        self.sim.data.qpos[ids[:,None],self.q_adr]=.25
        self.frame[ids]=0 if frames is None else frames
root=Path(__file__).resolve().parent
node=next(x for x in ast.parse((root/'robust_env.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='RobustEnv')
ns=dict(torch=torch,RecoveryEnv=Parent,filtered_target=filtered_target)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_RobustEnv','exec'),ns)
e=ns['RobustEnv'].__new__(ns['RobustEnv']);n=3
e.device='cpu';e.num_envs=n;e.q_adr=torch.arange(7,19);e.frame=torch.zeros(n,dtype=torch.long)
e.sim=NS(data=NS(qpos=torch.zeros(n,19),qvel=torch.ones(n,18),xfrc_applied=torch.zeros(n,2,6)))
e.filtered_target=torch.full((n,12),-.4);e.eval_mode=False;e.asymmetric_eval=False;e.measured_q=torch.stack([torch.arange(12)*.01,torch.arange(12)*-.01])
old=e.filtered_target.clone();ids=torch.tensor([0,2]);e.reset_envs(ids)
assert torch.equal(e.filtered_target[ids],e.sim.data.qpos[ids[:,None],e.q_adr])
assert torch.equal(e.filtered_target[1],old[1]);assert (e.sim.data.qvel[ids]==0).all()
e.base_context=torch.zeros(n,30);e.mass_scale=torch.ones(n);e.enc_bias=e.base_context[:,1:13];e.q0=torch.zeros(12);e.action_scale=.7
e.act_to_joint=torch.arange(11,-1,-1);e.tcfg=NS(dt=.02,max_step_reward=1.);e.decimation=4
seen=[];states=[]
def transition(*args):
    states.append(args[-1].clone())
    return dict(r=torch.zeros(n),hard=torch.zeros(n,dtype=torch.bool),terms={},obs=torch.zeros(n,2),goal=torch.zeros(n,1),com_pos=0,com_velocity=0,com_acceleration=0,center_acceleration=0)
e.task=NS(qmin=torch.full((12,),-2.),qmax=torch.full((12,),2.),transition_fast=transition)
e.state={};e._force_context=lambda:(torch.zeros(n,3),None);e._actuated_step=lambda target:seen.append(target.clone())
e.read_state=lambda:{'invalid_physics':torch.zeros(n,dtype=torch.bool)}
e.lib=NS(frame=lambda f:{k:None for k in ['pos','rot','q','v','w','qd']},remaining=lambda f:torch.full((n,),1000))
e.kin_c=lambda *args:None;e.ref_future=lambda f:None;e.prev_action=torch.zeros(n,12)
e.ep_len=torch.zeros(n,dtype=torch.long);e.max_ep_len=1000;e.gamma=.99;e.ep_disc_ret=torch.zeros(n);e.ep_disc_max=torch.zeros(n);e.ep_ret=torch.zeros(n);e.ep_id=torch.zeros(n,dtype=torch.long)
e.obs_hist=torch.zeros(n,2,2);e._noise=lambda n:torch.zeros(n,2);e._finish_resets=lambda ids:None
for i in range(12):
    action=torch.full((n,12),1. if i%2==0 else -.7)
    before=e.xi.clone();expected=advance_context(before,action,e.q0,e.action_scale,e.task.qmin,e.task.qmax)
    e.step(action)
    assert torch.equal(states[-1],before),'reward received post-filter instead of pre-filter context'
    assert torch.equal(e.xi,expected)
    assert all(torch.equal(x,expected[:,31:][:,e.act_to_joint]) for x in seen[-4:])
print(json.dumps(dict(passed=True,actual_partial_reset_memory=True,actual_step_pretransition_reward_context=True,physical_imagined_context_parity_steps=12,nonidentity_actuator_mapping=True,scope='Actual methods with CPU simulator double; real MuJoCo/GPU smoke and physical validation pending')))

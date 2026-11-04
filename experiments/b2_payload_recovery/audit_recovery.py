import json
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.task import flat_to_state
from supermimic.networks import Policy
from cycle_env import CycleLibrary
from guarded_policy import GuardedPolicy
from recovery_env import make_recovery_env

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
torch.set_num_threads(2);networks.AMP['enabled']=True
lib=CycleLibrary(str(ROOT/'experiments/b2_cycle/b2_full_cycle.npz'),'cuda')
env=make_recovery_env(lib,'cuda',8,evaluate=True,fixed_loads=[0,0,5,5,10,10,15,15],pushes=False)
env.reset();task=env.task;ids=torch.arange(8,device='cuda')
ck=torch.load(ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt',map_location='cuda',weights_only=False)
p=GuardedPolicy(210,195).cuda();p.load_state_dict(ck['policy'])
old=Policy(210,195,enc_hidden=(512,256,128),dec_hidden=(512,256,128)).cuda();old.load_state_dict(ck['policy'])
with torch.no_grad():
    inp=env.policy_inputs();identity=float((p(*inp)-old(*inp)).abs().max());assert identity==0.
    norm=p.obs_norm.mean.clone();p.obs_norm.update(torch.randn(100,210,device='cuda'));assert torch.equal(norm,p.obs_norm.mean)
    p.student.decoder[-1].bias.add_(100.)
    limit=float(((p(*inp)-old(*inp))*.7).abs().max());assert limit<=.040001
    p.load_state_dict(ck['policy'])
parity=[];anchor_errors=[]
with torch.no_grad():
    for initial in (199,649):
        frame=torch.full_like(ids,initial);env.reset_envs(ids,frame);env._finish_resets(ids,force=True)
        s0={k:v.clone() for k,v in env.state.items()};xi=env.xi.clone();prev=env.prev_action.clone()
        action=p(*env.policy_inputs());s1,info=env.step(action);ref=lib.frame(frame+1)
        expected=task.transition_analytics(s0,s1,ref,task.body_kin(ref),env.ref_future(frame+1),action,prev,xi)
        parity.append(float((expected['r']-info['reward']).abs().max()))
        next_xi=task.advance_context(s1,frame+1,xi)
        anchor_errors.append(float((next_xi[:,21:]-env.xi[:,21:]).abs().max()))
    assert max(parity)<1e-5 and max(anchor_errors)<1e-5,(parity,anchor_errors)
    before=env.anchor_context.clone()
    env.reset_envs(torch.tensor([0],device='cuda'),torch.tensor([0],device='cuda'))
    env._finish_resets(torch.tensor([0],device='cuda'),force=True)
    assert env.anchor_context[0].abs().max()==0 and torch.equal(before[1:],env.anchor_context[1:])
record=np.load(ROOT/'runs/b2_stable_cycle_v2_20261030_214533/payload_test_20261030_225905/0kg_push/rollout.npz')
state=flat_to_state(torch.tensor(record['state'][199,:8],device='cuda'))
ref=lib.frame(torch.full_like(ids,200))
xi=torch.zeros(8,30,device='cuda');xi[:,0]=1
xi=task.advance_context(state,torch.full_like(ids,200),xi)
terms,_=task.stability_terms(state,state,ref,task.body_kin(state),task.body_kin(state),task.body_kin(ref),xi)
zero_anchor=float(terms['hind_foot_anchor'].abs().max());assert zero_anchor<1e-6
s={k:v.clone() for k,v in state.items()}
s['pos']=(s['pos']+torch.tensor([.03,.03,0],device='cuda')).requires_grad_();s['v']=(s['v']+.04).requires_grad_()
terms,_=task.stability_terms(state,s,ref,task.body_kin(state),task.body_kin(s),task.body_kin(ref),xi)
loss=(terms['hind_foot_anchor']+terms['hind_foot_speed']).sum();loss.backward()
assert torch.isfinite(s['pos'].grad).all() and (s['pos'].grad[:,:2].abs()>0).all()
assert torch.isfinite(s['v'].grad).all() and (s['v'].grad.abs()>0).all()
assert (terms['hind_foot_anchor']>=-task.cfg.w_hind_anchor-1e-6).all()
assert (terms['hind_foot_speed']>=-task.cfg.w_hind_speed-1e-6).all()
result=dict(passed=True,source_actor_identity_max_error=identity,max_joint_correction_verified_rad=limit,
            frozen_policy_normalizers=True,real_imagined_reward_max_error=max(parity),anchor_capture_max_error=max(anchor_errors),
            reset_local_anchor_memory_verified=True,actual_stationary_foothold_penalty=zero_anchor,
            bounded_penalties_and_gradients_verified=True)
(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('RECOVERY_AUDIT '+json.dumps(result),flush=True)

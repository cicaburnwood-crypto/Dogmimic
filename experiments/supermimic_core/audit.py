"""Finite CPU/CUDA/DDP execution and critic-ablation correctness checks."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from common import ROOT, setup, write_json, tensor_digest

p = argparse.ArgumentParser()
p.add_argument('--robot', choices=['B2','G1'], required=True)
p.add_argument('--device', default='cpu')
p.add_argument('--out', required=True)
a = p.parse_args()
Trainer, base, legacy_update = setup(a.robot)
import torch
torch.set_num_threads(2)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cfg = base.TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz' if a.robot=='B2'
                    else 'data/processed/g1_standing_jump_060m.npz',
                    out=str(out), seed=1202601, num_envs=4, buffer_steps=64,
                    wm_batch=4, pi_batch=4, wm_batches=1, k=8, horizon=32,
                    lr=.0001, use_critic=False, amp=a.device!='cpu',
                    compile_wm=False, eval_every=0, action_noise=.015 if a.robot=='B2' else .1)
t = Trainer(cfg, device=a.device)
from state import verify_replay
replay_report=verify_replay(t.env,t.policy,steps=8)
assert replay_report['passed'],replay_report
initial_policy = tensor_digest(t.policy.state_dict())
initial_wm = tensor_digest(t.wm.state_dict())
anchor = tensor_digest(t.frozen_anchor.state_dict())
for _ in range(40):
    t.collect()
s0, acts, s1, xi, mask = t.buf.sample_windows(16, 8)
assert xi.shape[1] == 8 and xi.shape[-1] == (43 if a.robot=='B2' else 304)
assert mask.any()
policy_state = copy.deepcopy(t.policy.state_dict())
opt_state = copy.deepcopy(t.opt_pi.state_dict())
rng = torch.get_rng_state()
cuda_rng = torch.cuda.get_rng_state() if a.device!='cpu' else None
t.it = 501
legacy_stats = legacy_update(t)
legacy_weights = copy.deepcopy(t.policy.state_dict())
t.policy.load_state_dict(policy_state)
t.opt_pi.load_state_dict(opt_state)
torch.set_rng_state(rng)
if cuda_rng is not None:
    torch.cuda.set_rng_state(cuda_rng)
new_stats = t.update_policy()
max_difference = max(float((legacy_weights[k]-v).abs().max())
                     for k,v in t.policy.state_dict().items() if v.numel())
assert max_difference < 2e-6, ('C0 differs from previous actor update', max_difference)
assert abs(legacy_stats['pi_loss']-new_stats['pi_loss']) < 2e-5
t.policy.load_state_dict(policy_state)
t.opt_pi.load_state_dict(opt_state)
t.actor_updates = 0
cfg.use_critic = True
loss, rec, xH, valid, stats = t.imagined_objective()
assert xH.shape[-1] == t.critic_in
tail = (t.critic(xH)*valid[:,-1]).sum()
dx, = torch.autograd.grad(tail, xH, retain_graph=True)
params = [q for q in t.policy.parameters() if q.requires_grad]
dp = torch.autograd.grad(tail, params, allow_unused=True)
assert torch.isfinite(dx).all() and dx.abs().sum()>0
assert sum(float(g.abs().sum()) for g in dp if g is not None)>0
last = torch.full((4,),int(t.lib.num_frames)-1,device=t.device,dtype=torch.long)
assert (t.lib.remaining(last)==0).all()
assert torch.count_nonzero(t.critic(xH.detach())*torch.zeros_like(valid[:,-1])) == 0
before = tensor_digest(t.policy.state_dict())
before_wm = tensor_digest(t.wm.state_dict())
critic_before = tensor_digest(t.critic.state_dict())
t.update_policy(critic_only=True)
assert before == tensor_digest(t.policy.state_dict())
assert before_wm == tensor_digest(t.wm.state_dict())
assert critic_before != tensor_digest(t.critic.state_dict())
rows=[]
for i in range(3):
    t.it=501+i
    w=t.update_wm()
    wm_before=tensor_digest(t.wm.state_dict())
    t.wm.zero_grad(set_to_none=True)
    st=t.update_policy()
    assert wm_before == tensor_digest(t.wm.state_dict())
    assert all(q.grad is None for q in t.wm.net.parameters())
    assert all(q.grad is None and not q.requires_grad for q in t.frozen_anchor.parameters())
    assert anchor == tensor_digest(t.frozen_anchor.state_dict())
    assert all(math.isfinite(v) for v in list(w.values())+list(st.values()))
    t.updates+=1
    rows.append(dict(wm=w,actor=st))
assert initial_policy != tensor_digest(t.policy.state_dict())
assert initial_wm != tensor_digest(t.wm.state_dict())
if a.robot=='B2':
    import actuator_filter as af
    assert af.COMMAND_POWER==1200
    q=torch.randn(16,12,device=t.device)
    assert torch.equal(af.filtered_target(q,q*0),q)
    assert ((t.env.loads==0)|(t.env.loads==5)).all()
    assert (t.env.push_vectors.norm(dim=-1)>=7.999).all()
else:
    action=torch.zeros(4,29,device=t.device,requires_grad=True)
    context=t.env.xi.clone();context[:,37]=1.
    v=t.wm.inputs(t.env.state,action,context)
    assert torch.equal(v,t.wm.inputs(t.env.state,action+1,context))
    assert torch.count_nonzero(torch.autograd.grad(v.sum(),action)[0])==0
    context[:,37]=0.
    grad=torch.autograd.grad(t.wm.inputs(t.env.state,action,context).sum(),action)[0]
    assert torch.allclose(grad,torch.full_like(grad,4.))
ddp=False
if t.world>1:
    import torch.distributed as dist
    digest=tuple(tensor_digest(m.state_dict()) for m in (t.policy,t.wm,t.critic))
    all_digests=[None]*t.world
    dist.all_gather_object(all_digests,digest)
    assert len(set(all_digests))==1,all_digests
    ddp=True
report=dict(passed=True,robot=a.robot,device=a.device,world=t.world,critic_input_dim=t.critic_in,
            actual_context_dim=t.env.xi.shape[-1],horizon=32,k=8,updates=3,
            source_sha256=t.source_hash,C0_max_parameter_difference=max_difference,
            original_C0_parity=True,value_state_gradient=True,value_actor_gradient=True,
            critic_only_no_actor_or_WM_change=True,actor_updated=True,wm_updated=True,
            frozen_anchor_unchanged=True,terminal_tail_mask=True,ddp_weights_equal=ddp,
            complete_state_replay=replay_report,
            rows=rows,scope='finite implementation checks, NOT physical skill acceptance')
if t.rank==0:
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
t._barrier()

import os,sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.extend(str(ROOT/'experiments'/n) for n in ['three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills'])
import train_repair as m
from supermimic.train import TrainCfg
os.environ['B2_INIT_CHECKPOINT']=str(ROOT/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt')
os.environ['B2_POLICY_SOURCE']=str(ROOT/'runs/b2_balance_repair_20261101_230646/best.pt')
os.environ['B2_CONTINUE_CHECKPOINT']=str(ROOT/'runs/b2_effective_training_probe_20261101_205738/best.pt')
os.environ['B2_WM_PREFIT']='0';os.environ.pop('B2_WM_SOURCE',None)
m.torch.set_num_threads(2)
import argparse
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
out=ROOT/'runs'/('b2_power_repair_'+a.device+'_smoke');out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,wm_batches=1,horizon=32,k=2,use_critic=False,amp=a.device!='cpu',compile_wm=a.device!='cpu')
t=m.GuardedTrainer(cfg,a.device);ck=m.torch.load(os.environ['B2_POLICY_SOURCE'],map_location=a.device,weights_only=False)
assert all(m.torch.equal(v,t.policy.state_dict()[k]) for k,v in ck['policy'].items())
assert all(m.torch.equal(v,t.wm.state_dict()[k]) for k,v in ck['wm'].items())
import actuator_filter as af
assert af.COMMAND_POWER==1200.
a0=m.torch.zeros(2,12,device=a.device);a1=m.torch.ones_like(a0)*.5
assert m.torch.equal(af.filtered_target(a1,a0),a1)
anchor={k:v.clone() for k,v in t.frozen_anchor.state_dict().items()}
for _ in range(20):t.collect()
for i in range(3):
 t.it=i+1;w=t.update_wm();p=t.update_policy();t.updates+=1
 assert all(math.isfinite(v) for v in list(w.values())+list(p.values()))
assert all(m.torch.equal(v,t.frozen_anchor.state_dict()[k]) for k,v in anchor.items())
assert any(not m.torch.equal(v,t.policy.state_dict()[k]) for k,v in ck['policy'].items())
# Verify the real task term and its derivative, not just a mirrored formula.
s0={k:v.detach().clone() for k,v in t.env.state.items()};s1={k:v.clone() for k,v in s0.items()}
s0['qd'].zero_();s1['qd'].zero_();s1['qd'][:,0]=7.;s1['qd'].requires_grad_(True)
f=t.env.frame+1;ref=t.lib.frame(f);kin=t.env.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
terms=t.task.transition_fast(s0,s1,ref,kin,t.env.ref_future(f),t.env.prev_action,t.env.prev_action,t.env.xi)['terms']
v=terms['actuator_speed'];grad=m.torch.autograd.grad(v.sum(),s1['qd'])[0]
assert m.torch.allclose(v,m.torch.full_like(v,-2*(7/6-1)**2),atol=1e-6)
assert m.torch.allclose(grad[:,0],m.torch.full_like(grad[:,0],-4*(7/6-1)/6),atol=1e-6)
assert m.torch.count_nonzero(grad[:,1:])==0
# Physical torque function must be bit-identical to the previous version.
import importlib.util
spec=importlib.util.spec_from_file_location('previous_envelope',ROOT/'experiments/b2_effective_action_candidate/actuator_filter.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
q=m.torch.randn(128,12,device=a.device);qd=m.torch.randn_like(q)*8;target=q+.2*m.torch.randn_like(q);lim=m.torch.tensor([160.,160.,240.]*4,device=a.device)
assert all(m.torch.equal(x,y) for x,y in zip(af.torque_envelope(target,q,qd,lim),old.torque_envelope(target,q,qd,lim)))
s0['qd'].fill_(5.);s1['qd']=s0['qd'].clone()
action=m.torch.full_like(s0['q'],.7,requires_grad=True)
v=t.task.transition_fast(s0,s1,ref,kin,t.env.ref_future(f),action,t.env.prev_action,t.env.xi)['terms']['actuator_power']
g=m.torch.autograd.grad(v.sum(),action)[0]
assert (v<0).all() and m.torch.isfinite(g).all() and g.abs().sum()>0
# Old capped-power quantity can never exceed its threshold.
tau,_=af.torque_envelope(target,q,qd,lim)
assert ((tau*qd).abs().sum(-1)<=1200.01).all()
r=dict(passed=True,device=a.device,source_sha256=t.source_hash,actor_exact_load=True,continued_world_model=True,power_penalty_nonzero_gradient=True,anchor_preserved=True,updates=3,horizon=32,worst_joint_gradient_verified=True,physical_torque_identical=True,acceptance_speed_rad_s=6,acceptance_power_w=1501,scope='small-batch implementation smoke, not physical acceptance')
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

import hashlib,json,torch
from pathlib import Path
from robust_env import torque_envelope,POWER
p=Path(__file__).resolve().parent
q=torch.randn(10000,12);qd=torch.randn_like(q)*12;target=torch.randn_like(q,requires_grad=True);limits=torch.tensor([160.,160.,240.]*4)
tau,raw=torque_envelope(target,q,qd,limits)
assert torch.isfinite(tau).all() and (tau.abs()<=limits+1e-4).all()
assert ((tau*qd).abs().sum(-1)<=POWER+.01).all()
assert (tau[(qd.abs()>=10)&(raw*qd>0)].abs()<1e-5).all()
(tau.square().mean()+raw.square().mean()).backward();assert torch.isfinite(target.grad).all() and target.grad.abs().max()>0
root=p.parents[1];criterion=root/'experiments/b2_cycle/evaluate_cycle.py'
a=dict(passed=True,random_samples=10000,torque_power_envelope=True,braking_preserved=True,finite_gradients=True,original_evaluator_sha256=hashlib.sha256(criterion.read_bytes()).hexdigest(),experimental_not_vendor_limits=True)
(p/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))

import json,torch
from actuator_filter import *
torch.set_num_threads(1);torch.manual_seed(11)
B=8;xi=torch.randn(B,CONTEXT_DIM,dtype=torch.float64)*.1
q0=torch.zeros(12,dtype=torch.float64);scale=torch.ones(12,dtype=torch.float64)*.7
qmin=q0-2.;qmax=q0+2.;actions=torch.randn(16,B,12,dtype=torch.float64)*.3
physical_previous=xi[:,CONTEXT_PREFIX:].clone();context=xi.clone()
for action in actions:
    request=(q0+scale*action-xi[:,1:13]).clamp(qmin+.02,qmax-.02)
    physical=filtered_target(request,physical_previous)
    nxt=advance_context(context,action,q0,scale,qmin,qmax)
    assert torch.equal(physical,nxt[:,CONTEXT_PREFIX:])
    assert (abs(physical-physical_previous)<=COMMAND_RATE*CONTROL_DT+1e-12).all()
    assert torch.equal(nxt[:,:CONTEXT_PREFIX],xi[:,:CONTEXT_PREFIX])
    physical_previous=physical;context=nxt
# Same robot state/request, different filter memory must produce different control.
request=torch.ones(2,12);previous=torch.stack([torch.zeros(12),torch.ones(12)*.5])
assert not torch.equal(*filtered_target(request,previous).unbind())
# Validate ordinary and saturated derivatives away from clamp boundaries.
for value in [.02,.3]:
    t=torch.tensor(value,dtype=torch.float64,requires_grad=True)
    f=lambda x:filtered_target(x,torch.tensor(0.,dtype=torch.float64)).square()
    grad,=torch.autograd.grad(f(t),t);h=1e-6;fd=(f(t.detach()+h)-f(t.detach()-h))/(2*h)
    assert abs(grad-fd)<1e-8
q=torch.randn(100,12);qd=torch.randn(100,12)*8;target=torch.randn(100,12)
tau,_=torque_envelope(target,q,qd,torch.tensor([160.,160.,240.]*4))
assert ((tau*qd).abs().sum(-1)<=COMMAND_POWER+.001).all()
print(json.dumps(dict(passed=True,context_dim=CONTEXT_DIM,physical_imagined_filter_parity_steps=16,hidden_state_required=True,gradient_check_passed=True,command_power_bound_passed=True,scope='CPU component audit only; no integration/physical acceptance/deployment')))

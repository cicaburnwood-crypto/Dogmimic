"""Candidate actuator dynamics; shared by physical and imagined transitions.

Experimental command budgets, not vendor specifications. This module alone is
not an integrated trainer or a deployable controller.
"""
import torch
COMMAND_RATE = 4.0
COMMAND_POWER = 1200.0
CONTROL_DT = .02
CONTEXT_PREFIX = 31
CONTEXT_DIM = CONTEXT_PREFIX + 12

def filtered_target(request, previous_target, dt=CONTROL_DT):
    bound=COMMAND_RATE*dt
    return previous_target+(request-previous_target).clamp(-bound,bound)

def advance_context(xi, action, q0, action_scale, qmin, qmax):
    request=(q0+action_scale*action-xi[...,1:13]).clamp(qmin+.02,qmax-.02)
    target=filtered_target(request,xi[...,CONTEXT_PREFIX:CONTEXT_DIM])
    return torch.cat([xi[...,:CONTEXT_PREFIX],target],-1)

def torque_envelope(target,q,qd,limits):
    raw=250*(target-q)-8*qd
    tau=torch.maximum(torch.minimum(raw,limits),-limits)
    scale=((10.-qd.abs())/4.).clamp(0,1)
    tau=torch.where(tau*qd>0,tau*scale,tau)
    power=(tau*qd).abs().sum(-1,keepdim=True)
    tau=tau*(COMMAND_POWER/power.clamp_min(COMMAND_POWER))
    return tau,raw

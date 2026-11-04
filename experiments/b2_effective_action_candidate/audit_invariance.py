"""Check exact actuator-command equivalence versus current WM action input."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'b2_actuator_filtered_candidate'))
import torch
from actuator_filter import advance_context
B=16;torch.manual_seed(11)
xi=torch.zeros(B,43,dtype=torch.float64);xi[:,31:]=torch.randn(B,12,dtype=torch.float64)*.1
q0=torch.zeros(12,dtype=torch.float64);qmin=q0-3;qmax=q0+3;scale=.7
# Two requests on the same saturated branch have identical physical targets.
a=(xi[:,31:]+.2)/scale;b=(xi[:,31:]+.4)/scale
fa=advance_context(xi,a,q0,scale,qmin,qmax)[:,31:];fb=advance_context(xi,b,q0,scale,qmin,qmax)[:,31:]
assert torch.equal(fa,fb) and not torch.equal(a,b)
# Equivalent action representation keeps existing action units, but guarantees invariance.
def effective(action,context):
 target=advance_context(context,action,q0,scale,qmin,qmax)[:,31:]
 return (target-q0+context[:,1:13])/scale
assert torch.equal(effective(a,xi),effective(b,xi))
v=a.clone().requires_grad_(True);g,=torch.autograd.grad(effective(v,xi).sum(),v)
assert (g==0).all()
# Unsaturated action retains nonzero derivative; no straight-through approximation.
v=((xi[:,31:]+.01)/scale).clone().requires_grad_(True);g,=torch.autograd.grad(effective(v,xi).sum(),v)
assert torch.allclose(g,torch.ones_like(g))
print(json.dumps(dict(passed=True,equivalent_physical_targets=True,current_raw_action_inputs_differ=True,effective_action_inputs_identical=True,saturated_jacobian_zero=True,unsaturated_jacobian_one=True,scope='Exact actuator-map invariance; does not establish measured learned-model error or explain all failures')))

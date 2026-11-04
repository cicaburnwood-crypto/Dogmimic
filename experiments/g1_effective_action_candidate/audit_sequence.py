import torch,json
from actuator_sequence import consumed_targets
torch.manual_seed(11);torch.set_num_threads(2)
xi=torch.randn(9,304,dtype=torch.float64);xi[:,37]=torch.arange(9)/8
q0=torch.randn(29,dtype=torch.float64);scale=.7;a=torch.randn(9,29,dtype=torch.float64,requires_grad=True)
expected=[];queue=xi[:,42:303].reshape(9,9,29).clone();target=q0+scale*a-xi[:,1:30]
for j in range(4):
 queue[:,j]=target
 expected.append(queue[torch.arange(9),(j-torch.arange(9))%9])
expected=torch.stack(expected,1);actual=consumed_targets(xi,a,q0,scale)
assert torch.equal(actual,expected)
grad,=torch.autograd.grad(actual.sum(),a)
weights=(4-torch.arange(9)).clamp_min(0).double()[:,None]*scale
assert torch.allclose(grad,weights.expand(-1,29))
b=a.detach()+1;other=consumed_targets(xi,b,q0,scale)
assert torch.equal(actual[4:],other[4:])
print(json.dumps(dict(passed=True,all_nine_delays_exact=True,substep_order_exact=True,current_action_jacobian_matches_consumed_substeps=True,delay20_to40ms_no_immediate_action_effect=True,scope='Actuator queue mapping; no learned-model or physical task acceptance')))

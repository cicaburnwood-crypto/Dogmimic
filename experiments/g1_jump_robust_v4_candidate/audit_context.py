import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'engine'))
import torch
from supermimic.dynamic_context import advance,actual_force,DIM
xi=torch.zeros(9,DIM);q=torch.arange(9*29).reshape(1,9,29).expand(9,-1,-1).float();xi[:,42:303]=q.flatten(1);a=torch.ones(9,29)*7
next_x=advance(xi,a,torch.zeros(29),torch.ones(29))
for lag in range(9):
 ring=q[lag].clone();pos=0
 for _ in range(4):ring[pos]=a[lag];_sent=ring[(pos-lag)%9];pos=(pos+1)%9
 actual=ring[(torch.arange(9)+pos)%9]
 assert torch.equal(actual,next_x[lag,42:303].reshape(9,29))
for frame,expected in [(47,0),(48,1),(67,1),(68,0)]:
 xi[:,-1]=frame/200;xi[:,39:42]=1;assert (actual_force(xi)==expected).all()
a.requires_grad_();advance(xi,a,torch.zeros(29),torch.ones(29))[:,42:303].sum().backward();assert (a.grad==4).all()
print(json.dumps(dict(passed=True,queue_cases=9,pulse_boundaries=4,context_dim=DIM,queue_gradient=True)))

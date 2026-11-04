"""Run once with upstream PYTHONPATH and once with the adapter; compare all arrays."""
import sys
import numpy as np
import torch
from supermimic.policy.nets import Policy,PolicyCfg,Critic,CriticCfg
from supermimic.world_model.state_spec import StateSpec,StateSpecCfg
from supermimic.world_model.dynamics import DynamicsMLP
from supermimic.world_model.heads import analytic_observation,flatten_history_term_major
torch.set_num_threads(2);torch.manual_seed(19)
spec=StateSpec(StateSpecCfg(),'cpu')
s=torch.randn(4,71);s[:,3:7]=s[:,3:7]/s[:,3:7].norm(dim=-1,keepdim=True)
xi=torch.randn(4,33);a=torch.randn(4,29)
delta=torch.randn(4,70)
cfg=PolicyCfg(encoder_hidden=(32,16),decoder_hidden=(32,16))
pi=Policy(cfg);critic=Critic(CriticCfg(hidden=(32,16)),pcfg=cfg)
dyn=DynamicsMLP(hidden=(32,16))
obs=analytic_observation(s,a,xi,torch.zeros(29))
hist=flatten_history_term_major(obs[None].repeat(10,1,1));goal=torch.randn(4,680)
act=pi(hist,goal);feat=spec.canonical(s);value=critic(feat,goal,spec.normalize_xi(xi),torch.zeros(4,4))
(act.square().mean()+value.square().mean()).backward()
out=dict(canonical=feat,integrate=spec.integrate(s,delta),error=spec.error_vector(s,spec.integrate(s,delta)),
    obs=obs,history=hist,action=act,value=value,dynamics=dyn(feat,a,spec.normalize_xi(xi)),
    policy_gradient=torch.cat([p.grad.flatten() for p in pi.parameters()]),
    critic_gradient=torch.cat([p.grad.flatten() for p in critic.parameters()]))
np.savez(sys.argv[1],**{k:v.detach().numpy() for k,v in out.items()})

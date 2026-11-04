import copy,json,torch
from pathlib import Path
from stage_policy import from_checkpoint
from supermimic import networks
networks.AMP['enabled']=False
p=Path(__file__).resolve().parents[2];ck=torch.load(p/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt',map_location='cpu',weights_only=False)
policy=from_checkpoint(ck,'cpu');policy.teacher.requires_grad_(True);anchor=copy.deepcopy(policy).requires_grad_(False)
obs=torch.zeros(8,210);goal=torch.zeros(8,196);goal[:,-1]=0;goal[:,10]=1.;goal[:,13]=1.
with torch.no_grad():policy.teacher.decoder[-1].bias.add_(.01)
a,prior=policy.with_prior(obs,goal);fixed=anchor(obs,goal).detach();old=(a-prior).square().mean();new=(a-fixed).square().mean()
assert old.item()==0 and new.item()>0
new.backward();assert policy.teacher.decoder[-1].bias.grad.abs().max()>0
result=dict(passed=True,moving_prior_loss=old.item(),frozen_anchor_loss=new.item(),frozen_anchor_has_no_grad=all(x.grad is None for x in anchor.parameters()))
Path(__file__).with_name('audit_anchor.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

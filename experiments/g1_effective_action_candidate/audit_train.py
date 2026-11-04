import sys,json,copy,math,io,hashlib,argparse
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];v5=p.parent/'g1_jump_robust_v5_candidate'
sys.path.insert(1,str(p/'engine'))
import torch
from supermimic.train import Trainer,TrainCfg
from robust_env import RobustEnv
class CandidateTrainer(Trainer):
 def _make_env(self):
  e=RobustEnv(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma);self.tcfg=e.tcfg;return e
args=argparse.ArgumentParser();args.add_argument('--device',default='cpu');a=args.parse_args();torch.set_num_threads(2)
out=root/'runs'/('g1_effective_action_'+a.device+'_smoke');out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='data/processed/g1_standing_jump_060m.npz',out=str(out),num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,wm_batches=1,horizon=32,k=2,use_critic=False,amp=a.device!='cpu',compile_wm=a.device!='cpu')
t=CandidateTrainer(cfg,device=a.device)
source=root/'runs/g1_jump_robust_v5_20261101_195039/best.pt';raw=source.read_bytes();ck=torch.load(io.BytesIO(raw),map_location=a.device,weights_only=False)
t.policy.load_state_dict(ck['policy']);t.policy.obs_norm.until=0;t.policy.goal_norm.until=0
t.frozen_anchor=copy.deepcopy(t.policy).requires_grad_(False).eval()
xi=t.env.xi.clone();xi[:,37]=1.
a0=torch.zeros(4,29,device=a.device,requires_grad=True)
x=t.wm.inputs(t.env.state,a0,xi)
assert torch.equal(x,t.wm.inputs(t.env.state,a0+1,xi))
grad,=torch.autograd.grad(x.sum(),a0);assert (grad==0).all()
xi[:,37]=0.;grad,=torch.autograd.grad(t.wm.inputs(t.env.state,a0,xi).sum(),a0)
assert torch.allclose(grad,torch.full_like(grad,4.))
before={k:v.clone() for k,v in t.policy.state_dict().items()};wm={k:v.clone() for k,v in t.wm.net.state_dict().items()}
for _ in range(20):t.collect()
rows=[]
for i in range(3):
 t.it=i+1;w=t.update_wm();actor=t.update_policy();t.updates+=1
 assert all(math.isfinite(v) for v in list(w.values())+list(actor.values()))
 rows.append(dict(wm=w,actor=actor))
assert all(torch.equal(v,t.frozen_anchor.state_dict()[k]) for k,v in before.items())
assert any(not torch.equal(v,t.policy.state_dict()[k]) for k,v in before.items())
assert any(not torch.equal(v,t.wm.net.state_dict()[k]) for k,v in wm.items())
assert all(p.grad is None for p in t.frozen_anchor.parameters())
r=dict(passed=True,device=a.device,horizon=32,updates=3,environments=4,context_dim=t.env.xi.shape[-1],anchor_unchanged=True,actual_wm_delay_equivalence=True,delay_action_gradient_verified=True,wm_updated=True,actor_updated=True,source_iteration=ck['it'],source_sha256=hashlib.sha256(raw).hexdigest(),rows=rows,scope='Small-batch actual trainer smoke; no physical skill acceptance')
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='rows'}))

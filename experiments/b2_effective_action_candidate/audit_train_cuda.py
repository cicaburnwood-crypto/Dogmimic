"""Actual trainer collect/update smoke on CPU, compilation and AMP disabled."""
import os,json,math,hashlib,argparse
from pathlib import Path
import torch
import train_guarded as mod
from supermimic.train import TrainCfg
root=Path(__file__).resolve().parents[2]
os.environ['B2_INIT_CHECKPOINT']=str(root/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt')
os.environ['B2_POLICY_SOURCE']=str(root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt')
os.environ.pop('B2_WM_SOURCE',None);os.environ['B2_WM_PREFIT']='0'
torch.set_num_threads(2)
args=argparse.ArgumentParser();args.add_argument('--horizon',type=int,default=2);opt=args.parse_args()
out=root/'runs/b2_effective_cuda_training_audit';out=out.with_name(out.name+'_h'+str(opt.horizon));out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,wm_batches=1,horizon=opt.horizon,k=2,use_critic=False,amp=True,compile_wm=True)
t=mod.GuardedTrainer(cfg,device='cuda')
# Exact equivalence must hold in the real WM adapter, not only in a helper.
xi=t.env.xi.clone();xi[:,31:]=(t.task.qmin+t.task.qmax)/2
act1=(xi[:,31:]+.2-t.task.q0+xi[:,1:13])/t.task.action_scale
act2=(xi[:,31:]+.4-t.task.q0+xi[:,1:13])/t.task.action_scale
inp1=t.wm.inputs(t.env.state,act1,xi);inp2=t.wm.inputs(t.env.state,act2,xi)
assert torch.equal(inp1,inp2)
act1=act1.detach().requires_grad_(True)
grad,=torch.autograd.grad(t.wm.inputs(t.env.state,act1,xi).sum(),act1)
assert (grad==0).all()
anchor={k:v.clone() for k,v in t.frozen_anchor.state_dict().items()}
actor={k:v.clone() for k,v in t.policy.state_dict().items()}
wm={k:v.clone() for k,v in t.wm.net.state_dict().items()}
for _ in range(20):t.collect()
assert t.buf.xi.shape[-1]==43
rows=[]
for i in range(3):
 t.it=i+1;w=t.update_wm();p=t.update_policy();t.updates+=1
 assert all(math.isfinite(v) for v in list(w.values())+list(p.values()))
 rows.append(dict(wm=w,policy=p))
assert all(torch.equal(v,t.frozen_anchor.state_dict()[k]) for k,v in anchor.items())
assert any(not torch.equal(v,t.policy.state_dict()[k]) for k,v in actor.items())
assert any(not torch.equal(v,t.wm.net.state_dict()[k]) for k,v in wm.items())
assert all(p.grad is None for p in t.frozen_anchor.parameters())
d=dict(passed=True,device='cuda',horizon=opt.horizon,updates=3,environments=4,replay_context_dim=43,anchor_unchanged=True,actual_adapter_equivalence=True,saturated_action_gradient_zero=True,actor_updated=True,wm_updated=True,source_sha256=t.source_hash,rows=rows,scope='Actual small-batch trainer smoke only; no policy quality or full-batch throughput claim')
(out/'report.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))

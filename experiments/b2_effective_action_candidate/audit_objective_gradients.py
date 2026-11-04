"""Read-only CPU gradient diagnosis of saved Actor/WM; no optimizer steps."""
import os,inspect,textwrap,json,copy
from pathlib import Path
import torch
import sys
root=Path(__file__).resolve().parents[2]
sys.path.extend(str(root/'experiments'/n) for n in ['three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills'])
import train_guarded as mod
from supermimic.train import TrainCfg
root=Path(__file__).resolve().parents[2]
os.environ['B2_INIT_CHECKPOINT']=str(root/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt')
os.environ['B2_POLICY_SOURCE']=str(root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt')
os.environ.pop('B2_WM_SOURCE',None);os.environ['B2_WM_PREFIT']='0'
torch.set_num_threads(2)
out=root/'runs/b2_effective_gradient_diagnosis';out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=128,wm_batch=4,pi_batch=16,wm_batches=1,horizon=16,k=2,use_critic=False,amp=False,compile_wm=False)
t=mod.GuardedTrainer(cfg,device='cpu')
source=root/'runs/b2_effective_training_probe_20261101_205738/best.pt'
ck=torch.load(source,map_location='cpu',weights_only=False)
t.policy.load_state_dict(ck['policy']);t.wm.load_state_dict(ck['wm']);t.it=ck['it']
# Keep the original frozen reference, matching this training round.
for _ in range(100):t.collect()
code=textwrap.dedent(inspect.getsource(mod.GuardedTrainer.update_policy))
code=code[:code.index('    self.opt_pi.zero_grad')]+'''    params=[p for g in self.opt_pi.param_groups for p in g['params']]
    objectives=[-J.mean()/H,self.teacher_weight*trust.mean()/H]
    gradients=[]
    for obj in objectives:
        gs=torch.autograd.grad(obj,params,retain_graph=True,allow_unused=True)
        gradients.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,params)]))
    g0,g1=gradients
    return dict(reward_gradient_norm=float(g0.norm()),anchor_gradient_norm=float(g1.norm()),cosine=float(torch.nn.functional.cosine_similarity(g0[None],g1[None])),combined_norm=float((g0+g1).norm()),reward_loss=float(objectives[0]),anchor_loss=float(objectives[1]))
'''
ns=dict(vars(mod));exec(code,ns);t.update_policy=ns['update_policy'].__get__(t,type(t))
rows=[t.update_policy() for _ in range(3)]
r=dict(checkpoint=str(source),it=ck['it'],horizon=16,starts_per_batch=16,batches=rows,scope='CPU small sampled replay gradient diagnosis, no optimizer update, not physical acceptance')
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

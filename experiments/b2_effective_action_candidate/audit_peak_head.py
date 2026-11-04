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
seed=int(os.environ.get('B2_AUDIT_SEED','101'))
torch.manual_seed(seed)
torch.set_num_threads(2)
out=root/f'runs/b2_peak_head_feasibility';out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=256,wm_batch=4,pi_batch=16,wm_batches=1,horizon=16,k=2,use_critic=False,amp=False,compile_wm=False)
t=mod.GuardedTrainer(cfg,device='cpu')
source=root/'runs/b2_effective_training_probe_20261101_205738/best.pt'
ck=torch.load(source,map_location='cpu',weights_only=False)
t.policy.load_state_dict(ck['policy']);t.wm.load_state_dict(ck['wm']);t.it=ck['it']
# Freeze inference normalizers; collect independent physics transitions without updates.
t.policy.obs_norm.until=0;t.policy.goal_norm.until=0
t.env.training_loads=False
t.env.fixed_loads.copy_(torch.tensor([0.,5.,10.,15.]))
torch.manual_seed(seed)  # Trainer initialization sets its own seed; override before sampling.
t.env.reset()

from actuator_filter import torque_envelope

# Isolated supervised feasibility experiment; never updates Actor or original WM.
original=t.env._actuated_step
sub=[]
def traced(target):
    original(target)
    qd=t.env.sim.data.qvel[:,t.env.qd_adr][:,t.env.act_to_joint]
    sub.append(torch.stack([qd.abs().amax(-1)/6.,(t.env.sim.data.actuator_force*qd).abs().sum(-1)/1500.],-1).clone())
t.env._actuated_step=traced
original_step=t.env.step
X=[];Y=[]
def traced_step(action):
    sub.clear()
    s=t.env.state
    x=torch.cat([s['q'],s['qd'],s['v'],s['w'],s['rot'].flatten(1),action,t.env.xi],-1).detach().clone()
    state,info=original_step(action)
    X.append(x);Y.append(torch.stack(sub).amax(0))
    return state,info
t.env.step=traced_step
def collect(seed):
    X.clear();Y.clear();torch.manual_seed(seed);t.env.reset()
    for _ in range(1000):t.collect()
    return torch.cat(X).detach(),torch.cat(Y).detach()
x,y=collect(101);xt,yt=collect(202)
mean=x.mean(0);std=x.std(0).clamp_min(.1)
x=(x-mean)/std;xt=(xt-mean)/std
torch.manual_seed(919)
head=torch.nn.Sequential(torch.nn.Linear(x.shape[-1],128),torch.nn.ELU(),torch.nn.Linear(128,128),torch.nn.ELU(),torch.nn.Linear(128,2),torch.nn.Softplus())
opt=torch.optim.Adam(head.parameters(),lr=1e-3)
for step in range(1500):
    idx=torch.randint(len(x),(256,));pred=head(x[idx]);loss=(pred-y[idx]).square().mean()
    opt.zero_grad();loss.backward();opt.step()
with torch.no_grad():
    pred=head(xt);fast=yt[:,0]>1;hot=yt[:,1]>1
    report=dict(source=str(source),train_seed=101,test_seed=202,train_samples=len(x),test_samples=len(xt),normalized_rmse=(pred-yt).square().mean(0).sqrt().tolist(),overspeed=int(fast.sum()),missed_overspeed=int((fast&(pred[:,0]<=1)).sum()),false_speed_alarms=int(((~fast)&(pred[:,0]>1)).sum()),overpower=int(hot.sum()),missed_overpower=int((hot&(pred[:,1]<=1)).sum()),scope='Isolated supervised head, held-out seed sequential random training trajectories; no Actor updates, no deployment acceptance, no active training changes.')
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

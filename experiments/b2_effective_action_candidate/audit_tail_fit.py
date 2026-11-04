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
out=root/'runs/b2_wm_tail_fit_diagnosis';out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=256,wm_batch=4,pi_batch=16,wm_batches=1,horizon=16,k=2,use_critic=False,amp=False,compile_wm=False)
t=mod.GuardedTrainer(cfg,device='cpu')
source=root/'runs/b2_effective_training_probe_20261101_205738/best.pt'
ck=torch.load(source,map_location='cpu',weights_only=False)
t.policy.load_state_dict(ck['policy']);t.wm.load_state_dict(ck['wm']);t.it=ck['it']
# Freeze inference normalizers; collect independent physics transitions without updates.
t.policy.obs_norm.until=0;t.policy.goal_norm.until=0
t.env.training_loads=False
t.env.fixed_loads.copy_(torch.tensor([0.,5.,10.,15.]))
t.env.reset()
from supermimic.networks import state_error
torch.manual_seed(5719)
for _ in range(240):t.collect()
train=t.buf.sample_windows(4096,1)
t.buf=t._make_buffer();torch.manual_seed(9913);t.env.reset()
for _ in range(240):t.collect()
test=t.buf.sample_windows(4096,1)
def score(wm,data):
    s0,A,S1,xi,mask=data
    with torch.no_grad():
        truth=mod.flat_to_state(S1[:,0]);pred=wm.step(mod.flat_to_state(s0),A[:,0],xi[:,0])
        v=truth['qd'].abs().amax(-1);pv=pred['qd'].abs().amax(-1);rows=[]
        kg=xi[:,0,16]*15
        for load in [0,5,10,15]:
            m=mask&((kg-load).abs()<.01);fast=m&(v>6)
            rows.append(dict(kg=load,samples=int(m.sum()),rmse=float((pred['qd'][m]-truth['qd'][m]).square().mean().sqrt()),overspeed=int(fast.sum()),missed=int((fast&(pv<=6)).sum())))
        return rows
results={}
for name,tail in [('uniform',1.),('tail_weighted',5.)]:
    wm=copy.deepcopy(t.wm);wm.requires_grad_(True);opt=torch.optim.Adam(wm.net.parameters(),lr=1e-4)
    before=score(wm,test);torch.manual_seed(173)
    s0,A,S1,xi,mask=train;valid=mask.nonzero().flatten()
    for step in range(100):
        ids=valid[torch.randint(len(valid),(64,))]
        truth=mod.flat_to_state(S1[ids,0]);pred=wm.step(mod.flat_to_state(s0[ids]),A[ids,0],xi[ids,0])
        error=(state_error(pred,truth)/wm.out_norm.std).square().mean(-1)
        weight=1+(tail-1)*(truth['qd'].abs().amax(-1)>4).float()
        loss=(error*weight).sum()/weight.sum()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(wm.net.parameters(),1.);opt.step()
    results[name]=dict(before=before,after=score(wm,test))
r=dict(source=str(source),updates=100,batch=64,results=results,scope='CPU WM-only paired offline experiment, independent reset/collection for heldout, samples may repeat; no Actor training or physical acceptance')
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

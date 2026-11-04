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
out=root/'runs/b2_effective_prediction_diagnosis';out.mkdir(exist_ok=True)
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
for _ in range(240):t.collect()
with torch.no_grad():
    s0,A,S1,xi,mask=t.buf.sample_windows(4096,1)
    s=mod.flat_to_state(s0);real=mod.flat_to_state(S1[:,0])
    pred=t.wm.step(s,A[:,0],xi[:,0])
    err=(pred['qd']-real['qd']).square().mean(-1)
    actual=real['qd'].abs().amax(-1);expected=pred['qd'].abs().amax(-1)
    kg=xi[:,0,16]*15
    rows=[]
    for lo,hi in [(0,.01),(5,5.01),(10,10.01),(15,15.01)]:
        m=mask&(kg>=lo)&(kg<hi)
        if not m.any():continue
        fast=m&(actual>6)
        rows.append(dict(load_interval=[lo,hi],samples=int(m.sum()),joint_speed_rmse=float(err[m].mean().sqrt()),actual_peak=float(actual[m].max()),predicted_peak=float(expected[m].max()),actual_overspeed=int(fast.sum()),missed_overspeed=int((fast&(expected<=6)).sum()),predicted_overspeed=int((m&(expected>6)).sum())))
report=dict(checkpoint=str(source),iteration=ck['it'],rows=rows,scope='One-step prediction on newly collected fixed-load replay with other training randomization,4096 resamples with possible repeats; not independent full-cycle trials or acceptance')
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

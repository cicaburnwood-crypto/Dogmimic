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
out=root/f'runs/b2_substep_diagnosis_verified_seed{seed}';out.mkdir(exist_ok=True)
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
records=[];sub=[]
original=t.env._actuated_step
def traced(target):
    original(target)
    qd=t.env.sim.data.qvel[:,t.env.qd_adr][:,t.env.act_to_joint]
    sub.append(torch.stack([qd.abs().amax(-1),(t.env.sim.data.actuator_force*qd).abs().sum(-1)],-1).clone())
t.env._actuated_step=traced
original_step=t.env.step
def traced_step(action):
    sub.clear()
    state,info=original_step(action)
    peaks=torch.stack(sub).amax(0)
    endpoint=state['qd'].abs().amax(-1)
    records.append(torch.cat([peaks,endpoint[:,None]],-1).cpu())
    return state,info
t.env.step=traced_step
for _ in range(1000):t.collect()
a=torch.stack(records)
rows=[]
for i,kg in enumerate([0,5,10,15]):
    v=a[:,i];overs=v[:,0]>6
    rows.append(dict(kg=kg,transitions=len(v),substep_overspeed=int(overs.sum()),endpoint_missed_overspeed=int((overs&(v[:,2]<=6)).sum()),substep_peak_speed=float(v[:,0].max()),endpoint_peak_speed=float(v[:,2].max()),power_exceedances=int((v[:,1]>1501).sum()),substep_peak_power=float(v[:,1].max())))
report=dict(seed=seed,source=str(source),source_iteration=ck['it'],rows=rows,scope='1000 sequential training transitions per load; reset and randomized trajectories, not independent full-cycle acceptance. No policy updates. Power follows unchanged existing evaluation measurement convention.')
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

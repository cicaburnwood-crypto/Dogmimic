"""Read-only CPU gradient diagnosis of saved Actor/WM; no optimizer steps."""
import os,inspect,textwrap,json,copy
from pathlib import Path
import torch
import sys
root=Path(__file__).resolve().parents[2]
sys.path.extend(str(root/'experiments'/n) for n in ['three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills'])
import train_repair as mod
from supermimic.train import TrainCfg
root=Path(__file__).resolve().parents[2]
os.environ['B2_INIT_CHECKPOINT']=str(root/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt')
os.environ['B2_POLICY_SOURCE']=str(root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt')
os.environ.pop('B2_WM_SOURCE',None);os.environ['B2_WM_PREFIT']='0'
seed=int(os.environ.get('B2_AUDIT_SEED','101'))
torch.manual_seed(seed)
torch.set_num_threads(2)
out=root/'runs/b2_balance_physics_audit';out.mkdir(exist_ok=True)
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



e=mod.make_env(t.lib,'cpu',16,evaluate=True,kg=15.)
e.fixed_mass=1.;e.fixed_friction=1.;e.max_speed.zero_();e.max_power.zero_()
t.policy.load_state_dict(torch.load(root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt',map_location='cpu',weights_only=False)['policy'])
r,_=mod.evaluate(t.policy,e,seed=20261103)
report=dict(passed_implementation_check=r['completion_rate']>=.75,trials=16,success_rate=r['success_rate'],completion_rate=r['completion_rate'],peak_speed=float(e.max_speed.max()),peak_power=float(e.max_power.max()),source='own v3 actor',load_kg=15,pushes=True,scope='Candidate environment loaded-cycle diagnostic only; thresholds and physical evaluator unchanged. Not deployment acceptance.')
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

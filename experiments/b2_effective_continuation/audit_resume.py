import os,sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.extend(str(ROOT/'experiments'/n) for n in ['three_skills/engine','b2_payload_recovery','b2_payload','b2_cycle','three_skills'])
import train_continue as m
from supermimic.train import TrainCfg
os.environ['B2_INIT_CHECKPOINT']=str(ROOT/'runs/b2_payload_stage_recovery_v6_20261031_002818/best.pt')
os.environ['B2_POLICY_SOURCE']=str(ROOT/'runs/b2_actuator_robust_v3_20261101_194429/best.pt')
os.environ['B2_CONTINUE_CHECKPOINT']=str(ROOT/'runs/b2_effective_training_probe_20261101_205738/best.pt')
os.environ['B2_WM_PREFIT']='0';os.environ.pop('B2_WM_SOURCE',None)
m.torch.set_num_threads(2)
import argparse
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
out=ROOT/'runs'/('b2_effective_continue_'+a.device+'_smoke');out.mkdir(exist_ok=True)
cfg=TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz',out=str(out),num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,wm_batches=1,horizon=16,k=2,use_critic=False,amp=a.device!='cpu',compile_wm=a.device!='cpu')
t=m.Trainer(cfg,a.device);ck=m.torch.load(os.environ['B2_CONTINUE_CHECKPOINT'],map_location=a.device,weights_only=False)
assert all(m.torch.equal(v,t.policy.state_dict()[k]) for k,v in ck['policy'].items())
assert all(m.torch.equal(v,t.wm.state_dict()[k]) for k,v in ck['wm'].items())
anchor={k:v.clone() for k,v in t.frozen_anchor.state_dict().items()}
for _ in range(20):t.collect()
for i in range(3):
 t.it=i+1;w=t.update_wm();p=t.update_policy();t.updates+=1
 assert all(math.isfinite(v) for v in list(w.values())+list(p.values()))
assert all(m.torch.equal(v,t.frozen_anchor.state_dict()[k]) for k,v in anchor.items())
assert any(not m.torch.equal(v,t.policy.state_dict()[k]) for k,v in ck['policy'].items())
r=dict(passed=True,device=a.device,source_sha256=t.continue_hash,anchor_sha256=t.anchor_source_hash,actor_wm_exact_load=True,anchor_preserved=True,updates=3,horizon=16,scope='small-batch implementation smoke, not physical acceptance')
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

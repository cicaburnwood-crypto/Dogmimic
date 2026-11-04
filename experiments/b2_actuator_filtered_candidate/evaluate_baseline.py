import json,hashlib,time
from pathlib import Path
import torch
import train_guarded
from cycle_env import CycleLibrary
from stage_policy import from_checkpoint
from robust_env import make_env,SPEED,POWER
from evaluate_cycle import evaluate,CRITERIA
from supermimic import networks
root=Path(__file__).resolve().parents[2];torch.set_num_threads(2);networks.AMP['enabled']=True
source=root/'runs/b2_actuator_robust_v3_20261101_194429/best.pt'
# Snapshot bytes once so asynchronous source updates cannot change hash attribution.
raw=source.read_bytes();import io
ck=torch.load(io.BytesIO(raw),map_location='cuda',weights_only=False);policy=from_checkpoint(ck,'cuda').eval()
lib=CycleLibrary('experiments/b2_cycle/b2_full_cycle.npz','cuda');env=make_env(lib,'cuda',64,evaluate=True,kg=0.)
out=root/'runs/b2_filtered_initial_baseline';out.mkdir(exist_ok=True);rows=[];start=time.time()
for kg in [0,5,10,15]:
 env.fixed_loads.fill_(kg);env.fixed_mass=1.;env.fixed_friction=1.;env.max_speed.zero_();env.max_power.zero_()
 with torch.no_grad():r,_=evaluate(policy,env,seed=20261103)
 r.update(kg=kg,peak_speed=float(env.max_speed.max()),peak_mechanical_power=float(env.max_power.max()))
 r['actuator_pass']=r['peak_speed']<=SPEED and r['peak_mechanical_power']<=POWER+1;rows.append(r)
 report=dict(source=str(source),source_it=ck['it'],source_sha256=hashlib.sha256(raw).hexdigest(),complete=len(rows)==4,conditions=rows,elapsed_s=time.time()-start,criteria=CRITERIA,scope='Initial filtered dynamics baseline; not trained candidate or full robustness acceptance')
 (out/'report.json').write_text(json.dumps(report,indent=2));print(kg,round(64*r['success_rate']),r['peak_speed'],r['actuator_pass'],flush=True)

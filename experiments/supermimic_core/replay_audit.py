import json
import argparse
from common import setup
from state import verify_replay,EnvironmentSnapshot,warp_arrays
p=argparse.ArgumentParser();p.add_argument('--robot',default='G1');a=p.parse_args()
T,b,_=setup(a.robot)
c=b.TrainCfg(motions='data/processed/g1_standing_jump_060m.npz' if a.robot=='G1' else 'experiments/b2_cycle/b2_full_cycle.npz',out='runs/supermimic_core_v3_audit/replay',num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,compile_wm=False)
t=T(c,'cuda')
print(json.dumps(verify_replay(t.env,t.policy)),flush=True)
# Verify that the complete captured state is bit-exact before any recomputation.
s=EnvironmentSnapshot(t.env)
import torch,warp as wp
with torch.no_grad():
 for _ in range(3):t.env.step(t.policy(*t.env.policy_inputs()))
s.restore()
for group,obj in [('model',t.env.sim.wp_model),('data',t.env.sim.wp_data)]:
 diffs=[]
 for k,v in warp_arrays(obj):
  vv=wp.to_torch(v).cpu();orig=s.arrays[group][k]
  if not torch.equal(vv,orig):diffs.append(k)
 print(group,'inexact fields',diffs,flush=True)
print(json.dumps(verify_replay(t.env,t.policy)),flush=True)

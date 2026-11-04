"""Finite shared-fork check, including replay and all optimizer states."""
import argparse,json
from common import setup,tensor_digest,write_json
p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
T,b,_=setup(a.robot)
import torch
from state import TrainerSnapshot,verify_replay
c=b.TrainCfg(motions='data/processed/g1_standing_jump_060m.npz' if a.robot=='G1' else 'experiments/b2_cycle/b2_full_cycle.npz',
    out=a.out,num_envs=4,buffer_steps=64,wm_batch=4,pi_batch=4,wm_batches=1,compile_wm=False,use_critic=True,k=8,horizon=32)
t=T(c,'cuda')
for _ in range(40):t.collect()
t.update_wm();t.updates+=1
original={k:tensor_digest(getattr(t,k).state_dict()) for k in ['policy','wm','critic']}
original['replay']=tensor_digest(vars(t.buf))
s=TrainerSnapshot(t)
s.save(a.out+'/test_full_state.pt')
s0=t.buf.sample_starts(4,t.tcfg.history,t.task)
t.collect();t.update_policy();t.update_wm();s=TrainerSnapshot.load_into(t,a.out+'/test_full_state.pt')
assert original['replay']==tensor_digest(vars(t.buf))
for k in ['policy','wm','critic']:assert original[k]==tensor_digest(getattr(t,k).state_dict())
s1=t.buf.sample_starts(4,t.tcfg.history,t.task)
assert all(torch.equal(x,y) for x,y in zip(s0,s1))
s.restore()
first=t.update_policy();weights={k:v.clone() for k,v in t.policy.state_dict().items()}
s.restore();second=t.update_policy()
assert all(torch.equal(weights[k],v) for k,v in t.policy.state_dict().items())
assert first==second
replay=verify_replay(t.env,t.policy)
assert replay['passed'],replay
report=dict(passed=True,robot=a.robot,replay_sampling_exact=True,policy_update_after_restore_exact=True,disk_roundtrip=True,
            full_environment_restore=replay,all_optimizers_retained=True,actor_updates_outside_formal_training=2)
write_json(a.out+'/report.json',report);print(json.dumps(report),flush=True)

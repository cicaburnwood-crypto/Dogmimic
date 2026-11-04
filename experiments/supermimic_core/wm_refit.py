"""The single permitted bounded online-WM-only repair; Actor bytes stay fixed."""
import argparse,json,time
from pathlib import Path
from common import ROOT,DESIGN,setup,read_json,write_json,tensor_digest,sha256

p=argparse.ArgumentParser();p.add_argument('--robot',required=True,choices=['B2','G1']);p.add_argument('--out',required=True)
a=p.parse_args();Trainer,base,_=setup(a.robot)
import torch
torch.set_num_threads(2)
m=read_json(DESIGN/'experiment_matrix.json')['robots'][a.robot];b=m['batches'];hp=m['hyperparameters'];out=Path(a.out)
cfg=base.TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz' if a.robot=='B2' else 'data/processed/g1_standing_jump_060m.npz',
    out=str(out),seed=4203101,num_envs=b['envs_per_gpu'],buffer_steps=512,wm_batch=b['wm_batch'],pi_batch=b['wm_batch'],
    wm_batches=b['wm_batches'],k=8,horizon=32,lr=hp['base_lr'],action_noise=hp['action_noise_std'],use_critic=False,compile_wm=True,eval_every=0)
t=Trainer(cfg,'cuda');before=tensor_digest(t.policy.state_dict());anchor=tensor_digest(t.frozen_anchor.state_dict())
t.core_phase='D0_single_WM_refit';start=time.time()
for _ in range(128):t.collect()
loops=0
while t.wm_gradient_steps<2000:
    if (out/'STOP').exists() or (out.parent/'STOP').exists():break
    if time.time()-start>1200:break
    t.collect();loops+=1;t.update_credit+=.4
    while t.update_credit>=1 and t.wm_gradient_steps<2000:
        t.update_credit-=1;stats=t.update_wm();t.updates+=1
    if loops%100==0 and t.rank==0:
        row=dict(phase=t.core_phase,loops=loops,wm_gradient_steps=t.wm_gradient_steps,actor_updates=0,elapsed=time.time()-start)
        write_json(out/'progress.json',row);print(json.dumps(row),flush=True)
assert before==tensor_digest(t.policy.state_dict()) and anchor==tensor_digest(t.frozen_anchor.state_dict())
t.it=loops;t.train_time=time.time()-start
if t.rank==0:
    t.save(str(out/'wm_refit.pt'))
    write_json(out/'result.json',dict(robot=a.robot,actor_updates=0,wm_gradient_steps=t.wm_gradient_steps,
        source_sha256=t.source_hash,checkpoint_sha256=sha256(out/'wm_refit.pt'),policy_hash=before,
        elapsed=time.time()-start,loops=loops,physics_transitions=t.env_steps*t.world,
        independent_training_seed=4203101,diagnostic_holdout_not_consumed=True,one_attempt_only=True))
t._barrier()

"""Measure regression causes on preserved trajectories and validate the DR path."""
import json
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.networks import Policy
from supermimic.task import flat_to_state
from cycle_env import CycleLibrary
from evaluate_cycle import evaluate
from payload_env import make_payload_env

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
torch.set_num_threads(2);networks.AMP['enabled']=True
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
source=ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt'
oldrun=ROOT/'runs/b2_payload_stable_cycle_v3_20261030_232447'
test=ROOT/'runs/b2_stable_cycle_v2_20261030_214533/payload_test_20261030_225905'
ck=torch.load(source,map_location='cuda',weights_only=False)
arch=ck['policy_architecture']
def actor(state):
    p=Policy(210,195,enc_hidden=tuple(arch['encoder']),dec_hidden=tuple(arch['decoder'])).cuda()
    p.load_state_dict(state);return p.eval()
policy=actor(ck['policy'])
lib=CycleLibrary(str(ROOT/'experiments/b2_cycle/b2_full_cycle.npz'),'cuda')
env=make_payload_env(lib,'cuda',64,evaluate=True,fixed_loads=10.,pushes=True)
env.reset();task=env.task
summaries=[]
with torch.no_grad():
    for kg in (0,5,10,15):
        ar=np.load(test/f'{kg}kg_push/rollout.npz')
        idx=np.arange(0,len(ar['state']),5)
        current=flat_to_state(torch.tensor(ar['state'][idx].reshape(-1,37),device='cuda'))
        before=flat_to_state(torch.tensor(ar['state'][np.maximum(idx-1,0)].reshape(-1,37),device='cuda'))
        frames=torch.tensor(np.repeat(idx+1,64),device='cuda')
        ref=lib.frame(frames);kin=task.body_kin(current);rkin=task.body_kin(ref)
        xi=torch.zeros(len(frames),21,device='cuda');xi[:,0]=1;xi[:,16]=kg/15
        terms,metrics=task.stability_terms(before,current,ref,task.body_kin(before),kin,rkin,xi)
        hold=(ref['phase']==2)&(ref['hold_weight']>.99)
        feet=task.foot_positions(kin);rf=task.foot_positions(rkin)
        absolute=(feet[hold,2:,:2]-rf[hold,2:,:2]).norm(dim=-1)
        rates={k:float(v[hold].mean()) for k,v in terms.items() if k in ('hind_foot_anchor','hind_foot_speed','com_acceleration','hold_com_velocity')}
        shift=(feet[hold,2:,:2]-rf[hold,2:,:2]).mean(0)
        sm=dict(kg=kg,mean_absolute_hind_reference_error_m=float(absolute.mean()),mean_hind_reference_offset_xy=shift.cpu().tolist(),
                max_positive_reward_rate=task.cfg.max_step_reward/task.cfg.dt,hold_reward_rates=rates,
                measured_mean_hind_drift_m=float(np.mean([t['hind_foot_xy_drift_m'] for t in json.loads((test/f'{kg}kg_push/evaluation.json').read_text())['trials_detail']])))
        summaries.append(sm)
        print('REFERENCE_CONFLICT '+json.dumps(sm),flush=True)
    # A non-reset environment must retain its exact physical state through another
    # environment's mass/constant update.
    env.sim.data.qvel[:]=.02*torch.randn_like(env.sim.data.qvel)
    q=env.sim.data.qpos.clone();v=env.sim.data.qvel.clone()
    env.reset_envs(torch.tensor([0],device='cuda'),torch.tensor([0],device='cuda'))
    env._finish_resets(torch.tensor([0],device='cuda'),force=True)
    state_error=float(max((q[1:]-env.sim.data.qpos[1:]).abs().max(),(v[1:]-env.sim.data.qvel[1:]).abs().max()))
    assert state_error==0.,state_error
    # Evaluate the old best actor in the actual randomization implementation,
    # with fixed masses spanning the full training interval and its light pushes.
    n=128
    e=make_payload_env(lib,'cuda',n,evaluate=True,fixed_loads=torch.tensor([0.]*32+[5.]*32+[10.]*32+[15.]*32,device='cuda'),pushes=True)
    captured=[]
    class Capture:
        def eval(self):policy.eval()
        def train(self):policy.train()
        def __call__(self,obs,goal):
            if len(captured)<156:captured.append((obs.detach().clone(),goal.detach().clone()))
            return policy(obs,goal)
    r,arr=evaluate(Capture(),e,seed=20261112)
    dynamic=[]
    for j,kg in enumerate((0,5,10,15)):
        tr=r['trials_detail'][j*32:(j+1)*32]
        dynamic.append(dict(kg=kg,completed=sum(t['completed'] for t in tr),cycle=sum(t['cycle'] for t in tr),strict=sum(t['success'] for t in tr),trials=32))
    print('DYNAMIC_PHYSICS_BASELINE '+json.dumps(dynamic),flush=True)
    drift=[]
    obs=torch.cat([x[0] for x in captured[::5]]);goal=torch.cat([x[1] for x in captured[::5]])
    a0=policy(obs,goal)
    for it in (1000,2000):
        changed=torch.load(oldrun/f'ckpt_{it:07d}.pt',map_location='cuda',weights_only=False)
        p=actor(changed['policy']);a=p(obs,goal)
        with_old_norm={k:(ck['policy'][k] if k.startswith(('obs_norm.','goal_norm.')) else v) for k,v in changed['policy'].items()}
        p.load_state_dict(with_old_norm);fixed_norm=p(obs,goal)
        drift.append(dict(it=it,joint_command_rms_difference_rad=float(((a-a0)*.7).square().mean().sqrt()),
                          joint_command_max_difference_rad=float(((a-a0)*.7).abs().max()),
                          with_original_normalizers_rms_rad=float(((fixed_norm-a0)*.7).square().mean().sqrt())))
    print('ACTOR_DRIFT '+json.dumps(drift),flush=True)
result=dict(source_checkpoint=str(source),failed_run=str(oldrun),reference_conflict=summaries,
            other_env_state_preserved_through_mass_recompute=state_error==0.,dynamic_physics_baseline=dynamic,actor_drift=drift,
            interpretation='Large absolute reference-foothold penalty conflicts with an already stationary stance; unconstrained actor updates lose rise/return skill. Model-bias contribution not isolated.')
(HERE/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print('DIAGNOSIS_SAVED',flush=True)

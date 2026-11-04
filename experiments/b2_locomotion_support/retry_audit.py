"""Check command sampling, reward calibration and unchanged physical acceptance."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import torch
from task import CommandEnv,CommandCfg,support_probability,gait_penalties

HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'b2_locomotion_walk'
assert (HERE/'gait.py').read_bytes()==(OLD/'gait.py').read_bytes()
def functions(path,names):
    tree=ast.parse(path.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names}
assert functions(HERE/'evaluate.py',('cases','evaluate'))==functions(OLD/'evaluate.py',('cases','evaluate'))
torch.set_num_threads(2);torch.manual_seed(20261103)
env=CommandEnv('cuda',2048,evaluate=True,randomize=False,max_speed=2.,yaw_std=1.5)
env.reset();ids=torch.arange(env.num_envs,device='cuda');samples=[]
for _ in range(50):env._sample_targets(ids);samples.append(env.targets.clone())
cmd=torch.cat(samples)
assert cmd[:,:2].norm(dim=-1).max()<=2.000001 and cmd[:,1].abs().max()<=.600001
moving=cmd[:,:2].norm(dim=-1)>.05;fast=(cmd[:,:2].norm(dim=-1)>1.4)&moving
assert moving.float().mean()>.5 and fast.float().mean()>.15
assert cmd[:,2].abs().max()>4. and ((cmd[:,0]<0)&moving).any()

# Saved rollouts are used only to calibrate reward geometry against physics.
# No saved policy or trajectories are supplied to the new optimizer/replay.
run=HERE.parents[1]/'runs/b2_locomotion_walk_fresh_20261101_011622'
z=np.load(run/'best_rollout.npz');n=len(z['case_index'])
gap=z['foot_clearance_m'].reshape(-1,n,4);force=z['contact_forces_n'].reshape(-1,n,4)
vel=np.gradient(gap,.005,axis=0);idx=np.r_[700:1100,2100:2500]
g=torch.tensor(gap[idx]);v=torch.tensor(vel[idx]);truth=torch.tensor(force[idx]>5.)
old=torch.sigmoid((.006-g)/.003);new=support_probability(g,v,CommandCfg())
def stats(p,mask):
    t=truth[:,mask];q=p[:,mask];hit=q>.5
    return dict(precision=float(t[hit].float().mean()),recall=float(hit[t].float().mean()),
                predicted_count=float(q.sum(-1).mean()),actual_count=float(t.float().sum(-1).mean()),
                unloaded_mean_proxy=float(q[~t].mean()))
# Check both case subsets so an aggregate mean cannot hide directional errors.
# These are calibration diagnostics, not held-out policy-success evaluations.
groups={label:dict(old=stats(old,mask),new=stats(new,mask)) for label,mask in
        (('even_cases',z['case_index']%2==0),('odd_cases',z['case_index']%2==1))}
for group in groups.values():
    assert group['new']['precision']>group['old']['precision']+.10
    assert group['new']['recall']>.85
    assert group['new']['unloaded_mean_proxy']<.4*group['old']['unloaded_mean_proxy']
out=dict(acceptance_unchanged=True,gait_sha256=hashlib.sha256((HERE/'gait.py').read_bytes()).hexdigest(),
         fixed_cases_and_physical_evaluation_ast_unchanged=True,command_count=len(cmd),
         xy_max=float(cmd[:,:2].norm(dim=-1).max()),abs_y_max=float(cmd[:,1].abs().max()),
         angular_samples_max=float(cmd[:,2].abs().max()),sample_fraction_above_1p4m_s=float(fast.float().mean()),
         physical_support_calibration=groups,calibration_only_no_policy_or_replay_initialization=True)
(HERE/'retry_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('RETRY_AUDIT '+json.dumps(out),flush=True)

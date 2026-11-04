import json
from pathlib import Path
import torch
from supermimic import networks
from supermimic.networks import Policy
from cycle_env import CycleLibrary
from recovery_env import make_recovery_env
from stage_env import StageTask
from stage_policy import StagePolicy,install_phase_command

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
torch.set_num_threads(2);networks.AMP['enabled']=True
lib=CycleLibrary(str(ROOT/'experiments/b2_cycle/b2_full_cycle.npz'),'cuda')
env=make_recovery_env(lib,'cuda',8,evaluate=True,fixed_loads=10.,pushes=False,task_class=StageTask)
env.reset();ids=torch.arange(8,device='cuda')
ck=torch.load(ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt',map_location='cuda',weights_only=False)
p=StagePolicy(210,195).cuda();p.load_state_dict(ck['policy'])
old=Policy(210,195,enc_hidden=(512,256,128),dec_hidden=(512,256,128)).cuda();old.load_state_dict(ck['policy'])
checks=[]
with torch.no_grad():
    for frame,limit in ((0,0.),(200,.008),(650,.04)):
        env.reset_envs(ids,torch.full_like(ids,frame));env._finish_resets(ids,force=True)
        obs,goal=env.policy_inputs();assert goal.shape[-1]==196
        assert torch.equal(goal[:,-1],lib.phase[env.frame+1].float())
        p.load_state_dict(ck['policy'])
        initial=float((p(obs,goal)-old(obs,goal[:,:195])).abs().max());assert initial==0.
        p.hind_head.bias.fill_(100.);p.quad_head.bias.fill_(100.)
        delta=float(((p(obs,goal)-old(obs,goal[:,:195]))*.7).abs().max())
        assert abs(delta-limit)<1e-6
        checks.append(dict(frame=frame,initial_error=initial,max_delta_rad=delta))
    p.load_state_dict(ck['policy']);p.quad_head.bias.fill_(100.)
    for frame in (0,25,50,100,174,200,450):
        env.reset_envs(ids,torch.full_like(ids,frame));env._finish_resets(ids,force=True)
        obs,goal=env.policy_inputs()
        assert torch.equal(p(obs,goal),old(obs,goal[:,:195]))
    means=p.goal_norm.mean.clone();p.goal_norm.update(goal)
    assert torch.equal(means,p.goal_norm.mean)
p.load_state_dict(ck['policy'])
env.reset_envs(ids,torch.full_like(ids,650));env._finish_resets(ids,force=True)
p(*env.policy_inputs()).sum().backward()
assert p.quad_head.weight.grad.abs().max()>0 and p.hind_head.weight.grad.abs().max()<1e-6
assert all(x.grad is None for x in p.teacher.parameters())
# Evaluation installs the same phase command without changing physical metrics.
from cycle_env import make_cycle_env
evaluation=make_cycle_env(lib,'cuda',8,evaluate=True)
install_phase_command(evaluation.task)
s=env.state;future=env.ref_future(env.frame)
assert torch.equal(evaluation.task.goal(s,future),env.task.goal(s,future))
result=dict(passed=True,source_identity_and_limits=checks,initial_stance_rise_and_hindstand_isolated_from_return_head=True,
            training_evaluation_command_parity=True,phase_command_source='reference sequence, first future frame',
            frozen_teacher_and_normalizers=True,return_head_gradient_isolation=True)
(HERE/'audit_stage.json').write_text(json.dumps(result,indent=2)+'\n')
print('STAGE_AUDIT '+json.dumps(result),flush=True)

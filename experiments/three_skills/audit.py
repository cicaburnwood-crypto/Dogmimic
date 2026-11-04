"""Check actual model FK, simulator transitions, and fresh WM/policy gradients."""
import os,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.motion_lib import MotionLibrary
from supermimic.task import state_to_flat,flat_to_state,NJ,OBS_DIM,XI_DIM
from supermimic.networks import WorldModel,Policy,state_error,STATE_DELTA_DIM
from skill_env import make_env,IS_B2

torch.set_num_threads(2)
name=os.environ['SM_TASK'];here=Path(__file__).parent
lib=MotionLibrary(str(here/'targets'/f'{name}.npz'),'cuda')
env=make_env(lib,'cuda',32,evaluate=True);s=env.reset()
assert state_to_flat(s).shape==(32,13+2*NJ)
back=flat_to_state(state_to_flat(s))
assert all(torch.allclose(s[k],back[k],atol=1e-5) for k in back)
ids=torch.tensor([0,int(lib.lengths[0])//2,int(lib.lengths[0])-1],device='cuda')
ref=lib.frame(ids);kin=env.task.body_kin(ref);m=env.mj_model;d=mujoco.MjData(m)
maxerr=0.
for i in range(3):
    d.qpos[:3]=ref['pos'][i].cpu();d.qpos[3:7]=ref['quat'][i].cpu();d.qpos[7:]=ref['q'][i].cpu();mujoco.mj_forward(m,d)
    root=env.kin.body_names[0]
    bids=[env._find_body(n) for n in env.kin.body_names]
    maxerr=max(maxerr,float(np.max(abs(d.xpos[bids]-kin[0][i].cpu().numpy()))))
assert maxerr<1e-5,maxerr
actuator_joints=[m.joint(int(m.actuator_trnid[a,0])).name for a in range(m.nu)]
for a,j in enumerate(env.act_to_joint.tolist()):assert actuator_joints[a].split('/')[-1]==env.joint_names[j]
feat=env.task.state_features(s)
wm=WorldModel(env.task,feat.shape[-1],XI_DIM,hidden=(128,128)).cuda()
policy=Policy(OBS_DIM*env.tcfg.history,lib.q.shape[-1]*2*env.tcfg.future+15*env.tcfg.future,enc_hidden=(128,),dec_hidden=(128,)).cuda()
obs,goal=env.policy_inputs();a=policy(obs,goal);pred=wm.step(s,a,env.xi)
nxt,info=env.step(a.detach())
loss=state_error(pred,nxt).square().mean();loss.backward()
assert torch.isfinite(loss) and any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in wm.net.parameters())
assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in policy.parameters())
# Real contact dynamics under nominal rest action for two seconds.
ids=torch.arange(32,device='cuda');env.reset_envs(ids,torch.zeros_like(ids));env._finish_resets(ids,force=True)
terms=0
for _ in range(100):
    _,inf=env.step(torch.zeros(32,NJ,device='cuda'));terms+=int(inf['terminated'].sum())
result=dict(task=name,joints=NJ,state_dim=13+2*NJ,delta_dim=STATE_DELTA_DIM,obs_dim=OBS_DIM,goal_dim=goal.shape[-1],fk_max_abs_error_m=maxerr,finite_world_model_loss=float(loss),gradient_path_verified=True,invalid_physics=int(env.invalid_physics_count),zero_action_terminated_transitions=terms,actuator_joint_mapping_verified=True)
(here/f'audit_{name}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

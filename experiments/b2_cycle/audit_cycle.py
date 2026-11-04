import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from cycle_env import CycleLibrary,make_cycle_env

HERE=Path(__file__).parent;torch.set_num_threads(2)
lib=CycleLibrary(str(HERE/'b2_full_cycle.npz'),'cuda')
env=make_cycle_env(lib,'cuda',8,evaluate=True);env.reset();task=env.task
ids=torch.arange(8,device='cuda');idx=torch.full_like(ids,200)
ref=lib.frame(idx);kin=task.body_kin(ref);xi=env.xi.clone()
pos,vel=task.centroid(ref,kin)
assert lib.phase[0]==0 and lib.phase[-1]==4
assert set(lib.phase.cpu().tolist())=={0,1,2,3,4}
assert lib.num_frames*lib.dt<env.cfg.episode_length_s
assert torch.allclose(lib.q[0],lib.q[-1]) and torch.allclose(lib.root_pos[0],lib.root_pos[-1])
# Independent MuJoCo centroid and centroid-velocity calculation during rising.
r=lib.frame(torch.tensor([105],device='cuda'));P,V=task.centroid(r)
m=env.mj_model;d=mujoco.MjData(m)
d.qpos[:3]=r['pos'][0].cpu();d.qpos[3:7]=r['quat'][0].cpu();d.qpos[7:]=r['q'][0].cpu()
d.qvel[:3]=r['v'][0].cpu();d.qvel[3:6]=(r['rot'][0].T@r['w'][0]).cpu();d.qvel[6:]=r['qd'][0].cpu()
mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
perr=float(np.max(abs(P[0].cpu().numpy()-d.subtree_com[1])))
verr=float(np.max(abs(V[0].cpu().numpy()-d.subtree_linvel[1])))
assert perr<1e-5 and verr<1e-5,(perr,verr)
axis_costs=[]
for axis in range(3):
    for sign in (-1,1):
        s0={k:v.clone() for k,v in ref.items()};s1={k:v.clone() for k,v in ref.items()}
        v=s1['v'].clone();v[:,axis]+=sign*task.cfg.dt;v.requires_grad_(True);s1['v']=v
        terms,metrics=task.stability_terms(s0,s1,ref,task.body_kin(s0),task.body_kin(s1),kin,xi)
        cost=terms['com_acceleration']+terms['center_acceleration']
        assert (cost<0).all();cost.sum().backward()
        assert torch.isfinite(v.grad).all() and (v.grad[:,axis].abs()>0).all()
        assert torch.allclose(metrics['com_acceleration'][:,axis],torch.full((8,),float(sign),device='cuda'),atol=1e-4)
        axis_costs.append(float(cost.mean()))
assert max(axis_costs)-min(axis_costs)<1e-5,axis_costs
s0={k:v.clone() for k,v in ref.items()};s0['v']+=.2
terms,metrics=task.stability_terms(s0,s0,ref,task.body_kin(s0),task.body_kin(s0),kin,xi)
assert torch.max(metrics['com_acceleration'].abs())<1e-6
assert torch.max(terms['com_acceleration'].abs())<1e-6
# Reward parity between actual step and differentiable imagined-transition head.
env.reset_envs(ids,idx);env._finish_resets(ids,force=True)
s0={k:v.clone() for k,v in env.state.items()};prev=env.prev_action.clone();a=torch.zeros(8,12,device='cuda')
s1,info=env.step(a);ref1=lib.frame(idx+1)
expected=task.transition_analytics(s0,s1,ref1,task.body_kin(ref1),env.ref_future(idx+1),a,prev,xi)
assert torch.allclose(info['reward'],expected['r'],atol=1e-5)
assert torch.isfinite(info['reward']).all() and int(env.invalid_physics_count)==0
result=dict(centroid_position_max_error_m=perr,centroid_velocity_max_error_m_s=verr,
            all_six_axis_directions_penalized=True,acceleration_gradient_verified=True,isotropic_axis_costs=axis_costs,
            constant_velocity_zero_acceleration=True,real_imagined_reward_parity=True,finite_real_physics=True,
            complete_return_reference=True,frames=lib.num_frames,duration_s=(lib.num_frames-1)*lib.dt)
(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

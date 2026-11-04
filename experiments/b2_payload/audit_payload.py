"""Physics, centroid, reward-gradient, and changing-force replay checks."""
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from cycle_env import CycleLibrary
from evaluate_payloads import build_model,inertia_in_body
from payload_env import make_payload_env
from train_payload import TimedContextReplay,PayloadTrainer,PayloadWorldModel
from supermimic.rotations import yaw_mat

HERE=Path(__file__).resolve().parent
torch.set_num_threads(2)
(HERE/'models').mkdir(exist_ok=True)
models={}
for kg in (0,5,10,15):
    path=HERE/'models'/f'b2_payload_{kg}kg.xml'
    build_model(kg,path);models[kg]=mujoco.MjModel.from_xml_path(str(path))
lib=CycleLibrary(str(HERE.parent/'b2_cycle/b2_full_cycle.npz'),'cuda')
loads=[0,0,5,5,10,10,15,15]
env=make_payload_env(lib,'cuda',8,evaluate=True,fixed_loads=loads,pushes=False)
env.reset();ids=torch.arange(8,device='cuda');task=env.task
mass=env.sim.model.body_mass.sum(-1)
np.testing.assert_allclose(mass.cpu(),task.base_total_mass+np.array(loads),atol=2e-5,rtol=0)
np.testing.assert_allclose(env.sim.model.body_subtreemass[:,1].cpu(),mass.cpu(),atol=2e-5,rtol=0)
inertia_errors=[];position_errors=[];velocity_errors=[]
reference=lib.frame(torch.full_like(ids,105))
p,v=task.centroid(reference,payload_kg=env.loads)
for i,kg in enumerate(loads):
    m=models[kg];q=env.sim.model.body_iquat[i,1].cpu().numpy().astype(float)
    R=np.empty(9);mujoco.mju_quat2Mat(R,q/np.linalg.norm(q));R=R.reshape(3,3)
    I=R@np.diag(env.sim.model.body_inertia[i,1].cpu().numpy())@R.T
    inertia_errors.append(float(abs(I-inertia_in_body(m,1)).max()))
    d=mujoco.MjData(m)
    d.qpos[:3]=reference['pos'][i].cpu();d.qpos[3:7]=reference['quat'][i].cpu();d.qpos[7:]=reference['q'][i].cpu()
    d.qvel[:3]=reference['v'][i].cpu();d.qvel[3:6]=(reference['rot'][i].T@reference['w'][i]).cpu();d.qvel[6:]=reference['qd'][i].cpu()
    mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
    position_errors.append(float(abs(p[i].cpu().numpy()-d.subtree_com[1]).max()))
    velocity_errors.append(float(abs(v[i].cpu().numpy()-d.subtree_linvel[1]).max()))
assert max(inertia_errors)<2e-6,inertia_errors
assert max(position_errors)<2e-5 and max(velocity_errors)<2e-5
impulse_errors=[]
for axis in range(3):
    env.reset_envs(ids,torch.zeros_like(ids))
    env.sim.data.qpos[:,2]=3.;env.sim.data.qvel[:]=0.
    env._finish_resets(ids,force=True)
    env.sim.data.ctrl[:]=env.q0[env.act_to_joint]
    env.sim.data.xfrc_applied[:]=0.;env.sim.data.xfrc_applied[1::2,1,axis]=10.
    for _ in range(4):env.sim.step()
    _,vel=task.centroid(env.read_state(),payload_kg=env.loads)
    delta=(vel[1::2]-vel[::2]).cpu().numpy()
    expected=np.zeros((4,3));expected[:,axis]=.2/(task.base_total_mass+np.array([0,5,10,15]))
    impulse_errors.append(float(abs(delta-expected).max()))
assert max(impulse_errors)<2e-5,impulse_errors
env.sim.data.xfrc_applied[:]=0.
frame=torch.full_like(ids,200);env.reset_envs(ids,frame);env._finish_resets(ids,force=True)
s0={k:v.clone() for k,v in env.state.items()};xi=env.xi.clone();prev=env.prev_action.clone()
action=torch.zeros(8,12,device='cuda');s1,info=env.step(action)
ref=lib.frame(frame+1)
expected=task.transition_analytics(s0,s1,ref,task.body_kin(ref),env.ref_future(frame+1),action,prev,xi)
torch.testing.assert_close(info['reward'],expected['r'],atol=1e-5,rtol=1e-5)
assert torch.isfinite(info['reward']).all() and int(env.invalid_physics_count)==0
gradient_checks={}
for f,keys in ((200,('hind_foot_anchor','hind_foot_speed')),(700,('return_com_anchor','return_com_speed','return_feet_speed'))):
    ref=lib.frame(torch.full_like(ids,f));s={k:v.clone() for k,v in ref.items()}
    s['pos']=(s['pos']+torch.tensor([.025,.025,.025],device='cuda')).requires_grad_()
    s['v']=(s['v']+.05).requires_grad_()
    terms,_=task.stability_terms(ref,s,ref,task.body_kin(ref),task.body_kin(s),task.body_kin(ref),xi)
    cost=sum(terms[k] for k in keys).sum();assert cost<0
    cost.backward()
    assert torch.isfinite(s['pos'].grad).all() and torch.isfinite(s['v'].grad).all()
    assert (s['pos'].grad[:,:2].abs()>0).all() and (s['v'].grad.abs()>0).all()
    gradient_checks[str(f)]=dict(position_gradient_norm=float(s['pos'].grad.norm()),velocity_gradient_norm=float(s['v'].grad.norm()))
axis_costs=[]
ref=lib.frame(torch.full_like(ids,200))
for axis in range(3):
    for sign in (-1,1):
        s={k:v.clone() for k,v in ref.items()};s['v'][:,axis]+=sign*.02
        terms,metrics=task.stability_terms(ref,s,ref,task.body_kin(ref),task.body_kin(s),task.body_kin(ref),xi)
        np.testing.assert_allclose(metrics['com_acceleration'][:,axis].cpu(),sign,atol=1e-4)
        axis_costs.append(float((terms['com_acceleration']+terms['center_acceleration']).mean()))
assert max(axis_costs)-min(axis_costs)<1e-5
random_env=make_payload_env(lib,'cuda',64,evaluate=True,pushes=True)
random_env.reset();old=random_env.loads.clone();random_env.reset()
assert (random_env.loads>=5).all() and (random_env.loads<=15).all() and (old!=random_env.loads).all()
assert ((random_env.push_end-random_env.push_start)>=6).all() and ((random_env.push_end-random_env.push_start)<=12).all()
norm=random_env.push_vectors.norm(dim=-1)
assert (norm>=7.999).all() and (norm<=16.001).all()
random_env.frame[:]=random_env.push_start[:,0]
fx,remaining=random_env._force_context()
np.testing.assert_allclose(random_env.xi[:,17:20].cpu(),(fx/16).cpu(),atol=1e-7)
xt=PayloadTrainer.context_at(random_env.xi,12,.02)
assert xt[:,17:20].abs().max()==0 and xt[:,20].max()<1e-6
buf=TimedContextReplay(1,16,21,'cuda')
for j in range(8):
    z=torch.zeros(1,37,device='cuda');a=torch.full((1,12),float(j),device='cuda');context=torch.zeros(1,21,device='cuda');context[:,17]=j
    buf.add(z,a,z,context,torch.tensor([j],device='cuda'),torch.tensor([1],device='cuda'))
_,actions,_,contexts,valid=buf.sample_windows(8,4)
assert valid.all() and torch.equal(actions[:,:,0],contexts[:,:,17])
state=dict(rot=yaw_mat(torch.tensor([np.pi/2],device='cuda')))
ctx=torch.zeros(1,21,device='cuda');ctx[:,18]=1.
canon=PayloadWorldModel.canonical_context(state,ctx)
torch.testing.assert_close(canon[:,17:20],torch.tensor([[1.,0.,0.]],device='cuda'),atol=1e-6,rtol=0)
result=dict(passed=True,mass_and_derived_constants_verified=True,centroid_position_max_error_m=max(position_errors),
            centroid_velocity_max_error_m_s=max(velocity_errors),inertia_max_error_kg_m2=max(inertia_errors),
            impulse_max_error_m_s=max(impulse_errors),all_xyz_acceleration_penalties_verified=True,
            real_imagined_reward_parity=True,foot_and_return_gradients=gradient_checks,
            loads_resampled_each_reset=True,load_min=float(random_env.loads.min()),load_max=float(random_env.loads.max()),
            force_min_n=float(norm.min()),force_max_n=float(norm.max()),timed_force_replay_verified=True,
            transient_imagined_push_expiry_verified=True,heading_frame_force_context_verified=True)
(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PAYLOAD_AUDIT '+json.dumps(result),flush=True)

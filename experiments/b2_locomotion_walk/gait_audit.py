"""Regression checks for airborne costs, contact-force measurement and gait gating."""
import json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from task import CommandEnv,CommandCfg,gait_penalties
from gait import ContactRecorder,summarize_gait,GAIT_CRITERIA

torch.set_num_threads(2)
gaps=torch.tensor([[0.,0.,0.,0.],[.08,0.,0.,0.],[.08,0.,0.,.08],
                   [.08,.08,0.,0.],[.08,.08,.08,.08],[.16,.16,.16,.16]],requires_grad=True)
costs=gait_penalties(gaps,torch.zeros(6),torch.ones(6),CommandCfg())
penalty=-sum(costs.values())
assert penalty[0]<penalty[1]<penalty[2]<penalty[3]<penalty[4]<penalty[5],penalty
penalty.sum().backward()
assert torch.isfinite(gaps.grad).all() and (gaps.grad[5]>0).all()
compiled=torch.compile(gait_penalties,dynamic=False)
cg=gaps.detach().requires_grad_();ct=compiled(cg,torch.zeros(6),torch.ones(6),CommandCfg())
(-sum(ct.values())).sum().backward()
assert torch.allclose(-sum(ct.values()),penalty,atol=1e-5)
assert torch.allclose(cg.grad,gaps.grad,atol=1e-4,rtol=1e-5)

env=CommandEnv('cuda',16,evaluate=True,randomize=False);env.reset();env.contact_recorder=ContactRecorder(env)
with torch.no_grad():
    for _ in range(150):env.step(torch.zeros(16,12,device='cuda'))
force,gap,vz,nonfoot=env.contact_recorder.sample()
cpu=mujoco.MjData(env.mj_model)
cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy();cpu.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
cpu.ctrl[:]=env.sim.data.ctrl[0].cpu().numpy();mujoco.mj_forward(env.mj_model,cpu)
foot_ids=env.contact_recorder.feet;cpu_force=np.zeros(4);cpu_nonfoot=0.
for i in range(cpu.ncon):
    wrench=np.zeros(6);mujoco.mj_contactForce(env.mj_model,cpu,i,wrench)
    contact=cpu.contact[i]
    for j,foot in enumerate(foot_ids):
        if set(contact.geom)=={foot,env.contact_recorder.floor}:cpu_force[j]+=max(0.,wrench[0])
    if env.contact_recorder.floor in contact.geom and not any(f in contact.geom for f in foot_ids):cpu_nonfoot+=max(0.,wrench[0])
gpu_force=force[0].cpu().numpy();weight=env.mj_model.body_mass.sum()*9.81
assert np.all(cpu_force>20) and np.all(gpu_force>20),(cpu_force,gpu_force)
assert np.allclose(gpu_force,cpu_force,rtol=.03,atol=2.),(gpu_force,cpu_force)
assert math.isclose(float(nonfoot[0]),cpu_nonfoot,rel_tol=.03,abs_tol=2.)
assert abs(float((force.sum(-1)+nonfoot).mean())-weight)/weight<.02

# A genuinely elevated robot has zero load-bearing foot contact.
ids=torch.arange(16,device='cuda');s=env.read_state();pos=s['pos'].clone();pos[:,2]+=.15
quat=env.sim.data.qpos[:,3:7].clone();zero=torch.zeros(16,3,device='cuda')
env.write_state(ids,pos,quat,s['q'],zero,zero,torch.zeros_like(s['q']))
env.sim.forward();air_force,air_gap,_,air_other=env.contact_recorder.sample()
assert (air_force==0).all() and (air_gap>.1).all() and (air_other==0).all()

# One stable sequence passes; one 100 ms four-foot flight must fail.
f=np.ones((100,4,2,4),dtype=np.float32)*200;g=np.zeros_like(f);v=np.zeros((100,4,2),dtype=np.float32)
f[50:55,:,1]=0;g[50:55,:,1]=.05
metrics,passed=summarize_gait(f,g,v,np.arange(25,100),np.zeros_like(v),np.array([False,True]))
assert passed.tolist()==[True,False]
assert math.isclose(float(metrics['longest_airborne_s'][1]),.1)
_,dragged=summarize_gait(np.ones_like(f)*200,np.zeros_like(g),v,np.arange(25,100),np.zeros_like(v),np.array([False,True]))
assert dragged.tolist()==[True,False]
alternating=np.ones_like(f)*200
for t in range(100):
    if t%20<10:alternating[t,:,:, [0,3]]=0
    else:alternating[t,:,:, [1,2]]=0
_,walking=summarize_gait(alternating,np.zeros_like(g),v,np.arange(25,100),np.zeros_like(v),np.array([True,True]))
assert walking.all()
out=dict(penalty_order=['stand','one_swing','diagonal_swing','front_pair_swing','all_air_8cm','all_air_16cm'],
         penalties=penalty.detach().tolist(),airborne_downward_gradient=gaps.grad[5].tolist(),
         compiled_reward_and_gradient_match=True,static_cpu_foot_force_n=cpu_force.tolist(),
         static_gpu_foot_force_n=gpu_force.tolist(),expected_weight_n=weight,
         static_cpu_nonfoot_force_n=cpu_nonfoot,static_gpu_nonfoot_force_n=float(nonfoot[0]),
         lifted_pose_zero_contact_verified=True,synthetic_100ms_flight_rejected=True,
         continuous_four_foot_drag_rejected=True,synthetic_diagonal_support_accepted=True,criteria=GAIT_CRITERIA)
Path(__file__).with_name('gait_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('GAIT_AUDIT '+json.dumps(out),flush=True)

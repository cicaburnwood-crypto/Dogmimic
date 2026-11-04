"""Validate B2 solver observables, replay context and SuperMimic gradients."""
import json,math,os
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.motion_lib import MotionLibrary
def forbidden(*a,**k):raise AssertionError('This task uses no motion library')
MotionLibrary.__init__=forbidden
from task import CommandEnv,XI_DIM
from contract import HISTORY
from gait_clock import GOAL_DIM,policy_goal,advance_phase
from policy import make_policy
from replay import CommandReplay
from physics_head import PhysicsHead
from reward_physics import geometry_slots
from supermimic.networks import WorldModel
from supermimic.task import flat_to_state,state_to_flat
from gait import ContactRecorder

def main():
    torch.set_num_threads(2);torch.manual_seed(20261104)
    env=CommandEnv('cuda',16,evaluate=True,randomize=False);s=env.reset()
    assert env.lib is None and all(torch.count_nonzero(s[k])==0 for k in ('v','w','qd'))
    assert torch.equal(s['q'],env.q0.expand(16,-1))
    obs,goal=env.policy_inputs();assert obs.shape==(16,420) and goal.shape==(16,GOAL_DIM)
    assert torch.all(goal[:,3:]==1.)
    env.contact_recorder=ContactRecorder(env)
    with torch.no_grad():
        for _ in range(150):env.step(torch.zeros(16,12,device='cuda'))
    aux=env.last_aux[0].cpu().numpy();cpu=mujoco.MjData(env.mj_model)
    cpu.qpos[:]=env.sim.data.qpos[0].cpu().numpy();cpu.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
    cpu.ctrl[:]=env.sim.data.ctrl[0].cpu().numpy();mujoco.mj_forward(env.mj_model,cpu)
    slots=geometry_slots(env.mj_model);cpu_forces=np.zeros((12,3))
    for i in range(cpu.ncon):
        wrench=np.zeros(6);mujoco.mj_contactForce(env.mj_model,cpu,i,wrench)
        contact=cpu.contact[i];world_force=contact.frame.reshape(3,3).T@wrench[:3]
        for side,g in enumerate(contact.geom):
            if slots[g]>=0:cpu_forces[slots[g]]+=(2*side-1)*world_force
    gpu_forces=env.reward_recorder.total[0].cpu().numpy()
    np.testing.assert_allclose(gpu_forces,cpu_forces,rtol=.03,atol=2.)
    assert np.all(gpu_forces[:4,2]>20.)
    torque=np.zeros(12);torque[env.act_to_joint.cpu().numpy()]=cpu.actuator_force
    np.testing.assert_allclose(aux[:12],torque,rtol=.03,atol=2.)
    assert np.array_equal(aux[24:],(np.linalg.norm(cpu_forces[4:],axis=-1)>.1).astype(float))
    weight=env.mj_model.body_mass.sum()*9.81
    print('STATIC_FORCE_DIAG',gpu_forces.tolist(),cpu_forces.tolist(),weight,env.state['pos'][0].tolist(),env.state['v'][0].tolist(),env.mj_model.opt.gravity.tolist(),flush=True)
    # Zero-action PD settles with two calf contacts; it is NOT a passing gait.
    # Include measured non-foot ground reactions in Newton's force balance.
    assert abs(gpu_forces[:,2].sum()-weight)/weight<.02,(gpu_forces[:,2],weight)
    assert aux[24:].sum()==2.,'Calf collisions must not be lost or mistaken for feet'
    force,_,_,_=env.contact_recorder.sample()
    np.testing.assert_allclose(gpu_forces[:4,2],force[0].cpu().numpy(),rtol=.001,atol=.1)
    ids=torch.arange(16,device='cuda');s=env.read_state();pos=s['pos'].clone();pos[:,2]+=.2
    quat=env.sim.data.qpos[:,3:7].clone();zero=torch.zeros(16,3,device='cuda')
    env.write_state(ids,pos,quat,s['q'],zero,zero,torch.zeros_like(s['q']));env.sim.forward()
    airborne=env.reward_recorder.sample(env.read_state());assert (airborne[:,12:]==0).all()
    env.reset();buf=CommandReplay(16,64,XI_DIM,'cuda')
    with torch.no_grad():
      for t in range(64):
        env.set_command([.3,.1,.2]);a=.04*torch.randn(16,12,device='cuda')
        flat=state_to_flat(env.state);xi=env.xi.clone();ep=env.ep_id.clone();cmd=env.commands.clone()
        ctx=torch.cat([env.phase[:,None],env.reward_prev_contacts.float(),env.lastlast_action],-1).clone()
        nxt,info=env.step(a);buf.add(flat,a,state_to_flat(nxt),xi,cmd,ctx,env.last_aux,ep)
    s0,A,S1,X,Y,valid=buf.sample_context_windows(16,8)
    assert torch.isfinite(Y).all() and valid.any()
    torch.testing.assert_close(X[valid,1:,20:],A[valid,:-1],atol=1e-6,rtol=0.)
    flat,xi,cmd,hist,prev,ctx=buf.sample_starts(16,HISTORY,env.task)
    torch.testing.assert_close(prev,xi[:,20:])
    phase=ctx[:,0];contacts=ctx[:,1:5].bool();lastlast=ctx[:,5:];s0=flat_to_state(flat)
    policy=make_policy().cuda();feats=env.task.features(s0)
    wm=WorldModel(env.task,feats.shape[-1],XI_DIM,hidden=(64,64)).cuda()
    physics=PhysicsHead(feats.shape[-1]+12+XI_DIM).cuda()
    a=policy(hist.flatten(1),policy_goal(cmd,phase));pred=wm.step(s0,a,xi)
    model_input=wm.inputs(s0,a,xi);pred_aux=physics(model_input)
    # Sanity check differentiability for the actual public reward path.
    result=env.task.command_fast(pred,s0,cmd,a,prev,lastlast,xi[:,1:13],advance_phase(phase,cmd),pred_aux,contacts)
    loss=-result['r'].mean();loss.backward()
    gradient=sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None)
    assert math.isfinite(gradient) and gradient>0,gradient
    env.task.compile_sizes.add(16)
    with torch.no_grad():compiled=env.task.command_fast(pred,s0,cmd,a,prev,lastlast,xi[:,1:13],advance_phase(phase,cmd),pred_aux,contacts)
    torch.testing.assert_close(compiled['r'],result['r'],rtol=3e-4,atol=1e-7)
    randomized=CommandEnv('cuda',32,evaluate=True,randomize=True);randomized.reset()
    assert randomized.mass_scale.min()<1 and randomized.mass_scale.max()>1 and randomized.delay.max()==2
    result=dict(passed=True,no_motion_library=True,standard_pose_zero_velocity=True,actor_observation_dim=420,goal_dim=GOAL_DIM,
                actor_inputs='joint q/dq, IMU gyro/gravity, action history, XYZ command, internally generated gait clock',
                no_ground_truth_velocity_or_contact_actor_input=True,static_cpu_force_xyz=cpu_forces.tolist(),
                static_gpu_force_xyz=gpu_forces.tolist(),torque_policy_order_verified=True,
                expected_weight_n=weight,zero_action_calf_collisions=int(aux[24:].sum()),zero_action_is_not_a_passing_policy=True,airborne_zero_force=True,acceptance_force_consistency=True,
                replay_action_context_pass=True,finite_auxiliary_targets=True,policy_gradient_through_world_model=gradient,
                compiled_public_reward_matches=True,randomization_preserved=True,
                b2_morphology=dict(height=env.task.height,width=env.task.width,length=env.task.length,foot_radius=.032))
    Path(__file__).with_name('physics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

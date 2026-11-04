"""Export the pinned, user-selected policy and check numpy inference parity.

Run in the training checkout's CPU Python; never creates a DDS publisher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--runtime-dir',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    root=Path(a.root);task=root/'experiments/b2_tilt_locomotion_repair'
    sys.path.insert(0,str(task));sys.path.insert(0,a.runtime_dir)
    import contract as c
    from policy import make_policy
    from supermimic import networks
    from tilt_runtime import TiltLocomotionController,advance_command,policy_goal,constrain_request
    expected='96dbb589f9fafb39e7d25dea3e4566d9a13e32788ef39ab685489d1617b9ecb4'
    assert digest(a.checkpoint)==expected,'Wrong source checkpoint'
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['it']==12000
    networks.AMP['enabled']=False
    actor=make_policy();actor.load_state_dict(ck['policy']);actor.eval()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in ck['policy'].items()})
    xml=ET.parse(task/'b2_tilt_locomotion.xml')
    joints={e.attrib['name']:e for e in xml.iter('joint') if 'name' in e.attrib}
    limits=[[float(x) for x in joints[name].attrib['range'].split()] for name in c.POLICY_JOINTS]
    meta=dict(skill='tilt_locomotion',label='B2 tilted locomotion v2 · best 12000',iteration=12000,
        checkpoint_iteration=12000,checkpoint_sha256=expected,weights_sha256=digest(output/'weights.npz'),
        backend='supermimic-vibe',obs_dim=420,goal_dim=16,history=10,dt=.02,
        q0=c.Q0,action_scale=c.ACTION_SCALE,policy_to_sdk=c.POLICY_TO_SDK,joint_limits=limits,
        kp=c.KP,kd=c.KD,torque_limits=[200.,200.,300.]*4,
        physical_eval_passed=False,hardware_validated=False,simulation_strict_passed=914,simulation_strict_total=1312,
        source_checkpoint=str(a.checkpoint),observation_order='q-q0, qd, body gyro, projected gravity, previous action; oldest-first 10 frames',
        command_order=['vx','vy','yaw_rate','roll','pitch'],tilt_limit_degrees=20.,dpad_rate_degrees_s=5.,
        goal_normalization_clip=None,obs_normalization_clip=10.,source_task_hash=digest(task/'task.py'),source_contract_hash=digest(task/'contract.py'))
    (output/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
    runtime=TiltLocomotionController(output)
    rng=np.random.default_rng(20261104)
    obs=rng.normal(0,.4,(512,420)).astype(np.float32)
    contexts=rng.normal(0,.2,(512,13)).astype(np.float32)
    contexts[:,12]=rng.uniform(0,1,512)
    goals=c.policy_goal(torch.from_numpy(contexts))
    with torch.no_grad():reference=actor(torch.from_numpy(obs),goals).numpy()
    error=float(np.max(np.abs(reference-runtime.actor(obs,goals.numpy()))));assert error<3e-5,error
    state=np.zeros(13,np.float32);state[12]=.25;state_t=torch.from_numpy(state.copy())
    max_context=max_goal=0.
    for step in range(2000):
        if step%50==0:
            request=np.array([rng.uniform(-4,4),rng.uniform(-1,1),rng.uniform(-2,2),rng.uniform(-c.MAX_ANGLE,c.MAX_ANGLE),rng.uniform(-c.MAX_ANGLE,c.MAX_ANGLE)],np.float32)
            state[7:12]=constrain_request(request);state_t[7:12]=c.constrain_targets(torch.from_numpy(request))
        max_goal=max(max_goal,float(np.max(np.abs(policy_goal(state)-c.policy_goal(state_t).numpy()))))
        state=advance_command(state);state_t=c.advance_command(state_t)
        max_context=max(max_context,float(np.max(np.abs(state-state_t.numpy()))))
    assert max_context<2e-5 and max_goal<1e-4,(max_context,max_goal)
    # Full history/action/joint-permutation test, changing all inputs each step.
    history=None;previous=np.zeros(12,np.float32);ctx=torch.zeros(13);ctx[12]=.25
    max_target=0.
    from runtime import rotation
    for step in range(300):
        q=np.array(c.Q0,np.float32)+rng.normal(0,.05,12).astype(np.float32)
        dq=rng.normal(0,.2,12).astype(np.float32);gyro=rng.normal(0,.03,3).astype(np.float32)
        quat=np.array([1.,.02*np.sin(step/50),.02*np.cos(step/50),0.],np.float32);quat/=np.linalg.norm(quat)
        request=constrain_request([1.,.2,.2,.2*np.sin(step/60),.2*np.cos(step/60)])
        o=np.concatenate((q-np.array(c.Q0,np.float32),dq,gyro,-rotation(quat)[2],previous))
        history=np.tile(o,(10,1)) if history is None else np.concatenate((history[1:],o[None]))
        ctx[7:12]=torch.from_numpy(request);goal=c.policy_goal(ctx)
        with torch.no_grad():act=actor(torch.from_numpy(history.reshape(1,-1)),goal[None])[0].numpy()
        target=np.clip(np.array(c.Q0)+np.array(c.ACTION_SCALE)*act,np.array(limits)[:,0]+.02,np.array(limits)[:,1]-.02)
        qs=np.empty(12,np.float32);ds=qs.copy();qs[c.POLICY_TO_SDK]=q;ds[c.POLICY_TO_SDK]=dq
        got,debug=runtime.step(qs,ds,gyro,quat,request)
        max_target=max(max_target,float(np.max(np.abs(got[c.POLICY_TO_SDK]-target))))
        previous=act;ctx=c.advance_command(ctx)
    assert max_target<2e-5,max_target
    result=dict(source_sha256=expected,iteration=ck['it'],samples=512,actor_max_abs_error=error,
        command_steps=2000,context_max_abs_error=max_context,goal_max_abs_error=max_goal,
        full_runtime_steps=300,target_max_abs_error=max_target,physical_acceptance=False,
        runtime_sha256=digest(Path(a.runtime_dir)/'tilt_runtime.py'),weights_sha256=meta['weights_sha256'])
    (output/'export_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

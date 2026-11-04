"""CPU-only export of our checkpoints; original Torch implementation is the oracle."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['SM_ROBOT'] = 'b2'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from runtime import Actor, Controller, rotation, clock_goal, slew, constrain, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--skill', choices=['hindstand','locomotion'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--evaluation')
    parser.add_argument('--training-experiment')
    parser.add_argument('--label')
    parser.add_argument('--actor-iteration',type=int)
    args = parser.parse_args()
    project, out = Path(args.project), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    hind = args.skill == 'hindstand'
    experiment = args.training_experiment or ('b2_payload_recovery' if hind else 'b2_locomotion_wtw_repair')
    for path in reversed([project/'experiments'/experiment,
                          project/'experiments/b2_payload_recovery',
                          project/'experiments/b2_payload',
                          project/'experiments/b2_cycle',
                          project/'experiments/three_skills',
                          project/'experiments/three_skills/engine']):
        sys.path.insert(0,str(path))
    from supermimic import networks
    networks.AMP['enabled'] = False
    run = project/'runs'/('b2_payload_stage_recovery_v6_20261031_002818' if hind
                         else 'b2_locomotion_wtw_repair_20261101_110346')
    checkpoint = run/'best.pt'
    expected = ('14c5f8ba3eac277e8aeb865c5b5deee7e93d669162de49c5e719892e84a78208' if hind
                else '1bd416935f6e33606396c5f934ddcb3a02332121ea933a26f0702ec8d542dfb6')
    if args.checkpoint:
        assert args.checkpoint_sha256 and args.evaluation, 'Explicit checkpoint needs its SHA256 and evaluation'
        checkpoint=Path(args.checkpoint);expected=args.checkpoint_sha256
    assert sha256(checkpoint) == expected, 'Selected checkpoint changed'
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    evaluation = json.loads(Path(args.evaluation or run/'best_eval.json').read_text())
    if args.checkpoint:
        assert evaluation['checkpoint_sha256']==expected, 'Evaluation belongs to another checkpoint'
    if hind:
        from stage_policy import from_checkpoint
        policy = from_checkpoint(ck, 'cpu').eval()
    else:
        from policy import make_policy
        policy = make_policy().cpu().eval()
        policy.load_state_dict(ck['policy'])
    arrays = {k:v.detach().cpu().numpy() for k,v in ck['policy'].items()}
    np.savez(out/'weights.npz', **arrays)
    limits = [[-.87,.87],[-.94,4.69],[-2.82,-.43]]*4
    meta = dict(skill=args.skill,task=ck['task_kind'],iteration=ck['it'],
                checkpoint=str(checkpoint),checkpoint_sha256=expected,
                weights_sha256=sha256(out/'weights.npz'),
                obs_dim=210 if hind else 420,goal_dim=196 if hind else 7,
                history=5 if hind else 10,policy_dt=.02,obs_per_frame=42,
                q0=([0.,.7,-1.4] if hind else [0.,.67,-1.3])*4,
                action_scale=([.7]*12 if hind else [.45,.8,.8]*4),
                policy_to_sdk=[3,4,5,0,1,2,9,10,11,6,7,8],
                policy_joint_names=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                                    for joint in ('hip','thigh','calf')],
                joint_limits=limits,torque_limits=[200.,200.,300.]*4,kp=250.,kd=8.,
                motor_mode=10,official_policy=False,hardware_validated=False,
                physical_eval_passed=False,actuation_enabled=False,
                source_checkpoint=ck.get('source_checkpoint'),
                source_sha256=ck.get('source_sha256'),
                notes=['Shadow deployment only. No motor-command publisher.',
                       'Simulation selection is not hardware validation.'],
                exported_unix=time.time())
    meta.update(label=args.label or f'{args.skill} {ck["it"]}',
                checkpoint_iteration=ck['it'],actor_iteration=args.actor_iteration,
                actor_updates=ck.get('actor_updates'),physical_eval_passed=False)
    if hind:
        source = project/'experiments/b2_cycle/b2_full_cycle.npz'
        with np.load(source,allow_pickle=False) as r:
            qpos,qvel=r['qpos'],r['qvel']
            reference=dict(pos=qpos[:,:3],rot=np.stack([rotation(q) for q in qpos[:,3:7]]),
                           quat=qpos[:,3:7],q=qpos[:,7:],v=qvel[:,:3],w=qvel[:,3:6],
                           qd=qvel[:,6:],phase=r['phase'])
        np.savez(out/'reference.npz',**reference)
        meta.update(reference_sha256=sha256(out/'reference.npz'),
                    original_reference_sha256=sha256(source),
                    reference_frames=len(qpos),reference_seconds=(len(qpos)-1)*.02,
                    reference_source='Our procedural full stand-rise-hold-descend-recover motion',
                    root_position_required=True,root_position_under_low_level_control_verified=False,
                    target_clipping_added_for_deployment=True)
        meta['notes'].append('Training did not clip hindstand targets; deployment clips to joint limits with a 0.02 rad margin.')
        if experiment=='b2_light_payload':
            meta.update(iteration=args.actor_iteration or ck['it'],payload_kg=[0,5],
                torque_limits=[160.,160.,240.]*4,
                torque_envelope=dict(motoring_full_speed_rad_s=6.,motoring_zero_speed_rad_s=10.,command_power_w=1200.),
                target_clipping_added_for_deployment=False,
                notes=['Selected C1 candidate; iteration excludes 500 WM-only prefit loops.',
                       'Training target clipping and torque envelope preserved; no target slew.',
                       'Actuator budgets are experimental, not manufacturer ratings.',
                       'Simulation task success does not mean physical acceptance or hardware validation.'])
    else:
        meta.update(command=dict(max_norm_xy=2.,max_abs_y=.6,yaw_cap=None),
                    linear_acceleration=1.5,angular_acceleration=3.,
                    gait_frequency_hz=2.,gait_duty=.65,swing_height_m=.06,
                    root_position_required=False,prior_iterations=ck.get('prior_iterations'),
                    simulation_strict_passed=evaluation['core_passed'],
                    simulation_strict_total=evaluation['core_total'])
    (out/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'simulation_best_eval.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    actor = Actor(out)
    rng = np.random.default_rng(20261101)
    prefix = 'teacher.' if hind else ''
    obs = (arrays[prefix+'obs_norm.mean']+rng.normal(size=(256,meta['obs_dim']))*
           np.sqrt(arrays[prefix+'obs_norm.var']+.0001)*1.5).astype(np.float32)
    goal = (arrays[prefix+'goal_norm.mean']+rng.normal(size=(256,meta['goal_dim']-int(hind)))*
            np.sqrt(arrays[prefix+'goal_norm.var']+.0001)*1.5).astype(np.float32)
    if hind:
        goal=np.concatenate([goal,rng.integers(0,5,(256,1)).astype(np.float32)],axis=1)
        # Exercise both learned correction heads, not only their near-zero gates.
        goal[:64,21:33]=0
        goal[:32,10]=-.35;goal[:32,13]=-.35
        goal[32:64,10]=.7;goal[32:64,13]=.7;goal[32:64,195]=4
    with torch.no_grad(): expected_action=policy(torch.from_numpy(obs),torch.from_numpy(goal)).numpy()
    actual=actor(obs,goal)
    batch_error=float(np.max(np.abs(expected_action-actual)))
    single_error=max(float(np.max(np.abs(actor(o[None],g[None])[0]-e)))
                     for o,g,e in zip(obs,goal,expected_action))
    assert max(batch_error,single_error)<2e-5,(batch_error,single_error)
    np.savez(out/'parity_inputs.npz',obs=obs,goal=goal,expected_action=expected_action)
    audit=dict(actor_cases=len(obs),batch_max_abs_error=batch_error,
               single_max_abs_error=single_error,tolerance=2e-5,
               oracle='Original checkpoint policy in CPU float32; no retraining or policy conversion approximation')
    if hind:
        from supermimic.task import Task
        controller=Controller(out)
        controller.anchor_position=reference['pos'][0].copy()
        controller.anchor_rotation=np.eye(3,dtype=np.float32)
        errors=[]
        for frame in range(0,len(qpos)-1,7):
            ids=np.minimum(frame+np.arange(1,6),len(qpos)-1)
            pos=reference['pos'][frame]+rng.normal(0,.03,3).astype(np.float32)
            rot=reference['rot'][frame]
            s={'pos':torch.from_numpy(pos[None]),'rot':torch.from_numpy(rot[None])}
            future={k:torch.from_numpy(reference[k][ids][None]) for k in ['pos','rot','q','qd','v','w']}
            expected_goal=Task.goal(None,s,future)[0].numpy()
            expected_goal=np.r_[expected_goal,float(reference['phase'][ids[0]])].astype(np.float32)
            actual_goal=controller.reference_goal(pos,rot,frame)
            errors.append(float(np.max(np.abs(expected_goal-actual_goal))))
        audit.update(reference_goal_cases=len(errors),reference_goal_max_abs_error=max(errors))
        assert max(errors)<2e-6,max(errors)
    else:
        from gait_clock import policy_goal
        from contract import constrain_command,slew_command
        cmds=rng.normal(size=(512,3)).astype(np.float32)*3
        cmds[:32]=0;phases=rng.uniform(0,1,512).astype(np.float32)
        torch_goals=policy_goal(torch.from_numpy(cmds),torch.from_numpy(phases)).numpy()
        clock_error=float(np.max(np.abs(np.stack([clock_goal(c,p) for c,p in zip(cmds,phases)])-torch_goals)))
        command_error=max(float(np.max(np.abs(constrain(c)-constrain_command(*c)))) for c in cmds)
        slew_error=max(float(np.max(np.abs(slew(np.array([.2,.1,.3],dtype=np.float32),c)-
                                            slew_command([.2,.1,.3],c)))) for c in cmds)
        assert clock_error<2e-6 and command_error<2e-6 and slew_error<2e-6
        audit.update(clock_cases=512,clock_max_abs_error=clock_error,
                     command_max_abs_error=command_error,slew_max_abs_error=slew_error)
    if hind and experiment=='b2_light_payload':
        from actuator_filter import torque_envelope
        from pd_adapter import saturated_pd_target
        q=rng.uniform(-2,2,(1024,12)).astype('f')
        qd=rng.uniform(-12,12,(1024,12)).astype('f')
        target=rng.uniform(-3,3,(1024,12)).astype('f')
        limits=np.asarray(meta['torque_limits'],dtype='f')
        expected_tau,_=torque_envelope(torch.from_numpy(target),torch.from_numpy(q),
                                      torch.from_numpy(qd),torch.from_numpy(limits))
        actual=[]
        for t,p,v in zip(target,q,qd):
            wire,_,limited=saturated_pd_target(t.tolist(),p.tolist(),v.tolist(),250.,8.,limits.tolist(),meta['torque_envelope'])
            assert np.max(abs(250*(np.array(wire)-p)-8*v-np.array(limited)))<1e-4
            actual.append(limited)
        error=float(np.max(abs(np.asarray(actual)-expected_tau.numpy())))
        assert error<2e-4,error
        np.savez(out/'envelope_inputs.npz',target=target,q=q,dq=qd,expected_tau=expected_tau.numpy())
        audit.update(torque_envelope_cases=len(q),torque_envelope_max_abs_error=error)
    audit['passed']=True
    (out/'export_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(skill=args.skill,iteration=ck['it'],audit=audit)))


if __name__=='__main__':
    main()

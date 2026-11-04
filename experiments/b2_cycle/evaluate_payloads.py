"""Test the frozen B2 policy with actual rigid back payloads and matched pushes.

Payload inertia is merged into the trunk, preserving the policy's joint/body
layout. The visible collision box is massless because its mass is accounted for
in the explicit trunk inertial. No policy, reference, gains, or limits change.
"""
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from supermimic import networks
from supermimic.networks import Policy
from supermimic.task import state_to_flat
from cycle_env import CycleLibrary, make_cycle_env
from evaluate_cycle import CRITERIA, evaluate
from evaluate_pushes import compact, schedule, verify_force

HERE = Path(__file__).resolve().parent
BASE_XML = HERE.parent / 'three_skills' / 'b2_train.xml'
PAYLOAD_CENTER = np.array([0., 0., .18])
PAYLOAD_HALF_SIZE = np.array([.18, .12, .10])


def fmt(x):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(x).ravel())


def inertia_in_body(model, body):
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = rot.reshape(3, 3)
    return rot @ np.diag(model.body_inertia[body]) @ rot.T


def shift_inertia(mass, displacement):
    return mass * (np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement))


def build_model(kg, destination):
    assert kg >= 0
    base = mujoco.MjModel.from_xml_path(str(BASE_XML))
    body = base.body('base_link').id
    tree = ET.parse(BASE_XML)
    trunk = tree.getroot().find(".//body[@name='base_link']")
    mass = float(base.body_mass[body])
    center = base.body_ipos[body].copy()
    combined_center = (mass * center + kg * PAYLOAD_CENTER) / (mass + kg)
    combined_inertia = inertia_in_body(base, body)
    if kg:
        box_inertia = np.diag(kg / 3 * (np.sum(PAYLOAD_HALF_SIZE ** 2) - PAYLOAD_HALF_SIZE ** 2))
        combined_inertia += shift_inertia(mass, center - combined_center)
        combined_inertia += box_inertia + shift_inertia(kg, PAYLOAD_CENTER - combined_center)
        inertial = trunk.find('inertial')
        inertial.attrib.clear()
        inertial.set('mass', str(mass + kg))
        inertial.set('pos', fmt(combined_center))
        # Supply the principal frame directly: MuJoCo's fullinertia conversion
        # uses a less accurate eigensolver for nearly aligned principal axes.
        eigenvalues, principal_axes = np.linalg.eigh(combined_inertia)
        if np.linalg.det(principal_axes) < 0:
            principal_axes[:, 0] *= -1
        principal_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(principal_quaternion, principal_axes.ravel())
        inertial.set('diaginertia', fmt(eigenvalues))
        inertial.set('quat', fmt(principal_quaternion))
        ET.SubElement(trunk, 'geom', name='back_payload', type='box',
                      size=fmt(PAYLOAD_HALF_SIZE), pos=fmt(PAYLOAD_CENTER),
                      mass='0', density='0', contype='1', conaffinity='1', group='1',
                      rgba='.95 .46 .08 1')
    tree.write(destination, encoding='unicode')
    loaded = mujoco.MjModel.from_xml_path(str(destination))
    np.testing.assert_allclose(loaded.body_mass.sum(), base.body_mass.sum() + kg, atol=1e-10, rtol=0)
    np.testing.assert_allclose(loaded.body_ipos[body], combined_center, atol=1e-12, rtol=0)
    inertia_error = float(np.max(abs(inertia_in_body(loaded, body) - combined_inertia)))
    np.testing.assert_allclose(inertia_in_body(loaded, body), combined_inertia, atol=1e-10, rtol=0)
    assert (loaded.nq, loaded.nv, loaded.nu, loaded.nbody) == (base.nq, base.nv, base.nu, base.nbody)
    for field in ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange',
                  'actuator_trnid', 'jnt_range', 'dof_damping', 'dof_armature'):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(base, field))
    assert np.linalg.eigvalsh(combined_inertia).min() > 0
    # Verify total-system COM with the displaced load at three trunk inclinations.
    db, dl = mujoco.MjData(base), mujoco.MjData(loaded)
    com_errors = []
    for pitch in (0., -.6, -1.45):
        db.qpos[:3] = [0, 0, 1.5]
        db.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
        db.qpos[7:] = np.tile([0., .7, -1.4], 4)
        dl.qpos[:] = db.qpos
        mujoco.mj_forward(base, db)
        mujoco.mj_forward(loaded, dl)
        payload_world = db.xpos[body] + db.xmat[body].reshape(3, 3) @ PAYLOAD_CENTER
        expected = (base.body_mass.sum() * db.subtree_com[body] + kg * payload_world) / loaded.body_mass.sum()
        com_errors.append(float(np.max(np.abs(expected - dl.subtree_com[body]))))
    assert max(com_errors) < 1e-12
    return dict(payload_kg=kg, model_path=str(destination.resolve()),
                original_total_mass_kg=float(base.body_mass.sum()), total_mass_kg=float(loaded.body_mass.sum()),
                payload_center_in_trunk_m=PAYLOAD_CENTER.tolist(), payload_size_m=(2 * PAYLOAD_HALF_SIZE).tolist(),
                trunk_mass_kg=float(loaded.body_mass[body]), trunk_com_in_body_m=loaded.body_ipos[body].tolist(),
                trunk_inertia_in_body_kg_m2=combined_inertia.tolist(), system_com_audit_max_error_m=max(com_errors),
                inertia_reconstruction_max_error_kg_m2=inertia_error,
                mass_inertia_and_actuator_audit_passed=True,
                mounting='rigid, centered on dorsal trunk; actual box collision; inertia merged into trunk')


def phase_at(t):
    return 'four_leg' if t < 1 else 'rise' if t < 3.5 else 'hindstand' if t < 9.5 else 'return' if t < 12.5 else 'settle'


def failures(result, arr):
    counts = Counter()
    for i, trial in enumerate(result['trials_detail']):
        dead = np.flatnonzero(~arr['alive'][:, i])
        trial['first_failure_s'] = float(arr['time_s'][dead[0]]) if len(dead) else None
        reasons = []
        if len(dead):
            reasons.append('terminated_during_' + phase_at(trial['first_failure_s']))
        else:
            if trial['stand_pose_fraction'] < CRITERIA['hold_pose_fraction_min']:
                reasons.append('hindstand_pose')
            if trial['return_pose_fraction'] < CRITERIA['return_pose_fraction_min']:
                reasons.append('returned_pose')
            if trial['hind_foot_xy_drift_m'] > CRITERIA['foot_xy_drift_max']:
                reasons.append('hind_foot_drift')
            for section in ('stand', 'returned'):
                for key, value in trial[section].items():
                    if key + '_max' in CRITERIA and value > CRITERIA[key + '_max']:
                        reasons.append(section + '_' + key)
        trial['failure_reasons'] = reasons
        counts.update(reasons)
    result['failure_reason_counts'] = dict(counts)


@torch.no_grad()
def run_condition(policy, lib, model_path, n, seed, planned, with_push):
    env = make_cycle_env(lib, 'cuda', n, evaluate=True, model_path=model_path)
    if getattr(policy, 'reference_phase_command', False):
        from stage_policy import install_phase_command
        install_phase_command(env.task)
    force = torch.tensor(planned, device='cuda')
    live = torch.ones(n, device='cuda', dtype=torch.bool)
    applied, initial = [], []
    base_step = env.step
    counter = 0

    def step(action):
        nonlocal counter, live
        if not initial:
            initial.append(state_to_flat(env.state).cpu().numpy().copy())
        f = force[counter] * live[:, None] if with_push else torch.zeros(n, 3, device='cuda')
        env.sim.data.xfrc_applied[:] = 0.
        env.sim.data.xfrc_applied[:, 1, :3] = f
        applied.append(f.cpu().numpy().copy())
        state, info = base_step(action)
        live &= ~info['terminated']
        counter += 1
        return state, info

    env.step = step
    result, arr = evaluate(policy, env, seed=seed)
    env.sim.data.xfrc_applied[:] = 0.
    arr['force_world'] = np.asarray(applied)
    assert counter == len(planned)
    failures(result, arr)
    return result, arr, initial[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--push-seed', type=int, default=20261110)
    args = p.parse_args()
    start = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    torch.set_num_threads(2)
    networks.AMP['enabled'] = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lib = CycleLibrary(str(HERE / 'b2_full_cycle.npz'), 'cuda')
    ck = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    arch = ck['policy_architecture']
    if arch.get('type') in ('guarded_residual', 'phase_guarded_residual', 'stage_guarded_residual'):
        from guarded_policy import from_checkpoint
        policy = from_checkpoint(ck, 'cuda')
    else:
        policy = Policy(ck['policy']['obs_norm.mean'].numel(), ck['policy']['goal_norm.mean'].numel(),
                        enc_hidden=tuple(arch['encoder']), dec_hidden=tuple(arch['decoder'])).cuda()
        policy.load_state_dict(ck['policy'])
    planned, events = schedule(args.trials, lib.num_frames - 1, .02, args.push_seed)
    (out / 'push_schedule.json').write_text(json.dumps(events, indent=2) + '\n')
    summary = dict(checkpoint=args.checkpoint,
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   policy_frozen=True, reference_unchanged=True, actuator_limits_and_gains_unchanged=True,
                   payloads_kg=[0, 5, 10, 15], trials_per_condition=args.trials, seed=args.seed, push_seed=args.push_seed,
                   force_range_n=[8, 16], duration_range_s=[.12, .24], pushes_per_trial=5,
                   application='world XYZ force at trunk COM; zero imposed torque; same absolute forces for all loads',
                   inference_precision='bfloat16 MLP, float32 physics/state',
                   render_trial=0, render_trial_selection='fixed before observing outcomes',
                   pairing='identical initial states and force schedules; independent GPU contact reductions',
                   criteria=CRITERIA, models=[], conditions=[])
    summary['source_sha256'] = {str(path.relative_to(HERE.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), HERE/'cycle_env.py', HERE/'evaluate_cycle.py',
                                            HERE/'evaluate_pushes.py', HERE.parent/'three_skills'/'skill_env.py',
                                            HERE/'b2_full_cycle.npz', BASE_XML)}
    # Audit every physical model before running any of the expensive rollouts.
    for kg in summary['payloads_kg']:
        model_path = out / 'models' / f'b2_payload_{kg}kg.xml'
        model_info = build_model(kg, model_path)
        summary['models'].append(model_info)
    original_initial = None
    for model_info in summary['models']:
        kg, model_path = model_info['payload_kg'], Path(model_info['model_path'])
        model_info['force_audit'] = verify_force(lib, 'cuda', model_path=model_path)
        print('MODEL_AUDIT ' + json.dumps(model_info), flush=True)
        for pushed in (False, True):
            name = f'{kg}kg_' + ('push' if pushed else 'no_push')
            dest = out / name
            dest.mkdir()
            started = time.time()
            result, arr, initial = run_condition(policy, lib, model_path, args.trials, args.seed, planned, pushed)
            if original_initial is None:
                original_initial = initial
            initial_error = float(abs(initial - original_initial).max())
            assert initial_error == 0., (name, initial_error)
            assert all(torch.equal(v, ck['policy'][k]) for k, v in policy.state_dict().items()), 'Policy changed'
            result.update(condition=name, payload_kg=kg, random_push=pushed, total_mass_kg=model_info['total_mass_kg'],
                          it=int(ck['it']), source_iteration=int(ck['source_iteration']), checkpoint=args.checkpoint,
                          model_path=str(model_path), policy_frozen=True, initial_state_max_difference=initial_error,
                          elapsed_seconds=time.time() - started, rollout_path=str(dest / 'rollout.npz'),
                          force_range_n=[8, 16] if pushed else [0, 0], pushes_per_trial=5 if pushed else 0)
            np.savez_compressed(dest / 'rollout.npz', **arr, dt=.02)
            (dest / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
            summary['conditions'].append(compact(result))
            summary['elapsed_seconds'] = time.time() - start
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print('RESULT ' + json.dumps(compact(result)), flush=True)
    summary['completed'] = True
    summary['elapsed_seconds'] = time.time() - start
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (HERE / 'latest_payload_evaluation.json').write_text(json.dumps(dict(out=str(out), summary=str(out / 'summary.json')), indent=2) + '\n')
    print('SAVED ' + str(out), flush=True)


if __name__ == '__main__':
    main()

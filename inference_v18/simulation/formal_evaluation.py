"""Physical evaluation with unchanged goals, bounds, and simultaneous dwell."""
from joint import *
from formal_control import atomic_json
import time


@torch.no_grad()
def evaluate_suite(manager, lower, out, iteration, name, n=16, seed=411,
                   randomize=False, env_order=(0, 2, 1)):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    was_training = manager.training
    manager.eval()
    rows = []
    begun = time.time()
    try:
        for stage in env_order:
            atomic_json(out / 'eval_progress.json', dict(iteration=iteration, suite=name,
                        stage=stage, completed_strata=len(rows), trials_per_stratum=n,
                        updated_unix=time.time(), phase='physical_evaluation'))
            torch.manual_seed(seed + stage)
            env = JointEnv('cuda', n, randomize=randomize, compile=False, autoreset=False)
            env.reset()
            goal = (sample_goals(env, n, stage) if stage != 1 else
                    sample_goals(env, n, 1, arm_fraction=.45, yaw_limit=math.pi / 3, radius_max=.6))
            previous = torch.zeros(n, 12, device='cuda')
            alive = torch.ones(n, dtype=torch.bool, device='cuda')
            dwell = torch.zeros(n, device='cuda')
            maximum_dwell = dwell.clone()
            rejected = dwell.clone()
            maximum_command = torch.zeros(n, 12, device='cuda')
            initial = errors(env.arm, env.state, goal)[0]
            best_distance = initial.clone()
            failures = []
            for step in range(600):
                action = manager(env.state, env.context, goal, previous, env.xi)
                env.upper = action
                env.set_target(commands(action))
                state, info = env.step(lower(*env.policy_inputs()))
                distance, angle, _ = errors(env.arm, state, goal)
                newly_failed = info['terminated'] & alive
                if newly_failed.any():
                    details = env.arm.safety_details(state, state['aq'])
                    for case in newly_failed.nonzero().flatten().tolist():
                        failures.append(dict(case=case, time_s=(step + 1) * DT,
                            arm_fault=bool(env.last_arm_fault[case]),
                            invalid_physics=bool(env.last_step_invalid[case]),
                            height_m=float(state['pos'][case, 2]),
                            joint_violation_rad=float((torch.relu(env.arm.lower-state['aq'][case]) +
                                torch.relu(state['aq'][case]-env.arm.upper)).max()),
                            pair_violation_m=float(details[4][case]),
                            worst_pair_index=int(details[5][case].argmin())))
                best_distance = torch.minimum(best_distance, distance)
                alive &= ~info['terminated']
                hit = (distance <= .01) & (angle <= math.radians(10)) & alive
                dwell = torch.where(hit, dwell + DT, torch.zeros_like(dwell))
                maximum_dwell = torch.maximum(maximum_dwell, dwell)
                rejected += (~safe_arm_action(env.arm, state, env.xi, action)[2]).float()
                maximum_command = torch.maximum(maximum_command, action.abs())
                previous = action
                if step % 100 == 0:
                    atomic_json(out / 'eval_progress.json', dict(iteration=iteration,
                        suite=name, stage=stage, physics_step=step, completed_strata=len(rows),
                        updated_unix=time.time(), phase='physical_evaluation'))
            rows.append(dict(stage=stage, seed=seed+stage, trials=n, randomized=randomize,
                success=int(((maximum_dwell >= 1.-1e-6) & alive).sum()), alive=int(alive.sum()),
                initial_position_error=float(initial.mean()),
                final_position_error=float(distance.mean()),
                best_position_error=float(best_distance.mean()),
                final_orientation_error_deg=float(torch.rad2deg(angle).mean()), duration_s=12.,
                max_joint_hold_s=maximum_dwell.tolist(), final_position_errors_m=distance.tolist(),
                final_orientation_errors_deg=torch.rad2deg(angle).tolist(),
                safe_proposal_rejection_count=rejected.tolist(),
                max_abs_normalized_commands=maximum_command.tolist(), failures=failures))
            del env
        atomic_json(out / f'eval_{name}_{iteration:06d}.json', dict(iteration=iteration,
            suite=name, rows=rows, seconds=time.time()-begun, accepted=False,
            criterion=dict(position_m=.01, orientation_deg=10, simultaneous_hold_s=1)))
        return rows
    finally:
        manager.train(was_training)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

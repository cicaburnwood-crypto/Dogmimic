"""Record comparable physical episodes for native baselines, teacher and candidate."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from controllers import BASELINES, M1Controller, M20SController
from sim import (DT, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE,
                 M1_SETTLE_KD, M1_SETTLE_KP, M1_STAND,
                 M20S_FLAT, M20S_NAMES, RUNTIME, Robot)

HOLD = 50
DRIVE = 175
STOP = 150
STEPS = HOLD + DRIVE + STOP
# Seeds vary the initial base velocity. The task IDs match across all controllers.
TASKS = (
    ("stand", 0, 0, 0), ("forward_20", .20, 0, 0), ("forward_40", .40, 0, 0),
    ("forward_60", .60, 0, 0), ("reverse_20", -.20, 0, 0), ("reverse_40", -.40, 0, 0),
    ("left_yaw", 0, 0, .35), ("right_yaw", 0, 0, -.35),
    ("left_arc", .35, 0, .25), ("right_arc", .35, 0, -.25),
    ("left_side", 0, .15, 0), ("right_side", 0, -.15, 0),
    ("forward_30", .30, 0, 0), ("reverse_30", -.30, 0, 0),
    ("diagonal_left", .20, .10, .20), ("diagonal_right", .20, -.10, -.20),
)


def settle(robot, home, height, kp, kd, rng):
    robot.reset(home, height)
    robot.data.qvel[:3] = rng.uniform(-0.015, 0.015, 3)
    for _ in range(60):
        robot.step(home, kp, kd)


def run_episode(robot, controller, target, rng, teacher, noise):
    if teacher:
        kp, kd = controller.kp, controller.kd
        settle(robot, controller.default, .47, kp, kd, rng)
    else:
        kp, kd = M1_KP, M1_KD
        settle(robot, M1_STAND, .54, M1_SETTLE_KP, M1_SETTLE_KD, rng)
    controller.reset()
    commands = np.zeros((STEPS, 3), np.float32)
    commands[HOLD:HOLD+DRIVE] = target
    states = np.empty((STEPS+1, 45), np.float32)
    actions = np.zeros((STEPS, 16), np.float32)
    obs_dim = 0 if teacher else controller.history.shape[-1]
    obs = np.zeros((STEPS, obs_dim), np.float32)
    feature = np.zeros((STEPS, 11), np.float32)
    states[:] = robot.state()
    valid = 0
    for t, command in enumerate(commands):
        if not robot.valid():
            break
        if teacher:
            desired = controller.desired(robot, command)
            actions[t] = controller.previous
        else:
            observation = controller.observe(robot, command)
            raw = controller.action(observation)
            effective = raw + rng.normal(0, noise, 16) if noise else raw
            obs[t] = observation
            actions[t] = effective
            desired = M1_HOME + M1_SCALE * effective
        robot.step(desired, kp, kd)
        states[t+1] = robot.state()
        if not robot.valid():
            break
        gravity, wheel, support = robot.tip_and_wheels()
        feature[t] = np.r_[robot.velocity(), gravity[:2], wheel, support.mean(), states[t+1, 2]]
        valid += 1
    states[valid+1:] = states[valid]
    if not np.isfinite(states[:valid+1]).all():
        raise RuntimeError("Non-finite physics state")
    return states, actions, obs, feature, commands, valid


def summarize(task_ids, feature, commands, valid):
    metrics = []
    for idx in sorted(set(task_ids)):
        mask = np.asarray(task_ids) == idx
        data = feature[mask]
        truth = commands[mask]
        lengths = valid[mask]
        scores = []
        stop_ok = []
        for episode, target, n in zip(data, truth, lengths, strict=True):
            start = HOLD + 50
            end = min(n, HOLD + DRIVE)
            if end > start:
                scores.append((episode[start:end, :3] - target[start:end]) ** 2)
            stop = episode[HOLD+DRIVE:n, :3]
            stable = (np.linalg.norm(stop[:, :2], axis=1) <= .05) & (abs(stop[:, 2]) <= .10)
            # First entry by 1.5 s, with a consecutive full second of quiet samples.
            begins = [i for i in range(max(0, len(stable)-49)) if stable[i:i+50].all()]
            stop_ok.append(bool(begins and begins[0] <= 75))
        rmse = np.sqrt(np.mean(np.concatenate(scores), axis=0)).tolist() if scores else None
        metrics.append(dict(task=TASKS[idx][0], count=int(mask.sum()), completed=int((lengths == STEPS).sum()),
                            rmse=rmse, stopped=int(sum(stop_ok))))
    return metrics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("cts", "classic", "m20s", "candidate"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(len(TASKS))))
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    teacher = args.kind == "m20s"
    if args.kind == "candidate" and not args.checkpoint:
        parser.error("--checkpoint is required for a candidate")
    if teacher and args.noise:
        parser.error("teacher rollouts preserve vendor actions")
    robot = Robot(M20S_FLAT, M20S_NAMES, .09) if teacher else Robot(M1_FLAT, M1_NAMES, .095)
    mode = "classic" if args.kind == "candidate" else args.kind
    controller = M20SController() if teacher else M1Controller(mode, args.checkpoint)
    model_path = (
        RUNTIME / "m20s-min/policy/policy.onnx" if teacher else
        args.checkpoint or RUNTIME / "m1/models" / BASELINES[args.kind]
    )
    records, task_ids, seeds = [], [], []
    for seed in args.seeds:
        for idx in args.tasks:
            target = np.array(TASKS[idx][1:])
            rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
            records.append(run_episode(robot, controller, target, rng, teacher, args.noise))
            task_ids.append(idx)
            seeds.append(seed)
    states, actions, obs, feature, commands, lengths = map(np.asarray, zip(*records, strict=True))
    manifest = dict(source=("DeepRoboticsLab/sdk_deploy/M20S" if teacher else args.kind),
                    policy_sha256=sha256(model_path.read_bytes()).hexdigest(),
                    dt=DT, noise=args.noise, metrics=summarize(task_ids, feature, commands, lengths))
    if teacher:
        manifest["commit"] = json.loads((RUNTIME / "m20s-min/manifest.json").read_text())["commit"]
        manifest["wrapper"] = "Headless adapter of vendor SDK runner; M20S flat MJCF"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, state=states, action=actions, obs=obs, feature=feature,
                        command=commands, length=lengths, task=task_ids, seed=seeds)
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out, "completed", int((lengths == STEPS).sum()), "/", len(lengths), flush=True)
    for row in manifest["metrics"]:
        print(row["task"], row["completed"], row["rmse"], row["stopped"], flush=True)


if __name__ == "__main__":
    main()

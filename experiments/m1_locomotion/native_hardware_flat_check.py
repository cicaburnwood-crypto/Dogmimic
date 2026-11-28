"""Check the hardware policy throughout the M1 remote's forward/reverse range."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort

from controllers import session
from native_rollout import build_actor
from native_world import Course
from sim import LEGS, M1_FLAT, M1_NAMES, Robot
from sim import M1_HARDWARE_STAND as STAND
from sim import M1_HARDWARE_STAND_KD as STAND_KD
from sim import M1_HARDWARE_STAND_KP as STAND_KP
from sim import M1_REMOTE_SMOOTH as SMOOTH
from sim import step_m1_handoff as step_policy

MOVE_START, MOVE_STOP, FINISH = 75, 460, 700


def settle(robot, seed):
    rng = np.random.default_rng(seed)
    pose = STAND.copy()
    pose[LEGS] += rng.uniform(-.05, .05, len(LEGS))
    robot.reset(pose, .55)
    for _ in range(175):
        robot.step(STAND, STAND_KP, STAND_KD)


def record(checkpoint, scene, requested, seed):
    ort.set_seed(seed)
    robot = Robot(scene, M1_NAMES, .095)
    settle(robot, seed)
    actor = build_actor("candidate", checkpoint)
    zero = np.zeros(3)
    command = zero.copy()
    desired = np.asarray(requested)
    moving_height, steady, after_release = [], [], []
    for tick in range(FINISH):
        moving = MOVE_START <= tick < MOVE_STOP
        goal = desired if moving else zero
        command = SMOOTH * command + (1. - SMOOTH) * goal
        action = actor.action(actor.observe(robot, command))
        step_policy(robot, action, tick)
        if not robot.valid():
            break
        z = robot.data.qpos[2]
        if moving:
            moving_height.append(float(z))
            if tick >= 200:
                steady.append(np.r_[z, robot.velocity()])
        if tick >= 580:
            after_release.append(float(z))

    result = {"completed": tick == FINISH - 1,
              "moving_min_height_m": min(moving_height, default=None),
              "stopped_min_height_m": min(after_release, default=None)}
    if steady:
        a = np.asarray(steady)
        result.update(steady_height_range_m=float(np.ptp(a[:, 0])),
                      forward_std_m_s=float(a[:, 1].std()),
                      mean_twist=a[:, 1:].mean(axis=0).tolist())
    return result


def compare(job):
    baseline, candidate, scene, command, seed = job
    return {"command": command, "seed": seed, "before": record(baseline, scene, command, seed),
            "after": record(candidate, scene, command, seed)}


def actions_unchanged(baseline, candidate, directory):
    actors = []
    for i, checkpoint in enumerate((baseline, candidate)):
        model = onnx.load(checkpoint)
        for index, node in enumerate(n for n in model.graph.node if n.op_type == "RandomNormalLike"):
            existing = [attr for attr in node.attribute if attr.name != "seed"]
            del node.attribute[:]
            node.attribute.extend(existing)
            node.attribute.append(onnx.helper.make_attribute("seed", 351. + index))
        path = directory / f"seeded-{i}.onnx"
        onnx.save(model, path)
        actors.append(session(path))
    rng = np.random.default_rng(99412)
    maximum = 0.
    for index in range(512):
        current = rng.uniform(-.5, .5, (1, 49)).astype(np.float32)
        history = rng.uniform(-.5, .5, (1, 245)).astype(np.float32)
        current[:, 5] = -.9
        current[:, 6] = rng.uniform(-.96, .96) if index < 256 else rng.uniform(-2.8, 2.8)
        if index >= 256:
            current[:, 7] = 0.
            current[:, 8] = rng.choice((-1., 1.)) * rng.uniform(.005, .20)
        inputs = {"obs": current, "obs_history": history}
        before, after = (actor.run(["actions"], inputs)[0] for actor in actors)
        maximum = max(maximum, float(np.max(np.abs(before-after))))
    return maximum


def acceptance(rows, equality):
    if equality > 1e-6 or not all(row["after"]["completed"] for row in rows):
        return False
    for row in rows:
        x, _, yaw = row["command"]
        result = row["after"]
        if yaw:
            baseline = row["before"]
            yaw_error = abs(result["mean_twist"][2] - yaw)
            baseline_error = abs(baseline["mean_twist"][2] - yaw)
            if yaw_error > baseline_error + .015 or abs(result["mean_twist"][0] - baseline["mean_twist"][0]) > .05:
                return False
            continue
        if -.60 < x < -.48 and (abs(result["mean_twist"][0] - x) > .05 or
                                 result["forward_std_m_s"] > .025):
            return False
        if abs(x) < .6:
            continue
        if (result["moving_min_height_m"] < .40 or result["stopped_min_height_m"] < .40 or
                result["steady_height_range_m"] > .03 or result["forward_std_m_s"] > .025 or
                abs(result["mean_twist"][0] - x) > .065 or abs(result["mean_twist"][1]) > .035):
            return False
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    drive = [(sign * speed, 0., 0.) for speed in (.35, .45, .50, .55, .60, .80, 1., 1.20, 1.40)
             for sign in (1, -1)]
    drive += [(-speed, 0., 0.) for speed in (.49, .51, .53, .58)]
    steer = [(1., 0., .06), (1., 0., -.10), (-1., 0., .08), (-.45, 0., .35)]
    with TemporaryDirectory(prefix="m1-hardware-flat-") as temp:
        root, jobs = Path(temp), []
        for seed, friction in ((88751, .84), (88917, .96), (88993, 1.09)):
            scene = root / f"scene-{seed}.xml"
            Course(friction=friction).write(scene, M1_FLAT)
            jobs.extend((args.baseline, args.candidate, scene, command, seed) for command in drive + steer)
        with ProcessPoolExecutor(max_workers=args.workers) as workers:
            records = list(workers.map(compare, jobs))
        equality = actions_unchanged(args.baseline, args.candidate, root)
    report = {"baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": sha256(args.candidate.read_bytes()).hexdigest(),
              "low_speed_and_turn_action_delta_max": equality,
              "passed": acceptance(records, equality), "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Flat hardware regression", report["passed"], "trials", len(records), "unmodified action delta", equality)
    if not report["passed"]:
        raise SystemExit(1)

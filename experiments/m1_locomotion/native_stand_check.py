"""Measure sustained standing after reset, driving, turning, and descending."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from native_rollout import build_actor, reset_at_start
from native_world import Course
from sim import DT, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, Robot

DRIVE_START = 60
LEVEL_STOP = 215
SETTLED = round(6. / DT)
HOLD = round(90. / DT)

ROUTES = (
    ("flat", 0., (0., 0., 0.)),
    ("forward", 0., (.4, 0., 0.)),
    ("reverse", 0., (-.4, 0., 0.)),
    ("turn", 0., (0., 0., -.35)),
    ("reverse_turn", 0., (-.35, 0., .30)),
    ("upper", .18, (0., 0., 0.)),
    ("upper", .20, (0., 0., 0.)),
    ("descent", .18, (.4, 0., 0.)),
    ("descent", .20, (.4, 0., 0.)),
)


def record(checkpoint, scene, route, course, target, seed, yaw, lateral):
    robot = Robot(scene, M1_NAMES, .095)
    upper_start = route in ("upper", "descent")
    reset_at_start(robot, np.random.default_rng(seed), np.pi + yaw if upper_start else yaw,
                   lateral, course.descending_start if upper_start else 0.,
                   .54 + (course.top if upper_start else 0.))
    actor = build_actor("candidate", checkpoint)
    trunk = robot.model.body("trunk").id
    support = robot.model.geom("landing").id if route == "upper" else robot.floor
    zero = np.zeros(3)
    target = np.asarray(target)
    command = zero.copy()
    alpha = .99**10 if upper_start else 0.
    release = 0 if route in ("flat", "upper") else None
    heights = []
    post_settle_drift = 0.
    trunk_contact = False
    for tick in range(2700 + HOLD):
        if release is None:
            if route == "descent":
                # All wheels must have reached the lower flat before releasing W.
                if robot.data.xpos[robot.wheel_bodies, 0].max() < course.edge - .80:
                    release = tick
            elif tick >= LEVEL_STOP:
                release = tick
            if release is None and tick == 2700:
                break
        if release is not None and tick - release >= SETTLED:
            if not heights:
                settled_xy = robot.data.qpos[:2].copy()
            heights.append(float(robot.data.qpos[2] - (course.top if route == "upper" else 0.)))
            drift = float(np.linalg.norm(robot.data.qpos[:2] - settled_xy))
            post_settle_drift = max(post_settle_drift, drift)
            for contact in robot.data.contact:
                if support in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == support else contact.geom1
                    trunk_contact |= robot.model.geom_bodyid[other] == trunk
            if tick - release >= HOLD:
                break
        goal = target if release is None and tick >= DRIVE_START else zero
        command = alpha * command + (1. - alpha) * goal
        action = actor.action(actor.observe(robot, command))
        robot.step(M1_HOME + M1_SCALE * action, M1_KP, M1_KD)
        if not robot.valid():
            break

    low, high = (min(heights), max(heights)) if heights else (None, None)
    completed = len(heights) == HOLD - SETTLED + 1
    passed = completed and low >= .40 and high - low <= .025 and not trunk_contact
    return {"completed": completed, "min_height_m": low, "max_height_m": high,
            "settled_drift_m": post_settle_drift if heights else None,
            "trunk_contact": bool(trunk_contact), "passed": bool(passed)}


def compare(job):
    baseline, candidate, *args = job
    _, route, course, _, seed, _, _ = args
    return {"route": route, "rise_m": course.rise, "seed": seed,
            "before": record(baseline, *args), "after": record(candidate, *args)}


def run(baseline, candidate, workers):
    settings = ((.88, .04, -.09), (.95, -.035, .06), (1.03, .012, .10))
    with TemporaryDirectory(prefix="m1-stand-check-") as temp:
        scenes, jobs = {}, []
        for i, (route, rise, target) in enumerate(ROUTES):
            for j, (friction, yaw, lateral) in enumerate(settings):
                course = Course(rise=rise, steps=10 if rise else 0, friction=friction)
                if course not in scenes:
                    path = Path(temp) / f"course-{len(scenes)}.xml"
                    course.write(path, M1_FLAT)
                    scenes[course] = path
                jobs.append((baseline, candidate, scenes[course], route, course, target,
                             88100 + 10*i + j, yaw, lateral))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(compare, jobs))
    return {"baseline_sha256": sha256(baseline.read_bytes()).hexdigest(),
            "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
            "measured_window_s": [SETTLED*DT, HOLD*DT],
            "passed": all(row["after"]["passed"] for row in rows), "records": rows}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    report = run(args.baseline, args.candidate, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for side in ("before", "after"):
        print(side, sum(row[side]["passed"] for row in report["records"]), "/", len(report["records"]))
    if not report["passed"]:
        raise SystemExit(1)

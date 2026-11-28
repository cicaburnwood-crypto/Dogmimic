"""Compare front-wheel clearance on the keyboard simulator's two stair courses."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort

from native_rollout import CLIMB_LIMIT, build_actor, reset_at_start
from native_world import FRONT_WHEELS, Course
from sim import (DT, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, M1_SETTLE_KD,
                 M1_SETTLE_KP, M1_STAND, Robot, body_matrix)

FRONT = list(FRONT_WHEELS)
REAR = [1, 3]


def record(checkpoint, scene, course, speed, idle, yaw, lateral, seed):
    ort.set_seed(0 if seed is None else seed)
    robot = Robot(scene, M1_NAMES, .095)
    if seed is None:
        robot.reset(M1_STAND, .54)
        for _ in range(60):
            robot.step(M1_STAND, M1_SETTLE_KP, M1_SETTLE_KD)
    else:
        reset_at_start(robot, np.random.default_rng(seed), yaw, lateral)
    actor = build_actor("candidate", checkpoint)
    step_ids = {robot.model.geom(f"step_{i+1}").id: i for i in range(course.steps)}
    front_bodies = robot.wheel_bodies[FRONT]
    front_legs = dict(zip(front_bodies, range(len(FRONT))))
    front_contact = np.zeros((course.steps, len(FRONT)), dtype=int)
    first_over = np.full(front_contact.shape, np.nan)
    rim_clearance = np.full_like(first_over, np.nan)
    risers = course.edge + np.arange(course.steps) * course.tread
    tops = course.rise * np.arange(1, course.steps + 1)
    probe_x = risers[:, None] - .055
    command = np.zeros(3)
    alpha = .99**10
    landed = False
    for tick in range(-idle, CLIMB_LIMIT):
        if tick >= 0:
            command[0] = alpha * command[0] + (1 - alpha) * speed
        action = actor.action(actor.observe(robot, command))
        previous = robot.data.xpos[front_bodies]
        robot.step(M1_HOME + M1_SCALE * action, M1_KP, M1_KD)
        if tick < 0:
            continue
        wheels = robot.data.xpos[robot.wheel_bodies]
        front = wheels[FRONT]
        probe = (previous[:, 0] < probe_x) & (front[:, 0] >= probe_x) & np.isnan(rim_clearance)
        fraction = ((probe_x - previous[:, 0]) / np.maximum(front[:, 0] - previous[:, 0], 1e-6)).clip(0., 1.)
        rim = previous[:, 2] + fraction*(front[:, 2] - previous[:, 2]) - robot.wheel_radius
        rim_clearance[probe] = (rim - tops[:, None])[probe]
        over = (front[None, :, 0] >= risers[:, None] + .015) & (front[None, :, 2] >= tops[:, None] + .070)
        elapsed = (tick + 1) * DT
        first_over[over & np.isnan(first_over)] = elapsed
        touching = np.zeros_like(front_contact)
        for contact in robot.data.contact:
            if abs(contact.frame[0]) < .55 or abs(contact.frame[2]) > .72:
                continue
            step = step_ids.get(contact.geom1)
            wheel_geom = contact.geom2
            if step is None:
                step = step_ids.get(contact.geom2)
                wheel_geom = contact.geom1
            if step is not None:
                leg = front_legs.get(robot.model.geom_bodyid[wheel_geom])
                if leg is not None:
                    touching[step, leg] = 1
        front_contact += touching
        if not robot.valid() or abs(robot.data.qpos[1]) > course.half_width:
            break
        if (wheels[REAR, 0].min() >= course.end + .05 and robot.data.qpos[2] > course.top + .30
                and body_matrix(robot.data.qpos[3:7])[2, 2] > .85):
            landed = True
            break

    def finite_values(values, decimals):
        return [[round(float(v), decimals) if np.isfinite(v) else None for v in row] for row in values]

    return {"landed": landed, "elapsed_s": round(elapsed, 2), "front_clearance_s": finite_values(first_over, 2),
            "front_rim_clearance_m": finite_values(rim_clearance, 4),
            "front_riser_contact_s": np.round(front_contact * DT, 2).tolist()}


def compare(job):
    baseline, candidate, scene, course, speed, idle, yaw, lateral, seed = job
    conditions = scene, course, speed, idle, yaw, lateral, seed
    return {"rise_cm": round(100 * course.rise), "speed_mps": speed, "idle_s": round(idle * DT, 2),
            "yaw_rad": yaw, "seed": seed, "course": course.context().tolist(),
            "before": record(baseline, *conditions), "after": record(candidate, *conditions)}


def summarize(records):
    summary = {}
    for name in ("before", "after"):
        results = [r[name] for r in records]
        landed = [r for r in results if r["landed"]]
        comparable = [r[name] for r in records if r["before"]["landed"] and r["after"]["landed"]]
        times = np.array([np.sum(r["front_riser_contact_s"], axis=0) for r in comparable])
        delays = np.zeros((len(comparable), 2))
        clearance = [[], []]
        for i, row in enumerate(comparable):
            for left, right in row["front_clearance_s"]:
                if left is not None and right is not None:
                    delays[i] = np.maximum(delays[i], [left-right, right-left])
            for step in row["front_rim_clearance_m"]:
                for leg, height in enumerate(step):
                    if height is not None:
                        clearance[leg].append(height)
        summary[name] = {"landed": len(landed), "count": len(results),
                         "paired_landed": len(comparable),
                         "front_contact_median_s": np.median(times, axis=0).tolist() if comparable else None,
                         "front_contact_p90_s": np.quantile(times, .9, axis=0).tolist() if comparable else None,
                         "front_behind_p90_s": np.quantile(delays, .9, axis=0).tolist() if comparable else None,
                         "front_rim_clearance_p10_m": [float(np.quantile(values, .1)) if values else None
                                                        for values in clearance]}
    return summary


def cases(suite_name):
    for rise in (.18, .20):
        course = Course(rise=rise, steps=10)
        speeds = (.35, .40) if suite_name == "selection" else (.25, .35, .40, .45)
        idle_times = (23,) if suite_name == "selection" else (0, 12, 47)
        for speed in speeds:
            for idle in idle_times:
                yield course, speed, idle, 0., 0., None
    if suite_name == "keyboard":
        return
    first_seed, pairs = (73000, 8) if suite_name == "selection" else (83000, 16)
    rng = np.random.default_rng(first_seed)
    for i in range(pairs):
        course = Course(rise=(.18, .20)[i % 2], tread=float(rng.choice((.24, .265, .29))), steps=10,
                        edge=rng.uniform(.86, 1.21), friction=float(rng.choice((.85, .95, 1.06))))
        speed = float(rng.choice((.25, .30, .35, .40, .45)))
        yaw, lateral = rng.uniform(.025, .106), rng.uniform(-.13, .13)
        for j, side in enumerate((-1, 1)):
            yield course, speed, 10, side*yaw, side*lateral, first_seed + 2*i+j


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--suite", choices=("keyboard", "selection", "final"), default="keyboard")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="m1-stair-check-") as temp:
        jobs = []
        scenes = {}
        for course, speed, idle, yaw, lateral, seed in cases(args.suite):
            if course not in scenes:
                path = Path(temp) / f"stairs-{len(scenes)}.xml"
                course.write(path, M1_FLAT)
                scenes[course] = path
            jobs.append((args.baseline, args.candidate, scenes[course], course, speed, idle, yaw, lateral, seed))
        with mp.get_context("fork").Pool(args.workers) as pool:
            records = pool.map(compare, jobs)
    report = {"suite": args.suite, "front_order": ["LF", "RF"], "rim_probe_before_edge_m": .055,
              "baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": sha256(args.candidate.read_bytes()).hexdigest(),
              "summary": summarize(records), "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)

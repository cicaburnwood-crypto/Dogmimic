"""Collect M1-only demonstrations and evaluate one policy on matched terrain suites."""
from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mujoco
import numpy as np
import onnxruntime as ort

from controllers import M1Controller
from native_terrain import GUIDE_PARAMETERS, descent_guidance, level_guidance, observe_terrain
from native_world import (ARCHIVE, ARCHIVE_ROOT, DESCENT, FLAT, MIXED, NATIVE_FILES, NATIVE_RUNTIME,
                          NATIVE_SHA256, STAIRS, Course)
from rollout import DRIVE, HOLD, STEPS, TASKS
from sim import (DT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, M1_SETTLE_KD, M1_SETTLE_KP,
                 M1_STAND, Robot, body_matrix)

KINDS = ("flat", "stairs", "mixed", "descent")
ROLL_PERIOD = .8
CLIMB_LIMIT = 1480
DESCENT_LIMIT = 2200
LANDING_LIMIT = 400


@dataclass(frozen=True)
class Trial:
    kind: int
    course: Course
    command: tuple
    seed: int
    label: str
    yaw: float = 0.
    lateral: float = 0.


def stage_teachers(archive=None):
    dest = NATIVE_RUNTIME / "teachers"
    paths = {key: dest / name for key, name in NATIVE_FILES.items()}
    if archive is None and all(
        path.is_file() and sha256(path.read_bytes()).hexdigest() == NATIVE_SHA256[key]
        for key, path in paths.items()
    ):
        return paths
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive or ARCHIVE) as bundle:
        for teacher, basename in NATIVE_FILES.items():
            payload = bundle.read(ARCHIVE_ROOT + "models/" + basename)
            if sha256(payload).hexdigest() != NATIVE_SHA256[teacher]:
                raise ValueError(f"Archive weight does not match the verified {teacher} policy")
            path = dest / basename
            if not path.exists() or path.read_bytes() != payload:
                path.write_bytes(payload)
    return paths


def suite(name):
    # Separate RNG streams for training, model holdout, candidate selection and final scoring.
    if name.startswith("descent-"):
        seed, count = {"descent-train": (94000, 76), "descent-holdout": (95000, 20),
                       "descent-selection": (96000, 24), "descent-final": (97000, 48)}[name]
        rng = np.random.default_rng(seed)
        training = name == "descent-train"
        for i in range(count):
            rise = ((.15, .18, .20, .20) if training else (.18, .20))[i % (4 if training else 2)]
            tread = float(rng.choice((.235, .25, .28, .30) if training else (.24, .265, .29)))
            course = Course(rise=rise, tread=tread, steps=int(rng.choice((8, 10))) if training else 10,
                            edge=rng.uniform(.92, 1.23), friction=float(rng.choice((.86, .95, 1.05))))
            speed = float(rng.choice((.32, .36, .40, .44)))
            yield Trial(DESCENT, course, (speed, 0., 0.), seed+i, f"descent_{round(rise*100)}cm",
                        rng.uniform(-.085, .085), rng.uniform(-.14, .14))
        return
    if name == "level-replay":
        rng = np.random.default_rng(92000)
        for i in range(48):
            speed = (.16, .24, .34, .40, .50, .60)[i % 6]
            command = ((-1 if (i//6) % 2 else 1)*speed, 0., 0.)
            yield Trial(FLAT, Course(friction=float(rng.choice((.81, .89, .97, 1.05)))),
                        command, 92000+i, "level_straight")
        return
    first_seed = {"train": 13000, "holdout": 23000, "selection": 33000,
                  "final": 43000, "aggregate": 53000, "front-lift": 63000}[name]
    sizes = {
        "train": (14, 144, 104), "holdout": (2, 18, 16), "selection": (2, 16, 12),
        "final": (4, 60, 30), "aggregate": (5, 48, 40), "front-lift": (1, 32, 24),
    }
    repeats, stair_cases, mixed_cases = sizes[name]
    rng = np.random.default_rng(first_seed)
    train = name in ("train", "aggregate")
    for repeat in range(repeats):
        for i, (label, *command) in enumerate(TASKS):
            friction = float(rng.choice((.79, .90, 1.02) if train else (.85, .95, 1.06)))
            yield Trial(FLAT, Course(friction=friction), tuple(command), first_seed+16*repeat+i, label)
    for kind, n in ((STAIRS, stair_cases), (MIXED, mixed_cases)):
        if name == "front-lift":
            rises = (.18, .20) if kind == STAIRS else (.15, .18, .20)
            for i in range(n // 2):
                rise = rises[i % len(rises)]
                course = Course(rise=rise, tread=float(rng.choice((.23, .25, .28, .30))),
                                steps=int(rng.choice((8, 10))), edge=rng.uniform(.83, 1.25),
                                friction=float(rng.choice((.80, .90, 1.03))))
                command = (float(rng.choice((.22, .28, .34, .40, .45))), 0., 0.)
                yaw, lateral = rng.uniform(.020, .092), rng.uniform(-.12, .12)
                for j, side in enumerate((-1, 1)):
                    yield Trial(kind, course, command, first_seed + 500*kind + 2*i+j,
                                f"{KINDS[kind]}_{round(rise*100)}cm", side*yaw, side*lateral)
            continue
        for i in range(n):
            if train:
                rises = (.12, .15, .18, .20) if kind == STAIRS else (.15, .18, .20)
                rise = rises[i % len(rises)]
                treads = (.23, .25, .28, .30)
                friction = float(rng.choice((.79, .90, 1.02)))
                yaw = rng.uniform(-.078, .078)
                steps = int(rng.choice((6, 8, 10), p=(.18, .24, .58)))
            else:
                rises = (.18, .20) if kind == STAIRS else (.15, .18, .20)
                rise = rises[i % len(rises)]
                treads = (.24, .265, .29)
                friction = float(rng.choice((.85, .95, 1.06)))
                yaw = (1 if rng.integers(2) else -1) * rng.uniform(.025, .103)
                steps = 10
            speed = float(rng.choice((.25, .35, .45)))
            course = Course(rise=rise, tread=float(rng.choice(treads)), steps=steps,
                            edge=rng.uniform(.83, 1.25), friction=friction)
            yield Trial(kind, course, (speed, 0., 0.), first_seed + 500*kind + i,
                        f"{KINDS[kind]}_{int(round(rise*100))}cm", yaw=yaw,
                        lateral=rng.uniform(-.15, .15))


def reset_at_start(robot, rng, yaw=0., lateral=0., start=0., height=.54):
    robot.reset(M1_STAND, height)
    robot.data.qpos[:2] = [start, lateral]
    robot.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    robot.data.qvel[:3] = rng.uniform(-.012, .012, 3)
    mujoco.mj_forward(robot.model, robot.data)
    for _ in range(60):
        robot.step(M1_STAND, M1_SETTLE_KP, M1_SETTLE_KD)


def wheel_contacts(robot, terrain_ids, wheel_by_body):
    mask = np.zeros(4, dtype=np.uint8)
    for contact in robot.data.contact:
        a, b = contact.geom1, contact.geom2
        if a in terrain_ids or b in terrain_ids:
            robot_geom = b if a in terrain_ids else a
            wheel = wheel_by_body.get(robot.model.geom_bodyid[robot_geom])
            if wheel is not None:
                mask[wheel] = 1
    return mask


def simulate(trial, path, actor, rng, noise=0., collect=False, guide=False):
    course, kind = trial.course, trial.kind
    descending = kind == DESCENT
    robot = Robot(path, M1_NAMES, .095)
    if descending:
        reset_at_start(robot, rng, np.pi+trial.yaw, trial.lateral, course.descending_start, course.top+.54)
    else:
        reset_at_start(robot, rng, trial.yaw, trial.lateral)
    actor.reset()
    terrain = {robot.floor} | {robot.model.geom(f"step_{i}").id for i in range(1, course.steps+1)}
    if course.steps:
        terrain.add(robot.model.geom("landing").id)
    wheel_by_body = {body: i for i, body in enumerate(robot.wheel_bodies)}
    rear = robot.wheel_bodies[[1, 3]]
    zero = np.zeros(3)
    requested = np.asarray(trial.command)
    command = zero.copy()
    travel_limit = DESCENT_LIMIT if descending else CLIMB_LIMIT
    max_ticks = STEPS if kind == FLAT else travel_limit + LANDING_LIMIT
    states = np.empty((max_ticks+1, 45), dtype=np.float32) if collect else None
    observations = np.zeros((max_ticks+1, 49), dtype=np.float32) if collect else None
    actions = np.zeros((max_ticks+1, 16), dtype=np.float32) if collect else None
    targets = np.zeros((max_ticks+1, 3), dtype=np.float32) if collect else None
    support = np.zeros((max_ticks+1, 4), dtype=np.uint8) if collect else None
    baseline = np.zeros((max_ticks+1, 16), dtype=np.float32) if guide else None
    tracked = []
    landed_tick, stop_tick = None, None
    quiet_run = 0
    quiet_start, n_steps = None, 0
    outcome = "timeout"
    travelled = 0.
    min_x = float(robot.data.qpos[0])
    max_lateral = 0.
    highest_rear_x = float(robot.data.xpos[rear, 0].min())
    if collect:
        states[0] = robot.state()
    for tick in range(max_ticks):
        rear_x = float(robot.data.xpos[rear, 0].min())
        pose = robot.data.qpos
        upright = body_matrix(pose[3:7])[2, 2]
        if kind == FLAT:
            goal = requested if HOLD <= tick < HOLD+DRIVE else zero
            if tick == HOLD+DRIVE:
                stop_tick = tick
            command = goal
        else:
            if descending:
                landed = (robot.data.xpos[robot.wheel_bodies, 0].max() < course.edge - .05
                          and .30 < pose[2] < .70 and upright > .85)
                stop_zone = pose[0] <= course.edge - .85
            else:
                landed = rear_x > course.end + .05 and pose[2] > course.top + .30 and upright > .85
                stop_zone = rear_x >= course.end + (.90 if kind == MIXED else .12)
            if landed_tick is None and landed:
                landed_tick = tick
            if landed_tick is None and tick >= travel_limit:
                outcome = "descent_timeout" if descending else "climb_timeout"
                break
            if landed_tick is not None and tick-landed_tick >= LANDING_LIMIT:
                outcome = "landing_timeout"
                break
            if landed_tick is not None and stop_tick is None and stop_zone:
                stop_tick = tick
            if tick < HOLD or stop_tick is not None:
                goal = zero
            elif landed_tick is not None and kind == MIXED:
                goal = np.array([.42, 0., 0.])
            else:
                goal = requested
            alpha = 0.99**10
            command = alpha*command + (1-alpha)*goal

        obs = actor.observe(robot, command)
        action = actor.action(obs)
        if guide:
            baseline[tick] = action
            profile = observe_terrain(robot)
            delta = (descent_guidance(command, profile) if descending
                     else level_guidance(command, profile, obs, action))
            action = (action+delta).astype(np.float32)
        if noise:
            action = (action + rng.normal(0., noise, size=16)).astype(np.float32)
        if guide or noise:
            actor.previous = action
        if collect:
            observations[tick] = obs
            actions[tick] = action
            targets[tick] = goal
            support[tick] = wheel_contacts(robot, terrain, wheel_by_body)
        robot.step(M1_HOME + M1_SCALE*action, M1_KP, M1_KD)
        x, y, z = robot.data.qpos[:3]
        if not robot.valid():
            outcome = "fall"
            break
        if kind != FLAT and abs(y) > course.half_width-.11:
            outcome = "lane_exit"
            break
        if kind in (STAIRS, MIXED) and x > course.end and z < course.top + .24:
            outcome = "platform_drop"
            break
        n_steps += 1
        pose = robot.state()
        if collect:
            states[n_steps] = pose
        velocity = robot.velocity()
        travelled = max(travelled, float(x))
        min_x = min(min_x, float(x))
        max_lateral = max(max_lateral, abs(float(y)))
        highest_rear_x = max(highest_rear_x, float(robot.data.xpos[rear, 0].min()))
        if kind == FLAT and HOLD+50 <= tick < HOLD+DRIVE:
            tracked.append((velocity-requested)**2)
        if stop_tick is not None:
            stable = np.hypot(*velocity[:2]) <= .05 and abs(velocity[2]) <= .10
            if descending:
                stable &= (z > .30 and x < course.edge - .65 and body_matrix(pose[3:7])[2, 2] > .85)
            elif kind != FLAT:
                stable &= (pose[2] > course.top + .30 and highest_rear_x > course.end
                           and body_matrix(pose[3:7])[2, 2] > .85)
            quiet_run = quiet_run+1 if stable else 0
            if quiet_run == 50 and quiet_start is None:
                quiet_start = (tick-stop_tick-49)*DT
            if kind != FLAT and quiet_start is not None:
                outcome = "completed"
                break
            if kind != FLAT and tick-stop_tick >= 200:
                outcome = "landing_not_stable"
                break
    if kind == FLAT and n_steps == STEPS:
        outcome = "completed"
    record = {"kind": KINDS[kind], "case": trial.label, "seed": trial.seed, "target": trial.command,
              "course": course.context().tolist(), "initial_yaw": trial.yaw, "initial_y": trial.lateral,
              "outcome": outcome, "steps": n_steps, "stop_s": quiet_start, "max_x_m": travelled,
              "rear_x_m": highest_rear_x, "landing_s": None if landed_tick is None else landed_tick*DT,
              "sse": np.sum(tracked, axis=0).tolist() if tracked else [0.,0.,0.], "samples": len(tracked)}
    if descending:
        record.update(downhill_distance_m=course.descending_start-min_x, max_lateral_m=max_lateral)
    if not collect:
        return record, None
    observations[n_steps:] = 0
    actions[n_steps:] = 0
    targets[n_steps:] = 0
    support[n_steps:] = wheel_contacts(robot, terrain, wheel_by_body)
    arrays = (states, observations, actions, targets, support)
    if guide:
        baseline[n_steps:] = 0
        arrays += (baseline,)
    return record, tuple(a[:n_steps+1].copy() for a in arrays)


def summarize(records):
    def group(rows):
        passed = sum(row["outcome"] == "completed" for row in rows)
        return {"count": len(rows), "completed": passed, "success": passed/len(rows) if rows else 0.,
                "stop_by_1_5": sum(r["stop_s"] is not None and r["stop_s"] <= 1.5 for r in rows)}
    groups = {"flat": group([r for r in records if r["kind"] == "flat"]),
              "mixed": group([r for r in records if r["kind"] == "mixed"])}
    flights = [r for r in records if r["kind"] == "stairs"]
    groups["stairs"] = {k: group([r for r in flights if r["case"] == k]) for k in sorted({r["case"] for r in flights})}
    down = [r for r in records if r["kind"] == "descent"]
    if down:
        groups["descent"] = {k: group([r for r in down if r["case"] == k]) for k in sorted({r["case"] for r in down})}
    flat = [r for r in records if r["kind"] == "flat"]
    for label in sorted({r["case"] for r in flat}) + ["all"]:
        rows = flat if label == "all" else [r for r in flat if r["case"] == label]
        n = sum(r["samples"] for r in rows)
        mse = np.sum([r["sse"] for r in rows], axis=0) / max(n, 1)
        row = {"vxy_rmse": float(np.sqrt(mse[:2].sum())) if n else None,
               "yaw_rmse": float(np.sqrt(mse[2])) if n else None}
        if label == "all":
            groups["flat"].update(row)
        else:
            groups["flat"].setdefault("tasks", {})[label] = row | group(rows)
    return groups


def acceptance(metrics):
    flat = metrics["flat"]
    return (flat["count"] >= 64 and flat["success"] >= .95
            and flat["vxy_rmse"] is not None and flat["vxy_rmse"] <= .07
            and flat["stop_by_1_5"]/flat["count"] >= .95
            and all(metrics["stairs"][k]["count"] >= 30 and metrics["stairs"][k]["success"] >= .90
                    for k in ("stairs_18cm", "stairs_20cm"))
            and metrics["mixed"]["count"] >= 30 and metrics["mixed"]["success"] >= .90)


def descent_acceptance(metrics):
    return all(metrics["descent"][case]["count"] >= 24 and metrics["descent"][case]["success"] >= .90
               for case in ("descent_18cm", "descent_20cm"))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, help="Read the teacher weights from this M1 archive")
    p.add_argument("--suite", choices=("train", "holdout", "selection", "final", "aggregate", "front-lift",
                                       "level-replay", "descent-train", "descent-holdout",
                                       "descent-selection", "descent-final"),
                   required=True)
    p.add_argument("--policy", choices=("teachers", "classic", "stairs"), default="teachers")
    p.add_argument("--candidate", type=Path)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--guide", action="store_true", help="Collect level-stance or downhill wheel guidance")
    p.add_argument("--noise", type=float, default=0.)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    teachers = stage_teachers(args.archive)
    if args.candidate and args.policy != "teachers":
        p.error("--candidate and --policy select different controllers")
    if not 0 <= args.noise < .10:
        p.error("invalid collection noise")
    if args.guide and (not args.collect or not args.candidate or args.suite not in
                       ("level-replay", "descent-train", "descent-holdout")):
        p.error("Guidance is for collecting level or descent training trajectories")
    cases = list(suite(args.suite))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records, sequences = [], []
    with TemporaryDirectory(prefix="m1-native-") as root:
        worlds = {}
        for trial in cases:
            if trial.course not in worlds:
                dest = Path(root) / f"course_{len(worlds):04d}.xml"
                trial.course.write(dest)
                worlds[trial.course] = dest
        paths = dict(teachers)
        if args.candidate:
            paths["candidate"] = args.candidate
        setup = (paths, args.candidate is not None, args.policy, args.noise, args.collect, args.guide)
        jobs = [(trial, worlds[trial.course]) for trial in cases]
        with mp.get_context("fork").Pool(args.workers, initializer=setup_worker, initargs=setup) as pool:
            for i, (record, seq) in enumerate(pool.imap(run_case, jobs), 1):
                records.append(record)
                if args.collect:
                    sequences.append(seq)
                if i % 32 == 0 or i == len(cases):
                    print(f"{args.suite} {i}/{len(cases)}", flush=True)
    metrics = summarize(records)
    model = {k: sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
    report = {"suite": args.suite, "policy": "candidate" if args.candidate else args.policy,
              "sha256": model, "dt": DT, "transitions": sum(r["steps"] for r in records),
              "stair_climb_limit_s": CLIMB_LIMIT*DT, "landing_limit_s": LANDING_LIMIT*DT,
              "noise": args.noise, "metrics": metrics, "records": records}
    if args.suite == "final":
        report["passed"] = acceptance(metrics)
    if args.suite.startswith("descent-"):
        report["descent_limit_s"] = DESCENT_LIMIT*DT
    if args.suite == "descent-final":
        report["passed"] = descent_acceptance(metrics)
    if args.guide:
        report["guidance"] = (GUIDE_PARAMETERS if args.suite.startswith("descent-")
                              else {"trunk_height_m": .465, "leg_action_smoothing": .46})
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if args.collect:
        lengths = np.array([r["steps"] for r in records], np.int32)
        fields = ("state", "obs", "action", "target", "contact")
        if args.guide:
            fields += ("baseline_action",)
        arrays = dict(zip(fields, map(np.concatenate, zip(*sequences, strict=True)), strict=True))
        with args.out.open("wb") as f:
            np.savez_compressed(f, **arrays, length=lengths, start=np.cumsum(np.r_[0, lengths[:-1]+1]),
                                kind=np.asarray([t.kind for t in cases]),
                                course=np.stack([t.course.context() for t in cases]),
                                seed=np.asarray([t.seed for t in cases]),
                                finished=np.asarray([r["outcome"] == "completed" for r in records]))
    summary = {k: v for k, v in report.items() if k != "records"}
    summary["metrics"]["flat"].pop("tasks", None)
    print(json.dumps(summary, indent=2), flush=True)


def setup_worker(paths, candidate, policy, noise, collect, guide):
    global _options, _actors, _paths
    names = ("candidate",) if candidate else (tuple(NATIVE_FILES) if policy == "teachers" else (policy,))
    _paths = paths
    _actors = {name: build_actor(name, paths[name]) for name in names} if collect else {}
    _options = candidate, policy, noise, collect, guide


def run_case(job):
    trial, path = job
    candidate, policy, noise, collect, guide = _options
    if candidate:
        name = "candidate"
    elif policy == "teachers":
        name = "classic" if trial.kind == FLAT else "stairs"
    else:
        name = policy
    rng = np.random.default_rng(trial.seed)
    if collect:
        actor = _actors[name]
    else:
        ort.set_seed(trial.seed)
        actor = build_actor(name, _paths[name])
    return simulate(trial, path, actor, rng, noise, collect, guide)


def build_actor(name, path):
    return M1Controller("classic", path, period=.7 if name == "classic" else ROLL_PERIOD, continuous_phase=True)


if __name__ == "__main__":
    main()

"""Collect and compare M1 Classic and Uni Exp with the board's remote input."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort

from controllers import M1Controller
from native_hardware_flat_check import SMOOTH, settle, step_policy
from native_terrain import TRUNK_COM
from native_world import FLAT, HARDWARE_CLASSIC_SHA256, NATIVE_SHA256, Course
from sim import LEGS, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, Robot, body_matrix

COHORTS = {"train": (241000, 4), "holdout": (242000, 2), "selection": (243000, 1), "final": (244000, 3)}
START, STOP, END = 65, 330, 475


class HardwareClassic(M1Controller):
    def reset(self):
        super().reset()
        self.first_observation = True

    def action(self, obs):
        if self.first_observation:
            self.history[:] = obs
            self.first_observation = False
        else:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = obs
        data = {"obs": obs[None, :], "obs_history": self.history.reshape(1, -1)}
        self.previous = self.model.run(["actions"], data)[0][0]
        return self.previous


def trials(cohort):
    first, repeats = COHORTS[cohort]
    speeds = (.16, .23, .30, .38, .46, .54, .64, .82, 1., 1.20, 1.40)
    if cohort in ("selection", "final"):
        speeds = (.20, .35, .45, .60, .80, 1., 1.20, 1.40)
    commands = [(sign * speed, 0., 0.) for speed in speeds for sign in (1, -1)]
    commands += [(0., 0., sign * .35) for sign in (1, -1)]
    commands += [(sign * .40, 0., turn * .23) for sign in (1, -1) for turn in (1, -1)]
    commands += [(0., sign * .22, 0.) for sign in (1, -1)]
    if cohort in ("train", "holdout"):
        commands += [(1., 0., -.12), (-1., 0., .12), (.6, .13, .15), (0., .4, -.28), (0., 0., 0.)]
    frictions = (.76, .89, .98, 1.10) if cohort == "train" else (.82, .96, 1.06)
    for repeat in range(repeats):
        period = .8 if cohort == "train" and repeat != 0 else .7
        for i, command in enumerate(commands):
            seed = first + repeat * 100 + i
            yield seed, Course(friction=frictions[repeat % len(frictions)]), command, period


def run_trial(job):
    path, classic, scene, seed, course, command, period, collect, noise, *profile = job
    ort.set_seed(seed)
    rng = np.random.default_rng(seed + 1)
    robot = Robot(scene, M1_NAMES, .095)
    settle(robot, seed)
    controller_type = HardwareClassic if classic else M1Controller
    controller = controller_type("classic", path, period=period if classic else .8, continuous_phase=True)
    zero = np.zeros(3)
    requested = np.asarray(command)
    schedule = profile[0] if profile else np.zeros((END+1, 3))
    if not profile:
        schedule[START:STOP] = requested
    yaw_ticks = np.flatnonzero(np.abs(schedule[:STOP, 2]) > 1e-6) if profile else np.empty(0)
    yaw_end = int(yaw_ticks[-1]) + 1 if len(yaw_ticks) else STOP
    filtered = zero.copy()
    episode, steady = [], []
    released_turn = []
    movement_height, stop_height = [], []
    disturbance = np.zeros(16)
    for tick in range(END + 1):
        goal = schedule[tick]
        filtered = SMOOTH * filtered + (1 - SMOOTH) * goal
        obs = controller.observe(robot, filtered)
        expert = controller.action(obs)
        if noise:
            disturbance = .71 * disturbance + rng.normal(0., noise, size=16)
        action = controller.previous = (expert + disturbance).astype(np.float32)
        if collect:
            episode.append((robot.state(), obs, action, np.asarray(goal), robot.tip_and_wheels()[2], expert))
        if tick == END:
            break
        if classic:
            robot.step(M1_HOME + M1_SCALE * action, M1_KP, M1_KD)
        else:
            step_policy(robot, action, tick)
        if not robot.valid():
            break
        xyz, quat = robot.data.qpos[:3], robot.data.qpos[3:7]
        centre_z = (xyz + body_matrix(quat) @ TRUNK_COM)[2]
        if START <= tick < STOP:
            movement_height.append(centre_z)
        if START + 100 <= tick < STOP and np.array_equal(goal, requested):
            steady.append(np.r_[centre_z, robot.velocity(), robot.data.qpos[robot.qadr[LEGS]],
                                robot.data.qvel[robot.vadr[LEGS]],
                                robot.data.xpos[robot.wheel_bodies, 2] - robot.wheel_radius,
                                robot.tip_and_wheels()[2]])
        if yaw_end <= tick < STOP:
            released_turn.append(robot.data.xpos[robot.wheel_bodies, 2] - robot.wheel_radius)
        if tick >= STOP + 100:
            stop_height.append(centre_z)
    row = {"seed": seed, "friction": course.friction, "command": list(command), "finished": tick == END,
           "min_moving_com_m": min(movement_height, default=None), "min_stopped_com_m": min(stop_height, default=None)}
    if steady:
        s = np.asarray(steady)
        height, contact = s[:, 28:32], s[:, 32:36] > .5
        lifted = (height >= .010) & ~contact
        row.update(com_m=float(s[:, 0].mean()), com_range_m=float(np.ptp(s[:, 0])),
                   twist=s[:, 1:4].mean(0).tolist(), twist_std=s[:, 1:4].std(0).tolist(),
                   mean_leg_q_rad=s[:, 4:16].mean(0).tolist(), leg_rate_rms_rad_s=float(np.sqrt((s[:, 16:28]**2).mean())),
                   wheel_contact_fraction=contact.mean(0).tolist(), wheel_clearance_p95_m=np.quantile(height,.95,axis=0).tolist(),
                   wheel_clearance_max_m=height.max(0).tolist(), wheel_repeated_lift_count=(lifted[1:] & ~lifted[:-1]).sum(0).tolist())
    if released_turn:
        row["turn_release_clearance_max_m"] = np.max(released_turn, axis=0).tolist()
    arrays = tuple(np.asarray(values, dtype=dtype) for values, dtype in zip(zip(*episode),
                   (np.float32, np.float32, np.float32, np.float32, np.uint8, np.float32), strict=True)) if collect else None
    return row, arrays


def save_replays(out, cases, results, report, collecting):
    out.parent.mkdir(parents=True, exist_ok=True)
    records, sequences = zip(*results)
    report["records"] = records
    if collecting:
        lengths = np.asarray([len(values[0]) - 1 for values in sequences], np.int32)
        fields = ("state", "obs", "action", "target", "contact", "baseline_action")
        arrays = dict(zip(fields, map(np.concatenate, zip(*sequences)), strict=True))
        np.savez_compressed(out.with_suffix(".npz"), **arrays, length=lengths,
                            start=np.cumsum(np.r_[0, lengths[:-1]+1]), kind=np.full(len(cases), FLAT),
                            course=np.stack([row[1].context() for row in cases]), seed=np.asarray([row[0] for row in cases]),
                            finished=np.asarray([row["finished"] for row in records]))
        report["transitions"] = int(lengths.sum())
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("classic", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--suite", choices=tuple(COHORTS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if sha256(args.classic.read_bytes()).hexdigest() != HARDWARE_CLASSIC_SHA256:
        raise ValueError("Expected the board's verified original Classic weight")
    collecting = args.suite in ("train", "holdout")
    if collecting and args.candidate:
        raise ValueError("Classic drives the reference replay")
    cases = list(trials(args.suite))
    with TemporaryDirectory(prefix="m1-classic-") as scratch, ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        scenes = {}
        for seed, course, command, period in cases:
            if course.friction not in scenes:
                scene = Path(scratch) / f"floor-{course.friction}.xml"
                course.write(scene, M1_FLAT)
                scenes[course.friction] = scene
            jobs.append((args.candidate or args.classic, args.candidate is None, scenes[course.friction],
                         seed, course, command, period, collecting, .040 if args.suite == "train" else 0.))
        results = []
        for i, result in enumerate(pool.map(run_trial, jobs), 1):
            results.append(result)
            if i == len(jobs) or i % 24 == 0:
                print(args.suite, f"{i}/{len(jobs)}", flush=True)
    report = {"suite": args.suite, "policy": "candidate" if args.candidate else "hardware_classic",
              "sha256": NATIVE_SHA256 | {"hardware_classic": HARDWARE_CLASSIC_SHA256}}
    if args.candidate:
        report["sha256"]["candidate"] = sha256(args.candidate.read_bytes()).hexdigest()
    save_replays(args.out, cases, results, report, collecting)


if __name__ == "__main__":
    main()

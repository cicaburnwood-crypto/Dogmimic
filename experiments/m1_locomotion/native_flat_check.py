"""Validate flat keyboard steering, side travel and standing after key release."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort

from native_rollout import build_actor, reset_at_start
from native_world import Course
from sim import DT, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, Robot

DIAGONAL = .40 / np.sqrt(2)
COMMANDS = {
    "idle": (0., 0., 0.), "W_slow": (.15, 0., 0.), "W": (.40, 0., 0.), "W_fast": (.60, 0., 0.),
    "S": (-.40, 0., 0.), "Q": (0., 0., .35), "E": (0., 0., -.35),
    "A": (0., .15, 0.), "D": (0., -.15, 0.),
    "W+Q": (.40, 0., .35), "W+E": (.40, 0., -.35),
    "S+Q": (-.40, 0., .35), "S+E": (-.40, 0., -.35),
    "W+A": (DIAGONAL, .15, 0.), "W+D": (DIAGONAL, -.15, 0.),
}
TRIALS = ((71201, .87, False), (72131, .98, False),
          (73291, .92, True), (74517, 1.06, True))
RELEASE = 230
DRIFT_START = RELEASE + 75
END = RELEASE + 550


def record(checkpoint, scene, command, seed, filtered):
    robot = Robot(scene, M1_NAMES, .095)
    reset_at_start(robot, np.random.default_rng(seed))
    ort.set_seed(seed)
    actor = build_actor("candidate", checkpoint)
    target = np.asarray(command)
    zero = np.zeros(3)
    velocity_command = zero
    measured = []
    alpha = .99**10 if filtered else 0.
    for tick in range(END):
        requested = target if 50 <= tick < RELEASE else zero
        velocity_command = alpha*velocity_command + (1-alpha)*requested
        action = actor.action(actor.observe(robot, velocity_command))
        robot.step(M1_HOME + action*M1_SCALE, M1_KP, M1_KD)
        if not robot.valid():
            return {"completed": False}
        if 105 <= tick < RELEASE:
            measured.append(robot.velocity())
        if tick == DRIFT_START:
            parked = robot.data.qpos[:2].copy()
    observed = np.stack(measured)
    drift = np.linalg.norm(robot.data.qpos[:2] - parked)
    return {"completed": True, "stand_drift_m": float(drift),
            "mean_twist": observed.mean(axis=0).tolist(), "yaw_std_radps": float(observed[:, 2].std()),
            "twist_rmse": np.sqrt(np.mean((observed-target)**2, axis=0)).tolist()}


def compare(job):
    baseline, candidate, scene, label, command, seed, filtered = job
    return {"keys": label, "command": command, "seed": seed, "filtered": filtered,
            "before": record(baseline, scene, command, seed, filtered),
            "after": record(candidate, scene, command, seed, filtered)}


def summarize(rows, name):
    valid = [row for row in rows if row[name]["completed"]]
    turn = [row[name] for row in valid if row["command"][2]]
    side = [row[name] for row in valid if row["command"][1]]
    straight = [row[name] for row in valid if row["keys"] in ("W_slow", "W", "W_fast", "S")]
    drift = [row[name]["stand_drift_m"] for row in valid]
    idle = [row[name]["stand_drift_m"] for row in valid if row["keys"] == "idle"]
    rms = lambda records, col: float(np.sqrt(np.mean([r["twist_rmse"][col]**2 for r in records]))) if records else None
    return {"completed": len(valid), "count": len(rows),
            "stand_drift_median_m": float(np.median(drift)) if drift else None,
            "stand_drift_p90_m": float(np.quantile(drift, .9)) if drift else None,
            "idle_drift_max_m": max(idle, default=None),
            "turn_yaw_std_radps": float(np.mean([r["yaw_std_radps"] for r in turn])) if turn else None,
            "turn_yaw_rmse_radps": rms(turn, 2), "side_rmse_mps": rms(side, 1),
            "straight_forward_rmse_mps": rms(straight, 0)}


def acceptance(before, after):
    return (before["completed"] == after["completed"] == after["count"]
            and after["stand_drift_median_m"] < .6*before["stand_drift_median_m"]
            and after["idle_drift_max_m"] <= .020
            and after["turn_yaw_std_radps"] < .75*before["turn_yaw_std_radps"]
            and after["side_rmse_mps"] < .85*before["side_rmse_mps"]
            and after["straight_forward_rmse_mps"] <= .07)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="m1-flat-keyboard-") as tmp:
        jobs = []
        for seed, friction, filtered in TRIALS:
            scene = Path(tmp) / f"{seed}.xml"
            Course(friction=friction).write(scene, M1_FLAT)
            jobs.extend((args.baseline, args.candidate, scene, label, command, seed, filtered)
                        for label, command in COMMANDS.items())
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(compare, jobs)
    summary = {name: summarize(rows, name) for name in ("before", "after")}
    report = {"baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": sha256(args.candidate.read_bytes()).hexdigest(),
              "stand_drift_window_s": [(DRIFT_START-RELEASE)*DT, (END-RELEASE-1)*DT],
              "passed": acceptance(summary["before"], summary["after"]), "summary": summary, "records": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], **summary}, indent=2), flush=True)

"""Compare trunk centre motion and joint stability in level forward/reverse driving."""
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
from sim import DT, LEGS, M1_FLAT, M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, Robot


def record(checkpoint, path, speed, seed):
    ort.set_seed(seed)
    robot = Robot(path,M1_NAMES,.095)
    reset_at_start(robot,np.random.default_rng(seed))
    actor = build_actor("candidate",checkpoint)
    trunk = robot.model.body("trunk").id
    zero = np.zeros(3)
    command = np.array([speed,0.,0.])
    height, joint_vel, twist, legs = [], [], [], []
    for tick in range(385):
        moving = 70 <= tick < 310
        action = actor.action(actor.observe(robot,command if moving else zero))
        robot.step(M1_HOME+M1_SCALE*action,M1_KP,M1_KD)
        if not robot.valid():
            return {"completed": False}
        if moving:
            height.append(float(robot.data.xipos[trunk,2]))
            joint_vel.append(robot.data.qvel[robot.vadr[LEGS]].copy())
            legs.append(action[LEGS])
            if tick >= 115:
                twist.append(robot.velocity())
    body_vz = np.diff(height)/DT
    return {"completed": True, "com_range_m": float(np.ptp(height)),
            "com_vz_rms_mps": float(np.sqrt(np.mean(body_vz**2))),
            "com_path_m": float(np.abs(np.diff(height)).sum()),
            "leg_velocity_rms_radps": float(np.sqrt(np.mean(np.square(joint_vel)))),
            "leg_action_difference_rms": float(np.sqrt(np.mean(np.diff(legs,axis=0)**2))),
            "velocity_mean_mps": float(np.mean(twist,axis=0)[0]),
            "velocity_rmse_mps": float(np.sqrt(np.mean((np.asarray(twist)[:,0]-speed)**2)))}


def compare(job):
    baseline, candidate, path, speed, seed = job
    return {"speed_mps": speed, "seed": seed, "before": record(baseline,path,speed,seed),
            "after": record(candidate,path,speed,seed)}


def summarize(rows, name):
    groups = {}
    for side, sign in (("forward",1),("reverse",-1)):
        group = [r[name] for r in rows if sign*r["speed_mps"] > 0]
        valid = [r for r in group if r["completed"]]
        keys = ("com_range_m", "com_vz_rms_mps", "com_path_m", "leg_velocity_rms_radps",
                "leg_action_difference_rms", "velocity_rmse_mps")
        groups[side] = {"completed": len(valid), "count": len(group)}
        groups[side].update({key: float(np.mean([r[key] for r in valid])) if valid else None for key in keys})
    return groups


def acceptance(summary):
    for side in ("forward","reverse"):
        before, after = (summary[name][side] for name in ("before","after"))
        if (after["completed"] != after["count"] or after["velocity_rmse_mps"] > .07
                or after["com_vz_rms_mps"] >= before["com_vz_rms_mps"]
                or after["leg_velocity_rms_radps"] >= before["leg_velocity_rms_radps"]):
            return False
    return True


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("baseline",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--suite",choices=("selection","final"),required=True)
    p.add_argument("--workers",type=int,default=6)
    args = p.parse_args()
    trials = (((98211,.91),(98319,1.02)) if args.suite=="selection" else
              ((98831,.86),(98993,.94),(99067,1.00),(99141,1.07)))
    with TemporaryDirectory(prefix="m1-level-check-") as tmp:
        jobs = []
        for seed, friction in trials:
            scene = Path(tmp)/f"scene-{seed}.xml"
            Course(friction=friction).write(scene,M1_FLAT)
            jobs.extend((args.baseline,args.candidate,scene,sign*speed,seed)
                        for speed in (.16,.25,.40,.52,.60) for sign in (-1,1))
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(compare,jobs)
    summary = {name: summarize(rows,name) for name in ("before","after")}
    report = {"suite": args.suite, "baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": sha256(args.candidate.read_bytes()).hexdigest(),
              "summary": summary, "records": rows, "passed": acceptance(summary)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], **summary},indent=2),flush=True)

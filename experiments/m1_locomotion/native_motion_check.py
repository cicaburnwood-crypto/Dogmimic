"""Measure front-leg reach and repeated single-wheel lift on matched MuJoCo trials."""
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
from sim import M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_SCALE, Robot, body_matrix

FRONT = [0,2]
WHEEL_X = np.array([.28614,-.28586,.28614,-.28586])
COMMANDS = [(speed,0.,0.) for speed in (.20,.40,.60)] + [
    (-speed,0.,yaw) for speed in (.25,.40,.60) for yaw in (.35,-.35)
]
TRIALS = ((8703,.88,False),(8819,.95,True),(8933,1.06,False),(9041,.90,True))


def record(checkpoint, course, command, seed, filtered):
    ort.set_seed(seed)
    robot = Robot(course,M1_NAMES,.095)
    reset_at_start(robot,np.random.default_rng(seed))
    actor = build_actor("candidate",checkpoint)
    requested = np.asarray(command)
    zero = np.zeros(3)
    raw = zero.copy()
    reach, height, velocities = [], [], []
    ticks = 525
    completed = True
    for tick in range(ticks):
        if tick < 50 or tick >= 375:
            target = zero
        elif tick < 85 and command[0] < 0 and filtered:
            target = np.array([command[0],0.,0.])
        else:
            target = requested
        raw = (.99**10)*raw+(1-.99**10)*target if filtered else target
        action = actor.action(actor.observe(robot,raw))
        robot.step(M1_HOME+M1_SCALE*action,M1_KP,M1_KD)
        if not robot.valid():
            completed = False
            break
        if 120 <= tick < 375:
            pose = robot.data.qpos
            wheels = robot.data.xpos[robot.wheel_bodies]
            body_wheels = (wheels-pose[:3]) @ body_matrix(pose[3:7])
            reach.append(body_wheels[:,0]-WHEEL_X)
            height.append(wheels[:,2]-.095)
            velocities.append(robot.velocity())
    if not reach:
        raise RuntimeError("Trial fell before the motion scoring window")
    reach,height,velocities = map(np.asarray,(reach,height,velocities))
    exclusive = (height>.045) & ((height>.045).sum(axis=1,keepdims=True)==1)
    lift_events = ((~exclusive[:-1]) & exclusive[1:]).sum(axis=0)
    reached = reach[:,FRONT].max(axis=1)
    return {"completed":completed,"steady_samples":len(reach),"velocity_mean":velocities.mean(axis=0).tolist(),
            "velocity_rmse":np.sqrt(np.mean((velocities-requested)**2,axis=0)).tolist(),
            "front_reach_mean_m":float(reached.mean()),"front_reach_p95_m":float(np.quantile(reached,.95)),
            "reach_per_leg_m":reach.mean(axis=0).tolist(),"wheel_lift_p95_m":np.quantile(height,.95,axis=0).tolist(),
            "single_lift_fraction":float(exclusive.any(axis=1).mean()),"single_lift_events":lift_events.tolist()}


def compare(job):
    baseline,candidate,course,command,seed,filtered = job
    return {"command":command,"seed":seed,"filtered":filtered,
            "before":record(baseline,course,command,seed,filtered),
            "after":record(candidate,course,command,seed,filtered)}


def summarize(rows):
    result = {}
    for command in COMMANDS:
        values = [r for r in rows if tuple(r["command"])==command]
        kind = f"{command[0]:+.2f}/{command[2]:+.2f}"
        result[kind] = {}
        for version in ("before","after"):
            episodes = [r[version] for r in values]
            result[kind][version] = {"completed":sum(e["completed"] for e in episodes),
                "mean_forward_mps":float(np.mean([e["velocity_mean"][0] for e in episodes])),
                "mean_yaw_radps":float(np.mean([e["velocity_mean"][2] for e in episodes])),
                "front_reach_cm":100*float(np.mean([e["front_reach_mean_m"] for e in episodes])),
                "front_reach_p95_cm":100*float(np.mean([e["front_reach_p95_m"] for e in episodes])),
                "max_wheel_lift_p95_cm":100*max(max(e["wheel_lift_p95_m"]) for e in episodes),
                "single_lift_pct":100*float(np.mean([e["single_lift_fraction"] for e in episodes]))}
    return result


def acceptance(summary):
    for command in COMMANDS:
        before,after = (summary[f"{command[0]:+.2f}/{command[2]:+.2f}"][name] for name in ("before","after"))
        if after["completed"] != len(TRIALS):
            return False
        if command[0]>0:
            if (after["front_reach_cm"] > .80*before["front_reach_cm"] or
                    abs(after["mean_forward_mps"]-command[0])>.07 or after["max_wheel_lift_p95_cm"]>2.5):
                return False
        elif (after["max_wheel_lift_p95_cm"]>8.0 or
              after["mean_yaw_radps"]*np.sign(command[2])<.15 or after["mean_forward_mps"]>=-.15):
            return False
    return True


if __name__=="__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("baseline",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    args = p.parse_args()
    with TemporaryDirectory(prefix="m1-motion-check-") as tmp:
        jobs=[]
        for seed,friction,filtered in TRIALS:
            course = Path(tmp)/f"floor_{seed}.xml"
            Course(friction=friction).write(course)
            jobs.extend((args.baseline,args.candidate,course,u,seed,filtered) for u in COMMANDS)
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(compare,jobs)
    summary = summarize(rows)
    report = {"baseline_sha256":sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256":sha256(args.candidate.read_bytes()).hexdigest(),
              "passed":acceptance(summary),"count":len(rows),"summary":summary,"records":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2),flush=True)

"""Measure sustained level travel, including continued W after a descent."""
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

START = 60
STEADY = 240
CONTINUE = 790


def record(checkpoint, path, course, speed, seed, from_stairs, yaw, lateral):
    ort.set_seed(seed)
    robot = Robot(path,M1_NAMES,.095)
    rng = np.random.default_rng(seed)
    if from_stairs:
        reset_at_start(robot,rng,np.pi+yaw,lateral,course.descending_start,course.top+.54)
    else:
        reset_at_start(robot,rng,yaw)
    actor = build_actor("candidate",checkpoint)
    trunk = robot.model.body("trunk").id
    command = np.zeros(3)
    onset = None if from_stairs else START
    alpha = .99**10 if from_stairs else 0.
    height, velocity = [], []
    for tick in range(2700):
        request = speed if tick >= START else 0.
        command[0] = alpha*command[0] + (1-alpha)*request
        action = actor.action(actor.observe(robot,command))
        robot.step(M1_HOME + M1_SCALE*action,M1_KP,M1_KD)
        x,y,_ = robot.data.qpos[:3]
        if not robot.valid() or (from_stairs and x>course.edge-.1 and abs(y)>course.half_width-.10):
            return {"completed": False, "time_s": tick*DT}
        if onset is None and robot.data.xpos[robot.wheel_bodies,0].max() < course.edge-.12:
            onset = tick
        if onset is not None and tick >= onset+STEADY:
            height.append(float(robot.data.xipos[trunk,2]))
            velocity.append(float(robot.velocity()[0]))
            if tick >= onset+CONTINUE:
                break
    if onset is None or len(velocity) != CONTINUE-STEADY+1:
        return {"completed": False, "time_s": tick*DT}
    v = np.asarray(velocity)
    return {"completed": True, "time_s": tick*DT, "height_range_m": float(np.ptp(height)),
            "speed_std_mps": float(v.std()), "speed_mean_mps": float(v.mean()),
            "speed_90_span_mps": float(np.quantile(v,.95)-np.quantile(v,.05)),
            "speed_mean_error_mps": abs(float(v.mean())-speed)}


def compare(job):
    before, after, *args = job
    _,course,speed,seed,from_stairs,yaw,lateral = args
    return {"route": "down_to_level" if from_stairs else "level", "rise_m": course.rise,
            "seed": seed, "target_mps": speed, "yaw_rad": yaw, "start_lateral_m": lateral,
            "before": record(before,*args), "after": record(after,*args)}


def summarize(rows, side):
    done = [r[side] for r in rows if r[side]["completed"]]
    summary = {"count": len(rows), "completed": len(done)}
    for key in ("height_range_m","speed_std_mps","speed_90_span_mps","speed_mean_error_mps"):
        values = [r[key] for r in done]
        summary[key] = {"median": float(np.median(values)), "p90": float(np.quantile(values,.9))} if values else None
    return summary


def acceptance(summary):
    level = summary["level"]["after"]
    transition = summary["down_to_level"]["after"]
    baseline = summary["down_to_level"]["before"]
    return (all(s["count"]==s["completed"] for s in (level,transition))
            and level["height_range_m"]["p90"] < .006
            and level["speed_std_mps"]["p90"] < .010
            and transition["height_range_m"]["p90"] < .010
            and transition["height_range_m"]["p90"] < .50*baseline["height_range_m"]["p90"]
            and transition["speed_std_mps"]["p90"] < .012
            and transition["speed_std_mps"]["p90"] < .40*baseline["speed_std_mps"]["p90"]
            and all(s["speed_mean_error_mps"]["p90"] < .07 for s in (level,transition)))


def cases():
    for seed,friction,yaw in ((80401,.88,0.),(80407,1.04,np.pi-.04)):
        for speed in (.20,.40,.60,-.20,-.40,-.60):
            yield Course(friction=friction),speed,seed,False,yaw,0.
    settings = ((.90,0.,.075),(.94,.039,-.12),(1.03,-.065,.14),(.86,.083,.06))
    for rise in (.18,.20):
        for i,(friction,yaw,y) in enumerate(settings):
            for j,speed in enumerate((.36,.40,.44)):
                course = Course(rise=rise,steps=10,friction=friction)
                yield course,speed,81000+int(rise*100)*100+3*i+j,True,yaw,y


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("baseline",type=Path)
    p.add_argument("candidate",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    args = p.parse_args()
    with TemporaryDirectory(prefix="m1-cruise-check-") as tmp:
        scenes, jobs = {}, []
        for course,*case in cases():
            if course not in scenes:
                path = Path(tmp)/f"course-{len(scenes)}.xml"
                course.write(path,M1_FLAT)
                scenes[course] = path
            jobs.append((args.baseline,args.candidate,scenes[course],course,*case))
        with mp.get_context("fork").Pool(args.workers) as pool:
            rows = pool.map(compare,jobs)
    summary = {route: {side: summarize([r for r in rows if r["route"]==route],side) for side in ("before","after")}
               for route in ("level","down_to_level")}
    report = {"baseline_sha256": sha256(args.baseline.read_bytes()).hexdigest(),
              "candidate_sha256": sha256(args.candidate.read_bytes()).hexdigest(),
              "steady_window_s": [STEADY*DT,CONTINUE*DT], "summary": summary,
              "records": rows, "passed": acceptance(summary)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], **summary},indent=2),flush=True)

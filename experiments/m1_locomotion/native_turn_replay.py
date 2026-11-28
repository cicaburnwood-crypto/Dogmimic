"""Collect M1 rolling-turn episodes and score contact through command changes."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from native_classic_replay import END, START, STOP, run_trial, save_replays
from native_world import Course, HARDWARE_CLASSIC_SHA256, NATIVE_SHA256
from sim import M1_FLAT


def trials(suite):
    if suite == "train":
        first, frictions = 363000, (.76, .89, 1.06)
        speeds, turns = (.15,.25,.40,.60,.73,-.20,-.40,-.60), (.12,.23,.35)
    elif suite == "holdout":
        first, frictions = 364000, (.82,1.03)
        speeds, turns = (.20,.35,.55,.70,-.34,-.53), (.175,.35)
    elif suite == "selection":
        first, frictions = 365000, (.86,1.05)
        speeds, turns = (.15,.40,.60,.70,-.40), (.17,.35)
    else:
        first, frictions = 366000, (.84,.96,1.09)
        speeds, turns = (.15,.27,.40,.58,.70,-.35,-.55), (.12,.175,.26,.35)
    commands = [(vx,0.,sign*yaw) for vx in speeds for yaw in turns for sign in (-1,1)]
    commands.extend((v,0.,0.) for v in (.20,.40,.70,-.40))
    if suite in ("train","holdout"):
        commands.extend(((.40,.12,.24),(.40,-.12,-.24),(0.,0.,.35),(0.,0.,-.35)))
    for group, friction in enumerate(frictions):
        for i, command in enumerate(commands):
            seed = first + group*200 + i
            delay = (0,32,68)[(i+group)%3]
            release = (0,45)[(i+group)%2]
            profile = np.zeros((END+1,3))
            profile[START:STOP] = command
            profile[START:START+delay,2] = 0.
            if release:
                profile[STOP-release:STOP,2] = 0.
            yield seed, Course(friction=friction), command, profile, delay, release


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("classic", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--suite", choices=("train","holdout","selection","final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=0, help="Give a new policy replay a separate seed range")
    args = parser.parse_args()
    if sha256(args.classic.read_bytes()).hexdigest() != HARDWARE_CLASSIC_SHA256:
        raise ValueError("Expected the board's original Loco Classic checkpoint")
    collect = args.suite in ("train","holdout")
    cases = [(seed+args.seed_offset, *case) for seed,*case in trials(args.suite)]
    with TemporaryDirectory(prefix="m1-turn-replay-") as work, ProcessPoolExecutor(args.workers) as pool:
        scenes, jobs = {}, []
        for seed, course, command, schedule, _, _ in cases:
            if course not in scenes:
                scenes[course] = Path(work)/f"floor-{course.friction}.xml"
                course.write(scenes[course], M1_FLAT)
            noise = (.025,.05,.085)[seed%3] if args.suite=="train" else 0.
            jobs.append((args.candidate or args.classic, args.candidate is None, scenes[course], seed,
                         course, command, .8 if collect else .7, collect, noise, schedule))
        results = []
        for case, (record, sequence) in zip(cases, pool.map(run_trial,jobs), strict=True):
            record.update(turn_delay_ticks=case[-2], turn_release_ticks=case[-1])
            results.append((record,sequence))
            if len(results)%24==0 or len(results)==len(jobs):
                print(args.suite, f"{len(results)}/{len(jobs)}", flush=True)
    report = {"suite":"rolling-turn-"+args.suite, "policy":"candidate" if args.candidate else "hardware_classic",
              "sha256":NATIVE_SHA256 | {"hardware_classic": HARDWARE_CLASSIC_SHA256}}
    if args.candidate:
        report["sha256"]["candidate"] = sha256(args.candidate.read_bytes()).hexdigest()
    save_replays(args.out, cases, results, report, collect)


if __name__=="__main__":
    main()

"""Benchmark protocol of Appendix F: every clip plays once from its first frame, foot
friction 1.0, no domain randomization or observation noise, deterministic policy.
A clip fails if SuperMimic's training termination triggers. Normalized return is the
undiscounted reward sum divided by the largest attainable sum for that clip.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from .motion_lib import MotionLibrary
from .sim_env import EnvCfg, TrackingSimEnv
from .task import GOAL_FRAME_DIM, OBS_DIM, TaskCfg


def make_eval_env(lib, num_envs, device):
  cfg = EnvCfg(num_envs=num_envs, episode_length_s=1e9, obs_noise=False, reset_noise=False, domain_rand=True)
  env = TrackingSimEnv(lib, cfg, TaskCfg(), device)
  env.eval_mode = True
  return env


@torch.no_grad()
def evaluate(act_fn, lib: MotionLibrary, env: TrackingSimEnv, clip_ids=None, verbose=False):
  device = env.device
  n = env.num_envs
  if clip_ids is None:
    clip_ids = torch.arange(lib.num_clips, device=device)
  # Longest first so chunks finish together.
  order = clip_ids[torch.argsort(lib.lengths[clip_ids], descending=True)]
  ret = torch.zeros(lib.num_clips, device=device)
  failed = torch.zeros(lib.num_clips, dtype=torch.bool, device=device)
  steps_done = torch.zeros(lib.num_clips, dtype=torch.long, device=device)
  tcfg = env.tcfg
  t0 = time.time()
  env.reset()
  for c0 in range(0, len(order), n):
    clips = order[c0 : c0 + n]
    k = len(clips)
    ids = torch.arange(k, device=device)
    env.reset_envs(ids, frames=lib.starts[clips])
    env._finish_resets(ids, force=True)
    active = torch.zeros(n, dtype=torch.bool, device=device)
    active[:k] = True
    acc = torch.zeros(n, device=device)
    nsteps = torch.zeros(n, dtype=torch.long, device=device)
    while active.any():
      obs, goal = env.policy_inputs()
      a = act_fn(obs, goal)
      _, info = env.step(a)
      acc += info["reward"] * active
      nsteps += active.long()
      fin = info["done"] & active
      if fin.any():
        idx = fin.nonzero().squeeze(-1)
        cl = clips[idx]
        ret[cl] = acc[idx]
        failed[cl] = info["terminated"][idx]
        steps_done[cl] = nsteps[idx]
        active[idx] = False
    if verbose:
      print(f"  clips {c0 + k}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
  sel = clip_ids
  max_ret = tcfg.max_step_reward * (lib.lengths[sel] - 1).float()
  norm = (ret[sel] / max_ret).clamp(max=1.0)
  return {
    "norm_return": norm.mean().item(),
    "norm_return_frame_weighted": (ret[sel].sum() / max_ret.sum()).item(),
    "failed": int(failed[sel].sum().item()),
    "num_clips": int(len(sel)),
    "per_clip_norm_return": norm.tolist(),
    "per_clip_failed": failed[sel].tolist(),
    "eval_seconds": time.time() - t0,
  }


def load_supermimic_policy(path, device):
  from .networks import Policy

  ck = torch.load(path, map_location=device)
  tc = TaskCfg()
  pol = Policy(OBS_DIM * tc.history, GOAL_FRAME_DIM * tc.future).to(device)
  pol.load_state_dict(ck["policy"])
  pol.eval()
  return pol, ck


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, help="run directory with ckpt_*.pt")
  ap.add_argument("--motions", default="data/processed/lafan1_50hz.npz")
  ap.add_argument("--num-envs", type=int, default=0)
  ap.add_argument("--watch", action="store_true", help="keep evaluating new checkpoints")
  ap.add_argument("--kind", default="supermimic", choices=["supermimic", "ppo"])
  ap.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
  ap.add_argument("--out", default="eval.jsonl")
  args = ap.parse_args()
  device = "cuda"
  from . import networks

  networks.AMP["enabled"] = False
  torch.backends.cuda.matmul.allow_tf32 = True
  lib = MotionLibrary(args.motions, device)
  n = args.num_envs or min(lib.num_clips, 4096)
  env = make_eval_env(lib, n, device)
  out_path = os.path.join(args.run, args.out)
  done = set()
  if os.path.exists(out_path):
    for line in open(out_path):
      done.add(json.loads(line)["ckpt"])
  while True:
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")))
    ckpts = [c for i, c in enumerate(ckpts) if i % args.every == args.every - 1 or c == ckpts[-1]]
    todo = [c for c in ckpts if os.path.basename(c) not in done]
    for c in todo:
      if args.kind == "supermimic":
        pol, ck = load_supermimic_policy(c, device)
        act = lambda o, g: pol(o, g)
        meta = {"it": ck["it"], "hours": ck["train_time"] / 3600, "env_steps": ck["env_steps"]}
      else:
        from .ppo import load_ppo_policy

        act, meta = load_ppo_policy(c, device)
      res = evaluate(act, lib, env, verbose=lib.num_clips > n)
      rec = {"ckpt": os.path.basename(c), **meta, **{k: v for k, v in res.items() if not k.startswith("per_clip")}}
      print(json.dumps(rec), flush=True)
      with open(out_path, "a") as f:
        f.write(json.dumps({**rec, "per_clip_norm_return": res["per_clip_norm_return"], "per_clip_failed": res["per_clip_failed"]}) + "\n")
      done.add(os.path.basename(c))
    if not args.watch:
      break
    if os.path.exists(os.path.join(args.run, "DONE")) and not todo:
      break
    time.sleep(60)


if __name__ == "__main__":
  main()

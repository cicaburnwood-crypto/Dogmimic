"""Reference point: PD setpoint = next reference joint angles (no feedback).
Also evaluates a checkpoint if given."""

import sys

import torch

from supermimic.evaluate import evaluate, load_supermimic_policy, make_eval_env
from supermimic.motion_lib import MotionLibrary


def main():
  dev = "cuda"
  lib = MotionLibrary("data/processed/lafan1_50hz.npz", dev)
  env = make_eval_env(lib, lib.num_clips, dev)

  def ff(obs, goal):
    nxt = lib.q[lib.clamp_to_clip(env.frame + 1, env.frame)]
    return (nxt - env.q0) / env.action_scale

  res = evaluate(ff, lib, env)
  print("feedforward:", {k: v for k, v in res.items() if not k.startswith("per_clip")})
  print("  per clip:", [round(x, 3) for x in res["per_clip_norm_return"]])
  zero = evaluate(lambda o, g: torch.zeros(o.shape[0], 29, device=dev), lib, env)
  print("zero action:", {k: v for k, v in zero.items() if not k.startswith("per_clip")})
  for path in sys.argv[1:]:
    pol, ck = load_supermimic_policy(path, dev)
    res = evaluate(lambda o, g: pol(o, g), lib, env)
    print(path, ck["it"], {k: v for k, v in res.items() if not k.startswith("per_clip")})


if __name__ == "__main__":
  main()

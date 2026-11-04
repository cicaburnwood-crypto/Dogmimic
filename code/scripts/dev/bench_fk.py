"""Benchmark FK (forward + backward) eager vs torch.compile."""

import time

import torch

from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg, TrackingSimEnv
from supermimic.task import TaskCfg


def bench(name, fn, n=20):
  for _ in range(3):
    fn()
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  print(f"{name}: {(time.time() - t0) / n * 1000:.2f} ms", flush=True)


def main():
  dev = "cuda"
  lib = MotionLibrary("data/processed/lafan1_50hz.npz", dev)
  env = TrackingSimEnv(lib, EnvCfg(num_envs=64), TaskCfg(), dev)
  kin = env.kin
  B = 4096
  idx = torch.randint(0, lib.num_frames, (B,), device=dev)
  ref = lib.frame(idx)
  args = [ref[k].clone().requires_grad_(True) for k in ("pos", "rot", "q", "v", "w", "qd")]

  def fwd(f):
    with torch.no_grad():
      f(*args)

  def fwdbwd(f):
    P, R, V, W = f(*args)
    (P.sum() + R.sum() + V.sum() + W.sum()).backward()

  from supermimic.kinematics import FastTreeKinematics
  fk2 = FastTreeKinematics(env.mj_model, dev, root_body=env._find_body("pelvis"))
  cf2 = torch.compile(fk2.forward, dynamic=False)
  t0 = time.time(); fwdbwd(cf2); print("compile2 time", time.time() - t0)
  bench("fast compiled fwd", lambda: fwd(cf2))
  bench("fast compiled fwd+bwd", lambda: fwdbwd(cf2))
  print("fast max diff", max((a - b).abs().max().item() for a, b in zip(kin.forward(*args), cf2(*args))))
  bench("eager fwd", lambda: fwd(kin.forward))
  bench("eager fwd+bwd", lambda: fwdbwd(kin.forward))
  cf = torch.compile(kin.forward, dynamic=False)
  t0 = time.time()
  fwdbwd(cf)
  print("compile time", time.time() - t0)
  bench("compiled fwd", lambda: fwd(cf))
  bench("compiled fwd+bwd", lambda: fwdbwd(cf))
  out_e = kin.forward(*args)
  out_c = cf(*args)
  print("max diff", max((a - b).abs().max().item() for a, b in zip(out_e, out_c)))
  # Reward + features compiled too.
  task = env.task
  s = {"pos": args[0], "rot": args[1], "q": args[2], "v": args[3], "w": args[4], "qd": args[5]}
  a = torch.zeros(B, 29, device=dev)

  def rew(s):
    k = task.body_kin(s)
    rk = kin.forward(ref["pos"], ref["rot"], ref["q"], ref["v"], ref["w"], ref["qd"])
    r, _ = task.reward(s, k, ref, rk, a, a)
    f = task.state_features(s, k)
    return r.sum() + f.sum()

  bench("eager reward+feat fwd+bwd", lambda: rew(s).backward())
  crew = torch.compile(rew, dynamic=False)
  t0 = time.time()
  crew(s).backward()
  print("compile time", time.time() - t0)
  bench("compiled reward+feat fwd+bwd", lambda: crew(s).backward())


if __name__ == "__main__":
  main()

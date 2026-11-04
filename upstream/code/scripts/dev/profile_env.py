"""Time the pieces of an env step."""

import sys
import time

import torch

from mjlab.managers.event_manager import RecomputeLevel
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg, TrackingSimEnv
from supermimic.task import TaskCfg


def timeit(name, fn, n=20):
  fn()
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  print(f"{name}: {(time.time() - t0) / n * 1000:.2f} ms")


def main():
  dev = "cuda"
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
  lib = MotionLibrary("data/processed/lafan1_50hz.npz", dev)
  env = TrackingSimEnv(lib, EnvCfg(num_envs=n), TaskCfg(), dev)
  env.reset()
  print("cuda graph:", env.sim.use_cuda_graph)
  timeit("sim.step x4", lambda: [env.sim.step() for _ in range(4)])
  timeit("sim.forward", env.sim.forward)
  timeit("set_const", lambda: env.sim.recompute_constants(RecomputeLevel.set_const))
  timeit("read_state", env.read_state)
  s = env.read_state()
  timeit("fk", lambda: env.task.body_kin(s))
  ids = torch.arange(0, n, 50, device=dev)
  timeit("reset_envs(2%)", lambda: env.reset_envs(ids))
  timeit("policy_inputs", env.policy_inputs)
  a = torch.zeros(n, 29, device=dev)
  timeit("env.step", lambda: env.step(a))


if __name__ == "__main__":
  main()

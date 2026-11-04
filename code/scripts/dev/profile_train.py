"""torch.profiler breakdown of one world-model update and one policy update."""

import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from supermimic.train import Trainer, TrainCfg


def main():
  cfg = TrainCfg(out="runs/profile", warmup_steps=32)
  tr = Trainer(cfg)
  for _ in range(cfg.warmup_steps):
    tr.collect()
  for _ in range(3):
    tr.update_wm()
    tr.update_policy()
  torch.cuda.synchronize()
  for name, fn in (("wm", tr.update_wm), ("policy", tr.update_policy), ("collect", tr.collect)):
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    print(name, "wall", time.time() - t0)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      fn()
      torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    ev = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in ev) / 1e3
    print(f"{name}: total self CUDA time {cuda_total:.1f} ms")


if __name__ == "__main__":
  main()

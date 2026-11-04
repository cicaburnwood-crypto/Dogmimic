"""One-line-per-run summary of training logs and benchmark evaluations."""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def last_json(path):
  if not os.path.exists(path):
    return None
  last = None
  with open(path) as f:
    for line in f:
      if line.strip():
        last = line
  return json.loads(last) if last else None


def first_cross(path, key, thr):
  if not os.path.exists(path):
    return None
  for line in open(path):
    d = json.loads(line)
    if d.get(key, -1) >= thr:
      return d["hours"]
  return None


for run in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
  if not os.path.isdir(run):
    continue
  name = os.path.basename(run)
  log = last_json(os.path.join(run, "log.jsonl"))
  if log is None:
    continue
  ev = last_json(os.path.join(run, "eval.jsonl"))
  best = None
  if os.path.exists(os.path.join(run, "eval.jsonl")):
    best = max(json.loads(l)["norm_return"] for l in open(os.path.join(run, "eval.jsonl")) if l.strip())
  t82 = first_cross(os.path.join(run, "log.jsonl"), "ep_norm_return", 0.82)
  s = f"{name:28s} it {log['it']:>7} h {log['hours']:6.2f} train_ret {log.get('ep_norm_return', 0):.3f}"
  if t82 is not None:
    s += f" (0.82 @ {t82:.2f} h)"
  if ev:
    s += f" | eval {ev['norm_return']:.3f} @ {ev['hours']:.2f} h, failed {ev['failed']}/{ev['num_clips']}, best {best:.3f}"
  print(s)

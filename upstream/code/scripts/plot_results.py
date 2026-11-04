"""Plots for the reproduction (static PNGs for notes/reports).

fig_lafan_return.png  — Fig. 3a analogue: benchmark normalized return vs training hours,
                        ours (A6000) solid, the paper's curves (H100) dashed.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Paper Fig. 3a (LAFAN1, 40 clips) curves, read from the TeX source.
PAPER_SM_1GPU = [(0.0021125, 0.001519), (0.077808, 0.524), (0.151899, 0.6166), (0.225991, 0.6672), (0.299996, 0.67), (0.373919, 0.696), (0.44801, 0.7656), (0.522101, 0.7565), (0.596276, 0.777), (0.670791, 0.779), (0.745049, 0.7909), (0.894413, 0.7894), (1.04353, 0.8102), (1.19272, 0.8331), (1.342, 0.8309), (1.49103, 0.8483), (1.79009, 0.8282), (2.08984, 0.8519), (2.38983, 0.8509), (2.69008, 0.8616), (2.98771, 0.8661)]
PAPER_PPO_1GPU = [(0.00236111, 0.001764), (0.24, 0.025203), (0.479583, 0.21044), (0.720972, 0.2256), (0.960417, 0.2352), (1.19861, 0.2651), (1.43987, 0.3023), (1.67875, 0.3301), (1.91973, 0.3662), (2.16125, 0.3731), (2.40778, 0.4068), (2.65042, 0.4044), (2.88959, 0.3948), (3.1318, 0.43), (3.37181, 0.4195), (3.61264, 0.449), (3.85361, 0.4474), (4.09305, 0.4878), (4.32972, 0.4612), (4.57042, 0.4728), (4.81069, 0.4795), (5.0457, 0.4707), (5.28334, 0.534), (5.52069, 0.5233), (5.75889, 0.4984), (5.99695, 0.5348), (7.18611, 0.5412), (8.38444, 0.5449), (9.62625, 0.5499), (11.9934, 0.5881), (14.0222, 0.6166), (18.6919, 0.602)]


def read_eval(run):
  p = os.path.join(ROOT, "runs", run, "eval.jsonl")
  if not os.path.exists(p):
    return []
  out = []
  for line in open(p):
    d = json.loads(line)
    out.append((d["hours"], d["norm_return"], d["failed"], d["num_clips"]))
  return sorted(out)


def style(ax):
  ax.set_facecolor(SURFACE)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
  ax.tick_params(colors=INK2, labelsize=9)
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)


def fig_lafan(out):
  fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
  style(ax)
  series = [
    ("smimic_lafan1_1gpu", "SuperMimic, 1×A6000 (ours)", BLUE, "-"),
    ("ppo_lafan1_1gpu", "PPO-Sim, 1×A6000 (ours)", ORANGE, "-"),
  ]
  for run, label, color, ls in series:
    pts = read_eval(run)
    if pts:
      ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, lw=2, marker="o", ms=4, label=label)
  ax.plot(*zip(*PAPER_SM_1GPU), "--", color=BLUE, lw=1.5, alpha=0.6, label="SuperMimic, 1×H100 (paper)")
  ax.plot(*zip(*PAPER_PPO_1GPU), "--", color=ORANGE, lw=1.5, alpha=0.6, label="PPO-Sim, 1 GPU (paper)")
  ax.axhline(0.82, color=INK2, lw=1, ls=":")
  ax.text(0.02, 0.825, "target 0.82", color=INK2, fontsize=8, transform=ax.get_yaxis_transform())
  ax.set_xlabel("Training wall-clock time (h)", color=INK)
  ax.set_ylabel("Normalized return (LAFAN1, 40 clips)", color=INK)
  ax.set_ylim(0, 1)
  ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
  ax.set_title("Benchmark return vs. training time", color=INK, fontsize=11, loc="left")
  fig.tight_layout()
  fig.savefig(out, dpi=150, facecolor=SURFACE)
  print("wrote", out)


if __name__ == "__main__":
  os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
  fig_lafan(os.path.join(ROOT, "results", "fig_lafan_return.png"))

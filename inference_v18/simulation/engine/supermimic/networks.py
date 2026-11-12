"""World model, policy and critic networks (sizes from Table A.3 of the paper)."""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

# Mixed precision for the MLP matmuls only (set by the trainer). Integration,
# kinematics, rewards and losses stay in fp32.
AMP = {"enabled": False}
# Data-parallel world size (set by the trainer); normalizer statistics are all-reduced.
DIST = {"world": 1}


def amp_ctx():
  if AMP["enabled"]:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
  return contextlib.nullcontext()


def run_mlp(net, x):
  with amp_ctx():
    y = net(x)
  return y.float()

from .rotations import axis_angle_to_mat, heading_mat, mat_to_axis_angle, mm, mtm, mtv, mv
from .task import NJ, Task

STATE_DELTA_DIM = 3 + 3 + 3 + 3 + NJ + NJ  # 70


def mlp(inp, hidden, out, act=nn.ELU, out_gain=1.0):
  layers = []
  d = inp
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  last = nn.Linear(d, out)
  if out_gain != 1.0:
    nn.init.orthogonal_(last.weight, gain=out_gain)
    nn.init.zeros_(last.bias)
  layers.append(last)
  return nn.Sequential(*layers)


class Normalizer(nn.Module):
  """Running mean/std. Statistics are buffers, so gradients pass through unchanged."""

  def __init__(self, dim, eps=1e-2, until=None, clip=10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros((), dtype=torch.float64))
    self.eps = eps
    self.until = until
    self.clip = clip

  @torch.no_grad()
  def update(self, x):
    x = x.reshape(-1, x.shape[-1]).float()
    if self.until is not None and self.count >= self.until:
      return
    if DIST["world"] > 1:
      # Identical statistics on every rank: global mean, then global centered variance.
      import torch.distributed as dist

      nt = torch.tensor([float(x.shape[0])], device=x.device, dtype=torch.float64)
      sm = x.sum(0).double()
      dist.all_reduce(nt)
      dist.all_reduce(sm)
      m = (sm / nt).float()
      ss = ((x - m) ** 2).sum(0).double()
      dist.all_reduce(ss)
      v = (ss / nt).float()
      n = float(nt.item())
    else:
      n = x.shape[0]
      m = x.mean(0)
      v = x.var(0, unbiased=False)
    tot = self.count + n
    w = n / tot
    delta = m - self.mean
    self.mean += (w * delta).float()
    self.var = (self.var * (1 - w) + v * w + delta**2 * (w * (1 - w))).float()
    self.count = tot

  @property
  def std(self):
    return torch.sqrt(self.var + self.eps**2)

  def forward(self, x):
    y = (x - self.mean) / self.std
    if self.clip is not None:
      y = y.clamp(-self.clip, self.clip)
    return y


class WorldModel(nn.Module):
  """Deterministic dynamics f(s, a, xi) -> s' with an MLP on heading-frame features.

  The network predicts the normalized state change; integration back to the world
  frame is analytic. Reward, observation and termination heads are analytic
  (supermimic.task), so only the dynamics is learned.
  """

  def __init__(self, task: Task, feat_dim: int, xi_dim: int, hidden=(2048, 2048, 2048)):
    super().__init__()
    self.task = task
    in_dim = feat_dim + NJ + xi_dim
    self.net = mlp(in_dim, hidden, STATE_DELTA_DIM, act=nn.SiLU, out_gain=0.01)
    self.in_norm = Normalizer(in_dim, eps=1e-3)
    self.out_norm = Normalizer(STATE_DELTA_DIM, eps=1e-4)
    self.xi_dim = xi_dim

  def inputs(self, s, a, xi, feats=None):
    if feats is None:
      feats = self.task.features_fast(s)
    return torch.cat([feats, a, xi], dim=-1)

  def step(self, s, a, xi, feats=None):
    x = self.inputs(s, a, xi, feats)
    d = run_mlp(self.net, self.in_norm(x)) * self.out_norm.std + self.out_norm.mean
    return apply_delta_c(s, d)


def state_delta(s, s1):
  """Change from s to s1 in the parameterization the world model predicts."""
  H = heading_mat(s["rot"])
  return torch.cat(
    [
      mtv(H, s1["pos"] - s["pos"]),
      mat_to_axis_angle(mtm(s["rot"], s1["rot"])),
      mtv(H, s1["v"] - s["v"]),
      mtv(s1["rot"], s1["w"]) - mtv(s["rot"], s["w"]),
      s1["q"] - s["q"],
      s1["qd"] - s["qd"],
    ],
    dim=-1,
  )


def apply_delta(s, d):
  H = heading_mat(s["rot"])
  R1 = mm(s["rot"], axis_angle_to_mat(d[..., 3:6]))
  w_loc = mtv(s["rot"], s["w"]) + d[..., 9:12]
  return {
    "pos": s["pos"] + mv(H, d[..., 0:3]),
    "rot": R1,
    "v": s["v"] + mv(H, d[..., 6:9]),
    "w": mv(R1, w_loc),
    "q": s["q"] + d[..., 12 : 12 + NJ],
    "qd": s["qd"] + d[..., 12 + NJ :],
  }


def state_error(pred, true):
  """Error of a predicted state, in the true state's heading frame (70 dims)."""
  H = heading_mat(true["rot"])
  return torch.cat(
    [
      mtv(H, pred["pos"] - true["pos"]),
      mat_to_axis_angle(mtm(true["rot"], pred["rot"])),
      mtv(H, pred["v"] - true["v"]),
      mtv(pred["rot"], pred["w"]) - mtv(true["rot"], true["w"]),
      pred["q"] - true["q"],
      pred["qd"] - true["qd"],
    ],
    dim=-1,
  )


class Policy(nn.Module):
  """Motion encoder -> 32-d tanh latent -> shared decoder with proprioception."""

  def __init__(self, obs_dim, goal_dim, latent=32, enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256)):
    super().__init__()
    self.obs_norm = Normalizer(obs_dim)
    self.goal_norm = Normalizer(goal_dim)
    self.encoder = mlp(goal_dim, enc_hidden, latent)
    self.decoder = mlp(latent + obs_dim, dec_hidden, NJ, out_gain=0.01)

  def encode(self, goal):
    return torch.tanh(run_mlp(self.encoder, self.goal_norm(goal)))

  def forward(self, obs, goal):
    z = self.encode(goal)
    return run_mlp(self.decoder, torch.cat([z, self.obs_norm(obs)], dim=-1))


class Critic(nn.Module):
  def __init__(self, in_dim, hidden=(256, 128)):
    super().__init__()
    self.norm = Normalizer(in_dim)
    self.net = mlp(in_dim, hidden, 1)

  def forward(self, x):
    return run_mlp(self.net, self.norm(x)).squeeze(-1)


apply_delta_c = torch.compile(apply_delta, dynamic=False)
state_error_c = torch.compile(state_error, dynamic=False)
state_delta_c = torch.compile(state_delta, dynamic=False)

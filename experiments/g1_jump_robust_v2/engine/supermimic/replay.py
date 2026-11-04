"""Per-environment ring buffer of simulator transitions.

Each slot holds (s_t, a_t, s_{t+1}, xi, frame_t, episode id). s_{t+1} is the
pre-reset successor, so the terminal transition of an episode is kept. A k-step
window is valid when all its slots share one episode id.
"""

from __future__ import annotations

import torch

from .task import NJ, flat_to_state

SDIM = 71


class ReplayBuffer:
  def __init__(self, num_envs, capacity, xi_dim, device):
    self.n, self.L, self.device = num_envs, capacity, device
    self.s = torch.zeros(capacity, num_envs, SDIM, device=device)
    self.s1 = torch.zeros(capacity, num_envs, SDIM, device=device)
    self.a = torch.zeros(capacity, num_envs, NJ, device=device)
    self.xi = torch.zeros(capacity, num_envs, xi_dim, device=device)
    self.frame = torch.zeros(capacity, num_envs, dtype=torch.long, device=device)
    self.ep = torch.full((capacity, num_envs), -1, dtype=torch.long, device=device)
    self.ptr = 0
    self.size = 0

  def add(self, s, a, s1, xi, frame, ep):
    p = self.ptr
    self.s[p], self.a[p], self.s1[p] = s, a, s1
    self.xi[p], self.frame[p], self.ep[p] = xi, frame, ep
    self.ptr = (p + 1) % self.L
    self.size = min(self.size + 1, self.L)

  def _age_index(self, age):
    """Ring index of the slot `age` steps before the newest one (age 0 = newest)."""
    return (self.ptr - 1 - age) % self.L

  def sample_windows(self, batch, k):
    """Returns s0 (B,71), actions (B,k,29), targets (B,k,71), xi (B,dim), mask (B,)."""
    assert self.size >= k
    env = torch.randint(0, self.n, (batch,), device=self.device)
    # Start age so the window [age, age-k+1] (oldest to newest) is in the buffer.
    age0 = torch.randint(k - 1, self.size, (batch,), device=self.device)
    ages = age0[:, None] - torch.arange(k, device=self.device)  # oldest first
    idx = self._age_index(ages)
    e = env[:, None].expand(-1, k)
    ep = self.ep[idx, e]
    mask = (ep == ep[:, :1]).all(-1) & (ep[:, 0] >= 0)
    return self.s[idx[:, 0], env], self.a[idx, e], self.s1[idx, e], self.xi[idx[:, 0], env], mask

  def sample_starts(self, batch, history, task):
    """Start states for imagination with their noise-free observation history."""
    env = torch.randint(0, self.n, (batch,), device=self.device)
    age = torch.randint(0, self.size, (batch,), device=self.device)
    t = self._age_index(age)
    s0 = self.s[t, env]
    xi = self.xi[t, env]
    frame = self.frame[t, env]
    ep0 = self.ep[t, env]
    # History slots j = 0..history-1 are ages age+history-1-j .. age.
    back = torch.arange(history - 1, -1, -1, device=self.device)
    ages = (age[:, None] + back).clamp(max=self.size - 1)
    idx = self._age_index(ages)
    e = env[:, None].expand(-1, history)
    same = (self.ep[idx, e] == ep0[:, None]) & ((age[:, None] + back) < self.size)
    # Slots before the episode start repeat its first observation.
    first = same.float().argmax(-1)
    idx = torch.where(same, idx, idx.gather(1, first[:, None]).expand(-1, history))
    s_h = self.s[idx, e]
    prev_idx = (idx - 1) % self.L
    prev_ok = (self.ep[prev_idx, e] == ep0[:, None]) & (idx != self._age_index(torch.full_like(idx, self.size - 1)))
    a_prev = torch.where(prev_ok[..., None], self.a[prev_idx, e], torch.zeros_like(self.a[prev_idx, e]))
    st = flat_to_state(s_h)
    bias = xi[:, None, 1 : 1 + NJ]
    obs_hist = task.obs_clean(st, a_prev, bias.expand(-1, history, -1))
    return s0, xi, frame, obs_hist, a_prev[:, -1]

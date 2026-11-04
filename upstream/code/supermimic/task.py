"""Motion-tracking MDP pieces shared by the simulator and the world-model imagination.

State (world frame, relative to the env origin):
  pos (B,3) root position      rot (B,3,3) root rotation
  q   (B,29) joint positions   v (B,3) root linear velocity
  w   (B,3) root angular velocity   qd (B,29) joint velocities
Flat layout (71): pos 3 | quat 4 | q 29 | v 3 | w 3 | qd 29

Everything here is differentiable so the analytic reward, observation and
termination heads can sit on top of the learned dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import TreeKinematics
from .rotations import heading_mat, mat_to_axis_angle, mat_to_quat, mm, mmt, mtm, mtv, mv, quat_to_mat, rot6d, yaw_from_mat, yaw_mat

TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

NJ = 29
OBS_DIM = NJ + NJ + 3 + 3 + NJ  # 93
GOAL_FRAME_DIM = 3 + 6 + NJ + NJ + 3 + 3  # 73
XI_DIM = 1 + NJ + 3


@dataclass
class TaskCfg:
  dt: float = 0.02
  history: int = 10
  future: int = 10
  # Reward (BeyondMimic, no self-collision term, joint-limit weight -2).
  w_body_pos: float = 1.0
  w_body_ori: float = 1.0
  w_body_lin: float = 1.0
  w_body_ang: float = 1.0
  w_root_pos: float = 0.5
  w_root_ori: float = 0.5
  w_action_rate: float = -0.1
  w_joint_limit: float = -2.0
  std_body_pos: float = 0.3
  std_body_ori: float = 0.4
  std_body_lin: float = 1.0
  std_body_ang: float = 3.14
  std_root_pos: float = 0.3
  std_root_ori: float = 0.4
  # Termination.
  term_z: float = 0.25
  term_tilt: float = 0.8
  soft_temp_z: float = 0.02
  soft_temp_tilt: float = 0.04
  # Observation noise (uniform half-widths).
  noise_q: float = 0.01
  noise_qd: float = 0.5
  noise_gyro: float = 0.2
  noise_grav: float = 0.05

  @property
  def max_step_reward(self) -> float:
    return self.dt * (
      self.w_body_pos + self.w_body_ori + self.w_body_lin + self.w_body_ang + self.w_root_pos + self.w_root_ori
    )


def flat_to_state(x: torch.Tensor) -> dict:
  return {
    "pos": x[..., 0:3],
    "rot": quat_to_mat(x[..., 3:7]),
    "q": x[..., 7:36],
    "v": x[..., 36:39],
    "w": x[..., 39:42],
    "qd": x[..., 42:71],
  }


def state_to_flat(s: dict) -> torch.Tensor:
  return torch.cat([s["pos"], mat_to_quat(s["rot"]), s["q"], s["v"], s["w"], s["qd"]], dim=-1)


class Task:
  def __init__(self, kin: TreeKinematics, q0, soft_limits, action_scale, cfg: TaskCfg, device, compile_sizes=()):
    self.kin = kin
    self.compile_sizes = set(compile_sizes)
    self._compiled = {}
    self.cfg = cfg
    self.device = device
    self.q0 = q0.to(device)
    self.qmin = soft_limits[:, 0].to(device)
    self.qmax = soft_limits[:, 1].to(device)
    self.action_scale = action_scale.to(device)
    self.tracked = torch.tensor(kin.body_index(TRACKED_BODIES), device=device)
    self.root = 0  # pelvis
    self.obs_noise = torch.cat(
      [
        torch.full((NJ,), cfg.noise_q),
        torch.full((NJ,), cfg.noise_qd),
        torch.full((3,), cfg.noise_gyro),
        torch.full((3,), cfg.noise_grav),
        torch.zeros(NJ),
      ]
    ).to(device)
    self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)

  # Kinematics.

  def body_kin(self, s: dict):
    return self.kin.forward(s["pos"], s["rot"], s["q"], s["v"], s["w"], s["qd"])

  # Observation.

  def obs_clean(self, s: dict, prev_action: torch.Tensor, enc_bias: torch.Tensor) -> torch.Tensor:
    gyro = mtv(s["rot"], s["w"])
    grav = mtv(s["rot"], self.gravity.expand_as(s["w"]))
    return torch.cat([s["q"] - self.q0 + enc_bias, s["qd"], gyro, grav, prev_action], dim=-1)

  def sample_obs_noise(self, shape) -> torch.Tensor:
    return (torch.rand(shape + (OBS_DIM,), device=self.device) * 2 - 1) * self.obs_noise

  # Goal: K future reference frames, expressed relative to the robot root.

  def goal(self, s: dict, ref_future: dict) -> torch.Tensor:
    """ref_future tensors have shape (B, K, ...). Returns (B, K * GOAL_FRAME_DIM)."""
    R = s["rot"].unsqueeze(-3)  # (B,1,3,3)
    dp = mtv(R, ref_future["pos"] - s["pos"].unsqueeze(-2))
    drot = rot6d(mtm(R, ref_future["rot"]))
    v_loc = mtv(ref_future["rot"], ref_future["v"])
    w_loc = mtv(ref_future["rot"], ref_future["w"])
    g = torch.cat([dp, drot, ref_future["q"], ref_future["qd"], v_loc, w_loc], dim=-1)
    return g.flatten(-2)

  # State features in the root heading frame (translation and yaw invariant).

  def state_features(self, s: dict, kin=None) -> torch.Tensor:
    if kin is None:
      kin = self.kin.forward(s["pos"], s["rot"], s["q"], s["v"], s["w"], s["qd"])
    P, Rb, V, W = kin
    H = heading_mat(s["rot"])
    Hb = H.unsqueeze(-3)
    rel = mtv(Hb, P[..., 1:, :] - s["pos"].unsqueeze(-2))
    vel = mtv(Hb, V[..., 1:, :])
    return torch.cat(
      [
        s["pos"][..., 2:3],
        rot6d(mtm(H, s["rot"])),
        mtv(H, s["v"]),
        mtv(s["rot"], s["w"]),
        s["q"] - self.q0,
        s["qd"],
        rel.flatten(-2),
        vel.flatten(-2),
      ],
      dim=-1,
    )

  # Reward and termination for the transition into s (compared to ref at the same time).

  def reward(self, s: dict, kin, ref: dict, ref_kin, action, prev_action):
    c = self.cfg
    P, Rb, V, W = kin
    rP, rR, rV, rW = ref_kin
    tr = self.tracked
    P, Rb, V, W = P[..., tr, :], Rb[..., tr, :, :], V[..., tr, :], W[..., tr, :]
    rP, rR, rV, rW = rP[..., tr, :], rR[..., tr, :, :], rV[..., tr, :], rW[..., tr, :]
    pa, Ra = s["pos"], s["rot"]
    rpa, rRa = ref["pos"], ref["rot"]
    # Reference bodies re-anchored at the robot root (xy from robot, z from ref, yaw aligned).
    D = yaw_mat(yaw_from_mat(mmt(Ra, rRa)))
    dpos = torch.cat([pa[..., :2], rpa[..., 2:3]], dim=-1)
    Db = D.unsqueeze(-3)
    rel_pos = dpos.unsqueeze(-2) + mv(Db, rP - rpa.unsqueeze(-2))
    rel_rot = mm(Db, rR)
    e_pos = ((rel_pos - P) ** 2).sum(-1).mean(-1)
    e_ori = (mat_to_axis_angle(mmt(rel_rot, Rb)) ** 2).sum(-1).mean(-1)
    e_lin = ((rV - V) ** 2).sum(-1).mean(-1)
    e_ang = ((rW - W) ** 2).sum(-1).mean(-1)
    e_rpos = ((rpa - pa) ** 2).sum(-1)
    e_rori = (mat_to_axis_angle(mmt(rRa, Ra)) ** 2).sum(-1)
    terms = {
      "body_pos": c.w_body_pos * torch.exp(-e_pos / c.std_body_pos**2),
      "body_ori": c.w_body_ori * torch.exp(-e_ori / c.std_body_ori**2),
      "body_lin": c.w_body_lin * torch.exp(-e_lin / c.std_body_lin**2),
      "body_ang": c.w_body_ang * torch.exp(-e_ang / c.std_body_ang**2),
      "root_pos": c.w_root_pos * torch.exp(-e_rpos / c.std_root_pos**2),
      "root_ori": c.w_root_ori * torch.exp(-e_rori / c.std_root_ori**2),
      "action_rate": c.w_action_rate * ((action - prev_action) ** 2).sum(-1),
      "joint_limit": c.w_joint_limit
      * (torch.relu(s["q"] - self.qmax) + torch.relu(self.qmin - s["q"])).sum(-1),
    }
    r = sum(terms.values()) * c.dt
    return r, terms

  def termination(self, s: dict, ref: dict):
    """Hard (sim) and soft (imagination) termination for state s against ref."""
    c = self.cfg
    ez = (s["pos"][..., 2] - ref["pos"][..., 2]).abs()
    # z component of gravity in the root frame is -R[2,2].
    etilt = (s["rot"][..., 2, 2] - ref["rot"][..., 2, 2]).abs()
    hard = (ez > c.term_z) | (etilt > c.term_tilt)
    soft = 1 - (1 - torch.sigmoid((ez - c.term_z) / c.soft_temp_z)) * (
      1 - torch.sigmoid((etilt - c.term_tilt) / c.soft_temp_tilt)
    )
    return hard, soft

  # Everything the loop needs about a (new) state, in one call so it can be compiled.

  def analytics(self, s, ref, ref_kin, ref_future, action, prev_action, enc_bias):
    kin = self.body_kin(s)
    r, terms = self.reward(s, kin, ref, ref_kin, action, prev_action)
    hard, soft = self.termination(s, ref)
    obs = self.obs_clean(s, action, enc_bias)
    feats = self.state_features(s, kin)
    goal = self.goal(s, ref_future)
    return {"kin": kin, "r": r, "terms": terms, "hard": hard, "soft": soft, "obs": obs, "feats": feats, "goal": goal}

  def features(self, s):
    return self.state_features(s, self.body_kin(s))

  def _maybe_compiled(self, name, fn, *args):
    if args[0]["pos"].shape[0] in self.compile_sizes:
      if name not in self._compiled:
        self._compiled[name] = torch.compile(fn, dynamic=False)
      return self._compiled[name](*args)
    return fn(*args)

  def analytics_fast(self, *args):
    return self._maybe_compiled("analytics", self.analytics, *args)

  def features_fast(self, s):
    return self._maybe_compiled("features", self.features, s)

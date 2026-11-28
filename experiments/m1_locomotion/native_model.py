"""Native M1 policy weights and terrain-conditioned SuperMimic dynamics."""
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply
from torch import nn
from torch.nn import functional as F

from native_world import (FRONT_ACTIONS, FRONT_DEFICIT, FRONT_FORWARD, FRONT_HIDDEN, FRONT_LIMIT,
                          FRONT_PITCH, FRONT_STALL, FRONT_WHEELS, MAX_STAIRS, observation_reflection)
from sim import DT, LEGS, WHEELS
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import GenCoordRepresentation


class WheelState(GenCoordRepresentation):
    LOSS_WEIGHT_CONFIG = {
        **GenCoordRepresentation.LOSS_WEIGHT_CONFIG,
        "linear_velocity": "nominal_error_linear_velocity",
        "angular_velocity": "nominal_error_angular_velocity",
    }

    def __init__(self, home):
        super().__init__(len(home), home)
        self.dt = DT
        self.substep_coeff = 21 / 40
        mask = home.new_ones(self.rep_state_dim)
        mask[10+WHEELS] = 0
        self.register_buffer("wheel_phase_mask", mask)
        self.register_buffer("legs", torch.as_tensor(LEGS, device=home.device))

    def represent(self, state):
        return super().represent(state) * self.wheel_phase_mask

    def apply_prediction(self, aux, rep, prediction, dt):
        next_aux, next_rep = super().apply_prediction(aux, rep, prediction, dt)
        return next_aux, next_rep * self.wheel_phase_mask

    def loss_blocks(self, rep, aux):
        blocks = super().loss_blocks(rep, aux)
        blocks["joint_pos"] = blocks["joint_pos"][..., self.legs]
        blocks["linear_velocity"] = rep[..., 4:7]
        blocks["angular_velocity"] = rep[..., 7:10]
        return blocks


@dataclass
class NativeDynamicsCfg(WorldModelLearnerCfg):
    nominal_error_linear_velocity: float = .13
    nominal_error_angular_velocity: float = .45


def heading_twist(rep, aux):
    quat = aux[..., 3:7]
    forward = matrix_from_quat(quat)[..., :, 0]
    forward_rate = torch.linalg.cross(quat_apply(quat, rep[..., 7:10]), forward)
    velocity = quat_apply(quat, rep[..., 4:7])
    x, y = forward[..., 0], forward[..., 1]
    norm = (x.square()+y.square()).clamp_min(1e-6)
    vx, vy = velocity[..., 0], velocity[..., 1]
    yaw = (x*forward_rate[..., 1] - y*forward_rate[..., 0]) / norm
    return torch.stack(((x*vx+y*vy)/norm.sqrt(), (x*vy-y*vx)/norm.sqrt(), yaw), dim=-1)


class TerrainDynamics(Dynamics):
    """Recompute terrain and support proxies from each predicted pose."""

    def __init__(self, representation, home, scale, hidden=(256, 256, 256)):
        super().__init__(representation, 16, 15, scale, home, hidden)
        self.register_buffer("rungs", torch.arange(MAX_STAIRS, device=home.device, dtype=home.dtype))
        self.register_buffer("hip_x", home.new_tensor([.28614, -.28586, .28614, -.28586]))
        self.register_buffer("hip_y", home.new_tensor([.06, .06, -.06, -.06]))
        self.register_buffer("leg_y", home.new_tensor([.21935, .21935, -.21735, -.21935]))
        self.register_buffer("geometry_scale", home.new_tensor([.20, .25, 1.]))

    def wheel_positions(self, rep, aux):
        # Joint origins and axes follow M1-robotdog-asm09. The wheel axis itself
        # does not change the position of the wheel centre.
        q = rep[..., 10:26].unflatten(-1, (4, 4))
        haa, hfe, knee = q[..., 0], q[..., 1], q[..., 2]
        x = self.hip_x - .23848*hfe.sin() - .36538*(hfe+knee).sin()
        z_plane = -.23848*hfe.cos() - .36538*(hfe+knee).cos()
        y = self.hip_y + self.leg_y*haa.cos() - z_plane*haa.sin()
        z = .058 + self.leg_y*haa.sin() + z_plane*haa.cos()
        local = torch.stack((x, y, z), dim=-1)
        return torch.matmul(local, matrix_from_quat(aux[..., 3:7]).transpose(-1, -2)) + aux[..., None, :3]

    def ground_height(self, xy, context):
        rise, tread, count, edge, half_width, _ = context.unbind(dim=-1)
        x, y = xy[..., 0], xy[..., 1]
        edges = edge[..., None] + tread[..., None]*self.rungs
        risers = ((x[..., None]-edges[..., None, :]) / .065 + .5).clamp(0., 1.)
        risers = risers * (self.rungs < count[..., None])[..., None, :]
        in_lane = ((half_width[..., None]-y.abs())/.08 + .5).clamp(0., 1.)
        deck_end = edge + tread*count + 2.7
        in_deck = ((deck_end[..., None]-x)/.08 + .5).clamp(0., 1.)
        return rise[..., None]*risers.sum(dim=-1)*in_lane*in_deck

    def front_risers(self, wheels, context):
        front = wheels[..., list(FRONT_WHEELS), :]
        rise, tread, count, edge, width, _ = (v[..., None, None] for v in context.unbind(-1))
        gap = edge + tread*self.rungs - front[..., 0, None]
        top = rise * (self.rungs + 1)
        bottom = front[..., 2, None] - .095
        approach = ((gap + .035)/.07).clamp(0., 1.) * ((.21 - gap)/.075).clamp(0., 1.)
        previous_tread = ((bottom - (top-rise) + .075)/.055).clamp(0., 1.)
        approach *= previous_tread * (self.rungs < count) * (front[..., 1, None].abs() < width)
        return gap, approach, top + .012 - bottom

    def front_riser_score(self, before, after, context):
        gap, near, initial_shortfall = self.front_risers(before, context)
        # Score each wheel against the same riser over this transition. The actor
        # cannot erase a clearance penalty by changing the selected riser.
        active = (near * (initial_shortfall/.055).clamp(0., 1.)).detach()
        at_face = ((.135-gap)/.060).clamp(0., 1.).detach()
        movement = after[..., list(FRONT_WHEELS), :] - before[..., list(FRONT_WHEELS), :]
        remaining = (initial_shortfall - movement[..., 2, None]).clamp(0., .24)
        gain = ((initial_shortfall.clamp(0., .24)-remaining)/.030).tanh()
        advance = (movement[..., 0, None]/.014).tanh()
        score = active * (.55*gain + .22*advance - .35*at_face*(remaining/.18))
        return score.sum(dim=-1).mean(dim=-1)

    def terrain(self, rep, aux, context):
        wheels = self.wheel_positions(rep, aux)
        xy = wheels[..., :2]
        ground = self.ground_height(xy, context)
        ahead = torch.stack((xy[..., 0]+.16, xy[..., 1]), dim=-1)
        preview = self.ground_height(ahead, context) - ground
        clearance = wheels[..., 2] - .095 - ground
        support = torch.sigmoid((.025-clearance)/.026)
        return clearance, preview, support

    def forward(self, rep_state, action, aux_state, sim_ctx):
        clearance, preview, support = self.terrain(rep_state, aux_state, sim_ctx)
        shape = torch.stack((sim_ctx[..., 0], sim_ctx[..., 1], sim_ctx[..., 5]), dim=-1) / self.geometry_scale
        context = torch.cat((clearance.clamp(-.35,.45)*5, preview*5, support, shape), dim=-1)
        return super().forward(rep_state, action, aux_state, context)


class Teacher(nn.Module):
    """The two verified graphs differ only in a direct history input to the actor."""

    def __init__(self, file, device):
        super().__init__()
        self.names = {}
        with np.load(file, allow_pickle=False) as weights:
            for name in weights.files:
                buffer = name.replace(".", "_")
                self.register_buffer(buffer, torch.from_numpy(weights[name]).to(device))
                self.names[name] = buffer
            width = weights["actor_critic.actor.0.weight"].shape[1]
        if width not in (68, 313):
            raise ValueError(f"Unexpected native actor width {width}")
        self.direct_history = width == 313

    def linear(self, x, layer):
        prefix = f"actor_critic.{layer}"
        w, b = (getattr(self, self.names[f"{prefix}.{param}"]) for param in ("weight", "bias"))
        return F.linear(x, w, b)

    def forward(self, obs, history, sample=False):
        intermediate = F.elu(self.linear(F.elu(self.linear(history, "encoder.0")), "encoder.2"))
        latent = self.linear(intermediate, "encode_mean_latent")
        if sample:
            variance = self.linear(intermediate, "encode_logvar_latent")
            latent = latent + (.5*variance).exp()*torch.randn_like(latent)
        velocity = self.linear(intermediate, "encode_mean_vel")
        values = [velocity, latent, obs]
        if self.direct_history:
            values.append(history)
        x = torch.cat(values, dim=-1)
        for layer in (0, 2, 4):
            x = F.elu(self.linear(x, f"actor.{layer}"))
        return self.linear(x, "actor.6")


def ramp(value, bounds):
    lo, hi = bounds
    return ((value-lo)/(hi-lo)).clamp(0., 1.)


class FrontLift(nn.Module):
    """Reflect observations so the same learned controller scores either front wheel."""

    def __init__(self, reference):
        super().__init__()
        order, sign = observation_reflection()
        self.register_buffer("mirror", torch.as_tensor(np.concatenate([order + 49*i for i in range(6)]),
                                                       device=reference.device))
        self.register_buffer("mirror_sign", reference.new_tensor(np.tile(sign, 6)))
        self.register_buffer("limit", reference.new_tensor(FRONT_LIMIT))
        embed = reference.new_zeros((len(FRONT_ACTIONS), 16))
        embed[torch.arange(len(FRONT_ACTIONS), device=reference.device), list(FRONT_ACTIONS)] = 1.
        self.register_buffer("embed", embed)
        self.network = nn.Sequential(nn.Linear(294, FRONT_HIDDEN), nn.ELU(), nn.Linear(FRONT_HIDDEN, 3)).to(reference.device)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def authority(self, obs, turn):
        deficit = obs[..., [28,32,36,40]]*.5 - obs[..., 21:25]
        front = deficit[..., [0,2]].amax(dim=-1, keepdim=True)
        rear = deficit[..., [1,3]].mean(dim=-1, keepdim=True)
        stall = ramp(front-rear, FRONT_STALL) * ramp(front, FRONT_DEFICIT)
        cue = torch.maximum(ramp(-obs[..., 3:4], FRONT_PITCH), stall)
        return cue * ramp(obs[..., 6:7], FRONT_FORWARD) * (1-turn)

    def forward(self, obs, history, bias, inverse_std, turn):
        raw = torch.cat((obs, history), dim=-1)
        paired = torch.stack((raw, raw[..., self.mirror]*self.mirror_sign), dim=-2)
        normalized = (paired-bias)*inverse_std
        return self.network(normalized).tanh() * self.limit * self.authority(obs, turn)[..., None]


class Student(nn.Module):
    """One stair-initialized actor with a trainable correction on all actuators."""

    def __init__(self, stairs, bias, inverse_std, device, initial_drive, front_lift=False):
        super().__init__()
        self.stairs = stairs
        self.register_buffer("bias", bias)
        self.register_buffer("inverse_std", inverse_std)
        self.register_buffer("action_limit", bias.new_tensor([.09, .11, .11, .13]*4))
        self.register_buffer("turn_gains", bias.new_tensor([5.,14.]))
        self.register_buffer("wheel_mask", bias.new_tensor([0.,0.,0.,1.]*4))
        self.drive = nn.Parameter(bias.new_tensor(initial_drive/.30).atanh())
        self.correction = nn.Sequential(nn.Linear(294, 144), nn.ELU(), nn.Linear(144, 16)).to(device)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.front_lift = FrontLift(bias) if front_lift else None

    @torch.no_grad()
    def load_base(self, file):
        layers = self.correction
        parameters = {"w1": layers[0].weight, "b1": layers[0].bias, "w2": layers[2].weight,
                      "b2": layers[2].bias, "mean": self.bias, "inverse_std": self.inverse_std}
        with np.load(file, allow_pickle=False) as checkpoint:
            for key, parameter in parameters.items():
                value = checkpoint[key]
                if tuple(parameter.shape) != value.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid base policy parameter {key}")
                parameter.copy_(torch.from_numpy(value))
            drive = float(checkpoint["forward_correction"])
            if not -.30 < drive < .30:
                raise ValueError("Invalid base wheel correction")
            self.drive.copy_(self.drive.new_tensor(drive/.30).atanh())
        self.correction.requires_grad_(False)
        self.drive.requires_grad_(False)

    def authority(self, obs):
        return (obs[..., 7:9].abs() * self.turn_gains).amax(dim=-1, keepdim=True).clamp(0., 1.)

    def manoeuvre_delta(self, obs, history):
        inp = (torch.cat((obs, history), dim=-1)-self.bias)*self.inverse_std
        return self.correction(inp).tanh()*self.action_limit*self.authority(obs)

    def straight_delta(self, obs):
        turn = self.authority(obs)
        speed_fraction = ((obs[..., 6:7]-.90)/.30).clamp(0., 1.)
        # A common wheel correction keeps straight acceleration from adding yaw.
        return .30*self.drive.tanh()*speed_fraction*(1-turn)*self.wheel_mask

    def imitation_loss(self, delta, target, flat, good):
        flat_error = F.smooth_l1_loss(delta, target, reduction="none", beta=.08).mean(dim=-1)
        step_error = delta.square().mean(dim=-1)
        return (flat_error*flat*good + 4.*step_error*(1-flat)).mean()

    def forward(self, data):
        obs = data["obs"]
        history = data["history"].flatten(-2)
        manoeuvre = self.manoeuvre_delta(obs, history)
        self._aux = DT*2.*self.imitation_loss(manoeuvre, data["anchor"], data["flat"], data["good"])
        action = self.stairs(obs, history, sample=True) + manoeuvre + self.straight_delta(obs)
        if self.front_lift is not None:
            front = self.front_lift(obs, history, self.bias, self.inverse_std, self.authority(obs))
            self._aux = self._aux + DT*.45*front.square().mean()
            action = action + front.flatten(-2) @ self.front_lift.embed
        return action

    def auxiliary_loss(self):
        return self._aux

    def save(self, file):
        layers = self.correction
        values = {"w1": layers[0].weight, "b1": layers[0].bias, "w2": layers[2].weight, "b2": layers[2].bias,
                  "mean": self.bias, "inverse_std": self.inverse_std, "limit": self.action_limit,
                  "forward_correction": .30*self.drive.tanh()}
        if self.front_lift is not None:
            layers = self.front_lift.network
            values.update(front_w1=layers[0].weight, front_b1=layers[0].bias,
                          front_w2=layers[2].weight, front_b2=layers[2].bias)
        np.savez(file, **{k: v.detach().cpu().numpy() for k,v in values.items()})

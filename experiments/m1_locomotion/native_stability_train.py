"""Train level body stabilization and descent using original SuperMimic BPTT."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from native_model import NativeDynamicsCfg, TerrainDynamics, WheelState, heading_twist, ramp
from native_terrain import (DESCENT_GAIN_SPAN, DESCENT_LEG_CAP, FLAT_CAP, GUIDE_PARAMETERS,
                            PROBES, PROFILE_HIDDEN, TERRAIN_SIZE, TRUNK_COM, WHEEL_SIGNS)
from native_train import Replay, file_digest, save_report, upstream_source, validate_world
from native_world import DESCENT, FLAT, NATIVE_SHA256
from sim import DT, LEGS, M1_HOME, M1_SCALE, WHEELS
from supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearner


class StabilityDynamics(TerrainDynamics):
    def __init__(self, representation, home, scale):
        super().__init__(representation, home, scale)
        self.register_buffer("probe_offsets", home.new_tensor(PROBES))
        self.register_buffer("trunk_offset", home.new_tensor(TRUNK_COM))

    def centre(self, aux):
        return aux[..., 2] + quat_apply(aux[..., 3:7], self.trunk_offset.expand_as(aux[..., :3]))[..., 2]

    def profile(self, rep, aux, context):
        rotation = matrix_from_quat(aux[..., 3:7])
        direction = F.normalize(rotation[..., :2, 0], dim=-1)
        points = aux[..., None, :2] + direction[..., None, :]*self.probe_offsets[:, None]
        heights = self.ground_height(points, context)
        offset = quat_apply(aux[..., 3:7], self.trunk_offset.expand_as(aux[..., :3]))
        omega = quat_apply(aux[..., 3:7], rep[..., 7:10])
        velocity = quat_apply(aux[..., 3:7], rep[..., 4:7])
        vz = (velocity + torch.linalg.cross(omega, offset))[..., 2]
        heading = torch.atan2(direction[..., 1], direction[..., 0])
        y = aux[..., 1]
        error = torch.pi + torch.atan(GUIDE_PARAMETERS["lane_gain"]*y) - heading
        return torch.cat((torch.stack((torch.ones_like(vz), heights.amax(-1)-heights.amin(-1),
                                      heights[..., -1]-heights[..., 0], aux[..., 2]+offset[..., 2]-heights[..., 2],
                                      vz), dim=-1),
                          heading_twist(rep, aux), torch.stack((torch.atan2(error.sin(), error.cos()), y), -1)), -1)


class StabilityActor(nn.Module):
    """Two bounded trainable corrections to the current M1 action."""

    def __init__(self, mean, inverse_std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("inverse_std", inverse_std)
        self.register_buffer("flat_cap", mean.new_tensor(FLAT_CAP))
        self.register_buffer("descent_legs_cap", mean.new_tensor(DESCENT_LEG_CAP))
        self.register_buffer("gain_span", mean.new_tensor(DESCENT_GAIN_SPAN))
        self.register_buffer("prior_gains", mean.new_tensor([GUIDE_PARAMETERS[k] for k in
                                                              ("heading_kp", "yaw_kd", "wheel_speed_kp")]))
        self.register_buffer("turn_units", mean.new_tensor([1., 4.]))
        self.register_buffer("wheel_signs", mean.new_tensor(WHEEL_SIGNS))
        self.register_buffer("wheel_embed", torch.eye(16, device=mean.device)[WHEELS])
        self.register_buffer("leg_embed", torch.eye(16, device=mean.device)[LEGS])
        self.flat = self.network(16)
        self.descent = self.network(15)

    def network(self, outputs):
        layers = nn.Sequential(nn.Linear(294+TERRAIN_SIZE, PROFILE_HIDDEN), nn.ELU(),
                               nn.Linear(PROFILE_HIDDEN, outputs)).to(self.mean.device)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return layers

    def branches(self, obs, history, terrain):
        raw = torch.cat((obs, history, terrain), -1)
        features = ((raw-self.mean)*self.inverse_std).clamp(-8., 8.)
        vx = obs[..., 6:7]*.5
        flat = self.flat(features).tanh()*self.flat_cap
        learned = self.descent(features).tanh()
        gains = self.prior_gains + learned[..., :3]*self.gain_span
        kp, kd, speed_kp = gains.unbind(-1)
        turn = (kp*terrain[..., 8] - kd*terrain[..., 7]).clamp(-.8, .8)
        drive = speed_kp*(vx[..., 0]-terrain[..., 5])
        wheel = (drive[..., None] + .275*turn[..., None]*self.wheel_signs)/.95
        descent = wheel@self.wheel_embed + (learned[..., 3:]*self.descent_legs_cap)@self.leg_embed
        return flat, descent

    def weights(self, obs, terrain):
        turn = (obs[..., 7:9].abs()*self.turn_units).amax(-1, keepdim=True)
        straight = terrain[..., :1]*(1-ramp(turn, (.015, .065)))
        tilt = obs[..., 3:5].abs().amax(-1, keepdim=True)
        level = (1-ramp(terrain[..., 1:2], (.012, .055)))*(1-ramp(tilt, (.05, .13)))
        flat = straight*level*ramp(obs[..., 6:7].abs(), (.06, .24))
        descent = straight*ramp(-terrain[..., 2:3], (.036, .110))*ramp(obs[..., 6:7], (.06, .23))
        return flat, descent

    def forward(self, data):
        obs, terrain = data["obs"], data["terrain"]
        flat, descent = self.branches(obs, data["history"].flatten(-2), terrain)
        flat_weight, descent_weight = self.weights(obs, terrain)
        base = data["reference"]
        action = base + flat_weight*flat + descent_weight*descent
        error = (action-data["demonstration"]).square().mean(-1)
        self._aux = DT*.75*error.mean()
        return action

    def auxiliary_loss(self):
        return self._aux

    def save(self, path):
        weights = {"mean": self.mean, "inverse_std": self.inverse_std}
        for name in ("flat", "descent"):
            layers = getattr(self, name)
            weights.update({f"{name}_w1": layers[0].weight, f"{name}_b1": layers[0].bias,
                            f"{name}_w2": layers[2].weight, f"{name}_b2": layers[2].bias})
        np.savez(path, **{key: value.detach().cpu().numpy() for key,value in weights.items()})


class GuidedWindows:
    def __init__(self, replay, horizon):
        self.replay = replay
        self.horizon = horizon
        data = replay.data
        valid = ((replay.elapsed >= 5) & (replay.elapsed+horizon < data["length"][replay.episode])
                 & data["guided"][replay.episode])
        forward = data["obs"][:, 6]
        level = valid & (replay.kind == FLAT)
        descent = valid & (replay.kind == DESCENT) & (forward > .32)
        course = data["course"][replay.episode]
        descent &= ((data["state"][:, 0] > course[:, 3]-.45)
                    & (data["state"][:, 0] < course[:, 3]+course[:, 1]*course[:, 2]+.55))
        self.pools = (replay.frame[level & (forward > .18)], replay.frame[level & (forward < -.18)],
                      replay.frame[descent])
        if any(not len(pool) for pool in self.pools):
            raise ValueError("Required forward, reverse and downhill demonstrations are missing")

    def sample(self, batch):
        replay = self.replay
        sizes = batch//4, batch//4, batch-2*(batch//4)
        starts = torch.cat([pool[torch.randint(len(pool), (n,), device=replay.device)]
                            for pool,n in zip(self.pools, sizes, strict=True)])
        return starts, starts[:, None]+torch.arange(self.horizon+1, device=replay.device)


class StabilityWorld:
    def __init__(self, dynamics, representation, replay, windows, home, cfg, height):
        self.dynamics, self.representation = dynamics, representation
        self.replay, self.windows, self.home = replay, windows, home
        self.cfg, self.height = cfg, height
        self.legs = torch.as_tensor(LEGS, device=home.device)
        self.wheels = torch.as_tensor(WHEELS, device=home.device)

    def sample_initial_states(self):
        starts, frames = self.windows.sample(self.cfg.batch_size)
        replay, d = self.replay, self.replay.data
        obs = d["obs"][frames]
        return TensorDict({"state": d["state"][starts], "context": d["course"][replay.episode[starts]],
                           "target": d["target"][frames], "obs_command": obs[..., 6:9],
                           "phase": obs[..., 41:49], "previous": obs[:, 0, 25:41],
                           "history": replay.history(starts), "reference": d["baseline_action"][frames],
                           "demonstration": d["action"][frames], "flat": replay.kind[starts] == FLAT},
                          batch_size=[len(starts)])

    def reset(self, data):
        self.data = data
        self.tick = 0
        self.rep = self.representation.represent(data["state"])
        self.aux = self.representation.extract_aux_state(data["state"])
        self.history = data["history"]
        self.last = data["previous"]
        return self.observation()

    def observation(self):
        rep, t, d = self.rep, self.tick, self.data
        self.obs = torch.cat((rep[:, 7:10]*.25, rep[:, 1:4], d["obs_command"][:, t],
                              rep[:, 10:26][:, self.legs]-self.home[self.legs],
                              rep[:, 26:42][:, self.wheels]*.05, self.last, d["phase"][:, t]), -1)
        self.profile = self.dynamics.profile(rep, self.aux, d["context"])
        return TensorDict({"obs": self.obs, "terrain": self.profile, "history": self.history,
                           "reference": d["reference"][:, t], "demonstration": d["demonstration"][:, t],
                           "critic": rep}, batch_size=[len(rep)])

    def step(self, action):
        old = self.aux
        previous = self.last
        context = self.data["context"]
        goal = self.data["target"][:, self.tick]
        self.aux, self.rep = self.dynamics(self.rep, action, old, context)
        self.last = action
        self.history = torch.cat((self.history[:, 1:], self.obs[:, None]), dim=1)
        self.tick += 1
        obs = self.observation()
        h, vz, vx, vy, yaw = self.profile[:, 3:8].unbind(-1)
        # Use the trunk's inertial centre, including its rotation about the root.
        centre_vz = (self.dynamics.centre(self.aux)-self.dynamics.centre(old))/DT
        qd = self.rep[:, 26:42][:, self.legs]
        slew = (action[:, self.legs]-previous[:, self.legs]).square().mean(-1)
        straight = ((vx-goal[:, 0])/.145).square() + (vy/.085).square() + (yaw/.22).square()
        flat = ((-straight).exp() - .35*(centre_vz/.13).square().clamp(max=4.)
                - .24*(qd/2.3).square().mean(-1).clamp(max=4.)
                - .10*(slew/.025).clamp(max=4.) - .20*((h-self.height)/.066).square().clamp(max=4.))
        forward = ((old[:, 0]-self.aux[:, 0])/.012).tanh()
        tracking = (-((vx-.73*goal[:, 0])/.175).square() - (vy/.17).square()).exp()
        impact = ((-vz-.25)/.32).relu().square().clamp(max=5.)
        excessive_pitch = ((self.rep[:, 1]-.54)/.15).relu().square()
        roll = (self.rep[:, 2]/.20).square()
        lane = (self.aux[:, 1]/context[:, 4]).square()
        descent = (.65*tracking + .50*forward - .30*impact - .20*roll - .22*excessive_pitch
                   - .24*lane - .12*(yaw/.52).square() - .06*(slew/.055).clamp(max=4.))
        termination = torch.maximum(torch.sigmoid((.235-h)/.024), torch.sigmoid((self.rep[:, 3]+.66)/.025))
        reward = torch.where(self.data["flat"], flat, descent) - .65*termination
        return obs, reward*DT, termination, {}


def warm_up(actor, dynamics, state, replay, windows, updates):
    optimizer = torch.optim.AdamW(actor.parameters(), lr=.00045)
    for i in range(1, updates+1):
        starts, _ = windows.sample(768)
        d = replay.data
        rows = d["state"][starts]
        obs = d["obs"][starts]
        terrain = dynamics.profile(state.represent(rows), state.extract_aux_state(rows),
                                   d["course"][replay.episode[starts]])
        flat, descent = actor.branches(obs, replay.history(starts).flatten(-2), terrain)
        delta = torch.where((replay.kind[starts] == FLAT)[:, None], flat, descent)
        prediction = d["baseline_action"][starts]+delta
        loss = F.smooth_l1_loss(prediction, d["action"][starts], beta=.035)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), .7)
        optimizer.step()
        if i in (1, updates) or i % 160 == 0:
            print("reference imitation", i, float(loss.detach()), flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    for key in ("train", "holdout", "base", "base-report", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--holdout-extra", type=Path, action="append", default=[])
    p.add_argument("--world-updates", type=int, default=2800)
    p.add_argument("--warm-up", type=int, default=480)
    p.add_argument("--actor-updates", type=int, default=512)
    p.add_argument("--seed", type=int, default=2418)
    args = p.parse_args()
    source = upstream_source()
    old = json.loads(args.base_report.read_text())
    digest = file_digest(args.base)
    if old.get("phase") != "complete" or old.get("teacher_policies") != NATIVE_SHA256:
        raise ValueError("Expected existing verified M1 training provenance")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    home = torch.as_tensor(M1_HOME, dtype=torch.float32, device=device)
    scale = torch.as_tensor(M1_SCALE, dtype=torch.float32, device=device)
    replay = Replay([args.train, *args.extra], device)
    holdout = Replay([args.holdout, *args.holdout_extra], device)
    if replay.seeds & holdout.seeds:
        raise ValueError("Training and holdout course seeds overlap")
    if not replay.has_descent or not holdout.has_descent:
        raise ValueError("Expected descent training data and a held-out descent set")
    if any(s["candidate"] != digest for s in replay.sources+holdout.sources if s["guidance"]):
        raise ValueError("The guidance references must use the deployed base ONNX")
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "training.json"
    report = {"phase": "world_model", "seed": args.seed, "supermimic": source, "teacher_policies": NATIVE_SHA256,
              "device": torch.cuda.get_device_name(), "front_lift": old.get("front_lift"), "base_sha256": digest,
              "base_training_sha256": file_digest(args.base_report), "train": replay.sources, "holdout": holdout.sources,
              "objectives": {"flat": ["trunk_com_vz", "trunk_com_height", "leg_joint_velocity", "leg_action_slew"],
                             "descent": ["downhill_progress", "vertical_impact", "roll", "pitch", "yaw", "lane_offset"]}}
    save_report(status, report)
    state = WheelState(home).to(device)
    dynamics = StabilityDynamics(state, home, scale).to(device)
    config = NativeDynamicsCfg(lr=.00085, batch_size=256, num_batches=1, horizon=8)
    wm = WorldModelLearner(dynamics, state, config, device)
    wm._captured_update = wm._update_from
    wm.update_normalization(*replay.normalizer_data())
    print("device", torch.cuda.get_device_name(), "world transitions", int(replay.active.sum()), flush=True)
    for i in range(1,args.world_updates+1):
        stats = wm.update([replay.windows(8, config.batch_size)[0]])
        if i == 1 or i % 200 == 0 or i == args.world_updates:
            row = {k:float(v) for k,v in stats.items()}
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError("World model diverged")
            print("world", i, row, flush=True)
    validation = validate_world(wm, holdout, config.horizon)
    if any(group["h8_twist_rmse"][0] > .14 for group in validation.values()):
        raise RuntimeError("Held-out longitudinal prediction exceeded the established threshold")
    report.update(phase="imitation", world_updates=args.world_updates, world_validation=validation)
    save_report(status, report)
    torch.save(dynamics.state_dict(), args.out/"dynamics.pt")
    print("world holdout", json.dumps(validation), flush=True)
    cfg = BpttPolicyLearnerCfg(lr=.00011, batch_size=256, horizon=12, gamma=.98, noise=0.)
    windows = GuidedWindows(replay, cfg.horizon)
    idx, _ = windows.sample(32768)
    rows = replay.data["state"][idx]
    with torch.no_grad():
        profile = dynamics.profile(state.represent(rows), state.extract_aux_state(rows),
                                   replay.data["course"][replay.episode[idx]])
        samples = torch.cat((replay.data["obs"][idx], replay.history(idx).flatten(-2), profile), -1)
        height = profile[replay.kind[idx] == FLAT, 3].median().item()
    actor = StabilityActor(samples.mean(0), samples.std(0).clamp_min(.065).reciprocal())
    warm_up(actor, dynamics, state, replay, windows, args.warm_up)
    actor.save(args.out/"reference.npz")
    report.update(phase="policy", imitation_updates=args.warm_up, reference_trunk_com_height_m=height,
                  guided_windows=[len(pool) for pool in windows.pools])
    save_report(status, report)
    world = StabilityWorld(dynamics, state, replay, windows, home, cfg, height)
    policy = BpttPolicyLearner(actor, world, cfg, device)
    policy._captured_update = policy._update_from
    for i in range(1,args.actor_updates+1):
        stats = policy.update()
        if i == 1 or i % 32 == 0 or i == args.actor_updates:
            row = {k:float(v) for k,v in stats.items()}
            if not np.isfinite(list(row.values())).all():
                raise RuntimeError("BPTT diverged")
            print("policy", i, row, flush=True)
        if i in (16, 48, 96, 144, 224, 352) or i == args.actor_updates:
            actor.save(args.out/f"policy-{i:04}.npz")
    report.update(phase="complete", actor_updates=args.actor_updates)
    save_report(status, report)


if __name__ == "__main__":
    main()

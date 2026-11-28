"""Fine-tune the real M1 Classic actor with original SuperMimic on level ground."""
from argparse import ArgumentParser
import json
from pathlib import Path

from mjlab.utils.lab_api.math import quat_apply
import numpy as np
from tensordict import TensorDict
import torch
from torch import nn
from torch.nn import functional as F

from native_model import NativeDynamicsCfg, Teacher, TerrainDynamics, WheelState, heading_twist
from native_terrain import TRUNK_COM
from native_train import Replay, file_digest, save_report, upstream_source
from native_world import CLASSIC_RETRAINED_KEYS, FLAT
from sim import DT, LEGS, M1_HOME, M1_SCALE, WHEELS
from supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearner

class FlatActor(Teacher):
    def __init__(self, weights, device):
        super().__init__(weights, device)
        for key in CLASSIC_RETRAINED_KEYS:
            name = self.names[key]
            value = getattr(self, name)
            delattr(self, name)
            self.register_parameter(name, nn.Parameter(value))

    def infer(self, obs, history):
        # Classic includes this frame. Uni Exp's C++ history ends one frame earlier.
        current_history = torch.cat((history[:, 1:], obs[:, None]), dim=1).flatten(1)
        return super().forward(obs, current_history)

    def forward(self, data):
        action = self.infer(data["obs"], data["history"])
        self.imitation = DT * 1.6 * F.smooth_l1_loss(action, data["expert"], beta=.20)
        return action

    def auxiliary_loss(self):
        return self.imitation

    def trained_weights(self):
        return {key: getattr(self, self.names[key]).detach().cpu().numpy().copy() for key in CLASSIC_RETRAINED_KEYS}


class FlatWindows:
    def __init__(self, replay, horizon):
        self.replay, self.horizon = replay, horizon
        data = replay.data
        valid = (replay.kind == FLAT) & (replay.elapsed >= 42) & (replay.elapsed+horizon < data["length"][replay.episode])
        x = data["obs"][:, 6].abs() * .5
        turning = (data["obs"][:, 7:9].abs() * data["obs"].new_tensor([.5, 4.])).amax(-1) > .07
        self.pools = (replay.frame[valid & (x > .09) & (x < .48) & ~turning],
                      replay.frame[valid & (x >= .48) & ~turning], replay.frame[valid & turning],
                      replay.frame[valid & (x <= .09) & ~turning])
        if any(not len(pool) for pool in self.pools):
            raise ValueError("Replay must cover slow and fast straight travel, turning and rest")

    def sample(self, batch):
        replay = self.replay
        sizes = (batch*3//8, batch//4, batch//4)
        starts = torch.cat([pool[torch.randint(len(pool), (n,), device=replay.device)]
                            for pool, n in zip(self.pools, (*sizes, batch-sum(sizes)), strict=True)])
        frames = starts[:, None] + torch.arange(self.horizon+1, device=replay.device)
        return starts, frames

    def world_batch(self, batch):
        starts, frames = self.sample(batch)
        replay, d = self.replay, self.replay.data
        return TensorDict({"state": d["state"][frames], "action": d["action"][frames],
                           "sim_ctx": d["course"][replay.episode[starts], None].expand(-1, self.horizon+1, -1)},
                          batch_size=[batch])


class FlatWorld:
    def __init__(self, dynamics, state, windows, home, cfg):
        self.dynamics, self.representation, self.windows, self.home, self.cfg = dynamics, state, windows, home, cfg
        self.legs = torch.as_tensor(LEGS, device=home.device)
        self.wheels = torch.as_tensor(WHEELS, device=home.device)
        self.trunk_offset = home.new_tensor(TRUNK_COM)
        self.units = home.new_tensor([.5, .5, 4.])
        self.velocity_error_scale = home.new_tensor([.18, .16, .35])

    def centre(self, aux):
        return aux[..., 2] + quat_apply(aux[..., 3:7], self.trunk_offset.expand_as(aux[..., :3]))[..., 2]

    def sample_initial_states(self):
        starts, frames = self.windows.sample(self.cfg.batch_size)
        replay = self.windows.replay
        data = replay.data
        obs = data["obs"][frames]
        reference = data["state"][frames]
        return TensorDict({"state": reference[:, 0], "context": data["course"][replay.episode[starts]],
                           "command": obs[..., 6:9], "phase": obs[..., 41:49], "previous": obs[:, 0, 25:41],
                           "history": replay.history(starts), "expert": data["baseline_action"][frames],
                           "height": self.centre(reference[..., :7]), "legs": reference[..., self.legs+7]},
                          batch_size=[len(starts)])

    def reset(self, data):
        self.data, self.tick = data, 0
        self.rep = self.representation.represent(data["state"])
        self.aux = self.representation.extract_aux_state(data["state"])
        self.history, self.previous = data["history"], data["previous"]
        return self.observation()

    def observation(self):
        rep, t, data = self.rep, self.tick, self.data
        self.obs = torch.cat((rep[:, 7:10]*.25, rep[:, 1:4], data["command"][:, t],
                              rep[:, self.legs+10]-self.home[self.legs], rep[:, self.wheels+26]*.05,
                              self.previous, data["phase"][:, t]), dim=-1)
        return TensorDict({"obs": self.obs, "history": self.history, "expert": data["expert"][:, t], "critic": rep},
                          batch_size=[len(rep)])

    def step(self, action):
        previous_height = self.centre(self.aux)
        old_action = self.previous
        target = self.data["command"][:, self.tick] * self.units
        self.aux, self.rep = self.dynamics(self.rep, action, self.aux, self.data["context"])
        self.history = torch.cat((self.history[:, 1:], self.obs[:, None]), dim=1)
        self.previous = action
        self.tick += 1
        height = self.centre(self.aux)
        vz = (height-previous_height)/DT
        h_error = (height-self.data["height"][:, self.tick])/.035
        q_error = (self.rep[:, self.legs+10]-self.data["legs"][:, self.tick])/.20
        leg_rate = self.rep[:, self.legs+26]/2.5
        leg_slew = (action[:, self.legs]-old_action[:, self.legs])/.18
        twist = heading_twist(self.rep, self.aux)
        tracking = (-((twist-target)/self.velocity_error_scale).square().mean(-1)).exp()
        turn = (target[:, 1:].abs().amax(-1)/.20).clamp(0., 1.)
        posture = .60*h_error.square().clamp(max=4.) + .24*q_error.square().mean(-1).clamp(max=4.)
        stability = .24*(vz/.14).square().clamp(max=4.) + (1.-.75*turn)*.16*leg_rate.square().mean(-1).clamp(max=4.)
        termination = torch.maximum(torch.sigmoid((.22-height)/.024), torch.sigmoid((self.rep[:, 3]+.66)/.025))
        reward = tracking - posture - stability - .08*leg_slew.square().mean(-1).clamp(max=4.) - termination
        return self.observation(), reward*DT, termination, {}


@torch.no_grad()
def predict_error(learner, holdout):
    learner.dynamics.eval()
    batch = holdout.world_batch(2048)
    state = learner.representation
    rep, aux = state.represent(batch["state"]), state.extract_aux_state(batch["state"])
    prediction, pose = learner._rollout(rep[:, 0], batch["action"], aux[:, 0], batch["sim_ctx"][:, 0])
    expected = heading_twist(rep[:, -1], aux[:, -1])
    actual = heading_twist(prediction[:, -1], pose[:, -1])
    return {"twist_rmse": (actual-expected).square().mean(0).sqrt().tolist(),
            "base_height_rmse_m": float((pose[:, -1, 2]-aux[:, -1, 2]).square().mean().sqrt())}


def show(phase, step, stats):
    row = {key: float(value) for key, value in stats.items()}
    if not np.isfinite(list(row.values())).all():
        raise RuntimeError(f"Nonfinite {phase} update {step}")
    print(phase, step, json.dumps(row), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "classic", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--world-start", type=Path)
    parser.add_argument("--world-updates", type=int, default=960)
    parser.add_argument("--imitation-updates", type=int, default=320)
    parser.add_argument("--actor-updates", type=int, default=192)
    parser.add_argument("--seed", type=int, default=271132)
    args = parser.parse_args()
    source = upstream_source()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    home = torch.as_tensor(M1_HOME, dtype=torch.float32, device=device)
    scale = torch.as_tensor(M1_SCALE, dtype=torch.float32, device=device)
    data, validation = (Replay([path], device) for path in (args.train, args.holdout))
    if data.seeds & validation.seeds:
        raise ValueError("The held-out episodes share training seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "training.json"
    report = {"phase": "world", "seed": args.seed, "supermimic": source, "device": torch.cuda.get_device_name(),
              "classic_initializer_sha256": file_digest(args.classic), "train": data.sources, "holdout": validation.sources,
              "objectives": ["Classic trunk COM height", "Classic leg pose", "vertical speed", "leg rate",
                             "leg action slew", "command tracking"], "updated_parameters": CLASSIC_RETRAINED_KEYS}
    save_report(status, report)
    state = WheelState(home).to(device)
    dynamics = TerrainDynamics(state, home, scale).to(device)
    if args.world_start:
        # The compatible earlier model has two extra buffers used only by its terrain profiler.
        checkpoint = torch.load(args.world_start, map_location=device, weights_only=True)
        dynamics.load_state_dict({key: checkpoint[key] for key in dynamics.state_dict()})
    wm_cfg = NativeDynamicsCfg(lr=.00052, batch_size=256, num_batches=1, horizon=8)
    wm = WorldModelLearner(dynamics, state, wm_cfg, device)
    wm._captured_update = wm._update_from
    wm.update_normalization(*data.normalizer_data())
    world_windows = FlatWindows(data, wm_cfg.horizon)
    for step in range(1, args.world_updates+1):
        stats = wm.update([world_windows.world_batch(wm_cfg.batch_size)])
        if step == 1 or step % 120 == 0 or step == args.world_updates:
            show("world", step, stats)
    world_error = predict_error(wm, FlatWindows(validation, wm_cfg.horizon))
    print("world holdout", json.dumps(world_error), flush=True)
    torch.save(dynamics.state_dict(), args.out / "dynamics.pt")
    report.update(phase="imitation", world_updates=args.world_updates, world_holdout=world_error)
    save_report(status, report)
    if world_error["base_height_rmse_m"] > .035 or world_error["twist_rmse"][0] > .16:
        raise RuntimeError("Forward or vertical predictions are too inaccurate for flat-policy BPTT")
    actor = FlatActor(args.classic, device)
    initial = actor.trained_weights()
    window = FlatWindows(data, 12)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=.00013)
    for step in range(1, args.imitation_updates+1):
        starts, _ = window.sample(512)
        guess = actor.infer(data.data["obs"][starts], data.history(starts))
        loss = F.smooth_l1_loss(guess, data.data["baseline_action"][starts], beta=.20)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 80 == 0 or step == args.imitation_updates:
            show("imitation", step, {"loss": loss})
    np.savez(args.out / "imitation.npz", **actor.trained_weights())
    report.update(phase="policy", imitation_updates=args.imitation_updates)
    save_report(status, report)
    cfg = BpttPolicyLearnerCfg(lr=.000045, batch_size=192, horizon=12, gamma=.98, noise=0.)
    world = FlatWorld(dynamics, state, window, home, cfg)
    learner = BpttPolicyLearner(actor, world, cfg, device)
    learner._captured_update = learner._update_from
    for step in range(1, args.actor_updates+1):
        stats = learner.update()
        if step == 1 or step % 24 == 0 or step == args.actor_updates:
            show("policy", step, stats)
        if step in (16, 48, 96, 144) or step == args.actor_updates:
            np.savez(args.out / f"policy-{step:04}.npz", **actor.trained_weights())
    final = actor.trained_weights()
    report.update(phase="complete", actor_updates=args.actor_updates,
                  parameter_change_l2={key: float(np.linalg.norm(final[key]-initial[key]))
                                       for key in CLASSIC_RETRAINED_KEYS})
    save_report(status, report)


if __name__ == "__main__":
    main()

"""Train M1 rolling turns with wheel support and yaw tracking in upstream SuperMimic."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from native_classic_train import FlatActor, FlatWindows, FlatWorld, predict_error, show
from native_model import NativeDynamicsCfg, TerrainDynamics, WheelState, heading_twist, ramp
from native_train import Replay, file_digest, save_report, upstream_source
from native_world import (CLASSIC_RETRAINED_KEYS, CONTACT_ACTION_LIMIT, CONTACT_CLEARANCE,
                          CONTACT_DIRECTION, CONTACT_SIDE, CONTACT_TRAVEL, CONTACT_YAW, contact_action_maps)
from sim import DT, M1_HOME, M1_SCALE
from supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearner

TRAVEL_GATE = (.06, .14)
SIDE_GATE = (.015, .080)


class ContactActor(FlatActor):
    def __init__(self, weights, checkpoint, device, refine_contact=False):
        super().__init__(weights, device)
        with np.load(checkpoint, allow_pickle=False) as saved, torch.no_grad():
            for key in CLASSIC_RETRAINED_KEYS:
                getattr(self, self.names[key]).copy_(torch.as_tensor(saved[key], device=device))
        self.refine_contact = refine_contact
        if refine_contact:
            self.requires_grad_(False)
            self.contact_gain = torch.nn.Parameter(torch.zeros(4, device=device))
            self.steering_gain = torch.nn.Parameter(torch.zeros(2, device=device))
            legs, wheels = contact_action_maps()
            self.register_buffer("contact_legs", torch.as_tensor(legs, device=device))
            self.register_buffer("contact_wheels", torch.as_tensor(wheels, device=device))
        variance = torch.zeros(49, device=device)
        variance[:6] = .0025
        variance[9:21] = .005
        variance[21:25] = .005
        self.register_buffer("sensor_noise", variance)

    def infer(self, obs, history):
        travel = ramp(obs[:, 6:7].abs()*.5, TRAVEL_GATE)
        rolling = travel * (1-ramp(obs[:, 7:8].abs()*.5, SIDE_GATE))
        obs_scale = 1.-rolling
        obs = torch.cat((obs[:, :41], obs[:, 41:]*obs_scale), dim=-1)
        history = torch.cat((history[..., :41], history[..., 41:]*obs_scale[:, None]), dim=-1)
        if self.training:
            obs = obs + torch.randn_like(obs)*self.sensor_noise
        return super().infer(obs, history)

    def forward(self, data):
        action = self.infer(data["obs"], data["history"])
        obs = data["obs"]
        if self.refine_contact:
            vx, vy, yaw = (obs[:, 6:9] * obs.new_tensor([.5, .5, 4.])).unbind(-1)
            active = (ramp(vx.abs(), CONTACT_TRAVEL) * ramp(yaw.abs(), CONTACT_YAW) *
                      (1 - ramp(vy.abs(), CONTACT_SIDE)))[:, None]
            height = data["wheel_elevation"]
            gap = (height - height.amin(-1, keepdim=True) - CONTACT_CLEARANCE).relu()
            knee = (gap * self.extension_gain()).clamp(max=CONTACT_ACTION_LIMIT)
            direction = ramp(vx, CONTACT_DIRECTION)
            reverse, forward = self.yaw_gain().unbind(0)
            steer = (reverse + direction * (forward - reverse)) * yaw
            action = action + active * (knee @ self.contact_legs + steer[:, None] * self.contact_wheels)
        # Keep an action prior, with less weight on the demonstrator's leg swing
        # during arcs. The rolling objective comes from predicted wheel contact.
        arc = ramp(obs[:,6].abs()*.5, TRAVEL_GATE) * ramp(obs[:,8].abs()*4., (.045,.12))
        err = F.smooth_l1_loss(action, data["expert"], beta=.14, reduction="none").mean(-1)
        self.imitation = DT * ((.38-.27*arc)*err).mean()
        return action

    def extension_gain(self):
        return 20. + 16. * self.contact_gain.tanh()

    def yaw_gain(self):
        return .27 * self.steering_gain.tanh()

    def trained_weights(self):
        weights = super().trained_weights()
        if self.refine_contact:
            weights["contact_gain"] = self.extension_gain().detach().cpu().numpy()
            weights["steering_gain"] = self.yaw_gain().detach().cpu().numpy()
        return weights


class ContactWorld(FlatWorld):
    def __init__(self, dynamics, state, windows, home, cfg, support_weight=1., tilt_weight=0.,
                 yaw_weight=1., refine_contact=False):
        super().__init__(dynamics, state, windows, home, cfg)
        self.support_weight = support_weight
        self.tilt_weight = tilt_weight
        self.yaw_weight = yaw_weight
        self.refine_contact = refine_contact

    def observation(self):
        observation = super().observation()
        if self.refine_contact:
            observation["wheel_elevation"] = self.dynamics.wheel_positions(self.rep, self.aux)[..., 2]
        return observation

    def step(self, action):
        previous_height = self.centre(self.aux)
        old_wheels = self.dynamics.wheel_positions(self.rep, self.aux)[..., 2]
        previous = self.previous
        target = self.data["command"][:,self.tick] * self.units
        self.aux, self.rep = self.dynamics(self.rep, action, self.aux, self.data["context"])
        self.history = torch.cat((self.history[:,1:], self.obs[:,None]), dim=1)
        self.previous = action
        self.tick += 1

        actual = heading_twist(self.rep, self.aux)
        h = self.centre(self.aux)
        wheels = self.dynamics.wheel_positions(self.rep, self.aux)[..., 2]
        clearance = (wheels-wheels.amin(-1, keepdim=True)-.006).relu()
        wheel_cost = (clearance/.010).square().clamp(max=7.)
        wheel_rate = ((wheels-old_wheels)/(DT*.14)).square().mean(-1).clamp(max=4.)
        arc = ramp(target[:,0].abs(), TRAVEL_GATE) * ramp(target[:,2].abs(), (.04,.12))
        arc *= 1-ramp(target[:,1].abs(), SIDE_GATE)
        support = self.support_weight*arc*(.55*wheel_cost.mean(-1) + .42*wheel_cost.amax(-1) + .14*wheel_rate)

        error = (actual-target) / actual.new_tensor([.16,.15,.14])
        tracking = (.52*error[:,0].square() + .30*error[:,1].square() +
                    self.yaw_weight*(.75+.65*arc)*error[:,2].square()).clamp(max=6.)
        centre = .23*((h-self.data["height"][:,self.tick])/.042).square().clamp(max=4.)
        vertical = .18*((h-previous_height)/(DT*.16)).square().clamp(max=4.)
        leg_motion = .08*(self.rep[:,26+self.legs]/2.3).square().mean(-1).clamp(max=4.)
        slew = .12*((action[:,self.legs]-previous[:,self.legs])/.18).square().mean(-1).clamp(max=4.)
        tilt = self.tilt_weight * arc * ((self.rep[:,1:3].abs()-.042).relu()/.065).square().mean(-1).clamp(max=4.)
        done = torch.maximum(torch.sigmoid((.22-h)/.024), torch.sigmoid((self.rep[:,3]+.66)/.025))
        return self.observation(), (1.-tracking-support-centre-vertical-leg_motion-slew-tilt-done)*DT, done, {}


@torch.no_grad()
def wheel_prediction_error(learner, windows):
    trial = windows.world_batch(2048)
    rep = learner.representation.represent(trial["state"])
    aux = learner.representation.extract_aux_state(trial["state"])
    predicted, pose = learner._rollout(rep[:,0], trial["action"], aux[:,0], trial["sim_ctx"][:,0])
    target_z = learner.dynamics.wheel_positions(rep[:,-1],aux[:,-1])[...,2]
    actual_z = learner.dynamics.wheel_positions(predicted[:,-1],pose[:,-1])[...,2]
    return (actual_z-target_z).square().mean(0).sqrt().tolist()


def main():
    p = ArgumentParser(description=__doc__)
    for key in ("train", "holdout", "straight", "classic", "initial", "world-start", "out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--world-updates",type=int,default=760)
    p.add_argument("--actor-updates",type=int,default=360)
    p.add_argument("--seed",type=int,default=74231)
    p.add_argument("--extra",type=Path,action="append",default=[],help="Additional replay from a previous checkpoint")
    p.add_argument("--support-weight",type=float,default=1.)
    p.add_argument("--tilt-weight",type=float,default=0.)
    p.add_argument("--yaw-weight",type=float,default=1.)
    p.add_argument("--policy-lr",type=float,default=.00012)
    p.add_argument("--refine-contact",action="store_true",help="Freeze the actor and train bounded wheel-contact and yaw corrections")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    home = torch.as_tensor(M1_HOME,dtype=torch.float32,device=device)
    scale = torch.as_tensor(M1_SCALE,dtype=torch.float32,device=device)
    data = Replay([args.train,args.straight,*args.extra],device)
    holdout = Replay([args.holdout],device)
    if data.seeds & holdout.seeds:
        raise ValueError("Training and holdout scenario seeds overlap")

    args.out.mkdir(parents=True,exist_ok=False)
    status = args.out/"training.json"
    report = {"phase":"world", "seed":args.seed, "supermimic":upstream_source(),
              "device":torch.cuda.get_device_name(), "train":data.sources, "holdout":holdout.sources,
              "initial_sha256":file_digest(args.initial), "support_weight":args.support_weight, "tilt_weight":args.tilt_weight,
              "policy_lr":args.policy_lr, "yaw_weight":args.yaw_weight, "refine_contact":args.refine_contact,
              "objectives":["all-wheel support",
              "per-wheel vertical rate", "trunk height and vertical rate", "forward and yaw tracking", "leg action slew"]}
    save_report(status,report)
    state = WheelState(home).to(device)
    dynamics = TerrainDynamics(state,home,scale).to(device)
    dynamics.load_state_dict(torch.load(args.world_start,map_location=device,weights_only=True))
    cfg = NativeDynamicsCfg(lr=.0004,batch_size=256,num_batches=1,horizon=8)
    learner = WorldModelLearner(dynamics,state,cfg,device)
    learner._captured_update = learner._update_from
    learner.update_normalization(*data.normalizer_data())
    windows = FlatWindows(data,cfg.horizon)
    for i in range(1,args.world_updates+1):
        stats = learner.update([windows.world_batch(cfg.batch_size)])
        if i==1 or i%100==0 or i==args.world_updates:
            show("world",i,stats)
    reserved = FlatWindows(holdout,cfg.horizon)
    accuracy = predict_error(learner,reserved)
    accuracy["wheel_height_rmse_m"] = wheel_prediction_error(learner,reserved)
    print("world holdout",json.dumps(accuracy),flush=True)
    if max(accuracy["wheel_height_rmse_m"])>.020 or accuracy["twist_rmse"][2]>.12:
        raise RuntimeError("Held-out wheel height or yaw prediction is too inaccurate")
    torch.save(dynamics.state_dict(),args.out/"dynamics.pt")
    report.update(phase="policy",world_updates=args.world_updates,world_holdout=accuracy)
    save_report(status,report)

    actor = ContactActor(args.classic,args.initial,device,args.refine_contact)
    initial = actor.trained_weights()
    policy_cfg = BpttPolicyLearnerCfg(lr=args.policy_lr,batch_size=256,horizon=16,gamma=.985,noise=0.)
    world = ContactWorld(dynamics,state,FlatWindows(data,policy_cfg.horizon),home,policy_cfg,args.support_weight,args.tilt_weight,
                         args.yaw_weight,args.refine_contact)
    policy = BpttPolicyLearner(actor,world,policy_cfg,device)
    policy._captured_update = policy._update_from
    for i in range(1,args.actor_updates+1):
        statistics = policy.update()
        if i==1 or i%24==0 or i==args.actor_updates:
            show("policy",i,statistics)
        if i in (16,32,64,96,144,216,288) or i==args.actor_updates:
            np.savez(args.out/f"policy-{i:04}.npz",**actor.trained_weights())
    parameters = actor.trained_weights()
    report.update(phase="complete",actor_updates=args.actor_updates,parameter_change_l2={
        key:float(np.linalg.norm(parameters[key]-initial[key])) for key in parameters})
    if args.refine_contact:
        report.update(contact_gain=parameters["contact_gain"].tolist(), steering_gain=parameters["steering_gain"].tolist())
    save_report(status,report)


if __name__=="__main__":
    main()

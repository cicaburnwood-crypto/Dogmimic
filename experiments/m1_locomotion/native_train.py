"""Train one M1 flat/stairs policy using the original SuperMimic learners."""
from argparse import ArgumentParser
from hashlib import file_digest as hash_file
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn import functional as F

from native_model import NativeDynamicsCfg, Student, Teacher, TerrainDynamics, WheelState, heading_twist
from native_world import DESCENT, FLAT, FRONT_WHEELS, MIXED, NATIVE_SHA256, STAIRS
from sim import DT, LEGS, M1_HOME, M1_SCALE, WHEELS
from supermimic.learners.policy_learner import BpttPolicyLearner, BpttPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearner

UPSTREAM = "abc124c1bd5976341034668af3d12c087f2f1f55"


def file_digest(path):
    with path.open("rb") as file:
        return hash_file(file, "sha256").hexdigest()


def upstream_source():
    classes = (WorldModelLearner, BpttPolicyLearner)
    paths = [Path(inspect.getfile(cls)).resolve() for cls in classes]
    source = paths[0].parents[3]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "src/supermimic"], text=True)
    if revision != UPSTREAM or dirty or any(not p.is_relative_to(source / "src/supermimic") for p in paths):
        raise RuntimeError("The run must use the pinned upstream SuperMimic sources")
    return {"repo": "HybridRobotics/supermimic", "commit": revision, "files": [str(p) for p in paths]}


class Replay:
    def __init__(self, paths, device):
        self.device = device
        keys = ("state", "obs", "action", "target", "contact", "baseline_action",
                "course", "length", "kind", "seed", "finished", "start", "guided")
        blocks = {key: [] for key in keys}
        self.sources = []
        offset = 0
        for path in paths:
            manifest = json.loads(path.with_suffix(".json").read_text())
            if any(manifest["sha256"].get(k) != digest for k,digest in NATIVE_SHA256.items()):
                raise ValueError(f"Dataset carries different teachers: {path}")
            self.sources.append({"file": path.name, "sha256": file_digest(path), "policy": manifest["policy"],
                                 "transitions": manifest["transitions"], "candidate": manifest["sha256"].get("candidate"),
                                 "guidance": manifest.get("guidance")})
            with np.load(path, allow_pickle=False) as data:
                length = data["length"]
                rows = int((length+1).sum())
                if int(length.sum()) != manifest["transitions"] or not np.array_equal(data["start"], np.cumsum(np.r_[0, length[:-1]+1])):
                    raise ValueError(f"Invalid episode boundaries in {path}")
                if any(data[key].shape[0] != rows for key in ("state", "obs", "action", "target", "contact")):
                    raise ValueError(f"Inconsistent frame count in {path}")
                for key in keys:
                    if key == "guided":
                        value = np.full(len(length), "guidance" in manifest, dtype=bool)
                    elif key == "baseline_action":
                        value = data[key] if key in data else data["action"]
                        if value.shape != (rows, 16):
                            raise ValueError(f"Invalid reference actions in {path}")
                    else:
                        value = data[key]
                    blocks[key].append(value + offset if key == "start" else value)
                offset += rows
        arrays = {k: np.concatenate(v) for k,v in blocks.items()}
        if len(np.unique(arrays["seed"])) != len(arrays["seed"]):
            raise ValueError("Episodes contain duplicate scenario seeds")
        self.seeds = set(arrays["seed"].tolist())
        self.has_descent = np.any(arrays["kind"] == DESCENT).item()
        self.data = {k: torch.from_numpy(v).to(device) for k,v in arrays.items()}
        counts = self.data["length"]+1
        self.episode = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
        self.frame = torch.arange(offset, device=device)
        self.elapsed = self.frame - self.data["start"][self.episode]
        self.kind = self.data["kind"][self.episode]
        self.active = self.elapsed < self.data["length"][self.episode]
        contact = self.data["contact"]
        self.contact_change = torch.cat((torch.zeros(1, device=device, dtype=torch.bool),
                                         (contact[:-1] != contact[1:]).any(-1)))
        self.front_risks = None
        self.cache = {}

    def normalizer_data(self):
        return self.data["state"][self.active], self.data["action"][self.active]

    @torch.no_grad()
    def index_front_risks(self, dynamics, state):
        self.front_risks = torch.zeros((len(self.frame), 2), device=self.device, dtype=torch.bool)
        eligible = self.active & (self.kind != FLAT) & (self.data["obs"][:, 6] > .30)
        for frames in self.frame[eligible].split(32768):
            rows = self.data["state"][frames]
            wheels = dynamics.wheel_positions(state.represent(rows), state.extract_aux_state(rows))
            _, near, shortfall = dynamics.front_risers(wheels, self.data["course"][self.episode[frames]])
            self.front_risks[frames] = (near * (shortfall/.08).clamp(0., 1.)).amax(-1) > .45
        self.cache.clear()
        return self.front_risks.sum(dim=0).tolist()

    def starts(self, horizon, batch):
        if horizon not in self.cache:
            valid = (self.elapsed >= 5) & (self.elapsed+horizon < self.data["length"][self.episode])
            kinds = (FLAT, STAIRS, MIXED, DESCENT) if self.has_descent else (FLAT, STAIRS, MIXED)
            groups = [self.frame[valid & (self.kind == k)] for k in kinds]
            changes = [self.frame[valid & self.contact_change & (self.kind == k)] for k in kinds[1:]]
            risks = []
            if self.front_risks is not None:
                risks = [tuple(self.frame[valid & (self.kind == k) & risk] for risk in self.front_risks.unbind(-1))
                         for k in kinds[1:]]
            fast = self.frame[valid & (self.kind == FLAT) & (self.data["obs"][:, 6] > 1.1)]
            if any(not len(rows) for rows in groups):
                raise ValueError("Replay requires floor, stair and transition windows")
            self.cache[horizon] = groups, changes, fast, risks
        groups, changes, fast, risks = self.cache[horizon]
        if self.has_descent:
            floor, descent = batch//3, batch//3
            stairs = (batch-floor-descent)//2
            sizes = (floor, stairs, batch-floor-descent-stairs, descent)
        else:
            floor = batch*2//5
            sizes = (floor, (batch-floor)//2, batch-floor-(batch-floor)//2)

        def sample(frames, size):
            return frames[torch.randint(len(frames), (size,), device=self.device)]

        n_fast = sizes[0]//6 if len(fast) else 0
        selected = [sample(groups[0], sizes[0]-n_fast)]
        if n_fast:
            selected.append(sample(fast, n_fast))
        for i, (rows, events, n) in enumerate(zip(groups[1:], changes, sizes[1:], strict=True)):
            n_contact = n//(5 if risks else 3) if len(events) else 0
            n_riser = n//6 if risks and all(len(pool) for pool in risks[i]) else 0
            selected.append(sample(rows, n-n_contact-2*n_riser))
            if n_contact:
                selected.append(sample(events, n_contact))
            if n_riser:
                selected.extend(sample(pool, n_riser) for pool in risks[i])
        return torch.cat(selected)

    def windows(self, horizon, batch):
        starts = self.starts(horizon, batch)
        episode = self.episode[starts]
        frames = starts[:, None]+torch.arange(horizon+1, device=self.device)
        return TensorDict({"state": self.data["state"][frames], "action": self.data["action"][frames],
                           "sim_ctx": self.data["course"][episode, None].expand(-1, horizon+1, -1)},
                          batch_size=[batch]), starts, frames

    def history(self, frames):
        steps = frames[:, None] - torch.arange(5, 0, -1, device=self.device)
        return self.data["obs"][steps]


@torch.no_grad()
def target_residual(student, classic, obs, history, flat):
    desired = classic(obs, history) - student.stairs(obs, history)
    limit = student.action_limit * student.authority(obs)
    return desired.clamp(-limit, limit) * flat[:, None]


def fit_imitation(student, classic, replay, out, updates):
    optimizer = torch.optim.AdamW(student.correction.parameters(), lr=.00045, weight_decay=.00003)
    for i in range(1, updates+1):
        frames = replay.starts(1, 1024)
        obs = replay.data["obs"][frames]
        history = replay.history(frames).flatten(-2)
        flat = (replay.kind[frames] == FLAT).float()
        good = replay.data["finished"][replay.episode[frames]].float()
        target = target_residual(student, classic, obs, history, flat)
        prediction = student.manoeuvre_delta(obs, history)
        loss = student.imitation_loss(prediction, target, flat, good)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.correction.parameters(), .5)
        optimizer.step()
        if i == 1 or i % 200 == 0 or i == updates:
            print("imitation", i, "loss", float(loss.detach()), flush=True)
        if i in (200, 600) or i == updates:
            student.save(out / f"imitation-{i:04}.npz")


def fit_front_lift(student, dynamics, state, replay, out, updates):
    front = student.front_lift
    optimizer = torch.optim.AdamW(front.parameters(), lr=.0007)
    for i in range(1, updates+1):
        frames = replay.starts(1, 768)
        obs = replay.data["obs"][frames]
        history = replay.history(frames).flatten(-2)
        turn = student.authority(obs)
        with torch.no_grad():
            rows = replay.data["state"][frames]
            wheels = dynamics.wheel_positions(state.represent(rows), state.extract_aux_state(rows))
            context = replay.data["course"][replay.episode[frames]]
            _, near, shortfall = dynamics.front_risers(wheels, context)
            need = (near * (shortfall/.14).clamp(0., 1.)).amax(-1)
            heights = wheels[:, list(FRONT_WHEELS), 2]
            lag = ((heights.flip(-1)-heights-.05)/.09).clamp(0., 1.)
            slow = ((.25-obs[:, [21,23]])/.11).clamp(0., 1.)
            wheel = .30*lag*slow * near.amax(-1)
            target = torch.stack((.045*need, -.085*need, wheel), dim=-1)
            target = target * front.authority(obs, turn)[..., None]
        predicted = front(obs, history, student.bias, student.inverse_std, turn)
        loss = F.smooth_l1_loss(predicted, target, beta=.045)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(front.parameters(), .5)
        optimizer.step()
        if i == 1 or i % 160 == 0 or i == updates:
            print("front imitation", i, "loss", float(loss.detach()), flush=True)
    student.save(out / "front-imitation.npz")


class M1World:
    def __init__(self, dynamics, state, replay, student, classic, home, cfg, front_reward=0.):
        self.dynamics = dynamics
        self.representation = state
        self.replay = replay
        self.student = student
        self.classic = classic
        self.home = home
        self.cfg = cfg
        self.front_reward = front_reward
        self.legs = torch.as_tensor(LEGS, device=home.device)
        self.wheels = torch.as_tensor(WHEELS, device=home.device)
        self.command_tolerance = home.new_tensor([.18, .13, .28])

    def sample_initial_states(self):
        window, starts, frames = self.replay.windows(self.cfg.horizon, self.cfg.batch_size)
        d = self.replay.data
        obs = d["obs"][frames]
        return TensorDict({"state": window["state"][:, 0], "context": window["sim_ctx"][:, 0],
                           "target": d["target"][frames], "obs_cmd": obs[..., 6:9], "phase": obs[..., 41:49],
                           "history": self.replay.history(starts), "previous": obs[:, 0, 25:41],
                           "flat": (self.replay.kind[starts] == FLAT).float(),
                           "good": d["finished"][self.replay.episode[starts]].float()}, batch_size=[len(starts)])

    def reset(self, data):
        self.batch = data
        self.tick = 0
        self.rep = self.representation.represent(data["state"])
        self.aux = self.representation.extract_aux_state(data["state"])
        self.history = data["history"]
        self.last = data["previous"]
        return self.observation()

    def observation(self):
        t = self.tick
        rep = self.rep
        self.obs = torch.cat((rep[:, 7:10]*.25, rep[:, 1:4], self.batch["obs_cmd"][:, t],
                              rep[:, 10:26][:, self.legs]-self.home[self.legs],
                              rep[:, 26:42][:, self.wheels]*.05, self.last, self.batch["phase"][:, t]), dim=-1)
        flat = self.batch["flat"]
        anchor = target_residual(self.student, self.classic, self.obs, self.history.flatten(-2), flat)
        return TensorDict({"obs": self.obs, "history": self.history, "critic": rep, "anchor": anchor,
                           "flat": flat, "good": self.batch["good"]}, batch_size=[len(rep)])

    def step(self, action):
        old_aux = self.aux
        old_action = self.last
        context = self.batch["context"]
        if self.front_reward:
            previous_wheels = self.dynamics.wheel_positions(self.rep, old_aux)
        self.aux, self.rep = self.dynamics(self.rep, action, old_aux, context)
        self.last = action
        self.history = torch.cat((self.history[:, 1:], self.obs[:, None]), dim=1)
        twist = heading_twist(self.rep, self.aux)
        goal = self.batch["target"][:, self.tick]
        error = ((twist-goal)/self.command_tolerance).square().mean(dim=-1)
        tracking = (-error).exp()
        low_speed = goal.abs().amax(dim=-1) < .03
        stopping = (-12*twist[:, :2].square().sum(dim=-1)-2*twist[:, 2].square()).exp()*low_speed
        rise, tread, steps, edge, width, _ = context.unbind(-1)
        x, y, _ = self.aux[:, :3].unbind(-1)
        climbing = (steps > 0) & (x >= edge-.35) & (x <= edge + tread*steps+.30) & (goal[:, 0] > .15)
        forward = ((x-old_aux[:, 0])/.012).tanh()
        _, _, support = self.dynamics.terrain(self.rep, self.aux, context)
        contact = support.mean(dim=-1)
        lateral = ((y.abs()/width.clamp_min(.1)).clamp(0,2)).square()
        progress = (.34*forward + .03*contact - .13*lateral)*climbing
        top = self.dynamics.ground_height(self.aux[:, None, :2], context).squeeze(-1)
        height = self.aux[:, 2]-top
        bad_pose = torch.sigmoid((.22-height)/.023)
        bad_tilt = torch.sigmoid((self.rep[:, 3]+.66)/.025)
        termination = torch.maximum(bad_pose, bad_tilt)
        effort = (old_action-action).square().mean(dim=-1)
        reward = 1.18*tracking + .19*stopping + progress - .012*effort - .5*termination
        if self.front_reward:
            wheels = self.dynamics.wheel_positions(self.rep, self.aux)
            score = self.dynamics.front_riser_score(previous_wheels, wheels, context)
            reward = reward + self.front_reward*score*(goal[:, 0] > .15)
        self.tick += 1
        return self.observation(), reward*DT, termination, {}


@torch.no_grad()
def validate_world(learner, replay, horizon):
    learner.dynamics.eval()
    window, starts, frames = replay.windows(horizon, 3072)
    state = learner.representation
    truth = state.represent(window["state"])
    aux = state.extract_aux_state(window["state"])
    context = window["sim_ctx"]
    pred, poses = learner._rollout(truth[:, 0], window["action"], aux[:, 0], context[:, 0])
    expected = heading_twist(truth[:, 1:], aux[:, 1:])
    predicted = heading_twist(pred, poses)
    true_wheels = learner.dynamics.wheel_positions(truth[:, -1], aux[:, -1])
    predicted_wheels = learner.dynamics.wheel_positions(pred[:, -1], poses[:, -1])
    front_error = (predicted_wheels-true_wheels)[:, list(FRONT_WHEELS), 2]
    persistence = heading_twist(truth[:, 0], aux[:, 0])
    _, _, support = learner.dynamics.terrain(pred[:, -1], poses[:, -1], context[:, 0])
    contacts = replay.data["contact"][frames[:, -1]] > 0
    metrics = {}
    kinds = [("flat", FLAT), ("stairs", STAIRS), ("mixed", MIXED)]
    if replay.has_descent:
        kinds.append(("descent", DESCENT))
    for label, kind in kinds:
        mask = replay.kind[starts] == kind
        error = predicted[mask, -1] - expected[mask, -1]
        rmse = error.square().mean(dim=0).sqrt()
        default = (persistence[mask]-expected[mask, -1]).square().mean(dim=0).sqrt()
        metrics[label] = {"h8_twist_rmse": rmse.cpu().tolist(), "persistence_rmse": default.cpu().tolist(),
                          "h8_height_rmse": float((pred[mask, -1,0]-truth[mask, -1,0]).square().mean().sqrt()),
                          "h8_front_wheel_z_rmse_m": front_error[mask].square().mean(0).sqrt().cpu().tolist(),
                          "wheel_contact_accuracy": float(((support[mask] > .5) == contacts[mask]).float().mean())}
    return metrics


def save_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def main():
    p = ArgumentParser(description=__doc__)
    for key in ("train", "holdout", "teachers", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--seed", type=int, default=541)
    p.add_argument("--world-updates", type=int, default=2200)
    p.add_argument("--imitation-updates", type=int, default=1000)
    p.add_argument("--actor-updates", type=int, default=144)
    p.add_argument("--initial-drive", type=float, default=.16, help="Straight-wheel correction at 0.6 m/s before BPTT")
    p.add_argument("--front-lift", action="store_true", help="Learn clearance and drive corrections for the two front legs")
    p.add_argument("--init-policy", type=Path, help="Retain the previously trained rolling and steering parameters")
    p.add_argument("--front-imitation-updates", type=int, default=480)
    p.add_argument("--front-reward", type=float, default=1.35, help="Per-wheel riser clearance reward weight in BPTT")
    p.add_argument("--eager", action="store_true")
    args = p.parse_args()
    if abs(args.initial_drive) >= .30:
        p.error("The straight wheel correction must begin inside its output bound")
    if args.init_policy and not args.front_lift:
        p.error("--init-policy is for continued front-lift training")
    if args.front_lift and (args.front_imitation_updates < 1 or not 0. < args.front_reward <= 5.):
        p.error("Front-lift training requires positive imitation updates and clearance reward")
    source = upstream_source()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    home = torch.as_tensor(M1_HOME, dtype=torch.float32, device=device)
    scale = torch.as_tensor(M1_SCALE, dtype=torch.float32, device=device)
    replay = Replay([args.train, *args.extra], device)
    holdout = Replay([args.holdout], device)
    if replay.seeds & holdout.seeds:
        raise ValueError("The world-model holdout overlaps the training data")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"seed": args.seed, "supermimic": source, "device": torch.cuda.get_device_name(),
              "teacher_policies": NATIVE_SHA256, "train": replay.sources, "holdout": holdout.sources,
              "actor_observation_sizes": [49, 245], "action_size": 16,
              "initial_drive": args.initial_drive, "phase": "world_model"}
    status_path = args.out / "training.json"
    save_report(status_path, report)
    states, actions = replay.normalizer_data()
    repr_ = WheelState(home).to(device)
    dynamics = TerrainDynamics(repr_, home, scale).to(device)
    if args.front_lift:
        counts = replay.index_front_risks(dynamics, repr_)
        if min(counts) == 0:
            raise ValueError("The replay is missing riser approaches for a front leg")
        report["front_lift"] = {"legs": ["LF", "RF"], "reward_weight": args.front_reward,
                                "riser_windows": dict(zip(("LF", "RF"), counts)),
                                "base_policy_sha256": file_digest(args.init_policy) if args.init_policy else None}
        save_report(status_path, report)
        print("front riser windows", counts, flush=True)
    wm_cfg = NativeDynamicsCfg(lr=.00085, batch_size=256, num_batches=1, horizon=8)
    wm = WorldModelLearner(dynamics, repr_, wm_cfg, device)
    if args.eager:
        wm._captured_update = wm._update_from
    wm.update_normalization(states, actions)
    print("device", torch.cuda.get_device_name(), "replay samples", len(actions), flush=True)
    for i in range(1, args.world_updates+1):
        values = wm.update([replay.windows(8, wm_cfg.batch_size)[0]])
        if i == 1 or i % 200 == 0 or i == args.world_updates:
            stats = {k: float(v) for k,v in values.items()}
            if not np.isfinite(list(stats.values())).all():
                raise RuntimeError("World model loss diverged")
            print("world", i, stats, flush=True)
    metrics = validate_world(wm, holdout, wm_cfg.horizon)
    report.update(world_updates=args.world_updates, world_validation=metrics, phase="imitation")
    print("world holdout", json.dumps(metrics), flush=True)
    save_report(status_path, report)
    torch.save(dynamics.state_dict(), args.out / "dynamics.pt")
    if any(group["h8_twist_rmse"][0] > .14 or not np.isfinite(group["h8_twist_rmse"]).all() for group in metrics.values()):
        raise RuntimeError("Forward-velocity world-model validation failed")
    classic, stairs = (Teacher(args.teachers / f"{key}.npz", device) for key in ("classic", "stairs"))
    idx = replay.starts(1, 40960)
    samples = torch.cat((replay.data["obs"][idx], replay.history(idx).flatten(-2)), dim=-1)
    student = Student(stairs, samples.mean(0), samples.std(0).clamp_min(.075).reciprocal(), device,
                      args.initial_drive, front_lift=args.front_lift)
    if args.init_policy:
        student.load_base(args.init_policy)
    else:
        fit_imitation(student, classic, replay, args.out, args.imitation_updates)
    if args.front_lift:
        fit_front_lift(student, dynamics, repr_, replay, args.out, args.front_imitation_updates)
        report["front_lift"]["imitation_updates"] = args.front_imitation_updates
    report.update(imitation_updates=0 if args.init_policy else args.imitation_updates, phase="policy")
    save_report(status_path, report)
    cfg = BpttPolicyLearnerCfg(lr=.00015 if args.front_lift else .00011, batch_size=256, horizon=12, gamma=.98, noise=0.)
    world = M1World(dynamics, repr_, replay, student, classic, home, cfg,
                    front_reward=args.front_reward if args.front_lift else 0.)
    policy = BpttPolicyLearner(student, world, cfg, device)
    if args.eager:
        policy._captured_update = policy._update_from
    for i in range(1, args.actor_updates+1):
        values = policy.update()
        if i in (1, 16) or i % 32 == 0 or i == args.actor_updates:
            stats = {k: float(v) for k,v in values.items()}
            if not np.isfinite(list(stats.values())).all():
                raise RuntimeError("BPTT policy training diverged")
            print("policy", i, stats, flush=True)
        if i in (16, 48, 96, 144, 256, 384) or i == args.actor_updates:
            student.save(args.out / f"policy-{i:04}.npz")
    report.update(actor_updates=args.actor_updates, phase="complete")
    save_report(status_path, report)


if __name__ == "__main__":
    main()

"""Isolated SuperMimic critic ablation. No physical criteria or robot control."""
import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / 'diagnostics/training_redesign_20261102'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def setup(robot):
    """Each subprocess has exactly one robot's import/physics namespace."""
    os.chdir(ROOT)
    if robot == 'B2':
        os.environ.update(SM_ROBOT='b2', SM_TASK='b2_light_payload', B2_WM_PREFIT='0')
        paths = ['b2_light_payload', 'three_skills/engine', 'b2_payload_recovery',
                 'b2_payload', 'b2_cycle', 'three_skills']
        sys.path[:0] = [str(ROOT / 'experiments' / p) for p in paths]
        import train_repair as old
        from train import SkillTrainer
        from supermimic import train as base
        from robust_env import make_env

        class Implementation(CriticCore, old.GuardedTrainer):
            robot_kind = 'B2'

            def __init__(self, cfg, device='cuda'):
                base.Trainer.__init__(self, cfg, device)
                self.core_init()

        return Implementation, base, old.GuardedTrainer.update_policy
    if robot != 'G1':
        raise ValueError(robot)
    sys.path[:0] = [str(ROOT / 'experiments/g1_boundary_repair'),
                    str(ROOT / 'experiments/g1_boundary_repair/engine')]
    from supermimic import train as base
    from robust_env import RobustEnv

    class Implementation(CriticCore, base.Trainer):
        robot_kind = 'G1'

        def _make_env(self):
            e = RobustEnv(self.lib, self.device, self.cfg.num_envs, self.cfg.gamma)
            self.tcfg = e.tcfg
            return e

        def __init__(self, cfg, device='cuda'):
            super().__init__(cfg, device)
            self.core_init()

    return Implementation, base, base.Trainer.update_policy


@contextlib.contextmanager
def isolated_rng(device):
    import numpy as np
    import torch
    py, np_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def tensor_digest(mapping):
    import torch
    h = hashlib.sha256()
    for key, value in sorted(mapping.items()):
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().contiguous()
            h.update(key.encode())
            h.update(str((v.shape, v.dtype)).encode())
            h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class CriticCore:
    """Shared on/off path; original BPTT and imagined lambda-return fitting."""
    def core_init(self):
        import torch
        from supermimic.networks import Critic
        m = read_json(DESIGN / 'experiment_matrix.json')['robots'][self.robot_kind]
        rel = Path(m['initialization']['checkpoint']).relative_to('/home/liudiwen/projects/supermimic_vibe')
        self.source_path = ROOT / rel
        self.source_hash = sha256(self.source_path)
        assert self.source_hash == m['initialization']['sha256']
        ck = torch.load(self.source_path, map_location=self.device, weights_only=False)
        self.source_metadata = ck
        self.source_iteration = int(ck['it'])
        self.policy.load_state_dict(ck['policy'], strict=True)
        self.wm.load_state_dict(ck['wm'], strict=True)
        assert all(torch.equal(v, self.policy.state_dict()[k]) for k, v in ck['policy'].items())
        assert all(torch.equal(v, self.wm.state_dict()[k]) for k, v in ck['wm'].items())
        if self.robot_kind == 'B2':
            self.policy.teacher.requires_grad_(True)
            self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr*.02,
                                          betas=(.9, .95), fused=True)
        self.policy.obs_norm.until = 0
        self.policy.goal_norm.until = 0
        self.frozen_anchor = copy.deepcopy(self.policy).requires_grad_(False).eval()
        self.actor_lr_scale, self.teacher_weight = .2, .05
        with isolated_rng(self.device):
            torch.manual_seed(self.cfg.seed + 701)
            goal = self.env.policy_inputs()[1]
            self.critic_in = self.feat_dim + goal.shape[-1] + self.env.xi.shape[-1] + 1
            self.critic = Critic(self.critic_in).to(self.device)
        self.opt_v = torch.optim.Adam(self.critic.net.parameters(), lr=self.cfg.lr, fused=True)
        if self.world > 1:
            import torch.distributed as dist
            for v in list(self.critic.parameters()) + list(self.critic.buffers()):
                dist.broadcast(v.data, 0)
        self.actor_updates = self.critic_fits = self.wm_gradient_steps = 0
        self.started_at = time.time()
        self.core_phase = 'initial'
        self._last_rollout = None
        self.source_metadata = {k: v for k, v in ck.items()
                                if k in ('policy_architecture', 'task_cfg', 'robot')}

    def _extra_state(self):
        m = dict(getattr(self, 'source_metadata', {}))
        m.update(task_kind=self.robot_kind.lower()+'_supermimic_core',
                 source_checkpoint=str(self.source_path), source_sha256=self.source_hash,
                 source_iteration=self.source_iteration,
                 initialization='own Actor and WM continued; fresh replay/optimizers; new critic',
                 critic_bootstrap=self.cfg.use_critic, context_dim=self.env.xi.shape[-1],
                 actor_updates=self.actor_updates, critic_fits=self.critic_fits,
                 wm_gradient_steps=self.wm_gradient_steps, core_phase=self.core_phase,
                 reference_sha256=sha256(self.cfg.motions),
                 algorithm='SuperMimic WM BPTT with controlled original critic',
                 external_teacher=False)
        return m

    def update_wm(self):
        out = super().update_wm()
        self.wm_gradient_steps += self.cfg.wm_batches
        return out

    def value_context(self, state, xi):
        if self.robot_kind == 'G1':
            return xi
        import torch
        from train_recovery import RecoveryWorldModel
        return torch.cat([RecoveryWorldModel.canonical_context(state, xi), xi[..., 30:43]], -1)

    def imagined_objective(self):
        import torch
        from supermimic.task import NJ, flat_to_state
        from supermimic.networks import run_mlp, apply_delta_c
        cfg, task, lib, wm = self.cfg, self.task, self.lib, self.wm
        B, H, K = cfg.pi_batch, cfg.horizon, self.tcfg.future
        s0, xi, frame, hist, prev = self.buf.sample_starts(B, self.tcfg.history, task)
        with torch.no_grad():
            raw = frame[:, None] + torch.arange(H+K+1, device=self.device)
            idx = lib.clamp_to_clip(raw, frame)
            valid = (raw == idx)[:, :H+1].float()
            remaining = lib.remaining(idx[:, :H+1])
            rem = remaining.float().clamp(max=100)/100.
            value_valid = valid * (remaining > 0)
            ref = lib.frame(idx)
            kin = self.env.kin_c(ref['pos'], ref['rot'], ref['q'], ref['v'], ref['w'], ref['qd'])
        at = lambda t: {k: v[:, t] for k, v in ref.items()}
        future = lambda t: {k: v[:, t+1:t+1+K] for k, v in ref.items()}
        kat = lambda t: tuple(v[:, t] for v in kin)
        state = flat_to_state(s0)
        hist = hist + task.sample_obs_noise((B, self.tcfg.history))
        an = task.analytics_fast(state, at(0), kat(0), future(0), prev, prev, xi[:, 1:1+NJ])
        alive, J = torch.ones(B, device=self.device), torch.zeros(B, device=self.device)
        anchor_cost = torch.zeros(B, device=self.device)
        rec = {k: [] for k in ('x', 'r', 'd', 'm', 'c')}
        context = xi
        terms = {}
        for t in range(H):
            if self.robot_kind == 'B2':
                physical = self.context_at(xi, t, self.tcfg.dt)
                xt = torch.cat([physical, context[:, 21:]], -1)
            else:
                xt = context
            goal, feats = an['goal'], an['feats']
            rec['x'].append(self.critic_input(feats, goal, self.value_context(state, xt), rem[:, t]).detach())
            action = self.policy(hist.flatten(1), goal)
            prior = self.frozen_anchor(hist.flatten(1), goal)
            delta = run_mlp(wm.net, wm.in_norm(wm.inputs(state, action, xt, feats)))*wm.out_norm.std + wm.out_norm.mean
            successor = apply_delta_c(state, delta)
            if self.robot_kind == 'B2':
                an = task.transition_fast(state, successor, at(t+1), kat(t+1), future(t+1), action, prev, xt)
                anchor_cost = anchor_cost + (((action-prior)*.7/.04).square().mean(-1))*valid[:, t+1]
            else:
                an = task.analytics_fast(successor, at(t+1), kat(t+1), future(t+1), action, prev, xi[:, 1:1+NJ])
                anchor_cost = anchor_cost + ((action-prior)*task.action_scale/.05).square().mean(-1)
            reward, death, mask = an['r'], an['soft'], valid[:, t+1]
            J = J + cfg.gamma**t * alive * reward * mask
            for k, v in [('r', reward*mask), ('d', death), ('m', mask), ('c', alive)]:
                rec[k].append(v.detach())
            for k, v in an['terms'].items():
                terms[k] = terms.get(k, 0.) + v.detach().mean()/H
            alive = alive*(1-death)
            if self.robot_kind == 'B2':
                from actuator_filter import advance_context
                context = task.advance_context(successor, idx[:, t+1], xt)
                target = advance_context(xt, action, task.q0, task.action_scale, task.qmin, task.qmax)[:, 31:43]
                context = torch.cat([context[:, :31], target], -1)
            else:
                from supermimic.dynamic_context import advance
                context = advance(context, action, task.q0, task.action_scale)
            hist = torch.cat([hist[:, 1:], (an['obs']+task.sample_obs_noise((B,)))[:, None]], 1)
            state, prev = successor, action
        if self.robot_kind == 'B2':
            context = torch.cat([self.context_at(xi, H, self.tcfg.dt), context[:, 21:]], -1)
        xH = self.critic_input(an['feats'], an['goal'], self.value_context(state, context), rem[:, H])
        assert xH.shape[-1] == self.critic_in
        tail = cfg.gamma**H * alive * self.critic(xH) * value_valid[:, H]
        if cfg.use_critic:
            J = J + tail
        coeff = .05 if self.robot_kind == 'B2' else .03
        loss = -J.mean()/H + coeff*anchor_cost.mean()/H
        return loss, rec, xH, value_valid, dict(pi_J=float(J.detach().mean()),
                    pi_loss=float(loss.detach()), anchor_loss=float(anchor_cost.detach().mean()/H),
                    img_survival=float(alive.detach().mean()), tail_mean=float(tail.detach().mean()),
                    **{'img_r_'+k: float(v) for k, v in terms.items()})

    def update_policy(self, critic_only=False):
        import torch
        from supermimic.train import lr_at
        for p in self.wm.net.parameters():
            p.requires_grad_(False)
        for p in self.critic.net.parameters():
            p.requires_grad_(False)
        try:
            loss, rec, xH, valid, stats = self.imagined_objective()
            if not critic_only:
                self.opt_pi.zero_grad(set_to_none=True)
                loss.backward()
                params = [p for g in self.opt_pi.param_groups for p in g['params']]
                self.sync_grads(params)
                norm = torch.nn.utils.clip_grad_norm_(params, .5 if self.robot_kind=='B2' else 1.)
                if self.robot_kind == 'G1':
                    torch.nn.utils.clip_grad_value_(params, .1)
                scale = .02 if self.robot_kind=='B2' else .1
                for g in self.opt_pi.param_groups:
                    g['lr'] = lr_at(self.cfg, self.it)*scale*g.get('lr_multiplier', 1.)
                self.opt_pi.step()
                self.actor_updates += 1
                stats.update(pi_gnorm=float(norm), actor_lr=self.opt_pi.param_groups[0]['lr'])
            xH = xH.detach()
        finally:
            for p in self.wm.net.parameters():
                p.requires_grad_(True)
            for p in self.critic.net.parameters():
                p.requires_grad_(True)
        if self.cfg.use_critic or critic_only:
            with isolated_rng(self.device):
                torch.manual_seed(self.cfg.seed + 1000007 + self.critic_fits)
                stats.update(super().update_critic(rec, xH, valid))
            self.critic_fits += 1
        self._last_rollout = None
        assert all(__import__('math').isfinite(v) for v in stats.values()), stats
        return stats


"""Physical 5--15 kg episodic payload randomization and light XYZ force pulses."""
from dataclasses import dataclass
from pathlib import Path
import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel
from supermimic.sim_env import EnvCfg
from supermimic.rotations import mv, mat_to_quat
from cycle_env import CycleEnv, CycleTask, CycleCfg
from evaluate_payloads import BASE_XML, PAYLOAD_CENTER, PAYLOAD_HALF_SIZE, inertia_in_body

XI_DIM = 21  # original 16; load/15, force XYZ/16, remaining pulse seconds
HERE = Path(__file__).resolve().parent


@dataclass
class PayloadCfg(CycleCfg):
    w_hind_anchor: float = 4.
    hind_anchor_scale: float = .025
    w_hind_speed: float = 3.
    hind_speed_scale: float = .06
    w_loaded_hold_speed: float = 2.
    hold_speed_scale: float = .06
    w_return_com_anchor: float = 6.
    return_anchor_scale: float = .025
    w_return_com_speed: float = 4.
    return_speed_scale: float = .04
    w_return_feet_speed: float = 2.
    return_feet_speed_scale: float = .06


class PayloadTask(CycleTask):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        m = kw['mj_model']  # Always the original unloaded model.
        self.base_total_mass = float(m.body_mass.sum())
        self.base_trunk_mass = float(m.body_mass[1])
        self.payload_center = torch.tensor(PAYLOAD_CENTER, device=self.device, dtype=torch.float32)

    def centroid(self, s, kin=None, com_offset=None, payload_kg=None):
        p, v = super().centroid(s, kin)
        if payload_kg is None:
            payload_kg = torch.zeros_like(p[..., 0])
        offset = mv(s['rot'], self.payload_center)
        pp = s['pos'] + offset
        pv = s['v'] + torch.cross(s['w'], offset, dim=-1)
        total = self.base_total_mass + payload_kg
        p = (self.base_total_mass * p + payload_kg[..., None] * pp) / total[..., None]
        v = (self.base_total_mass * v + payload_kg[..., None] * pv) / total[..., None]
        if com_offset is not None:
            root_offset = mv(s['rot'], com_offset) * ((self.base_trunk_mass + payload_kg) / total)[..., None]
            p = p + root_offset
            v = v + torch.cross(s['w'], root_offset, dim=-1)
        return p, v

    def foot_velocities(self, kin):
        offset = mv(kin[1][..., self.feet, :, :], self.foot_offset)
        return kin[2][..., self.feet, :] + torch.cross(kin[3][..., self.feet, :], offset, dim=-1)

    def stability_terms(self, s0, s1, ref, kin0, kin1, refkin, xi):
        c = self.cfg
        hold = ref['hold_weight']
        gain = c.transition_stability_fraction + (1 - c.transition_stability_fraction) * hold
        kg = xi[..., 16] * 15.
        p0, v0 = self.centroid(s0, kin0, xi[..., 13:16], kg)
        p1, v1 = self.centroid(s1, kin1, xi[..., 13:16], kg)
        rp, _ = self.centroid(ref, refkin, payload_kg=kg)
        com_acc = (v1 - v0) / c.dt
        center_acc = (s1['v'] - s0['v']) / c.dt
        terms = dict(
            com_acceleration=-gain*c.w_com_acc*self.robust_norm_cost(com_acc,c.com_acc_scale),
            center_acceleration=-gain*c.w_center_acc*self.robust_norm_cost(center_acc,c.com_acc_scale),
            hold_com_velocity=-hold*c.w_com_speed*v1.square().sum(-1),
            hold_center_velocity=-hold*c.w_center_speed*s1['v'].square().sum(-1),
            hold_angular_velocity=-hold*c.w_angular_speed*s1['w'].square().sum(-1),
            hold_joint_velocity=-hold*.002*s1['qd'].square().sum(-1),
            hold_com_drift=-hold*2*(p1-rp).square().sum(-1),
            stable_hold=hold*c.w_stable*torch.exp(-v1.square().sum(-1)/.12**2-s1['w'].square().sum(-1)/.3**2))
        hind = hold * (ref['phase'] == 2)
        returned = hold * (ref['phase'] == 4)
        feet, reference_feet = self.foot_positions(kin1), self.foot_positions(refkin)
        feet_v = self.foot_velocities(kin1)
        # Absolute reference footholds prevent slow walks that evade instantaneous
        # velocity/acceleration costs. No environment-only hidden anchor is used.
        terms.update(
            hind_foot_anchor=-hind*c.w_hind_anchor*self.robust_norm_cost((feet-reference_feet)[...,2:,:2],c.hind_anchor_scale).mean(-1),
            hind_foot_speed=-hind*c.w_hind_speed*self.robust_norm_cost(feet_v[...,2:,:],c.hind_speed_scale).mean(-1),
            loaded_hold_speed=-hold*c.w_loaded_hold_speed*self.robust_norm_cost(v1,c.hold_speed_scale),
            return_com_anchor=-returned*c.w_return_com_anchor*self.robust_norm_cost(p1-rp,c.return_anchor_scale),
            return_com_speed=-returned*c.w_return_com_speed*self.robust_norm_cost(v1,c.return_speed_scale),
            return_feet_speed=-returned*c.w_return_feet_speed*self.robust_norm_cost(feet_v,c.return_feet_speed_scale).mean(-1))
        return terms, dict(com_pos=p1, com_velocity=v1, com_acceleration=com_acc, center_acceleration=center_acc)


class PayloadEnv(CycleEnv):
    def __init__(self, lib, cfg, task_cfg, device, gamma=.99, model_path=None,
                 fixed_loads=None, pushes=True):
        super().__init__(lib, cfg, task_cfg, device, gamma, model_path=model_path or HERE/'models/b2_payload_10kg.xml')
        base = mujoco.MjModel.from_xml_path(str(BASE_XML))
        old = self.task
        self.task = PayloadTask(self.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,
                                task_cfg,device,mj_model=base,compile_sizes=(cfg.num_envs,) if cfg.compile else ())
        # Expand both primary and derived constants. No load change is allowed to
        # reach a physics step before set_const has refreshed its inverse weights.
        fields = ('body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass',
                  'dof_invweight0','body_invweight0','tendon_length0','tendon_invweight0')
        self.sim.expand_model_fields(tuple(f for f in fields if f not in self.sim.expanded_fields))
        self.base_mass = float(base.body_mass[1])
        self.base_com = torch.tensor(base.body_ipos[1],device=device,dtype=torch.float64)
        self.base_inertia = torch.tensor(inertia_in_body(base,1),device=device,dtype=torch.float64)
        self.payload_center64 = torch.tensor(PAYLOAD_CENTER,device=device,dtype=torch.float64)
        h = torch.tensor(PAYLOAD_HALF_SIZE,device=device,dtype=torch.float64)
        self.box_unit_inertia = torch.diag((h.square().sum()-h.square())/3)
        self.eye = torch.eye(3,device=device,dtype=torch.float64)
        self.loads = torch.zeros(self.num_envs,device=device)
        self.fixed_loads = None if fixed_loads is None else torch.as_tensor(fixed_loads,device=device,dtype=torch.float32).expand(self.num_envs).clone()
        self.pushes = pushes
        self.push_start = torch.zeros(self.num_envs,5,device=device,dtype=torch.long)
        self.push_end = self.push_start.clone()
        self.push_vectors = torch.zeros(self.num_envs,5,3,device=device)
        self.randomizations = 0
        self.pushed_transitions = 0

    def _force_context(self):
        active = (self.frame[:,None] >= self.push_start) & (self.frame[:,None] < self.push_end)
        force = (active[...,None]*self.push_vectors).sum(1)
        remaining = (active*(self.push_end-self.frame[:,None])).sum(1).float() * self.tcfg.dt
        return force, remaining

    @property
    def xi(self):
        force, remaining = self._force_context()
        return torch.cat([super().xi,self.loads[:,None]/15.,force/16.,remaining[:,None]],-1)

    def _randomize(self, ids):
        super()._randomize(ids)
        k = len(ids)
        kg = (5.+10.*torch.rand(k,device=self.device)) if self.fixed_loads is None else self.fixed_loads[ids]
        self.loads[ids] = kg
        mass = kg.double()
        center = (self.base_mass*self.base_com+mass[:,None]*self.payload_center64)/(self.base_mass+mass[:,None])
        def shift(m, delta):
            return m[:,None,None]*(delta.square().sum(-1)[:,None,None]*self.eye-delta[:,:,None]*delta[:,None,:])
        inertia = self.base_inertia + mass[:,None,None]*self.box_unit_inertia
        inertia = inertia + shift(torch.full_like(mass,self.base_mass),self.base_com-center)
        inertia = inertia + shift(mass,self.payload_center64-center)
        moments, axes = torch.linalg.eigh(inertia)
        sign = torch.where(torch.linalg.det(axes)<0,-1.,1.)
        axes[:,:,0] *= sign[:,None]
        quat = mat_to_quat(axes)
        self.sim.model.body_mass[ids,1] = (self.base_mass+mass).float()
        self.sim.model.body_ipos[ids,1] = center.float()+self.com_offset[ids]
        self.sim.model.body_inertia[ids,1] = moments.float()
        self.sim.model.body_iquat[ids,1] = quat.float()
        self._need_set_const = True
        self.randomizations += k
        lo = torch.tensor([1.5,4.5,7.2,10.2,13.1],device=self.device)
        hi = torch.tensor([2.7,5.5,8.2,11.5,13.8],device=self.device)
        begin = ((lo+(hi-lo)*torch.rand(k,5,device=self.device))/self.tcfg.dt).round().long()
        self.push_start[ids] = begin
        self.push_end[ids] = begin+torch.randint(6,13,(k,5),device=self.device)
        direction = torch.randn(k,5,3,device=self.device)
        direction = direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        self.push_vectors[ids] = direction*(8.+8.*torch.rand(k,5,1,device=self.device)) if self.pushes else 0.
        self.sim.data.xfrc_applied[ids] = 0.

    def _finish_resets(self, ids, force=False):
        if self._need_set_const:
            self.sim.recompute_constants(RecomputeLevel.set_const)
            self._need_set_const = False
        super()._finish_resets(ids,force=force)

    def _sample_start_frames(self, k):
        frames,_ = super()._sample_start_frames(k)
        u = torch.rand(k,device=self.device)
        frames[u<.5] = 0
        final = (u>=.5)&(u<.75)
        frames[final] = torch.randint(475,self.lib.num_frames-2,(int(final.sum()),),device=self.device)
        return frames,self.lib.bin_of_frame[frames]

    def step(self, action):
        force,_ = self._force_context()
        self.sim.data.xfrc_applied[:] = 0.
        self.sim.data.xfrc_applied[:,1,:3] = force
        self.pushed_transitions += int((force.norm(dim=-1)>0).sum())
        return super().step(action)


def make_payload_env(lib,device,n,gamma=.99,evaluate=False,fixed_loads=None,pushes=True):
    cfg = EnvCfg(num_envs=n,episode_length_s=20.,domain_rand=not evaluate,obs_noise=not evaluate,
                 reset_noise=not evaluate,compile=not evaluate,set_const_every=1,
                 friction_range=(.85,1.15),enc_bias_range=(-.003,.003),com_range=((-0.005,.005),)*3,
                 pose_range=((-0.01,.01),(-.01,.01),(-.003,.003),(-.02,.02),(-.02,.02),(-.02,.02)),
                 joint_pos_range=(-.02,.02),vel_range=((-0.05,.05),)*6)
    env = PayloadEnv(lib,cfg,PayloadCfg(),device,gamma,fixed_loads=fixed_loads,pushes=pushes)
    env.eval_mode = evaluate
    return env

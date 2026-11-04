"""Pre-hardware simulator adapters; the reward is exactly paper Table A.2.

Training imports only the frozen community reproduction, never a repair branch.
B2 geometry/PD are the pre-hardware model. Payload and finite force pulses are
task context, not rewards, policy inputs, auxiliary heads or a teacher.
"""
from pathlib import Path
import mujoco
import torch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sim.sim import Simulation
from mjlab.managers.event_manager import RecomputeLevel
from sm_original.sim_env import TrackingSimEnv, EnvCfg
from sm_original.task import Task, TaskCfg, OBS_DIM
from sm_original.kinematics import FastTreeKinematics
from sm_original.rotations import mat_to_quat

HERE = Path(__file__).resolve().parent


class G1Env(TrackingSimEnv):
    """Original jump start exposure, with the unmodified paper reward/DR."""
    def _randomize(self, ids):
        super()._randomize(ids)
        # Implement the stated coefficient under MuJoCo max mixing.
        if self.cfg.domain_rand:
            self.sim.model.geom_friction[ids,:,0]=self.friction[ids,None]

    def _sample_start_frames(self, k):
        frames, bins = super()._sample_start_frames(k)
        beginning = torch.rand(k, device=self.device) < .25
        frames[beginning] = 0
        return frames, self.lib.bin_of_frame[frames]


class B2Env(TrackingSimEnv):
    def __init__(self, lib, cfg, task_cfg, device, gamma=.99):
        self.lib, self.cfg, self.tcfg, self.device, self.gamma = lib, cfg, task_cfg, device, gamma
        self.num_envs = n = cfg.num_envs
        m = mujoco.MjModel.from_xml_path(str(HERE/'models/b2_train.xml'))
        self.sim = Simulation(num_envs=n, cfg=SimulationCfg(nconmax=48, njmax=320,
            mujoco=MujocoCfg(timestep=.005, iterations=10, ls_iterations=20)), model=m, device=device)
        self.mj_model = m = self.sim.mj_model
        assert (m.nq, m.nv, m.nu) == (19, 18, 12)
        self.kin = FastTreeKinematics(m, device, root_body=1)
        self.kin_c = torch.compile(self.kin.forward, dynamic=False) if cfg.compile else self.kin.forward
        self.decimation = 4
        self.root_qadr = self.root_vadr = 0
        self.q_adr = torch.arange(7,19,device=device)
        self.qd_adr = torch.arange(6,18,device=device)
        self.act_to_joint = torch.tensor(m.actuator_trnid[:,0]-1,device=device)
        self.joint_names = [m.joint(j).name for j in range(1,13)]
        self.q0 = torch.tensor([0.,.7,-1.4]*4,device=device)
        self.action_scale = torch.full((12,),.7,device=device)
        limits = torch.tensor(m.jnt_range[1:],device=device,dtype=torch.float32)
        self.task = Task(self.kin,self.q0,limits,self.action_scale,task_cfg,device,
                         compile_sizes=(n,) if cfg.compile else ())
        self.foot_geoms = torch.tensor([m.geom(x+'_foot_collision').id for x in ('FL','FR','RL','RR')],device=device)
        self.torso_body = 1
        fields = ('geom_friction','body_mass','body_ipos','body_inertia','body_iquat','body_subtreemass',
                  'dof_invweight0','body_invweight0','tendon_length0','tendon_invweight0')
        self.sim.expand_model_fields(fields)
        self.default_torso_ipos = torch.tensor(m.body_ipos[1],dtype=torch.float32,device=device)
        self.friction = torch.ones(n,device=device)
        self.enc_bias = torch.zeros(n,12,device=device)
        self.com_offset = torch.zeros(n,3,device=device)
        self.frame = torch.zeros(n,dtype=torch.long,device=device)
        self.ep_len = self.frame.clone(); self.ep_id = self.frame.clone(); self._next_ep = n
        self.max_ep_len = int(cfg.episode_length_s/task_cfg.dt)
        self.prev_action = torch.zeros(n,12,device=device)
        self.obs_hist = torch.zeros(n,task_cfg.history,OBS_DIM,device=device)
        self.bin_G = torch.zeros(lib.num_bins,device=device)
        self.bin_visits = self.bin_G.clone(); self.bin_past_w = self.bin_G.clone()
        self.ep_bin = self.frame.clone()
        self.ep_disc_ret = torch.zeros(n,device=device)
        self.ep_disc_max = self.ep_disc_ret.clone(); self.ep_ret = self.ep_disc_ret.clone()
        self.eval_mode = False; self.state = None; self.goal_buf = None
        self.loads = torch.zeros(n,device=device)
        self.mass_scale = torch.ones(n,device=device)
        self.fixed_load = None
        self.push_start = torch.zeros(n,5,device=device)
        self.push_end = self.push_start.clone()
        self.push_vectors = torch.zeros(n,5,3,device=device)
        self.defaults = {key:torch.tensor(getattr(m,key),device=device,dtype=torch.float32)
                         for key in ('body_mass','body_ipos','body_inertia','body_iquat')}
        self.base_mass = float(m.body_mass[1])
        self.base_com = torch.tensor(m.body_ipos[1],device=device,dtype=torch.float64)
        rot = torch.empty(9,dtype=torch.float64).numpy()
        mujoco.mju_quat2Mat(rot,m.body_iquat[1])
        R = torch.tensor(rot.reshape(3,3),device=device)
        self.base_inertia = R @ torch.diag(torch.tensor(m.body_inertia[1],device=device)) @ R.T
        self.payload_center = torch.tensor([0.,0.,.18],device=device,dtype=torch.float64)
        half = torch.tensor([.18,.12,.10],device=device,dtype=torch.float64)
        self.box_unit_inertia = torch.diag((half.square().sum()-half.square())/3)
        self.eye = torch.eye(3,device=device,dtype=torch.float64)

    @property
    def xi(self):
        # [friction,bias12,COM3,load/5,mass, five(start,end,Fxyz/16),time_s]
        schedule = torch.cat([self.push_start[:,:,None],self.push_end[:,:,None],self.push_vectors/16.],-1)
        return torch.cat([self.friction[:,None],self.enc_bias,self.com_offset,self.loads[:,None]/5.,
                          self.mass_scale[:,None],schedule.flatten(1),self.frame[:,None]*self.tcfg.dt],-1)

    def _randomize(self, ids):
        super()._randomize(ids)
        k = len(ids)
        for key,value in self.defaults.items():
            getattr(self.sim.model,key)[ids] = value
        kg = torch.where(torch.rand(k,device=self.device)<.5,0.,5.) if self.fixed_load is None else torch.full((k,),self.fixed_load,device=self.device)
        self.loads[ids] = kg
        scale = .9+.2*torch.rand(k,device=self.device) if not self.eval_mode else torch.ones(k,device=self.device)
        self.mass_scale[ids] = scale
        mass = kg.double()
        center = (self.base_mass*self.base_com+mass[:,None]*self.payload_center)/(self.base_mass+mass[:,None])
        def shift(mass, delta):
            return mass[:,None,None]*(delta.square().sum(-1)[:,None,None]*self.eye-delta[:,:,None]*delta[:,None,:])
        inertia = self.base_inertia + mass[:,None,None]*self.box_unit_inertia
        inertia = inertia + shift(torch.full_like(mass,self.base_mass),self.base_com-center)
        inertia = inertia + shift(mass,self.payload_center-center)
        moments, axes = torch.linalg.eigh(inertia)
        axes[:,:,0] *= torch.where(torch.linalg.det(axes)<0,-1.,1.)[:,None]
        self.sim.model.body_mass[ids,1] = (self.base_mass+mass).float()
        self.sim.model.body_ipos[ids,1] = center.float()+self.com_offset[ids]
        self.sim.model.body_inertia[ids,1] = moments.float()
        self.sim.model.body_iquat[ids,1] = mat_to_quat(axes).float()
        self.sim.model.body_mass[ids] *= scale[:,None]
        self.sim.model.body_inertia[ids] *= scale[:,None,None]
        # Set floor and feet together: MuJoCo's max mixing must not erase low mu.
        self.sim.model.geom_friction[ids,:,0] = self.friction[ids,None]
        lo = torch.tensor([1.5,4.5,7.2,10.2,13.1],device=self.device)
        hi = torch.tensor([2.7,5.5,8.2,11.5,13.8],device=self.device)
        start = ((lo+(hi-lo)*torch.rand(k,5,device=self.device))/self.tcfg.dt).round()*self.tcfg.dt
        self.push_start[ids] = start
        self.push_end[ids] = start+torch.randint(6,13,(k,5),device=self.device)*self.tcfg.dt
        direction = torch.randn(k,5,3,device=self.device)
        direction /= direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        self.push_vectors[ids] = direction*(8.+8.*torch.rand(k,5,1,device=self.device))
        self.sim.data.xfrc_applied[ids] = 0.
        self._need_set_const = True

    def _finish_resets(self, ids, force=False):
        if self._need_set_const:
            self.sim.recompute_constants(RecomputeLevel.set_const)
            self._need_set_const = False
        super()._finish_resets(ids,force)

    def _sample_start_frames(self, k):
        frames,_ = super()._sample_start_frames(k)
        frames[torch.rand(k,device=self.device)<.5] = 0
        return frames,self.lib.bin_of_frame[frames]

    def step(self, action):
        t = self.frame[:,None]*self.tcfg.dt
        active = (t >= self.push_start)&(t < self.push_end)
        force = (active[:,:,None]*self.push_vectors).sum(1)
        self.sim.data.xfrc_applied.zero_()
        self.sim.data.xfrc_applied[:,1,:3] = force
        return super().step(action)


def make_env(robot, lib, device, n, gamma=.99, compile=True, evaluate=False):
    cfg = EnvCfg(num_envs=n,episode_length_s=20. if robot=='B2' else 10.,
                 compile=compile,obs_noise=not evaluate,reset_noise=not evaluate,
                 domain_rand=True,set_const_every=1)
    if robot=='B2':
        # Pre-hardware small pose jitter; retain light-load/mass/push task scope.
        cfg.friction_range=(.4,1.3)
        cfg.enc_bias_range=(-.003,.003);cfg.com_range=((-0.005,.005),)*3
        cfg.joint_pos_range=(-.02,.02)
        cfg.pose_range=((-0.01,.01),(-.01,.01),(-.003,.003),(-.02,.02),(-.02,.02),(-.02,.02))
        cfg.vel_range=((-0.05,.05),)*6
        env=B2Env(lib,cfg,TaskCfg(),device,gamma)
    else:
        # Paper Table A.3 DR, pre-hardware jump-specific reset jitter only.
        cfg.joint_pos_range=(-.03,.03)
        cfg.pose_range=((-0.02,.02),(-.02,.02),(-.005,.005),(-.03,.03),(-.03,.03),(-.05,.05))
        cfg.vel_range=((-0.1,.1),)*6
        env=G1Env(lib,cfg,TaskCfg(),device,gamma)
    env.eval_mode=evaluate
    return env

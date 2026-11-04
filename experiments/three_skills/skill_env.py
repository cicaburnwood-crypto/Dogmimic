from dataclasses import dataclass
from pathlib import Path
import os
import mujoco
import torch
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.task import Task,TaskCfg,NJ,OBS_DIM
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import mv
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sim.sim import Simulation

IS_B2=os.environ.get('SM_ROBOT')=='b2'

@dataclass
class SkillCfg(TaskCfg):
    history:int=5
    future:int=5
    w_root_pos:float=1.5
    w_root_ori:float=2.
    w_action_rate:float=-.025
    std_root_pos:float=.22
    std_root_ori:float=.45
    term_z:float=.32
    w_foot_position:float=1.0
    w_foot_height:float=1.0
    @property
    def max_step_reward(self):
        return super().max_step_reward+self.dt*(self.w_foot_position+self.w_foot_height)

class SkillTask(Task):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        names=('FL_calf','FR_calf','RL_calf','RR_calf') if IS_B2 else ('left_ankle_roll_link','right_ankle_roll_link')
        self.feet=torch.tensor(self.kin.body_index(names),device=self.device)
        # B2 feet are fixed sphere offsets from calf body origins.
        self.foot_offset=torch.tensor([0.,0.,-.35],device=self.device)
    def foot_positions(self,kin):
        p=kin[0][...,self.feet,:]
        if IS_B2:p=p+mv(kin[1][...,self.feet,:,:],self.foot_offset)
        return p
    def reward(self,s,kin,ref,rkin,action,prev):
        r,terms=super().reward(s,kin,ref,rkin,action,prev)
        dp=self.foot_positions(kin)-self.foot_positions(rkin)
        terms['foot_position']=self.cfg.w_foot_position*torch.exp(-dp.square().sum(-1).mean(-1)/.12**2)
        terms['foot_height']=self.cfg.w_foot_height*torch.exp(-dp[...,2].square().mean(-1)/.05**2)
        r=r+self.cfg.dt*(terms['foot_position']+terms['foot_height'])
        if 'invalid_physics' in s:r=torch.where(s['invalid_physics'],0.,r)
        return r,terms
    def termination(self,s,ref):
        hard,soft=super().termination(s,ref)
        if IS_B2:
            # Rzz is expected to be small during rear standing. Judge orientation
            # against the reference, and terminate sideways rolls/body collapse.
            side=s['rot'][...,2,1].abs()
            hard=hard|(side>.65)|(s['pos'][...,2]<.23)
            soft=1-(1-soft)*(1-torch.sigmoid((side-.65)/.05))
        if 'invalid_physics' in s:
            hard=hard|s['invalid_physics'];soft=torch.where(s['invalid_physics'],1.,soft)
        return hard,soft

class SkillEnv(TrackingSimEnv):
    def _sample_start_frames(self,k):
        frames,bins=super()._sample_start_frames(k)
        start=torch.rand(k,device=self.device)<.4
        frames[start]=0;bins[start]=0
        return frames,bins
    def read_state(self):
        s=super().read_state()
        bad=~torch.stack([torch.isfinite(v).flatten(1).all(1) for v in s.values()]).all(0)
        # Also reject physically explosive finite states, before normalizers see them.
        bad=bad|(s['qd'].abs().amax(-1)>300)|(s['v'].abs().amax(-1)>50)
        self.invalid_physics_count=getattr(self,'invalid_physics_count',torch.zeros((),device=self.device))+bad.sum()
        ref=self.lib.frame(self.frame)
        for k in s:s[k]=torch.where(bad.reshape((-1,)+(1,)*(s[k].ndim-1)),ref[k],s[k])
        s['invalid_physics']=bad
        return s
    def step(self,action):
        state,info=super().step(action)
        self.last_step_invalid=state['invalid_physics']
        return state,info

class B2Env(SkillEnv):
    def __init__(self,lib,cfg,task_cfg,device,gamma=.99,model_path=None):
        self.lib,self.cfg,self.tcfg,self.device,self.gamma=lib,cfg,task_cfg,device,gamma
        self.num_envs=n=cfg.num_envs
        m=mujoco.MjModel.from_xml_path(str(model_path or Path(__file__).parent/'b2_train.xml'))
        self.sim=Simulation(num_envs=n,cfg=SimulationCfg(nconmax=48,njmax=320,mujoco=MujocoCfg(timestep=.005,iterations=10,ls_iterations=20)),model=m,device=device)
        self.mj_model=m=self.sim.mj_model
        assert m.nq==19 and m.nv==18 and m.nu==12
        self.kin=FastTreeKinematics(m,device,root_body=1)
        self.kin_c=torch.compile(self.kin.forward,dynamic=False) if cfg.compile else self.kin.forward
        self.decimation=4;self.root_qadr=0;self.root_vadr=0
        self.q_adr=torch.arange(7,19,device=device);self.qd_adr=torch.arange(6,18,device=device)
        self.act_to_joint=torch.tensor(m.actuator_trnid[:,0]-1,device=device)
        self.joint_names=[m.joint(j).name for j in range(1,13)]
        self.q0=torch.tensor([0.,.7,-1.4]*4,device=device)
        self.action_scale=torch.full((12,),.7,device=device)
        lim=torch.tensor(m.jnt_range[1:],device=device,dtype=torch.float32)
        self.task=SkillTask(self.kin,self.q0,lim,self.action_scale,task_cfg,device,compile_sizes=(n,) if cfg.compile else ())
        self.foot_geoms=torch.tensor([m.geom(x+'_foot_collision').id for x in ('FL','FR','RL','RR')],device=device)
        self.torso_body=1
        if cfg.domain_rand:self.sim.expand_model_fields(('geom_friction','body_ipos'))
        self.default_torso_ipos=torch.tensor(m.body_ipos[1],dtype=torch.float32,device=device)
        self.friction=torch.ones(n,device=device);self.enc_bias=torch.zeros(n,12,device=device);self.com_offset=torch.zeros(n,3,device=device)
        self.frame=torch.zeros(n,dtype=torch.long,device=device)
        self.ep_len=self.frame.clone();self.ep_id=self.frame.clone();self._next_ep=n
        self.max_ep_len=int(cfg.episode_length_s/task_cfg.dt)
        self.prev_action=torch.zeros(n,12,device=device)
        self.obs_hist=torch.zeros(n,task_cfg.history,OBS_DIM,device=device)
        self.bin_G=torch.zeros(lib.num_bins,device=device);self.bin_visits=self.bin_G.clone();self.bin_past_w=self.bin_G.clone()
        self.ep_bin=self.frame.clone();self.ep_disc_ret=torch.zeros(n,device=device);self.ep_disc_max=self.ep_disc_ret.clone();self.ep_ret=self.ep_disc_ret.clone()
        self.eval_mode=False;self.state=None;self.goal_buf=None

def make_env(lib,device,n,gamma=.99,evaluate=False):
    cfg=EnvCfg(num_envs=n,episode_length_s=15.,domain_rand=not evaluate,obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate,
               friction_range=(.85,1.15),enc_bias_range=(-.003,.003),com_range=((-0.005,.005),)*3,
               pose_range=((-0.01,.01),(-.01,.01),(-.003,.003),(-.02,.02),(-.02,.02),(-.02,.02)),
               joint_pos_range=(-.02,.02),vel_range=((-0.05,.05),)*6)
    tc=SkillCfg()
    env=(B2Env if IS_B2 else SkillEnv)(lib,cfg,tc,device,gamma)
    if not IS_B2:
        old=env.task
        env.task=SkillTask(env.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,tc,device,compile_sizes=() if evaluate else (n,))
    env.eval_mode=evaluate
    return env

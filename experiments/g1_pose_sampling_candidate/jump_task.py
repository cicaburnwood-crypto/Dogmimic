from dataclasses import dataclass
import torch
from supermimic.task import Task, TaskCfg
from supermimic.sim_env import TrackingSimEnv, EnvCfg

@dataclass
class JumpTaskCfg(TaskCfg):
    w_root_pos: float = 1.5
    std_root_pos: float = .15
    w_action_rate: float = -.03
    w_feet: float = 2.
    w_feet_height: float = 2.
    std_feet: float = .10
    std_feet_height: float = .03
    @property
    def max_step_reward(self):
        return super().max_step_reward+self.dt*(self.w_feet+self.w_feet_height)

class JumpTask(Task):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feet=torch.tensor(self.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link')),device=self.device)
    def reward(self,s,kin,ref,ref_kin,action,prev_action):
        r,terms=super().reward(s,kin,ref,ref_kin,action,prev_action)
        dp=kin[0][...,self.feet,:]-ref_kin[0][...,self.feet,:]
        feet=self.cfg.w_feet*torch.exp(-dp.square().sum(-1).mean(-1)/self.cfg.std_feet**2)
        height=self.cfg.w_feet_height*torch.exp(-dp[...,2].square().mean(-1)/self.cfg.std_feet_height**2)
        terms.update(feet_position=feet,feet_height=height)
        return r+self.cfg.dt*(feet+height),terms

class JumpEnv(TrackingSimEnv):
    def _sample_start_frames(self,k):
        frames,bins=super()._sample_start_frames(k)
        beginning=torch.rand(k,device=self.device)<.25
        frames[beginning]=0;bins[beginning]=0
        return frames,bins

def make_env(lib,device,num_envs,gamma=.99,evaluate=False):
    cfg=EnvCfg(num_envs=num_envs,episode_length_s=10.,friction_range=(.7,1.1),
               joint_pos_range=(-.03,.03),pose_range=((-0.02,.02),(-.02,.02),(-.005,.005),(-.03,.03),(-.03,.03),(-.05,.05)),
               vel_range=((-0.1,.1),(-.1,.1),(-.1,.1),(-.1,.1),(-.1,.1),(-.1,.1)),
               obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate)
    tc=JumpTaskCfg()
    env=JumpEnv(lib,cfg,tc,device,gamma)
    old=env.task
    env.task=JumpTask(env.kin,old.q0,torch.stack([old.qmin,old.qmax],dim=-1),old.action_scale,tc,device,compile_sizes=() if evaluate else (num_envs,))
    env.eval_mode=evaluate
    return env

from dataclasses import dataclass
import torch
from supermimic.task import Task,TaskCfg
from supermimic.sim_env import TrackingSimEnv,EnvCfg

@dataclass
class SprintTaskCfg(TaskCfg):
    w_root_pos: float=1.
    std_root_pos: float=.3
    w_action_rate: float=-.03
    w_speed: float=2.
    w_feet: float=1.
    w_feet_height: float=1.
    @property
    def max_step_reward(self):
        return super().max_step_reward+self.dt*(self.w_speed+self.w_feet+self.w_feet_height)

class SprintTask(Task):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feet=torch.tensor(self.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link')),device=self.device)
    def reward(self,s,kin,ref,ref_kin,action,prev_action):
        r,terms=super().reward(s,kin,ref,ref_kin,action,prev_action)
        dp=kin[0][...,self.feet,:]-ref_kin[0][...,self.feet,:]
        speed=self.cfg.w_speed*torch.exp(-(s['v'][...,:2]-ref['v'][...,:2]).square().sum(-1)/1.0**2)
        feet=self.cfg.w_feet*torch.exp(-dp.square().sum(-1).mean(-1)/.18**2)
        height=self.cfg.w_feet_height*torch.exp(-dp[...,2].square().mean(-1)/.05**2)
        terms.update(speed=speed,feet_position=feet,feet_height=height)
        reward=r+self.cfg.dt*(speed+feet+height)
        if 'invalid_physics' in s:reward=torch.where(s['invalid_physics'],0.,reward)
        return reward,terms
    def termination(self,s,ref):
        hard,soft=super().termination(s,ref)
        if 'invalid_physics' in s:
            hard=hard|s['invalid_physics'];soft=torch.where(s['invalid_physics'],1.,soft)
        return hard,soft

class SprintEnv(TrackingSimEnv):
    def read_state(self):
        s=super().read_state()
        bad=~torch.stack([torch.isfinite(v).flatten(1).all(1) for v in s.values()]).all(0)
        self.invalid_physics_count=getattr(self,'invalid_physics_count',torch.zeros((),device=self.device))+bad.sum()
        ref=self.lib.frame(self.frame)
        # Invalid physics is a failed terminal transition. Replace only its numeric
        # payload so NaNs cannot poison normalizers before the environment resets.
        for k in s:
            s[k]=torch.where(bad.reshape((-1,)+(1,)*(s[k].ndim-1)),ref[k],s[k])
        s['invalid_physics']=bad
        return s
    def step(self,action):
        state,info=super().step(action)
        self.last_step_invalid=state['invalid_physics']
        return state,info

def make_env(lib,device,num_envs,gamma=.99,evaluate=False):
    cfg=EnvCfg(num_envs=num_envs,episode_length_s=15.,friction_range=(.8,1.2),
       joint_pos_range=(-.03,.03),pose_range=((-0.02,.02),(-.02,.02),(-.005,.005),(-.03,.03),(-.03,.03),(-.04,.04)),
       vel_range=((-0.2,.2),(-.1,.1),(-.1,.1),(-.1,.1),(-.1,.1),(-.1,.1)),
       com_range=((-0.01,.01),(-.01,.01),(-.01,.01)),obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate)
    tc=SprintTaskCfg();env=SprintEnv(lib,cfg,tc,device,gamma);old=env.task
    env.task=SprintTask(env.kin,old.q0,torch.stack([old.qmin,old.qmax],dim=-1),old.action_scale,tc,device,compile_sizes=() if evaluate else (num_envs,))
    env.eval_mode=evaluate
    return env

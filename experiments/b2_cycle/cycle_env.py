from dataclasses import dataclass
import numpy as np
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.sim_env import EnvCfg
from supermimic.rotations import mv
from skill_env import B2Env,SkillTask,SkillCfg

class CycleLibrary(MotionLibrary):
    def __init__(self,path,device,*args,**kwargs):
        super().__init__(path,device,*args,**kwargs)
        d=np.load(path,allow_pickle=False)
        assert self.num_clips==1 and len(d['phase'])==self.num_frames
        self.phase=torch.tensor(d['phase'],device=device)
        self.hold_weight=torch.tensor(d['hold_weight'],device=device)
    def frame(self,idx):
        return {**super().frame(idx),'phase':self.phase[idx],'hold_weight':self.hold_weight[idx]}

@dataclass
class CycleCfg(SkillCfg):
    w_action_rate:float=-.06
    com_acc_scale:float=3.
    w_com_acc:float=.35
    w_center_acc:float=.15
    w_com_speed:float=3.
    w_center_speed:float=1.
    w_angular_speed:float=.5
    w_stable:float=2.
    transition_stability_fraction:float=.08
    @property
    def max_step_reward(self):
        return super().max_step_reward+self.dt*self.w_stable

class CycleTask(SkillTask):
    def __init__(self,*args,mj_model,**kw):
        super().__init__(*args,**kw)
        self.mass_weight=torch.tensor(mj_model.body_mass[1:]/mj_model.body_mass[1:].sum(),device=self.device,dtype=torch.float32)
        self.inertial_offset=torch.tensor(mj_model.body_ipos[1:],device=self.device,dtype=torch.float32)
        assert len(self.mass_weight)==self.kin.nb

    def centroid(self,s,kin=None,com_offset=None):
        if kin is None:kin=self.body_kin(s)
        P,R,V,W=kin
        offset=mv(R,self.inertial_offset)
        pos=((P+offset)*self.mass_weight[:,None]).sum(-2)
        vel=((V+torch.cross(W,offset,dim=-1))*self.mass_weight[:,None]).sum(-2)
        if com_offset is not None:
            root_offset=mv(s['rot'],com_offset)*self.mass_weight[0]
            pos=pos+root_offset
            vel=vel+torch.cross(s['w'],root_offset,dim=-1)
        return pos,vel

    @staticmethod
    def robust_norm_cost(x,scale):
        # Isotropic in world XYZ; linear tails keep large acceleration gradients
        # useful, without the exploding quadratic cost of early bad rollouts.
        return 2*(torch.sqrt(1+x.square().sum(-1)/(scale*scale))-1)

    def stability_terms(self,s0,s1,ref,kin0,kin1,refkin,xi):
        c=self.cfg;hold=ref['hold_weight']
        gain=c.transition_stability_fraction+(1-c.transition_stability_fraction)*hold
        p0,v0=self.centroid(s0,kin0,xi[...,-3:])
        p1,v1=self.centroid(s1,kin1,xi[...,-3:])
        rp,rv=self.centroid(ref,refkin)
        com_acc=(v1-v0)/c.dt
        center_acc=(s1['v']-s0['v'])/c.dt
        terms=dict(
            com_acceleration=-gain*c.w_com_acc*self.robust_norm_cost(com_acc,c.com_acc_scale),
            center_acceleration=-gain*c.w_center_acc*self.robust_norm_cost(center_acc,c.com_acc_scale),
            hold_com_velocity=-hold*c.w_com_speed*v1.square().sum(-1),
            hold_center_velocity=-hold*c.w_center_speed*s1['v'].square().sum(-1),
            hold_angular_velocity=-hold*c.w_angular_speed*s1['w'].square().sum(-1),
            hold_joint_velocity=-hold*.002*s1['qd'].square().sum(-1),
            hold_com_drift=-hold*2*(p1-rp).square().sum(-1),
            stable_hold=hold*c.w_stable*torch.exp(-v1.square().sum(-1)/.12**2-s1['w'].square().sum(-1)/.3**2),
        )
        return terms,dict(com_pos=p1,com_velocity=v1,com_acceleration=com_acc,center_acceleration=center_acc)

    def transition_analytics(self,s0,s1,ref,refkin,future,action,prev,xi):
        out=self.analytics(s1,ref,refkin,future,action,prev,xi[...,1:13])
        extra,metrics=self.stability_terms(s0,s1,ref,self.body_kin(s0),out['kin'],refkin,xi)
        out['r']=out['r']+sum(extra.values())*self.cfg.dt
        if 'invalid_physics' in s1:out['r']=torch.where(s1['invalid_physics'],0.,out['r'])
        out['terms'].update(extra);out.update(metrics)
        return out
    def transition_fast(self,*args):
        return self._maybe_compiled('transition_analytics',self.transition_analytics,*args)

class CycleEnv(B2Env):
    def __init__(self,lib,cfg,task_cfg,device,gamma=.99,model_path=None):
        super().__init__(lib,cfg,task_cfg,device,gamma,model_path=model_path)
        old=self.task
        self.task=CycleTask(self.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,task_cfg,device,
                            mj_model=self.mj_model,compile_sizes=(cfg.num_envs,) if cfg.compile else ())

    def _sample_start_frames(self,k):
        frames,bins=super()._sample_start_frames(k)
        # Explicitly expose the previously unseen descent and final settling.
        u=torch.rand(k,device=self.device)
        return_mask=u<.3
        frames[return_mask]=torch.randint(475,self.lib.num_frames-2,(int(return_mask.sum()),),device=self.device)
        bins=self.lib.bin_of_frame[frames]
        return frames,bins

    def step(self,action):
        # Same transition boundary in real physics and in the learned model:
        # compute acceleration before terminal resets, never across reset jumps.
        s0=self.state
        self.sim.data.ctrl[:]=(self.q0+self.action_scale*action-self.enc_bias)[:,self.act_to_joint]
        for _ in range(self.decimation):self.sim.step()
        s1=self.read_state();self.last_step_invalid=s1['invalid_physics']
        next_frame=self.frame+1;ref=self.lib.frame(next_frame)
        refkin=self.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
        out=self.task.transition_fast(s0,s1,ref,refkin,self.ref_future(next_frame),action,self.prev_action,self.xi)
        r,hard=out['r'],out['hard'];self.ep_len+=1
        truncated=((self.lib.remaining(next_frame)<=0)|(self.ep_len>=self.max_ep_len))&~hard
        done=hard|truncated
        disc=self.gamma**(self.ep_len-1).float()
        self.ep_disc_ret+=disc*r;self.ep_disc_max+=disc*self.tcfg.max_step_reward;self.ep_ret+=r
        info=dict(reward=r,terms=out['terms'],terminated=hard,truncated=truncated,frame=self.frame.clone(),ep_id=self.ep_id.clone(),ep_len=self.ep_len.clone(),done=done,
                  com_pos=out['com_pos'],com_velocity=out['com_velocity'],com_acceleration=out['com_acceleration'],center_acceleration=out['center_acceleration'])
        self.frame=next_frame;self.prev_action=action.clone()
        self.obs_hist=torch.cat([self.obs_hist[:,1:],(out['obs']+self._noise(self.num_envs))[:,None]],1)
        self.goal_buf=out['goal']
        ids=done.nonzero().squeeze(-1)
        if len(ids):
            ratio=self.ep_disc_ret[ids]/self._max_disc_return(ids)
            b=self.ep_bin[ids];alpha=self.cfg.bin_ema
            new=torch.where(self.bin_visits[b]>0,(1-alpha)*self.bin_G[b]+alpha*ratio,ratio)
            self.bin_G[b]=new.clamp(0,1);self.bin_visits[b]+=1
            potential=torch.minimum(self.lib.remaining(self.frame[ids]-self.ep_len[ids]),torch.full_like(ids,self.max_ep_len)).float()
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),done_terminated=hard[ids].clone(),done_bin=b,
                        done_norm_return=self.ep_ret[ids]/(self.tcfg.max_step_reward*potential.clamp_min(1)))
            self.reset_envs(ids)
        self._finish_resets(ids)
        return s1,info

def make_cycle_env(lib,device,n,gamma=.99,evaluate=False,model_path=None):
    cfg=EnvCfg(num_envs=n,episode_length_s=20.,domain_rand=not evaluate,obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate,
               friction_range=(.85,1.15),enc_bias_range=(-.003,.003),com_range=((-0.005,.005),)*3,
               pose_range=((-0.01,.01),(-.01,.01),(-.003,.003),(-.02,.02),(-.02,.02),(-.02,.02)),joint_pos_range=(-.02,.02),vel_range=((-0.05,.05),)*6)
    env=CycleEnv(lib,cfg,CycleCfg(),device,gamma,model_path=model_path);env.eval_mode=evaluate
    return env

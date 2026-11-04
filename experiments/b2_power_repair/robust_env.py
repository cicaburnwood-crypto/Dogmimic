"""Balance repair with unchanged physical torque envelope and acceptance. Experimental actuator envelope, not a claim about Unitree firmware limits."""
from dataclasses import dataclass
import torch,mujoco,json
from pathlib import Path
from supermimic.sim_env import EnvCfg
from recovery_env import RecoveryEnv,RecoveryCfg
from stage_env import StageTask
XI_DIM=43
SPEED=6.;POWER=1500.;ACCEL=180.

from actuator_filter import filtered_target, torque_envelope

class RobustTask(StageTask):
    def advance_context(self,s,frame,xi):
        return torch.cat([super().advance_context(s,frame,xi),xi[...,30:]],-1)
    def transition_analytics(self,s0,s1,ref,refkin,future,action,prev,xi):
        out=super().transition_analytics(s0,s1,ref,refkin,future,action,prev,xi)
        target=(self.q0+self.action_scale*action-xi[...,1:13]).clamp(self.qmin+.02,self.qmax-.02)
        target=filtered_target(target,xi[...,31:43],self.cfg.dt)
        limits=torch.tensor([160.,160.,240.]*4,device=action.device)
        tau,raw=torque_envelope(target,s0['q'],s0['qd'],limits)
        tau1,raw1=torque_envelope(target,s1['q'],s1['qd'],limits)
        speed=torch.maximum(s0['qd'].abs().amax(-1),s1['qd'].abs().amax(-1))
        # The old loss was identically zero: capped power<=1200 < threshold1500.
        # Penalize uncapped mechanical demand, plus pre-torque/post-speed power.
        demand=torch.maximum((raw*s0['qd']).abs().sum(-1),(raw1*s1['qd']).abs().sum(-1))
        crossing=(tau*s1['qd']).abs().sum(-1)
        power=torch.maximum(demand,crossing)
        terms=dict(actuator_speed=-2.*(speed/SPEED-1).relu().square(),
                   actuator_acceleration=-.5*(((s1['qd']-s0['qd']).abs()/(self.cfg.dt*ACCEL)-1).relu().square().mean(-1)),
                   actuator_power=-2.*((power/POWER-1).relu().square()),
                   actuator_saturation=-.2*((raw-tau)/limits).square().mean(-1))
        out['r']=out['r']+self.cfg.dt*sum(terms.values());out['terms'].update(terms)
        return out

class RobustEnv(RecoveryEnv):
    def __init__(self,*args,**kw):
        super().__init__(*args,task_class=RobustTask,**kw)
        self.filtered_target=self.sim.data.qpos[:,self.q_adr].clone()
        self.mass_scale=torch.ones(self.num_envs,device=self.device)
        self.nominal_mass=self.sim.model.body_mass[0].clone();self.nominal_inertia=self.sim.model.body_inertia[0].clone()
        self.limits=torch.tensor([160.,160.,240.]*4,device=self.device)[self.act_to_joint]
        self.max_speed=torch.zeros(self.num_envs,device=self.device);self.max_power=self.max_speed.clone()
        self.training_loads=kw.get('fixed_loads') is None
        if self.training_loads:self.fixed_loads=torch.zeros(self.num_envs,device=self.device)
        self.measured_q=torch.tensor(json.loads(Path(__file__).with_name('measured_initial_q.json').read_text()),device=self.device)
        self.fixed_mass=None;self.fixed_friction=None;self.asymmetric_eval=False
    @property
    def xi(self):return torch.cat([super().xi,self.mass_scale[:,None],self.filtered_target],-1)
    def _randomize(self,ids):
        self.sim.model.body_mass[ids]=self.nominal_mass;self.sim.model.body_inertia[ids]=self.nominal_inertia
        if self.training_loads:
            self.fixed_loads[ids]=torch.where(torch.rand(len(ids),device=self.device)<.25,0.,5.+10.*torch.rand(len(ids),device=self.device))
        super()._randomize(ids)
        scale=.9+.2*torch.rand(len(ids),device=self.device) if self.fixed_mass is None else torch.full((len(ids),),self.fixed_mass,device=self.device)
        self.mass_scale[ids]=scale;self.sim.model.body_mass[ids]*=scale[:,None];self.sim.model.body_inertia[ids]*=scale[:,None,None]
        if self.fixed_friction is not None:
            self.friction[ids]=self.fixed_friction;self.sim.model.geom_friction[ids[:,None],self.foot_geoms,0]=self.fixed_friction
        # Equal-priority MuJoCo contacts use the larger friction: randomize floor too.
        self.sim.model.geom_friction[ids,:,0]=self.friction[ids,None]
        self._need_set_const=True
    def _finish_resets(self,ids,force=False):
        from payload_env import PayloadEnv
        # Evaluation may perturb qpos after reset_envs; synchronize reset memory here too.
        self.filtered_target[ids]=self.sim.data.qpos[ids[:,None],self.q_adr]
        PayloadEnv._finish_resets(self,ids,force)
        self.anchor_context[ids]=0.
        self.anchor_context[:]=self.task.advance_context(self.state,self.frame,self.xi)[:,21:30]
    def reset_envs(self,ids,frames=None):
        super().reset_envs(ids,frames)
        start=ids[self.frame[ids]==0]
        if len(start) and (not self.eval_mode or self.asymmetric_eval):
            bounds=torch.tensor([.04,.25,.12]*4,device=self.device)
            delta=(torch.rand(len(start),12,device=self.device)*2-1)*bounds
            self.sim.data.qpos[start[:,None],self.q_adr]+=delta
            chosen=start[::2]
            self.sim.data.qpos[chosen[:,None],self.q_adr]=self.measured_q[torch.arange(len(chosen),device=self.device)%2]
            self.sim.data.qvel[start]=0.
        # Reset memory after all initial-pose perturbations, in policy joint order.
        self.filtered_target[ids]=self.sim.data.qpos[ids[:,None],self.q_adr]
    def _actuated_step(self,target):
        q=self.sim.data.qpos[:,self.q_adr][:,self.act_to_joint];qd=self.sim.data.qvel[:,self.qd_adr][:,self.act_to_joint]
        tau,_=torque_envelope(target,q,qd,self.limits)
        self.sim.data.ctrl[:]=q+(tau+8*qd)/250
        self.sim.step()
        actual_qd=self.sim.data.qvel[:,self.qd_adr][:,self.act_to_joint]
        self.max_speed=torch.maximum(self.max_speed,actual_qd.abs().amax(-1))
        self.max_power=torch.maximum(self.max_power,(self.sim.data.actuator_force*actual_qd).abs().sum(-1))

    def step(self,action):
        # Same transition boundary in real physics and in the learned model:
        # compute acceleration before terminal resets, never across reset jumps.
        s0=self.state
        xi0=self.xi
        force,_=self._force_context()
        self.pushed_transitions+=int((force.norm(dim=-1)>0).sum())
        self.sim.data.xfrc_applied[:]=0.;self.sim.data.xfrc_applied[:,1,:3]=force
        target=(self.q0+self.action_scale*action-self.enc_bias).clamp(self.task.qmin+.02,self.task.qmax-.02)
        self.filtered_target=filtered_target(target,self.filtered_target,self.tcfg.dt)
        target=self.filtered_target[:,self.act_to_joint]
        for _ in range(self.decimation):self._actuated_step(target)
        s1=self.read_state();self.last_step_invalid=s1['invalid_physics']
        next_frame=self.frame+1;ref=self.lib.frame(next_frame)
        refkin=self.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
        out=self.task.transition_fast(s0,s1,ref,refkin,self.ref_future(next_frame),action,self.prev_action,xi0)
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

def make_env(lib,device,n,gamma=.99,evaluate=False,kg=None):
    cfg=EnvCfg(num_envs=n,episode_length_s=20.,domain_rand=True,obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate,set_const_every=1,friction_range=(.4,1.3),enc_bias_range=(-.003,.003),com_range=((-0.01,.01),)*3,joint_pos_range=(-.02,.02),vel_range=((0.,0.),)*6)
    env=RobustEnv(lib,cfg,RecoveryCfg(),device,gamma,fixed_loads=kg,pushes=True)
    env.eval_mode=evaluate
    if evaluate:env.cfg.domain_rand=False
    return env

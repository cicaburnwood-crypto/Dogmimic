"""Command-conditioned four-foot attitude task on the pinned vibe engine."""
import json
import math
from dataclasses import dataclass
import mujoco
import torch
from bootstrap import HERE
from contract import *
from supermimic.task import Task, TaskCfg
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import mv, mtv
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sim.sim import Simulation
from mjlab.managers.event_manager import RecomputeLevel

XI_DIM=32  # mu,bias12,COM3,mass,kp,kd,delay/2,previous applied action12

@dataclass
class AttitudeCfg(TaskCfg):
    history:int=HISTORY
    future:int=0
    noise_q:float=.005
    noise_qd:float=.25
    noise_gyro:float=.08
    noise_grav:float=.02
    @property
    def max_step_reward(self):return 8.*DT

class AttitudeTask(Task):
    def __init__(self,*args,height,foot_offsets,**kwargs):
        super().__init__(*args,**kwargs)
        self.height=height
        self.feet=torch.tensor(self.kin.body_index(('FL_calf','FR_calf','RL_calf','RR_calf')),device=self.device)
        self.foot_offsets=torch.as_tensor(foot_offsets,device=self.device,dtype=torch.float32)
        self.torque_limits=torch.tensor([200.,200.,300.]*4,device=self.device)

    def foot_state(self,kin):
        offset=mv(kin[1][...,self.feet,:,:],self.foot_offsets)
        return kin[0][...,self.feet,:]+offset,kin[2][...,self.feet,:]+torch.cross(kin[3][...,self.feet,:],offset,dim=-1)

    def command_analytics(self,s,context,action,previous,xi):
        kin=self.body_kin(s);feet,velocity=self.foot_state(kin)
        error=angle_error(angles(s['rot']),context[...,:2])
        omega=mtv(s['rot'],s['w']);rate_error=omega-body_rate(context)
        bias=xi[...,1:13]
        torque=KP*xi[...,17:18]*(self.q0+self.action_scale*action-bias-s['q'])-KD*xi[...,18:19]*s['qd']
        demand=(torque*s['qd']).abs().sum(-1)
        gap=feet[...,2]-.032
        stationary=(context[...,4:6].abs().amax(-1)<math.radians(.1)).float()
        terms=dict(
            attitude=4.*torch.exp(-error.square().sum(-1)/math.radians(8)**2),
            attitude_error=-.7*(error/MAX_ANGLE).square().sum(-1),
            angular_rate=.8*torch.exp(-rate_error.square().sum(-1)/.35**2),
            angular_rate_error=-.12*rate_error.square().sum(-1),
            height=.8*torch.exp(-(s['pos'][...,2]-self.height).square()/.04**2),
            height_error=-12.*(s['pos'][...,2]-self.height).square(),
            stationary=.8*torch.exp(-s['v'].square().sum(-1)/.10**2),
            translation=-.8*s['v'].square().sum(-1),
            contact=.8*torch.exp(-gap.square().mean(-1)/.02**2),
            foot_height=-3.*(gap/.08).square().mean(-1),
            slip=-.5*(velocity[...,:2]/.2).square().sum(-1).mean(-1),
            neutral_posture=.8*stationary*torch.exp(-(s['q']-self.q0).square().mean(-1)/.25**2),
            action_rate=-.025*(action-previous).square().sum(-1),
            effort=-.04*(torque/self.torque_limits).square().mean(-1),
            peak_velocity=-.3*torch.relu((s['qd'].abs().amax(-1)-4.)/2.).square(),
            power_demand=-.15*torch.relu((demand-900.)/600.).square(),
            joint_limits=-2.*(torch.relu(self.qmin-s['q'])+torch.relu(s['q']-self.qmax)).sum(-1),
        )
        tilt=angles(s['rot']).abs().amax(-1)
        hard=(s['pos'][...,2]<.25)|(tilt>math.radians(55))
        soft=1-(1-torch.sigmoid((.25-s['pos'][...,2])/.02))*(1-torch.sigmoid((tilt-math.radians(55))/.04))
        reward=sum(terms.values())*DT-2.*soft
        return dict(r=reward,hard=hard,soft=soft,terms=terms,obs=self.obs_clean(s,action,bias),feats=self.state_features(s,kin),goal=context)

    def command_fast(self,*args):return self._maybe_compiled('attitude',self.command_analytics,*args)

class AttitudeEnv(TrackingSimEnv):
    def __init__(self,device,n,evaluate=False,randomize=True,compile=True,autoreset=True):
        self.device=device;self.num_envs=n;self.gamma=.99;self.eval_mode=evaluate;self.autoreset=autoreset
        self.tcfg=AttitudeCfg();self.cfg=EnvCfg(num_envs=n,episode_length_s=24.,domain_rand=randomize,
            obs_noise=randomize,reset_noise=False,compile=compile,set_const_every=1)
        self.model_meta=json.loads((HERE/'model.json').read_text())
        m=mujoco.MjModel.from_xml_path(str(HERE/'b2_attitude.xml'))
        self.sim=Simulation(num_envs=n,cfg=SimulationCfg(nconmax=48,njmax=320,
            mujoco=MujocoCfg(timestep=SIM_DT,iterations=10,ls_iterations=20)),model=m,device=device)
        self.mj_model=m=self.sim.mj_model;self.decimation=4
        self.kin=FastTreeKinematics(m,device,root_body=1)
        self.root_qadr=self.root_vadr=0
        self.q_adr=torch.arange(7,19,device=device);self.qd_adr=torch.arange(6,18,device=device)
        self.act_to_joint=torch.tensor(m.actuator_trnid[:,0]-1,device=device)
        self.joint_names=[m.joint(j).name for j in range(1,13)]
        self.q0=torch.tensor(Q0,device=device);self.action_scale=torch.tensor(ACTION_SCALE,device=device)
        self.foot_geoms=torch.tensor([m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')],device=device)
        self.task=AttitudeTask(self.kin,self.q0,torch.tensor(m.jnt_range[1:],dtype=torch.float32,device=device),
            self.action_scale,self.tcfg,device,height=self.model_meta['standing_height_m'],
            foot_offsets=m.geom_pos[self.foot_geoms.cpu().numpy()],compile_sizes=(n,) if compile else ())
        fields=('geom_friction','body_ipos','body_mass','body_inertia','body_subtreemass','dof_invweight0',
                'body_invweight0','tendon_length0','tendon_invweight0','actuator_gainprm','actuator_biasprm')
        self.sim.expand_model_fields(fields)
        self.default_mass=torch.tensor(m.body_mass,device=device,dtype=torch.float32)
        self.default_inertia=torch.tensor(m.body_inertia,device=device,dtype=torch.float32)
        self.default_torso_ipos=torch.tensor(m.body_ipos[1],device=device,dtype=torch.float32)
        self.friction=torch.ones(n,device=device);self.enc_bias=torch.zeros(n,12,device=device);self.com_offset=torch.zeros(n,3,device=device)
        self.mass_scale=torch.ones(n,device=device);self.kp_scale=self.mass_scale.clone();self.kd_scale=self.mass_scale.clone()
        self.delay=torch.zeros(n,device=device,dtype=torch.long)
        self.frame=torch.zeros(n,device=device,dtype=torch.long);self.ep_len=self.frame.clone();self.ep_id=self.frame.clone();self._next_ep=n
        self.max_ep_len=round(self.cfg.episode_length_s/DT)
        self.prev_action=torch.zeros(n,12,device=device);self.obs_hist=torch.zeros(n,HISTORY,OBS,device=device)
        self.context=torch.zeros(n,GOAL_DIM,device=device)
        self.event_left=self.frame.clone();self.event_count=self.frame.clone();self.individual=torch.zeros(n,device=device,dtype=torch.bool)
        self.bin_G=torch.zeros(1,device=device);self.bin_visits=self.bin_G.clone();self.bin_past_w=self.bin_G.clone()
        self.ep_ret=torch.zeros(n,device=device)
        self.invalid_physics_count=torch.zeros((),device=device,dtype=torch.long);self.last_step_invalid=torch.zeros(n,device=device,dtype=torch.bool)
        self.state=None;self.goal_buf=self.context;self.contact_recorder=None;self.fixed_domain=None
        self._need_set_const=False

    @property
    def xi(self):
        return torch.cat([self.friction[:,None],self.enc_bias,self.com_offset,self.mass_scale[:,None],
                          self.kp_scale[:,None],self.kd_scale[:,None],self.delay[:,None]/2.,self.prev_action],-1)

    def _randomize(self,ids):
        k=len(ids);dev=self.device
        rand=self.cfg.domain_rand
        self.friction[ids]=.4+.9*torch.rand(k,device=dev) if rand else 1.
        self.enc_bias[ids]=.01*torch.rand(k,12,device=dev)-.005 if rand else 0.
        self.com_offset[ids]=.02*torch.rand(k,3,device=dev)-.01 if rand else 0.
        self.mass_scale[ids]=.9+.2*torch.rand(k,device=dev) if rand else 1.
        self.kp_scale[ids]=.85+.3*torch.rand(k,device=dev) if rand else 1.
        self.kd_scale[ids]=.8+.4*torch.rand(k,device=dev) if rand else 1.
        self.delay[ids]=torch.randint(0,3,(k,),device=dev) if rand else 0
        if self.fixed_domain:
            for name,values in self.fixed_domain.items():getattr(self,name)[ids]=values[ids]
        # Floor AND feet, so MuJoCo max-mixing does not erase low friction.
        self.sim.model.geom_friction[ids,:,0]=self.friction[ids,None]
        self.sim.model.body_ipos[ids,1]=self.default_torso_ipos+self.com_offset[ids]
        self.sim.model.body_mass[ids]=self.default_mass*self.mass_scale[ids,None]
        self.sim.model.body_inertia[ids]=self.default_inertia*self.mass_scale[ids,None,None]
        self.sim.model.actuator_gainprm[ids,:,0]=KP*self.kp_scale[ids,None]
        self.sim.model.actuator_biasprm[ids,:,1]=-KP*self.kp_scale[ids,None]
        self.sim.model.actuator_biasprm[ids,:,2]=-KD*self.kd_scale[ids,None]
        self._need_set_const=True

    def reset_envs(self,ids,frames=None):
        k=len(ids)
        if not k:return
        self.sim.reset(ids);self._randomize(ids)
        pos=torch.zeros(k,3,device=self.device);pos[:,2]=self.model_meta['standing_height_m']
        quat=torch.zeros(k,4,device=self.device);quat[:,0]=1
        q=self.q0.expand(k,-1).clone()
        if self.cfg.domain_rand:q+=.02*(torch.rand_like(q)-.5)
        if getattr(self,'nominal_stride',0):q[ids%self.nominal_stride==0]=self.q0
        z=torch.zeros(k,3,device=self.device)
        self.write_state(ids,pos,quat,q,z,z,torch.zeros_like(q))
        self.frame[ids]=0;self.ep_len[ids]=0;self.ep_ret[ids]=0
        self.ep_id[ids]=torch.arange(self._next_ep,self._next_ep+k,device=self.device);self._next_ep+=k
        self.prev_action[ids]=0;self.context[ids]=0;self.event_left[ids]=100;self.event_count[ids]=0
        self.individual[ids]=torch.rand(k,device=self.device)<.5

    def _finish_resets(self,ids,force=False):
        if self._need_set_const:
            self.sim.recompute_constants(RecomputeLevel.set_const);self._need_set_const=False
        self.state=self.read_state()
        if len(ids):
            state={k:v[ids] for k,v in self.state.items()}
            self.obs_hist[ids]=(self.task.obs_clean(state,self.prev_action[ids],self.enc_bias[ids])+self._noise(len(ids)))[:,None]
        self.goal_buf=self.context

    def _noise(self,n):
        noise=self.task.sample_obs_noise((n,)) if self.cfg.obs_noise else torch.zeros(n,OBS,device=self.device)
        if getattr(self,'nominal_stride',0) and n==self.num_envs:noise[::self.nominal_stride]=0
        return noise

    def set_target(self,target):
        value=torch.as_tensor(target,device=self.device,dtype=torch.float32).expand(self.num_envs,2)
        if not torch.isfinite(value).all() or (value.abs()>MAX_ANGLE+1e-6).any():raise ValueError('Invalid attitude command')
        self.context[:,4:6]=value
        self.goal_buf=self.context

    def prepare_commands(self):
        if self.eval_mode:return
        self.event_left-=1
        ids=(self.event_left<=0).nonzero().squeeze(-1)
        if len(ids):
            k=len(ids);dev=self.device
            axis=torch.randint(0,2,(k,),device=dev)
            degrees=(2*torch.rand(k,device=dev)-1)*MAX_ANGLE
            mix=torch.rand(k,device=dev)
            degrees=torch.where(mix<.25,torch.where(degrees>=0,MAX_ANGLE,-MAX_ANGLE),degrees)
            degrees=torch.where(mix>.9,(2*torch.rand(k,device=dev)-1)*math.radians(5),degrees)
            target=torch.zeros(k,2,device=dev);target[torch.arange(k,device=dev),axis]=degrees
            neutral=self.individual[ids]&(self.event_count[ids]%2==1)
            neutral|=(~self.individual[ids])&(self.event_count[ids]%4==3)
            target[neutral]=0
            self.context[ids,4:6]=target
            long=torch.randint(180,251,(k,),device=dev)
            short=torch.randint(60,151,(k,),device=dev)
            self.event_left[ids]=torch.where(self.individual[ids]|neutral,long,short)
            self.event_count[ids]+=1
        self.context[self.ep_len>=self.max_ep_len-175,4:6]=0
        self.goal_buf=self.context

    def step(self,action):
        action=action.clamp(-1.,1.);previous=self.prev_action.clone();xi=self.xi.clone()
        now=(self.q0+self.action_scale*action-self.enc_bias).clamp(self.task.qmin+.02,self.task.qmax-.02)
        old=(self.q0+self.action_scale*previous-self.enc_bias).clamp(self.task.qmin+.02,self.task.qmax-.02)
        peak_speed=torch.zeros(self.num_envs,device=self.device);peak_power=peak_speed.clone();contact=[]
        for j in range(self.decimation):
            before_velocity=self.sim.data.qvel[:,6:18].clone()
            ctrl=torch.where((j<self.delay)[:,None],old,now)
            self.sim.data.ctrl[:]=ctrl[:,self.act_to_joint]
            self.sim.step()
            # Every 5ms, evaluate power at both integration endpoints.
            qd=self.sim.data.qvel[:,6:18]
            torque=self.sim.data.qfrc_actuator[:,6:18]
            peak_speed=torch.maximum(peak_speed,qd.abs().amax(-1))
            peak_speed=torch.maximum(peak_speed,before_velocity.abs().amax(-1))
            power=torch.maximum((torque*qd).abs().sum(-1),(torque*before_velocity).abs().sum(-1))
            peak_power=torch.maximum(peak_power,power)
            if self.contact_recorder is not None:contact.append(self.contact_recorder.sample())
        if contact:self.last_contacts=tuple(torch.stack([x[i] for x in contact]) for i in range(4))
        s=self.read_state()
        bad=~torch.stack([torch.isfinite(v).flatten(1).all(1) for v in s.values()]).all(0)
        bad|=(s['qd'].abs().amax(-1)>150)|(s['v'].abs().amax(-1)>30)
        self.last_step_invalid=bad;self.invalid_physics_count+=bad.sum()
        for k,v in s.items():s[k]=torch.where(bad.reshape((-1,)+(1,)*(v.ndim-1)),self.state[k],v)
        self.context=advance_command(self.context)
        an=self.task.command_fast(s,self.context,action,previous,xi)
        r=torch.where(bad,-2.,an['r']);hard=an['hard']|bad
        self.ep_len+=1;self.frame+=1;self.ep_ret+=r
        trunc=(self.ep_len>=self.max_ep_len)&~hard;done=hard|trunc
        info=dict(reward=r,terminated=hard,truncated=trunc,done=done,terms=an['terms'],peak_speed=peak_speed,peak_power=peak_power)
        self.prev_action=action.clone()
        self.obs_hist=torch.cat([self.obs_hist[:,1:],(an['obs']+self._noise(self.num_envs))[:,None]],1)
        ids=done.nonzero().squeeze(-1) if self.autoreset else torch.empty(0,dtype=torch.long,device=self.device)
        if len(ids):
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),done_terminated=hard[ids].clone(),done_norm_return=self.ep_ret[ids]/(self.max_ep_len*self.tcfg.max_step_reward))
            self.reset_envs(ids)
        self._finish_resets(ids)
        return s,info


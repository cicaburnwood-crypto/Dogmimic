"""Reference-free, proprioceptive B2 locomotion with full start/stop episodes."""
import json
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from mjlab.managers.event_manager import RecomputeLevel
from supermimic.task import Task,TaskCfg,NJ
from supermimic.sim_env import TrackingSimEnv,EnvCfg
from supermimic.rotations import heading_mat,mtv,mv
from skill_env import B2Env
from contract import *
from gait_clock import advance_phase,policy_goal
from wtw_rewards import reward_terms,combine
from reward_physics import RewardContactRecorder

XI_DIM=32  # friction, bias12, COM3, trunk-mass scale, kp/kd scales, delay/2, prev-action12
HERE=Path(__file__).resolve().parent

@dataclass
class CommandCfg(TaskCfg):
    history:int=HISTORY
    future:int=0
    noise_q:float=.005
    noise_qd:float=.25
    noise_gyro:float=.08
    noise_grav:float=.02
    @property
    def max_step_reward(self):return 1.5*self.dt

class CommandTask(Task):
    def __init__(self,*args,standing_height,**kw):
        super().__init__(*args,**kw)
        self.height=standing_height
        self.feet=torch.tensor(self.kin.body_index(('FL_calf','FR_calf','RL_calf','RR_calf')),device=self.device)
        self.offset=torch.tensor([0.,8.6986e-05,-.35],device=self.device)
        from supermimic.task import flat_to_state
        s=torch.zeros(1,37,device=self.device);s[:,3]=1.;s[:,7:19]=self.q0
        fp,_=self.foot_state(self.body_kin(flat_to_state(s)))
        self.width=float((fp[0,[0,2],1].mean()-fp[0,[1,3],1].mean()).item())
        self.length=float((fp[0,:2,0].mean()-fp[0,2:,0].mean()).item())
    def foot_state(self,kin):
        off=mv(kin[1][...,self.feet,:,:],self.offset)
        return kin[0][...,self.feet,:]+off,kin[2][...,self.feet,:]+torch.cross(kin[3][...,self.feet,:],off,dim=-1)
    def command_analytics(self,s,previous_s,cmd,action,prev,lastlast,bias,phase,aux,contacts_prev):
        kin=self.body_kin(s);fp,fv=self.foot_state(kin)
        raw,contact=reward_terms(s,previous_s,cmd,action,prev,lastlast,bias,phase,aux,contacts_prev,
                                fp,fv,self.q0,self.action_scale,self.qmin,self.qmax,self.height,self.width,self.length)
        reward,terms=combine(raw)
        tilt=1-s['rot'][...,2,2]
        hard=(s['pos'][...,2]<.32)|(tilt>.6)
        soft=1-(1-torch.sigmoid((.32-s['pos'][...,2])/.025))*(1-torch.sigmoid((tilt-.6)/.04))
        return dict(r=reward,hard=hard,soft=soft,terms=terms,obs=self.obs_clean(s,action,bias),
                    feats=self.state_features(s,kin),goal=policy_goal(cmd,phase),contacts=contact)
    def command_fast(self,*args):return self._maybe_compiled('command',self.command_analytics,*args)

class CommandEnv(B2Env):
    def __init__(self,device,n,evaluate=False,max_speed=.5,yaw_std=.35,randomize=None):
        tc=CommandCfg();dr=not evaluate if randomize is None else randomize
        ec=EnvCfg(num_envs=n,episode_length_s=22.,domain_rand=dr,obs_noise=dr,reset_noise=False,
                  compile=not evaluate,set_const_every=1,friction_range=(.5,1.25),enc_bias_range=(-.005,.005),
                  com_range=((-0.015,.015),)*3)
        super().__init__(SimpleNamespace(num_bins=1),ec,tc,device,model_path=HERE/'b2_locomotion.xml')
        self.lib=None
        self.model_meta=json.loads((HERE/'model.json').read_text())
        self.q0=torch.tensor(Q0,device=device);self.action_scale=torch.tensor(ACTION_SCALE,device=device)
        limits=torch.tensor(self.mj_model.jnt_range[1:],device=device,dtype=torch.float32)
        self.task=CommandTask(self.kin,self.q0,limits,self.action_scale,tc,device,
                              standing_height=self.model_meta['standing_height_m'],compile_sizes=() if evaluate else (n,))
        self.eval_mode=evaluate
        self.commands=torch.zeros(n,3,device=device);self.targets=self.commands.clone()
        self.command_left=torch.zeros(n,dtype=torch.long,device=device)
        self.max_speed=max_speed;self.yaw_std=yaw_std
        self.mass_scale=torch.ones(n,device=device);self.kp_scale=self.mass_scale.clone();self.kd_scale=self.mass_scale.clone()
        self.delay=torch.zeros(n,dtype=torch.long,device=device)
        self.base_mass=float(self.mj_model.body_mass[1]);self.base_inertia=torch.tensor(self.mj_model.body_inertia[1],device=device,dtype=torch.float32)
        if dr:
            fields=('body_mass','body_inertia','body_subtreemass','dof_invweight0','body_invweight0',
                    'tendon_length0','tendon_invweight0','actuator_gainprm','actuator_biasprm')
            self.sim.expand_model_fields(tuple(f for f in fields if f not in self.sim.expanded_fields))
        self.invalid_physics_count=torch.zeros((),device=device,dtype=torch.long)
        self.last_step_invalid=torch.zeros(n,device=device,dtype=torch.bool)
        self.contact_recorder=None
        self.reward_recorder=RewardContactRecorder(self)
        self.phase=torch.full((n,),.25,device=device)
        self.reward_prev_contacts=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.lastlast_action=torch.zeros(n,12,device=device)
        self.last_aux=torch.zeros(n,32,device=device)
    read_state=TrackingSimEnv.read_state
    @property
    def xi(self):
        base=TrackingSimEnv.xi.fget(self)
        return torch.cat([base,self.mass_scale[:,None],self.kp_scale[:,None],self.kd_scale[:,None],
                          self.delay[:,None].float()/2.,self.prev_action],-1)
    def _randomize(self,ids):
        TrackingSimEnv._randomize(self,ids)
        k=len(ids)
        # Evaluation with randomize=True uses exactly these held-out physical ranges.
        if self.cfg.domain_rand:
            if self.eval_mode:
                self.friction[ids]=.5+.75*torch.rand(k,device=self.device)
                self.enc_bias[ids]=.01*torch.rand(k,12,device=self.device)-.005
                self.com_offset[ids]=.03*torch.rand(k,3,device=self.device)-.015
                gf=self.sim.model.geom_friction;vals=gf[ids][:,self.foot_geoms].clone()
                vals[...,0]=self.friction[ids,None];gf[ids[:,None],self.foot_geoms[None,:]]=vals
                self.sim.model.body_ipos[ids,1]=self.default_torso_ipos+self.com_offset[ids]
            self.mass_scale[ids]=.8+.4*torch.rand(k,device=self.device)
            self.kp_scale[ids]=.85+.3*torch.rand(k,device=self.device)
            self.kd_scale[ids]=.8+.4*torch.rand(k,device=self.device)
            self.delay[ids]=torch.randint(0,3,(k,),device=self.device)
            if self.eval_mode and getattr(self,'nominal_stride',0):
                nominal=ids[ids%self.nominal_stride==0]
                self.friction[nominal]=1.;self.enc_bias[nominal]=0;self.com_offset[nominal]=0
                self.mass_scale[nominal]=1.;self.kp_scale[nominal]=1.;self.kd_scale[nominal]=1.;self.delay[nominal]=0
                gf=self.sim.model.geom_friction;vals=gf[nominal][:,self.foot_geoms].clone();vals[...,0]=1.
                gf[nominal[:,None],self.foot_geoms[None,:]]=vals
                self.sim.model.body_ipos[nominal,1]=self.default_torso_ipos
            self.sim.model.body_mass[ids,1]=self.base_mass*self.mass_scale[ids]
            self.sim.model.body_inertia[ids,1]=self.base_inertia*self.mass_scale[ids,None]
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
        z=torch.zeros(k,3,device=self.device);q=self.q0.expand(k,-1).clone()
        self.write_state(ids,pos,quat,q,z,z,torch.zeros_like(q))
        self.frame[ids]=0;self.ep_len[ids]=0
        self.ep_id[ids]=torch.arange(self._next_ep,self._next_ep+k,device=self.device);self._next_ep+=k
        self.phase[ids]=.25;self.reward_prev_contacts[ids]=False;self.lastlast_action[ids]=0
        self.prev_action[ids]=0;self.ep_ret[ids]=0;self.commands[ids]=0;self.targets[ids]=0;self.command_left[ids]=0
    def _finish_resets(self,ids,force=False):
        if self.cfg.domain_rand and self._need_set_const:
            self.sim.recompute_constants(RecomputeLevel.set_const);self._need_set_const=False
        self.state=self.read_state()
        if len(ids):
            s={k:v[ids] for k,v in self.state.items()}
            self.obs_hist[ids]=(self.task.obs_clean(s,self.prev_action[ids],self.enc_bias[ids])+self._noise(len(ids)))[:,None]
        self.goal_buf=policy_goal(self.commands,self.phase)
    def _noise(self,k):
        return self.task.sample_obs_noise((k,)) if self.cfg.obs_noise else torch.zeros(k,OBS,device=self.device)
    def set_command(self,cmd):
        target=torch.as_tensor(cmd,device=self.device,dtype=torch.float32).expand(self.num_envs,3).clone()
        target[:,1].clamp_(-MAX_LATERAL,MAX_LATERAL)
        target[:,:2]*=(MAX_TRANSLATION/target[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1.)
        if not torch.isfinite(target).all():raise ValueError('Nonfinite command')
        self.targets[:]=target
        delta=target-self.commands
        self.commands[:,:2]+=delta[:,:2]*(LINEAR_ACCELERATION*DT/delta[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp(max=1.)
        self.commands[:,2]+=delta[:,2].clamp(-ANGULAR_ACCELERATION*DT,ANGULAR_ACCELERATION*DT)
        self.goal_buf=policy_goal(self.commands,self.phase)
    def _sample_targets(self,ids):
        k=len(ids);dev=self.device
        # Retain 25% low-speed rehearsal after expanding the command range.
        bound=torch.full((k,),self.max_speed,device=dev)
        bound[torch.rand(k,device=dev)<.25]=min(.5,self.max_speed)
        edge=torch.rand(k,device=dev)<.5
        speed=bound*torch.where(edge,.7+.3*torch.rand(k,device=dev),torch.rand(k,device=dev))
        vy=(torch.rand(k,device=dev)*2-1)*torch.minimum(speed,torch.full_like(speed,MAX_LATERAL))
        sign=torch.where(torch.rand(k,device=dev)<.5,-torch.ones_like(speed),torch.ones_like(speed))
        vx=sign*torch.sqrt(torch.clamp(speed.square()-vy.square(),min=0))
        # Cover pure forward/reverse, pure sideways, simultaneous xy and rotation.
        mode=torch.rand(k,device=dev);pure_x=mode<.22;pure_y=(mode>=.22)&(mode<.37)
        vy[pure_x]=0;vx[pure_x]=sign[pure_x]*speed[pure_x]
        vx[pure_y]=0;vy[pure_y]=sign[pure_y]*torch.minimum(speed[pure_y],torch.full_like(speed[pure_y],MAX_LATERAL))
        wz=torch.randn(k,device=dev)*self.yaw_std  # unbounded angular support, no clipping
        wz[torch.rand(k,device=dev)<.5]=0
        cmd=torch.stack([vx,vy,wz],-1)
        cmd[torch.rand(k,device=dev)<.15]=0
        spin=mode>.92;cmd[spin,:2]=0
        self.targets[ids]=cmd
        self.command_left[ids]=torch.randint(100,201,(k,),device=dev)
    def prepare_commands(self):
        if self.eval_mode:return
        t=self.ep_len*DT
        moving=((t>=1.5)&(t<7.5))|((t>=10.5)&(t<17.))
        self.command_left-=1
        change=(moving&(self.command_left<=0)).nonzero().squeeze(-1)
        if len(change):self._sample_targets(change)
        self.targets[~moving]=0;self.command_left[~moving]=0
        self.set_command(self.targets)
    def step(self,action):
        action=action.clamp(-1.,1.)
        now=(self.q0+self.action_scale*action-self.enc_bias).clamp(self.task.qmin+.02,self.task.qmax-.02)
        old=(self.q0+self.action_scale*self.prev_action-self.enc_bias).clamp(self.task.qmin+.02,self.task.qmax-.02)
        contact_samples=[]
        for j in range(self.decimation):
            ctrl=torch.where((j<self.delay)[:,None],old,now)
            self.sim.data.ctrl[:]=ctrl[:,self.act_to_joint];self.sim.step()
            if self.contact_recorder is not None:contact_samples.append(self.contact_recorder.sample())
        if contact_samples:
            self.last_contacts=tuple(torch.stack([s[i] for s in contact_samples]) for i in range(4))
        s=self.read_state()
        bad=~torch.stack([torch.isfinite(v).flatten(1).all(1) for v in s.values()]).all(0)
        bad|=(s['qd'].abs().amax(-1)>150)|(s['v'].abs().amax(-1)>30)
        self.last_step_invalid=bad;self.invalid_physics_count+=bad.sum()
        for k,v in s.items():s[k]=torch.where(bad.reshape((-1,)+(1,)*(v.ndim-1)),self.state[k],v)
        self.last_aux=self.reward_recorder.sample(s)
        bad |= ~torch.isfinite(self.last_aux).all(-1)
        self.last_step_invalid=bad
        self.last_aux=torch.nan_to_num(self.last_aux,nan=0.,posinf=0.,neginf=0.)
        phase=advance_phase(self.phase,self.commands)
        an=self.task.command_fast(s,self.state,self.commands,action,self.prev_action,self.lastlast_action,
                                  self.enc_bias,phase,self.last_aux,self.reward_prev_contacts)
        r=torch.where(bad,0.,an['r']);hard=an['hard']|bad
        self.phase=phase;self.reward_prev_contacts=an['contacts']
        self.lastlast_action=self.prev_action.clone()
        self.ep_len+=1;self.frame+=1;self.ep_ret+=r
        truncated=(self.ep_len>=self.max_ep_len)&~hard;done=hard|truncated
        info=dict(reward=r,terminated=hard,truncated=truncated,done=done,terms=an['terms'])
        self.prev_action=action.clone()
        self.obs_hist=torch.cat([self.obs_hist[:,1:],(an['obs']+self._noise(self.num_envs))[:,None]],1)
        ids=done.nonzero().squeeze(-1)
        if len(ids):
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),
                        done_terminated=hard[ids].clone(),done_norm_return=self.ep_ret[ids]/(self.max_ep_len*self.tcfg.max_step_reward))
            self.reset_envs(ids)
        self._finish_resets(ids)
        return s,info

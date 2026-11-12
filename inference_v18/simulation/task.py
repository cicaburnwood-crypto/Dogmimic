"""Simultaneous locomotion and body attitude on the pinned vibe engine."""
import json
import math
from dataclasses import dataclass
import mujoco
import torch
from bootstrap import HERE
from contract import *
from sampling import positive_huber, sample_targets as stratified_targets
from supermimic.task import Task, TaskCfg
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import mv, mtv, heading_mat
from arm import ArmController
from actuation import XI_DIM,advance_actuator,consumed_targets
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sim.sim import Simulation
from mjlab.managers.event_manager import RecomputeLevel



@dataclass
class TiltLocomotionCfg(TaskCfg):
    history:int=HISTORY
    future:int=0
    noise_q:float=.005
    noise_qd:float=.25
    noise_gyro:float=.08
    noise_grav:float=.02
    @property
    def max_step_reward(self):return 9.3*DT

class TiltLocomotionTask(Task):
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
        rp=angles(s['rot']);error=angle_error(rp,context[...,3:5])
        vel=heading_velocity(s['rot'],s['v']);rates=euler_rates(s['rot'],s['w'])
        verror=vel[...,:2]-context[...,:2];yawerror=rates[...,2]-context[...,2]
        omega=mtv(s['rot'],s['w']);rateerror=omega-target_body_rate(context)
        bias=xi[...,1:13]
        applied,act_next=advance_actuator(action,xi,self)
        target=self.q0+self.action_scale*applied
        requested=KP*xi[...,17:18]*(target-s['q'])-KD*xi[...,18:19]*s['qd']
        torque=requested.clamp(-self.torque_limits,self.torque_limits)
        stance,height,clock=gait(context)
        # Gait phase comes from command dynamics, not a teacher joint trajectory.
        # Separate actual-near-ground slip from desired stance velocity tracking.
        contact=torch.sigmoid((.034-feet[...,2])/.004).detach()
        active=moving(context).float();stopped=1-active
        foot_v2=velocity[...,:2].square().sum(-1)
        foot_speed=torch.sqrt(foot_v2+1e-10)
        attitude_norm=torch.sqrt(error.square().sum(-1)+1e-12)
        body_z=s['pos'][...,2]-(self.height+context[...,13])
        rel=mtv(heading_mat(s['rot'])[...,None,:,:],feet-s['pos'][...,None,:])
        widths=rel[..., [0,2],1]-rel[..., [1,3],1]
        width_tolerance=.06+.04*context[...,3].abs()/MAX_ANGLE+.03*context[...,1].abs()
        width_excess=torch.relu((widths-.383286).abs()-width_tolerance[...,None])
        width_imbalance=torch.relu((widths[...,0]-widths[...,1]).abs()-.08-.04*context[...,3].abs()/MAX_ANGLE)
        # Command-conditioned Cartesian foot sweep, not a joint-motion teacher.
        anchors=rel.new_tensor([[.3285,.191643], [.3285,-.191643],[-.3285,.191643],[-.3285,-.191643]])
        foot_motion=context[...,:2][...,None,:]+context[...,2,None,None]*torch.stack([-anchors[:,1],anchors[:,0]],-1)
        phase=foot_phases(context);d=duty(context)[...,None]
        swing=((phase-d)/(1-d)).clamp(0,1)
        ease=swing.square()*(3-2*swing)
        travel=torch.where(phase<d,d*.5-phase,d*(ease-.5))/frequency(context)[...,None]
        desired_xy=anchors+foot_motion*travel[...,None]*blend(context)[...,None,None]
        input_now=act_next[...,32:44];ready=xi[...,80]
        da=(input_now-xi[...,32:44])*self.action_scale
        dda=(input_now-2*xi[...,32:44]+xi[...,44:56])*self.action_scale
        terms=dict(
            linear_tracking=5.*torch.exp(-verror.square().sum(-1)/.5**2),
            linear_error=-1.*verror.square().sum(-1),
            yaw_tracking=1.5*torch.exp(-yawerror.square()/.4**2),
            yaw_error=-.2*yawerror.square(),
            attitude=2.*torch.exp(-error.square().sum(-1)/math.radians(4)**2),
            attitude_error=-.5*positive_huber(attitude_norm/math.radians(2)),
            body_rate=-.08*rateerror.square().sum(-1),
            height=1.5*torch.exp(-body_z.square()/.03**2)-.6*positive_huber(body_z.abs()/.03),
            vertical_velocity=-.5*(s['v'][...,2]-context[...,14]).square(),
            stance_velocity=-.4*(stance*positive_huber(foot_speed/.5)).mean(-1),
            contact_slip=-.4*(contact*positive_huber(foot_speed/.2)).sum(-1)/contact.sum(-1).clamp_min(1.),
            foot_clearance=-3.*((feet[...,2]-height).square()/clearance(context)[...,None].square()).mean(-1),
            foot_sweep=-1.*active*((rel[...,:2]-desired_xy).square().sum(-1)/.12**2).mean(-1),
            moving_width=-.5*active*positive_huber(width_excess/.05).mean(-1),
            rear_splay=-.4*active*positive_huber(width_excess[...,1]/.05),
            support_balance=-.25*active*positive_huber(width_imbalance/.05),
            stop_posture=.3*stopped*torch.exp(-context[...,13].square()/.03**2)*torch.exp(-(s['q']-self.q0).square().mean(-1)/.4**2),
            action_rate=-.08*ready*(da/.06).square().mean(-1),
            action_curvature=-.04*ready*(dda/.04).square().mean(-1),
            effort=-.04*(requested/self.torque_limits).square().mean(-1),
            joint_velocity=-.0005*s['qd'].square().sum(-1),
            mechanical_work=-.00004*(torque*s['qd']).abs().sum(-1),
            # Train a margin before the unchanged actual-speed acceptance limit.
            # The physical target filter is identical in collection and imagination.
            planar_overspeed=-.4*positive_huber(torch.relu(vel[...,:2].norm(dim=-1)-(P3_MAX_SPEED-.10))/.10),
            lateral_overspeed=-.4*positive_huber(torch.relu(vel[...,1].abs()-(MAX_LATERAL-.05))/.05),
            joint_limits=-2.*(torch.relu(self.qmin-s['q'])+torch.relu(s['q']-self.qmax)).sum(-1),
        )
        progress=torch.exp(-verror.square().sum(-1)/.5**2-yawerror.square()/.4**2)
        motion_gate=stopped+active*(.25+.75*progress)
        terms['attitude']=terms['attitude']*motion_gate
        terms['height']=1.5*torch.exp(-body_z.square()/.03**2)*motion_gate-.6*positive_huber(body_z.abs()/.03)
        tilt=rp.abs().amax(-1)
        hard=(s['pos'][...,2]<.25)|(tilt>math.radians(55))
        # A 29.6cm commanded root is only 4.6cm above the unchanged hard floor.
        # Localize the surrogate there instead of rewarding refusal to crouch.
        ztemp=.02-.012*((-context[...,13]-.10)/.10).clamp(0,1)
        soft=1-(1-torch.sigmoid((.25-s['pos'][...,2])/ztemp))*(1-torch.sigmoid((tilt-math.radians(55))/.04))
        reward=sum(terms.values())*DT-2.*soft
        return dict(r=reward,hard=hard,soft=soft,terms=terms,obs=self.obs_clean(s,action,bias),feats=self.state_features(s,kin),goal=policy_goal(context))

    def command_fast(self,*args):return self._maybe_compiled('tilt_locomotion',self.command_analytics,*args)

class TiltLocomotionEnv(TrackingSimEnv):
    def __init__(self,device,n,evaluate=False,randomize=True,compile=True,autoreset=True):
        self.device=device;self.num_envs=n;self.gamma=.99;self.eval_mode=evaluate;self.autoreset=autoreset
        self.tcfg=TiltLocomotionCfg();self.cfg=EnvCfg(num_envs=n,episode_length_s=32.,domain_rand=randomize,
            obs_noise=randomize,reset_noise=False,compile=compile,set_const_every=1)
        self.model_meta=json.loads((HERE/'model.json').read_text())
        m=mujoco.MjModel.from_xml_path(str(HERE/'b2_a3.xml'))
        self.sim=Simulation(num_envs=n,cfg=SimulationCfg(nconmax=256,njmax=1024,
            mujoco=MujocoCfg(timestep=SIM_DT,iterations=10,ls_iterations=20)),model=m,device=device)
        self.mj_model=m=self.sim.mj_model;self.decimation=4
        self.kin=FastTreeKinematics(mujoco.MjModel.from_xml_path(str(HERE/'leg_kinematics.xml')),device,root_body=1)
        self.root_qadr=self.root_vadr=0
        self.q_adr=torch.arange(7,19,device=device);self.qd_adr=torch.arange(6,18,device=device)
        self.act_to_joint=torch.tensor(m.actuator_trnid[:12,0]-1,device=device)
        self.joint_names=[m.joint(j).name for j in range(1,13)]
        self.q0=torch.tensor(Q0,device=device);self.action_scale=torch.tensor(ACTION_SCALE,device=device)
        self.foot_geoms=torch.tensor([m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')],device=device)
        self.task=TiltLocomotionTask(self.kin,self.q0,torch.tensor(m.jnt_range[1:13],dtype=torch.float32,device=device),
            self.action_scale,self.tcfg,device,height=self.model_meta['standing_height_m'],
            foot_offsets=m.geom_pos[self.foot_geoms.cpu().numpy()],compile_sizes=(n,) if compile else ())
        self.arm=ArmController(self.task,device);self.task.arm=self.arm;self.arm.compiled=compile
        self.arm_torque_limits=torch.tensor([97.,97.,40.,40.,40.,40.],device=device)
        self.arm_memory=self.arm.reset(n,randomize)
        self.last_arm_rejected=torch.zeros(n,dtype=torch.bool,device=device)
        self.last_arm_fault=self.last_arm_rejected.clone()
        self.arm_peak_speed=torch.zeros(n,device=device)
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
        self.context=torch.zeros(n,CONTEXT_DIM,device=device)
        self.actuator_memory=torch.zeros(n,61,device=device)
        self.curriculum_level=0
        self.event_left=self.frame.clone();self.event_count=self.frame.clone();self.individual=torch.zeros(n,device=device,dtype=torch.bool)
        self.bin_G=torch.zeros(1,device=device);self.bin_visits=self.bin_G.clone();self.bin_past_w=self.bin_G.clone()
        self.ep_ret=torch.zeros(n,device=device)
        self.invalid_physics_count=torch.zeros((),device=device,dtype=torch.long);self.last_step_invalid=torch.zeros(n,device=device,dtype=torch.bool)
        self.state=None;self.goal_buf=policy_goal(self.context);self.contact_recorder=None;self.fixed_domain=None
        self._need_set_const=False

    def read_state(self):
        s=super().read_state()
        s.update(aq=self.sim.data.qpos[:,19:25].clone(),aqd=self.sim.data.qvel[:,18:24].clone())
        return s

    @property
    def xi(self):
        return torch.cat([self.friction[:,None],self.enc_bias,self.com_offset,self.mass_scale[:,None],
                          self.kp_scale[:,None],self.kd_scale[:,None],self.delay[:,None]/2.,self.actuator_memory,self.arm_memory[:,:24],
                          self.context[:,7:9].norm(dim=-1,keepdim=True),self.context[:,9:10].abs()],-1)

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
        self.sim.model.actuator_gainprm[ids,:12,0]=KP*self.kp_scale[ids,None]
        self.sim.model.actuator_biasprm[ids,:12,1]=-KP*self.kp_scale[ids,None]
        self.sim.model.actuator_biasprm[ids,:12,2]=-KD*self.kd_scale[ids,None]
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
        self.actuator_memory[ids]=0
        self.sim.data.qpos[ids,19:25]=self.arm.home
        self.sim.data.qvel[ids,18:24]=0
        self.sim.data.ctrl[ids,12:18]=self.arm.home
        self.arm_memory[ids]=self.arm.reset(k,self.cfg.domain_rand)
        self.prev_action[ids]=0;self.context[ids]=0;self.context[ids,12]=.25;self.event_left[ids]=100;self.event_count[ids]=0
        self.individual[ids]=torch.rand(k,device=self.device)<.5

    def _finish_resets(self,ids,force=False):
        if self._need_set_const:
            self.sim.recompute_constants(RecomputeLevel.set_const);self._need_set_const=False
        self.state=self.read_state()
        if len(ids):
            state={k:v[ids] for k,v in self.state.items()}
            self.obs_hist[ids]=(self.task.obs_clean(state,self.prev_action[ids],self.enc_bias[ids])+self._noise(len(ids)))[:,None]
        self.goal_buf=policy_goal(self.context)

    def _noise(self,n):
        noise=self.task.sample_obs_noise((n,)) if self.cfg.obs_noise else torch.zeros(n,OBS,device=self.device)
        if getattr(self,'nominal_stride',0) and n==self.num_envs:noise[::self.nominal_stride]=0
        return noise

    def set_target(self,target):
        value=torch.as_tensor(target,device=self.device,dtype=torch.float32)
        if value.shape[-1]==5:value=torch.cat([value,torch.zeros_like(value[...,:1])],-1)
        value=value.expand(self.num_envs,6)
        if not torch.isfinite(value).all():raise ValueError('Nonfinite command')
        self.context=set_requested(self.context,value)
        self.goal_buf=policy_goal(self.context)

    def sample_targets(self,k):
        return stratified_targets(k,self.device,self.curriculum_level)

    def prepare_commands(self):
        if self.eval_mode:return
        self.event_left-=1
        ids=(self.event_left<=0).nonzero().squeeze(-1)
        if len(ids):
            target=self.sample_targets(len(ids))
            paired=self.individual[ids]
            reverse=paired&(self.event_count[ids]==1)
            target[reverse,:5]=-self.context[ids[reverse],7:12]
            target[reverse,5]=self.context[ids[reverse],15]
            target[paired&(self.event_count[ids]>=2)]=0
            target[(~paired)&((self.event_count[ids]%4)==3)]=0
            self.context[ids]=set_requested(self.context[ids],target)
            self.event_left[ids]=torch.randint(400,501,(len(ids),),device=self.device)
            self.event_count[ids]+=1
        finishing=self.ep_len>=self.max_ep_len-250
        self.context[finishing,7:12]=0;self.context[finishing,15]=0
        self.goal_buf=policy_goal(self.context)

    def step(self,action):
        action=action.clamp(-1.,1.);previous=self.prev_action.clone();xi=self.xi.clone()
        targets,next_xi=consumed_targets(action,xi,self.task)
        arm_target,arm_next,rejected=self.arm.advance_fast(self.state,xi)
        self.last_arm_target=arm_target.detach().clone()
        self.last_arm_rejected=rejected;arm_fault=torch.zeros_like(rejected)
        arm_peak=torch.zeros(self.num_envs,device=self.device)
        arm_min=torch.full_like(arm_peak,10.)
        peak_speed=torch.zeros(self.num_envs,device=self.device);peak_power=peak_speed.clone();contact=[]
        for j in range(self.decimation):
            before_velocity=self.sim.data.qvel[:,6:18].clone()
            ctrl=targets[:,j]
            self.sim.data.ctrl[:,:12]=ctrl[:,self.act_to_joint]
            # Genuine bounded actuator torque; never overwrite arm q/qdot.
            gravity=self.arm.gravity_torque(self.read_state(),self.mass_scale)
            # Affine actuator PD keeps velocity damping in MuJoCo's implicit
            # integration. This target offset adds gravity before force clipping.
            self.sim.data.ctrl[:,12:18]=arm_target+gravity/120.
            self.sim.step()
            arm_s=self.read_state()
            safe,dist,ground,tcp=self.arm.safety_fast(arm_s,arm_s['aq'])
            arm_fault|=~safe;arm_min=torch.minimum(arm_min,dist)
            arm_peak=torch.maximum(arm_peak,arm_s['aqd'].abs().amax(-1))
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
        r=torch.where(bad,-2.,an['r']);hard=an['hard']|bad|arm_fault
        self.last_arm_fault=arm_fault;self.arm_peak_speed=arm_peak;self.arm_min_clearance=arm_min
        self.ep_len+=1;self.frame+=1;self.ep_ret+=r
        trunc=(self.ep_len>=self.max_ep_len)&~hard;done=hard|trunc
        info=dict(reward=r,terminated=hard,truncated=trunc,done=done,terms=an['terms'],peak_speed=peak_speed,peak_power=peak_power)
        self.prev_action=action.clone()
        self.actuator_memory=next_xi[:,20:81].clone()
        self.arm_memory=arm_next.clone()
        self.obs_hist=torch.cat([self.obs_hist[:,1:],(an['obs']+self._noise(self.num_envs))[:,None]],1)
        ids=done.nonzero().squeeze(-1) if self.autoreset else torch.empty(0,dtype=torch.long,device=self.device)
        if len(ids):
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),done_terminated=hard[ids].clone(),done_norm_return=self.ep_ret[ids]/(self.max_ep_len*self.tcfg.max_step_reward))
            self.reset_envs(ids)
        self._finish_resets(ids)
        return s,info

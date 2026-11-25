"""One fresh 18-action policy, two goal interfaces, official WM actor-critic.

No robot IO, pretrained locomotion, orientation goal, or inference IK gate.
"""
import math
import torch
from torch import nn
from tensordict import TensorDict
from official_adapter import Representation, RobotDynamics, pack, unpack, POSITION_TOLERANCE, STILL_LIMITS, still_mask
from joint import JointEnv, ee, arm_view, wm_action, base_ground_clearance, clearance_reserve
from contract import DT, heading_velocity
from b2physics.rotations import mv, mtv


def smooth_norm(x, eps=1e-8):
    return (x.square().sum(-1)+eps).sqrt()


def observation(s, xi, goal, cmd, mode, prev, anchor):
    tip,_=ee(ARM,s)
    v=heading_velocity(s['rot'],s['v'])
    rel_tip=mtv(s['rot'],goal-tip)
    rel_base=mtv(s['rot'],goal-s['pos'])
    # Goal slots are zeroed in velocity mode; commands are zeroed in EE mode.
    task=torch.cat([mode[:,None],cmd*(1-mode[:,None]),rel_tip*mode[:,None],rel_base/3*mode[:,None]],-1)
    proprio=torch.cat([v/2,mtv(s['rot'],s['w']),-s['rot'][:,2,:],s['q'],s['qd']/10,
                      s['aq'],s['aqd'],s['pos'][:,2:3],prev,
                      xi[:,20:81],xi[:,93:105],mtv(s['rot'],s['pos']-anchor),
                      torch.log(xi[:,107:113].exp().clamp_min(1e-6))],-1)
    return torch.cat([task,proprio],-1)


# Bound once to the geometry of this task, not a mutable control mode.
ARM=None
def bind(arm):
    global ARM
    ARM=arm


class Actor(nn.Module):
    def __init__(self, dim, task):
        super().__init__()
        self.vel=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,32),nn.Tanh())
        self.ee=nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,32),nn.Tanh())
        self.net=nn.Sequential(nn.Linear(dim-10+33,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,18))
        nn.init.normal_(self.net[-1].weight,std=.002);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('positive',(task.qmax-.02-task.q0)/task.action_scale)
        self.register_buffer('negative',(task.q0-(task.qmin+.02))/task.action_scale)

    def forward(self, obs):
        x=obs['actor'];m=x[...,:1]
        z=(1-m)*self.vel(x[...,1:4])+m*self.ee(x[...,4:10])
        raw=self.net(torch.cat([m,z,x[...,10:]],-1)).tanh()
        leg=raw[...,:12]*torch.where(raw[...,:12]>=0,self.positive,self.negative)
        return torch.cat([leg,raw[...,12:]],-1)


class Critic(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        values=self.net(x);m=x[...,:1]
        return values[...,:1]*(1-m)+values[...,1:]*m


def rewards(physics,s,previous,xi,next_xi,u,prev,goal,cmd,mode,anchor,known):
    """Smooth objectives; strict acceptance remains a physical-substep test."""
    arm=physics.arm;task=physics.task
    tip,_=ee(arm,s);old_tip,_=ee(arm,previous)
    d=smooth_norm(tip-goal);old_d=smooth_norm(old_tip-goal)
    v=heading_velocity(s['rot'],s['v']);verr=v[:,:2]-cmd[:,:2]
    yaw_error=s['w'][:,2]-cmd[:,2]
    # Wide dense term plus near-command precision, no gait/reference rewards.
    vel=2*torch.exp(-verr.square().sum(-1)/.5**2)-.25*(torch.sqrt(1+(verr/.5).square())-1).sum(-1)
    vel+=.75*torch.exp(-yaw_error.square()/.4**2)-.1*(torch.sqrt(1+(yaw_error/.5).square())-1)
    vel-=.08*s['v'][:,2].square()+.02*s['w'][:,:2].square().sum(-1)
    vel-=.05*(s['aq']-arm.home).square().mean(-1)
    zero=(cmd.square().sum(-1)<1e-6).float()
    vel-=zero*.15*(s['q']-task.q0).square().mean(-1)

    # A velocity-only soft posture preference prevents rewarding a brief
    # forward dive as locomotion. Based on reference-free differentiable
    # locomotion objectives (Open-DiffLoco, Table I); not a new angle/height
    # bound, gait clock, pose teacher, or an EE-mode restriction.
    posture=torch.zeros_like(vel)
    if getattr(physics,'velocity_posture',False):
        posture=torch.exp(-10*(s['pos'][:,2]-task.height).square())+.5*s['rot'][:,2,2]
        posture-=.3*(s['q']-task.q0).square().sum(-1)
        posture-=.42*s['v'][:,2].square()+.03*s['w'][:,:2].square().sum(-1)
        vel+=posture

    # Potential-based term is discount consistent, smooth, and bounded in step return.
    phi=-torch.log1p(d.square());old_phi=-torch.log1p(old_d.square())
    progress=.3*(.99*phi-old_phi)
    arrival=torch.exp(-(d/.03).square())
    ee_speed=smooth_norm((tip-old_tip)/DT)
    still=torch.exp(-s['v'].square().sum(-1)/.03**2-s['w'].square().sum(-1)/.06**2
                    -s['qd'].square().mean(-1)/.15**2-s['aqd'].square().mean(-1)/.06**2
                    -ee_speed.square()/.015**2)
    position=1.5*torch.exp(-d)+.5*torch.exp(-(d/.10).square())+torch.exp(-(d/.03).square())
    position+=1.5*torch.exp(-(d/.015).square())*still
    # No target orientation term. Only qualified arm-alone training samples get an anchor cost.
    base_cost=.015*s['v'].square().sum(-1)+.01*s['w'].square().sum(-1)
    base_cost+=known*.2*((s['pos']-anchor)/.2).square().sum(-1)
    braking=arrival.detach()*(.08*torch.log1p(s['v'].square().sum(-1)/.01**2)
                            +.04*torch.log1p(s['w'].square().sum(-1)/.02**2)
                            +.03*torch.log1p(ee_speed.square()/.005**2))

    feet,fv=task.foot_state(task.body_kin(s))
    contact=torch.sigmoid((.034-feet[...,2])/.004).detach()
    slip=(contact*fv[...,:2].square().sum(-1)).sum(-1)/contact.sum(-1).clamp_min(1.)
    # Use consumed filtered targets, not merely the raw network outputs.
    rate=(next_xi[:,20:32]-xi[:,20:32])*task.action_scale
    curvature=(next_xi[:,32:44]-2*xi[:,32:44]+xi[:,44:56])*task.action_scale
    smooth=.02*(rate/.08).square().mean(-1)+.01*(curvature/.08).square().mean(-1)
    smooth+=.02*(u[:,12:]-prev[:,12:]).square().mean(-1)+.0003*s['qd'].square().mean(-1)
    desired=task.q0+next_xi[:,20:32]*task.action_scale
    torque=(250*xi[:,17:18]*(desired-s['q'])-8*xi[:,18:19]*s['qd']).clamp(-task.torque_limits,task.torque_limits)
    energy=.02*(torque/task.torque_limits).square().mean(-1)+.00001*(torque*s['qd']).abs().sum(-1)

    details=arm.safety_details(s,s['aq'])
    body_clear=base_ground_clearance(arm,s)
    joint_gap=torch.minimum(s['q']-task.qmin,task.qmax-s['q'])
    arm_gap=torch.minimum(s['aq']-arm.lower,arm.upper-s['aq'])
    # Signed physical margins, no Euler-angle or commanded-height termination.
    margins=torch.stack([body_clear,details[2]-arm.margin,details[5].amin(-1),
                         joint_gap.amin(-1),arm_gap.amin(-1)],-1)
    scales=margins.new_tensor([.003,.002,.002,.01,.01])
    # A valid adjacent geometry can sit 1-2mm from its configured margin.
    # A sigmoid centered on that margin assigns ~0.3 failure even to standing.
    # C1 onset gives zero hazard in the valid set, and saturates after violation;
    # the softplus barrier below still supplies anticipatory safety gradients.
    violation=torch.relu(-margins/scales)
    term=1-torch.exp(-violation.square().sum(-1))
    barrier=torch.nn.functional.softplus(-margins/scales).sum(-1)
    if getattr(physics,'finite_pair_barrier',False):
        # Zero additional cost outside a finite, positive safety reserve.
        # Keep signed physical margins and termination exactly unchanged.
        reserve=clearance_reserve(arm,s).clamp_min(.001)
        old_pair=torch.nn.functional.softplus(-margins[:,2]/.002)
        pair_cost=.5*torch.relu((reserve-details[5])/.002).square().amax(-1)
        barrier=barrier-old_pair+pair_cost
    costs=.1*slip+smooth+energy+.5*barrier
    # Sampling 20% EE starts does not by itself make velocity the first skill:
    # narrow EE precision terms can dominate the shared leg gradient. Warm only
    # the EE task objective; physical safety and the evaluation stay unchanged.
    ee_scale=getattr(physics,'ee_task_scale',1.)
    task_value=(1-mode)*vel+mode*ee_scale*(position-base_cost-braking)
    r=DT*(task_value-costs)+mode*ee_scale*progress-2*term
    terms=dict(velocity=vel,ee_position=position,progress=progress,base_cost=base_cost,
               braking=braking,slip=slip,smooth=smooth,barrier=barrier,distance=d,term=term,
               velocity_posture=posture)
    return r,term,terms


class ImaginedEnv:
    def __init__(self,wm,rep,physics):
        self.dynamics=wm;self.representation=rep;self.physics=physics
        self.critic_privileged=False
        # Positive upper bound includes potential differences over the bounded physical domain.
        self.max_step_reward=4.0
    def observations(self):
        x=observation(self.s,self.xi,self.goal,self.cmd,self.mode,self.prev,self.anchor)
        # The training critic must observe reward labels and randomized dynamics
        # that are deliberately absent from the deployment Actor interface.
        if self.critic_privileged:
            scale=torch.full_like(self.known[:,None],getattr(self.physics,'ee_task_scale',1.))
            critic=torch.cat([x,self.known[:,None],self.xi[:,:20],scale],-1)
        else:critic=x
        return TensorDict({'actor':x,'critic':critic},batch_size=[len(x)])
    def reset(self,start):
        self.rep=self.representation.represent(start['state']);self.aux=self.representation.extract_aux_state(start['state'])
        self.s=unpack(start['state']);self.xi=self.aux[:,7:]
        for key in ['goal','cmd','mode','prev','anchor','known']:setattr(self,key,start[key])
        return self.observations()
    def step(self,u):
        aux,rep=self.dynamics(self.rep,wm_action(u),self.aux,u.new_zeros((len(u),0)))
        s=unpack(self.representation.reconstruct_full_state(aux[:,:7],rep));xi=aux[:,7:]
        r,term,self.terms=rewards(self.physics,s,self.s,self.xi,xi,u,self.prev,self.goal,self.cmd,self.mode,self.anchor,self.known)
        self.s=s;self.rep=rep;self.aux=aux;self.xi=xi;self.prev=u
        return self.observations(),r,term


def sample_tasks(e,stage,ids=None):
    if ids is None:ids=torch.arange(e.num_envs,device=e.device)
    n=len(ids);dev=e.device;s={k:v[ids] for k,v in e.state.items()};p=ee(e.arm,s)[0]
    center=getattr(e,'episode_origin',e.state['pos'])[ids]
    mode=(torch.rand(n,device=dev)<[.20,.5,.6,.6][stage]).float()
    cmd=torch.zeros(n,3,device=dev);vmax=[.5,1.,2.,2.][stage]
    cmd[:,0]=(2*torch.rand(n,device=dev)-1)*vmax
    cmd[:,1]=(2*torch.rand(n,device=dev)-1)*min(vmax,1.)
    norm=cmd[:,:2].norm(dim=-1,keepdim=True);cmd[:,:2]*=torch.clamp(vmax/norm.clamp_min(1e-6),max=1)
    cmd[:,2]=(2*torch.rand(n,device=dev)-1)*[.5,.5,1.,1.][stage]
    level=getattr(e,'velocity_level',2)
    if stage==0 and level<2:
        # Stage A: first fore/aft, then cardinal motion, then combined yaw.
        # This changes only training-command exposure; all fixed tests remain.
        cmd.zero_();speed=.15+.35*torch.rand(n,device=dev)
        if level==0:
            sign=torch.where(torch.rand(n,device=dev)<.5,1.,-1.)
            cmd[:,0]=speed*sign
        else:
            cardinal=torch.randint(4,(n,),device=dev)
            direction=cmd.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
            cmd[:,:2]=speed[:,None]*direction[cardinal]
    cmd[torch.rand(n,device=dev)<.15]=0;cmd*=1-mode[:,None]
    goal=p.clone();known=torch.zeros(n,device=dev)
    # Cheap positive path witnesses; never an IK solve per Actor update.
    q=s['aq'].clone();ok=torch.ones(n,dtype=torch.bool,device=dev)
    amplitude=.08 if stage==0 else .35
    proposal=(q+amplitude*torch.randn_like(q)).clamp(e.arm.lower+.03,e.arm.upper-.03)
    for f in [0.,.25,.5,.75,1.]:ok &= e.arm.safety_fast(s,q+f*(proposal-q))[0]
    proposal=torch.where(ok[:,None],proposal,q)
    goal=ee(e.arm,dict(s,aq=proposal))[0];known[:]=ok.float()
    # Even after the robot travels outside the target cylinder, a task switch
    # cannot publish the current tip as an out-of-domain fallback goal.
    radial=goal[:,:2]-center[:,:2];radius=radial.norm(dim=-1,keepdim=True)
    inside=(radius[:,0]<=3)&(goal[:,2]>=e.z_min)&(goal[:,2]<=e.z_max)
    bounded_xy=center[:,:2]+radial*torch.clamp(3/radius.clamp_min(1e-6),max=1.)
    goal=torch.cat([bounded_xy,goal[:,2:3].clamp(e.z_min,e.z_max)],-1);known*=inside.float()
    if stage>=2:
        global_goal=(torch.rand(n,device=dev)<.65)&mode.bool()
        u=torch.rand(n,device=dev);r=3*torch.sqrt(u);theta=2*math.pi*torch.rand(n,device=dev)
        xyz=torch.stack([center[:,0]+r*theta.cos(),center[:,1]+r*theta.sin(),
                         e.z_min+(e.z_max-e.z_min)*torch.rand(n,device=dev)],-1)
        goal=torch.where(global_goal[:,None],xyz,goal);known[global_goal]=0
    return dict(goal=goal,cmd=cmd,mode=mode,anchor=s['pos'].clone(),known=known)


class Replay:
    """GPU ring of factual pre-reset transitions. No Python per-sample stack."""
    def __init__(self,capacity,envs,device):
        self.capacity=capacity;self.envs=envs;self.device=device;self.data={};self.count=0
    def add(self,**row):
        i=self.count%self.capacity
        for k,v in row.items():
            if k not in self.data:self.data[k]=torch.empty((self.capacity,*v.shape),dtype=v.dtype,device=v.device)
            self.data[k][i].copy_(v.detach())
        self.count+=1
    def sample_starts(self,n,ee_fraction=.5):
        size=min(self.count,self.capacity);chosen=[]
        for mode,count in [(0,n-round(n*ee_fraction)),(1,round(n*ee_fraction))]:
            candidates=(self.data['valid'][:size] & (self.data['mode'][:size]==mode)).nonzero()
            if not len(candidates):candidates=self.data['valid'][:size].nonzero()
            if not len(candidates):raise RuntimeError('Replay contains no valid physical starts')
            chosen.append(candidates[torch.randint(len(candidates),(count,),device=self.device)])
        ix=torch.cat(chosen);ix=ix[torch.randperm(n,device=self.device)];t,j=ix.unbind(-1)
        return TensorDict({k:self.data[k][t,j] for k in ['state','goal','cmd','mode','prev','anchor','known']},batch_size=[n])
    def wm_batch(self,n,horizon=8,terminal=False,precursor=False):
        size=min(self.count,self.capacity)
        if size<horizon:return None
        if precursor:
            candidates=(self.data['terminated'][:size]&self.data['valid'][:size]).nonzero()
            if not len(candidates):return None
            ends,j=candidates.unbind(-1)
            ts=(ends[:,None]-horizon+1+torch.arange(horizon,device=self.device)[None,:])%self.capacity
            if size<self.capacity:
                keep=ends>=horizon-1;ts=ts[keep];j=j[keep]
            if not len(j):return None
            ep=self.data['episode'][ts,j[:,None]];tick=self.data['tick'][ts,j[:,None]]
            valid=self.data['valid'][ts,j[:,None]].all(-1)&(ep==ep[:,:1]).all(-1)
            valid&=(tick==tick[:,:1]+torch.arange(horizon,device=self.device)[None,:]).all(-1)
            valid&=~self.data['done'][ts[:,:-1],j[:,None]].any(-1)
            ids=valid.nonzero().flatten()
            if not len(ids):return None
            ids=ids[torch.randint(len(ids),(n,),device=self.device)];ts=ts[ids];j=j[ids]
        elif terminal:
            candidates=(self.data['terminated'][:size]&self.data['valid'][:size]).nonzero()
            if len(candidates)==0:return None
            idx=candidates[torch.randint(len(candidates),(n,),device=self.device)];ts=idx[:,0:1];j=idx[:,1];horizon=1
        else:
            # Logical time is checked too: ring-wrap must never concatenate unrelated transitions.
            count=n*4;t=torch.randint(max(1,size-horizon+1),(count,),device=self.device)
            j=torch.randint(self.envs,(count,),device=self.device)
            ts=t[:,None]+torch.arange(horizon,device=self.device)[None,:]
            ep=self.data['episode'][ts,j[:,None]];tick=self.data['tick'][ts,j[:,None]]
            valid=self.data['valid'][ts,j[:,None]].all(-1)&(ep==ep[:,:1]).all(-1)
            valid&=(tick==tick[:,:1]+torch.arange(horizon,device=self.device)[None,:]).all(-1)
            valid&=~self.data['done'][ts[:,:-1],j[:,None]].any(-1)
            ids=valid.nonzero().flatten()
            if not len(ids):return None
            ids=ids[torch.randint(len(ids),(n,),device=self.device)];ts=ts[ids];j=j[ids]
        states=self.data['state'][ts,j[:,None]]
        last=self.data['next_state'][ts[:,-1],j]
        states=torch.cat([states,last[:,None]],1)
        actions=self.data['action'][ts,j[:,None]];actions=torch.cat([actions,actions[:,-1:]],1)
        return TensorDict({'state':states,'action':actions,'sim_ctx':states.new_zeros(n,horizon+1,0)},batch_size=[n,horizon+1])

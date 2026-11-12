"""P4-frozen whole-body manager and differentiable EE objective. No robot IO."""
from bootstrap import HERE,ROOT
import math,json,torch
from torch import nn
from contract import DT,tilt_limit,set_requested,advance_command,policy_goal
from supermimic.rotations import mv,mtv
from task import TiltLocomotionEnv
from policy import make_policy
from actuation import FilteredWorldModel,advance_actuator
import actuation
# Match the accepted deployment filter; physics WM is conditioned on applied targets.
w=2*math.pi*11/50;g=1/(1+math.tan(w/2));actuation.B=(g,-2*g*math.cos(w),g);actuation.A=(1.,actuation.B[1],2*g-1)

def ee(arm,s):
 p,r=arm.fk(s['aq']);local=p[...,-1,:]+.125*r[...,-1,:,2]
 return s['pos']+mv(s['rot'],local),s['rot']@r[...,-1,:,:]

def errors(arm,s,goal):
 p,r=ee(arm,s);dp=(p-goal[...,:3]).norm(dim=-1)
 relative=r.transpose(-1,-2)@goal[...,3:].reshape(-1,3,3)
 cos=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
 return dp,torch.acos(cos),cos

def observe(s,ctx,goal,last,arm=None,xi=None):
 rel=mtv(s['rot'],goal[:,:3]-s['pos'])/3
 rr=(s['rot'].transpose(-1,-2)@goal[:,3:].reshape(-1,3,3)).flatten(1)
 if arm is not None:
  ep,er=ee(arm,s)
  rel=mtv(s['rot'],goal[:,:3]-ep)/.5
  rr=(er.transpose(-1,-2)@goal[:,3:].reshape(-1,3,3)).flatten(1)
 return torch.cat([rel,rr,mtv(s['rot'],s['v'])/2,mtv(s['rot'],s['w']),-s['rot'][:,2,:],s['q'],s['qd']/10,s['aq'],s['aqd'],ctx,last,s['pos'][:,2:3],xi[:,93:105]],-1)

class Manager(nn.Module):
 def __init__(self,dim,arm=None):
  super().__init__();self.arm=arm;self.net=nn.Sequential(nn.Linear(dim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,12),nn.Tanh())
  nn.init.normal_(self.net[-2].weight,std=.02);nn.init.zeros_(self.net[-2].bias)
 def forward(self,s,ctx,goal,last,xi):return self.net(observe(s,ctx,goal,last,arm=self.arm,xi=xi))

def commands(u):
 xy=u[:,:2]*u.new_tensor([2.,1.]);xy=xy*(2/xy.norm(dim=-1,keepdim=True).clamp_min(2))
 height=torch.where(u[:,5]>=0,.05*u[:,5],.2*u[:,5]);lim=tilt_limit(height)
 return torch.cat([xy,.5*u[:,2:3],u[:,3:5]*lim[:,None],height[:,None]],-1)

def joint_command_margin(arm):
 # Training-trajectory directed tracking lag +20ms motion maximum, +5mrad,
 # rounded up; J2 has11boundary samples, so use conservative30mrad fallback.
 # Command standoff only. Actual joint bounds and sampled goal set unchanged.
 return arm.lower.new_tensor([.030,.030,.025,.020,.025,.025])

def arm_action(arm,s,xi,u):
 speed=torch.maximum(s['v'][:,:2].norm(dim=-1),xi[:,105]);speed=torch.maximum(speed,.4*torch.maximum(s['w'][:,2].abs(),xi[:,106]))
 scale,limit=arm.envelope(speed);last=xi[:,93:99];oldvel=xi[:,99:105]
 wanted=u[:,6:]*limit[:,None]
 # Empirical command standoff protects against measured motor tracking transients.
 lo=arm.lower+joint_command_margin(arm);hi=arm.upper-joint_command_margin(arm)
 gap_up=(hi-torch.maximum(last,s['aq'])).clamp_min(0)
 gap_down=(torch.minimum(last,s['aq'])-lo).clamp_min(0)
 v_up=(2*1.2*gap_up+(1.2*DT)**2).sqrt()-1.2*DT
 v_down=(2*1.2*gap_down+(1.2*DT)**2).sqrt()-1.2*DT
 wanted=wanted.clamp(-v_down,v_up)
 # Anticipate real motor tracking error and0.2s of outward physical velocity.
 # This only changes desired velocity; original acceleration, speed, and fixed
 # command-position bounds still apply below. No direct q/qdot assignment.
 tracking_down=torch.relu(last-s['aq']);tracking_up=torch.relu(s['aq']-last)
 inward_floor=(lo+tracking_down+.2*torch.relu(-s['aqd'])-last)/.05
 inward_ceiling=(hi-tracking_up-.2*torch.relu(s['aqd'])-last)/.05
 inward_floor=inward_floor.clamp(-limit[:,None],limit[:,None])
 inward_ceiling=inward_ceiling.clamp(-limit[:,None],limit[:,None])
 wanted=torch.minimum(torch.maximum(wanted,inward_floor),inward_ceiling)
 vel=oldvel+(wanted-oldvel).clamp(-1.2*DT,1.2*DT)
 proposed=last+DT*vel
 # Speed-dependent range constrains further motion, without snapping an extended arm.
 lo=arm.lower+joint_command_margin(arm);hi=arm.upper-joint_command_margin(arm)
 inward=proposed.clamp(lo,hi);target=last+(inward-last).clamp(-limit[:,None]*DT,limit[:,None]*DT)
 vel=(target-last)/DT
 return target,torch.cat([xi[:,81:93],target,vel,xi[:,105:107]],-1)


def clearance_reserve(arm,s):
 # Only arm-vs-moving-leg pairs need the larger movement allowance.
 # Physical safety thresholds remain in arm.safety(), unchanged.
 ids=arm.body[arm.pairs];leg_pair=((ids>=1)&(ids<=12)).any(-1)
 ordinary=arm.pair_margins>.0015
 return torch.where(ordinary,torch.where(leg_pair,.005,.002),0.)

def safe_arm_action(arm,s,xi,u):
 target,ext=arm_action(arm,s,xi,u)
 current=arm.safety_details(s,s['aq'])
 valid=current[0].clone()
 pair_buffer=clearance_reserve(arm,s)
 reserve=torch.minimum(current[5],pair_buffer)
 ground_reserve=torch.minimum(current[2]-arm.margin,torch.full_like(current[2],.002))
 for blend in [.5,1.]:
  details=arm.safety_details(s,s['aq']+blend*(target-s['aq']))
  valid=valid & details[0] & (details[5]>=reserve-1e-7).all(-1) & (details[2]-arm.margin>=ground_reserve-1e-7)
 target=torch.where(valid[:,None],target,xi[:,93:99])
 ext=torch.cat([ext[:,:12],target,torch.where(valid[:,None],ext[:,18:24],torch.zeros_like(target)),ext[:,24:]],-1)
 return target,ext,valid


def proposal_barrier(arm,s,xi,u):
 """Differentiable0.1s intent preview, before the hard command rejection.
 Guidance loss only: actual arm speed/acceleration/torque/safety stay unchanged.
 """
 speed=torch.maximum(s['v'][:,:2].norm(dim=-1),xi[:,105])
 speed=torch.maximum(speed,.4*torch.maximum(s['w'][:,2].abs(),xi[:,106]))
 _,limit=arm.envelope(speed)
 look=s['aq']+.1*u[:,6:]*limit[:,None]
 details=arm.safety_details(s,look)
 reserve=clearance_reserve(arm,s)
 pair_cost=torch.relu((reserve-details[5])/.002).square().sum(-1)
 ground_cost=torch.relu((arm.margin+.002-details[2])/.002).square()
 joint_cost=(torch.relu((arm.lower+joint_command_margin(arm)-look)/.02).square()+torch.relu((look-(arm.upper-joint_command_margin(arm)))/.02).square()).sum(-1)
 return pair_cost+ground_cost+joint_cost

def physical_state_barrier(arm,s,details=None):
 """Dimensionless soft costs; actual safety limits and command bounds unchanged.
 Include actual outward velocity in joint risk, rather than just command intent.
 All body/leg/arm geometry gradients reach the imagined physical state.
 """
 if details is None:details=arm.safety_details(s,s['aq'])
 pair_reserve=clearance_reserve(arm,s)
 pair_scale=pair_reserve.clamp_min(.002)
 pair=torch.relu((pair_reserve-details[5])/pair_scale).square().amax(-1)
 ground=torch.relu((arm.margin+.002-details[2])/.01).square()
 lower_risk=(arm.lower+.03+.15*torch.relu(-s['aqd'])-s['aq'])/.03
 upper_risk=(s['aq']-(arm.upper-.03)+.15*torch.relu(s['aqd']))/.03
 joint=(torch.relu(lower_risk).square()+torch.relu(upper_risk).square()).sum(-1)
 return pair+ground+joint

class JointEnv(TiltLocomotionEnv):
 def __init__(self,device,n,**kw):
  super().__init__(device,n,**kw);self.upper=torch.zeros(n,12,device=device)
  self.arm.advance_fast=self._controlled_arm
 def _controlled_arm(self,s,xi):
  target,ext,valid=safe_arm_action(self.arm,s,xi,self.upper)
  return target,ext,~valid

def reward(arm,s,goal,u,prev):
 dist,angle,cos=errors(arm,s,goal)
 workspace=torch.zeros_like(dist)
 joint=(torch.relu(arm.lower-s['aq'])+torch.relu(s['aq']-arm.upper)).sum(-1)
 tilt=torch.acos(s['rot'][:,2,2].clamp(-1,1));fall=torch.sigmoid((.25-s['pos'][:,2])/.015)+torch.sigmoid((tilt-.9)/.05)
 # Dense costs retain gradients at far targets. No privileged witness is a teacher.
 precision=torch.exp(-dist.square()/.01**2-(1-cos)/(1-math.cos(math.radians(10))))
 orientation_weight=3.0
 safe_angle=torch.acos(cos.clamp(-1+1e-6,1-1e-6))
 pose_cost=torch.logsumexp(torch.stack([dist/.01,safe_angle/math.radians(10)],-1),-1)-math.log(2)
 approach=torch.exp(-dist.square()/.10**2)
 # Intermediate pose shaping supplies gradients before the strict 1cm basin.
 middle=torch.exp(-dist.square()/.05**2-(1-cos)/(1-math.cos(math.radians(25))))
 body_cost=.15*torch.exp(-dist.square()/.35**2).detach()*u[:,:6].square().sum(-1)
 return (4*precision+4*middle+approach-pose_cost-body_cost-.015*(u-prev).square().mean(-1)-.002*u.square().mean(-1)-3*workspace-3*joint-5*fall)*DT

def sample_goals(env,n,stage,arm_fraction=None,yaw_limit=None,radius_max=None):
 """Feasible terminal configurations, not necessarily reachable without moving."""
 dev=env.device;arm=env.arm
 if stage==0:
  q=arm.home+.18*(arm.lower+(arm.upper-arm.lower)*torch.rand(n,6,device=dev)-arm.home)
  s=dict(pos=torch.zeros(n,3,device=dev),rot=torch.eye(3,device=dev).expand(n,3,3),q=env.q0.expand(n,12),aq=q)
  s['pos'][:,2]=env.task.height
 else:
  bank=torch.load(HERE/'pose_bank.pt',map_location=dev,weights_only=True);ids=torch.randint(len(bank['pos']),(n,),device=dev)
  s={k:v[ids].clone() for k,v in bank.items()};s['aq']=arm.lower+(arm.upper-arm.lower)*torch.rand(n,6,device=dev)
  if arm_fraction is not None:s['aq']=arm.home+arm_fraction*(s['aq']-arm.home)
 initial=dict(pos=torch.zeros(n,3,device=dev),rot=torch.eye(3,device=dev).expand(n,3,3),aq=arm.home.expand(n,6));initial['pos'][:,2]=env.task.height
 def admission():
  valid=arm.safety_fast(s,s['aq'])[0]
  if stage==0:
   pp,rr=ee(arm,s);dd,aa,_=errors(arm,initial,torch.cat([pp,rr.flatten(1)],-1));valid&=(dd>.025)|(aa>math.radians(12))
  return valid
 valid=admission()
 for _ in range(512):
  bad=~valid
  if not bad.any():break
  proposal=arm.lower+(arm.upper-arm.lower)*torch.rand(int(bad.sum()),6,device=dev)
  s['aq'][bad]=arm.home+.18*(proposal-arm.home) if stage==0 else (proposal if arm_fraction is None else arm.home+arm_fraction*(proposal-arm.home))
  valid=admission()
 if not valid.all():
  # Invalid sampling is an explicit error, never counted as a task failure.
  raise RuntimeError('Goal witness safety rejection budget exhausted')
 p,R=ee(arm,s)
 if stage:
  yaw=(torch.rand(n,device=dev)*2-1)*(math.pi if yaw_limit is None else yaw_limit);c=yaw.cos();sn=yaw.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
  rot=torch.stack([c,-sn,z,sn,c,z,z,z,o],-1).reshape(n,3,3)
  p=mv(rot,p);R=rot@R
  radius=torch.minimum(torch.full((n,),(.25 if stage==1 else 2.1) if radius_max is None else radius_max,device=dev),(3-p[:,:2].norm(dim=-1)).clamp_min(0))*torch.rand(n,device=dev).sqrt();theta=torch.rand(n,device=dev)*2*math.pi
  p[:,:2]+=radius[:,None]*torch.stack([theta.cos(),theta.sin()],-1)
 # Clipping would destroy feasibility; configured range includes <=.9m reach +2.1m translation.
 if (p[:,:2].norm(dim=-1)>3.001).any():raise RuntimeError('Goal beyond 3m sampling contract')
 return torch.cat([p,R.flatten(1)],-1)


def relabel_precision_goals(arm,s,original):
 """Actor-only goal relabeling; never modify factual WM samples or evaluation.
 Half original task goals, quarter current achieved pose, quarter nearby safe FK.
 No witness joint targets are given to the actor or used in its objective.
 """
 with torch.no_grad():
  n=len(original);out=original.clone();order=torch.randperm(n,device=original.device)
  hold=order[:n//4];near=order[n//4:n//2]
  p,r=ee(arm,s);achieved=torch.cat([p,r.flatten(1)],-1)
  out[hold]=achieved[hold]
  perturbed={k:v.clone() for k,v in s.items()}
  perturbed['aq']=(s['aq']+.06*torch.randn_like(s['aq'])).clamp(arm.lower,arm.upper)
  valid=arm.safety_fast(perturbed,perturbed['aq'])[0]
  p,r=ee(arm,perturbed);candidate=torch.cat([p,r.flatten(1)],-1)
  out[near]=torch.where(valid[near,None],candidate[near],achieved[near])
 return out

"""B2+A3 geometry and direct actuation for the joint policy."""

import math
import torch

import actuation
from actuation import advance_actuator
from b2physics.rotations import mv
from contract import DT
from task import TiltLocomotionEnv

# Match the original 11 Hz filter used in physical collection.
w = 2 * math.pi * 11 / 50
g = 1 / (1 + math.tan(w / 2))
actuation.B = (g, -2 * g * math.cos(w), g)
actuation.A = (1., actuation.B[1], 2 * g - 1)


def ee(arm,s):
 p,r=arm.fk(s['aq']);local=p[...,-1,:]+.125*r[...,-1,:,2]
 return s['pos']+mv(s['rot'],local),s['rot']@r[...,-1,:,:]

def arm_view(u):return torch.cat([torch.zeros_like(u[...,:6]),u[...,12:18]],-1)

def wm_action(u):return torch.cat([u,u.new_zeros((*u.shape[:-1],2))],-1)

def base_ground_clearance(arm,s):
 ids=arm.body==0
 center=s['pos'][:,None,:]+(s['rot'][:,None,:,:]@arm.bc[ids,:,None]).squeeze(-1)
 rotation=s['rot'][:,None,:,:]@arm.br[ids]
 return (center[...,2]-(rotation[...,2,:].abs()*arm.bh[ids]).sum(-1)).amin(-1)

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

class JointEnv(TiltLocomotionEnv):
 def __init__(self,device,n,**kw):
  super().__init__(device,n,**kw);self.upper=torch.zeros(n,18,device=device)
  self.arm.advance_fast=self._controlled_arm
  self.task.physical_body_clearance=lambda s:base_ground_clearance(self.arm,s)
 def _controlled_arm(self,s,xi):
  target,ext,valid=safe_arm_action(self.arm,s,xi,arm_view(self.upper))
  return target,ext,~valid

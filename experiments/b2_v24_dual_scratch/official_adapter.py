"""B2+A3 representation and actuator adapter for the unmodified official WM.
V24's fresh Actor, Critic, rewards and imagined environment are in dual.py.
The legacy namespace supplies simulation geometry only; no P4 policy is loaded.
"""
from bootstrap import HERE
import torch
from torch import nn
from supermimic.models import Dynamics
from supermimic.representations import GenCoordRepresentation
from joint import ee, safe_arm_action, advance_actuator, DT, base_ground_clearance
from b2physics.rotations import quat_to_mat
from mjlab.utils.lab_api.math import quat_from_matrix as mat_to_quat

def pack(s, xi):
 return torch.cat([s['pos'],mat_to_quat(s['rot']),s['q'],s['aq'],s['v'],s['w'],s['qd'],s['aqd'],xi],-1)

def unpack(full):
 return dict(pos=full[...,:3],rot=quat_to_mat(full[...,3:7]),q=full[...,7:19],aq=full[...,19:25],v=full[...,25:28],w=full[...,28:31],qd=full[...,31:43],aqd=full[...,43:49])

class Representation(GenCoordRepresentation):
 LOSS_WEIGHT_CONFIG={**GenCoordRepresentation.LOSS_WEIGHT_CONFIG,"root_linear_velocity":"nominal_root_linear_velocity","root_angular_velocity":"nominal_root_angular_velocity","leg_velocity":"nominal_leg_velocity","arm_pos":"nominal_error_arm_pos","arm_vel":"nominal_error_arm_vel","ee_pos":"nominal_error_ee_pos","response_arm_pos":"nominal_response_arm_pos","response_arm_vel":"nominal_response_arm_vel","response_ee":"nominal_response_ee","response_root_velocity":"nominal_response_root_velocity","response_leg_velocity":"nominal_response_leg_velocity"}
 def __init__(self,q0,arm=None,clearance_loss=False):
  super().__init__(18,q0);self.dt=DT;self.substep_coeff=.625;self.arm=arm;self.rep_state_dim=133
  self.clearance_loss=clearance_loss
  if clearance_loss:self.LOSS_WEIGHT_CONFIG={**self.LOSS_WEIGHT_CONFIG,"body_clearance":"nominal_body_clearance"}
 def represent(self,full):
  from supermimic.utils.gencoord_helpers import compute_gencoord_state
  return torch.cat([compute_gencoord_state(full[...,:49]),full[...,69:156]],-1)
 def apply_prediction(self,aux,rep,pred,dt):
  a,r=super().apply_prediction(aux[...,:7],rep[...,:46],pred,dt)
  return a,torch.cat([r,rep[...,46:]],-1)
 def reconstruct_full_state(self,aux,rep):return super().reconstruct_full_state(aux[...,:7],rep[...,:46])
 def joint_state(self,rep):return super().joint_state(rep[...,:46])
 def extract_aux_state(self,full):return torch.cat([full[...,:7],full[...,49:162]],-1)
 def loss_blocks(self,rep,aux):
  rep=rep[...,:46]
  blocks=super().loss_blocks(rep,aux[...,:7]);blocks['root_linear_velocity']=rep[...,4:7];blocks['root_angular_velocity']=rep[...,7:10];blocks['leg_velocity']=rep[...,28:40];blocks['arm_pos']=rep[...,22:28];blocks['arm_vel']=rep[...,40:46]
  if self.arm is not None:
   full=self.reconstruct_full_state(aux[...,:7],rep);shape=full.shape[:-1];st=unpack(full.reshape(-1,49));blocks['ee_pos']=ee(self.arm,st)[0].reshape(*shape,3)
  else:blocks['ee_pos']=aux[...,:3]*0
  for source,name in [('arm_pos','response_arm_pos'),('arm_vel','response_arm_vel'),('ee_pos','response_ee'),('root_linear_velocity','response_root_velocity'),('leg_velocity','response_leg_velocity')]:
   if getattr(self,'paired_response',False):
    grouped=blocks[source].reshape(-1,self.response_group_size,*blocks[source].shape[1:]);blocks[name]=(grouped-grouped[:,:1]).reshape_as(blocks[source])
   else:blocks[name]=blocks[source]*0
  if self.clearance_loss:
   full=self.reconstruct_full_state(aux[...,:7],rep);shape=full.shape[:-1]
   blocks['body_clearance']=base_ground_clearance(self.arm,unpack(full.reshape(-1,49))).reshape(*shape,1)
  return blocks

class RobotDynamics(nn.Module):
 """Exact expanded-donor input contract; V22 analytic actuator state retained."""
 def __init__(self,rep,task,hidden=(2048,2048,2048)):
  super().__init__();self.rep=rep;self.task=task
  scale=torch.cat([task.action_scale,torch.ones(8,device=task.device)])
  offset=torch.cat([task.q0,torch.zeros(8,device=task.device)])
  self.core=Dynamics(rep,20,26,scale,offset,hidden)
 @property
 def state_normalizer(self):return self.core.state_normalizer
 @property
 def action_normalizer(self):return self.core.action_normalizer
 def normalizers(self):return self.core.normalizers()
 def proc(self,a):return self.core.proc(a)
 def forward(self,rep,a,aux,sim_ctx):
  a=torch.cat([a[...,:12],a[...,12:18].clamp(-1,1),a[...,18:20]],-1)
  xi=torch.cat([aux[...,7:112],a[...,18:20],aux[...,114:120]],-1)
  # The donor places actuator memory in the represented state and requests in action.
  donor_rep=torch.cat([rep[...,:46],xi[...,20:107]],-1)
  context=torch.cat([xi[...,:20],xi[...,107:113]],-1)
  state=unpack(self.rep.reconstruct_full_state(aux[...,:7],rep))
  _,leg_next=advance_actuator(a[...,:12],xi,self.task)
  arm_u=torch.cat([torch.zeros_like(a[...,:6]),a[...,12:18]],-1)
  _,arm_next,_=safe_arm_action(self.rep.arm,state,xi,arm_u)
  na,nr=self.core(donor_rep,a,aux[...,:7],context)
  nxi=torch.cat([leg_next[...,:81],arm_next[...,:24],a[...,18:20],xi[...,107:113]],-1)
  return torch.cat([na,nxi],-1),torch.cat([nr[...,:46],nxi[...,20:107]],-1)

POSITION_TOLERANCE=.015
# Numerical standstill tolerances; every condition must hold simultaneously.
STILL_LIMITS=dict(body_linear_m_s=.01,body_angular_rad_s=.02,leg_joint_rad_s=.05,arm_joint_rad_s=.02,ee_linear_m_s=.005)
def motion_values(arm,s,previous):
 return dict(body_linear=s['v'].norm(dim=-1),body_angular=s['w'].norm(dim=-1),leg_joint=s['qd'].abs().amax(-1),arm_joint=s['aqd'].abs().amax(-1),ee_linear=(ee(arm,s)[0]-ee(arm,previous)[0]).norm(dim=-1)/DT)
def still_mask(values):
 limits=list(STILL_LIMITS.values())
 return torch.stack([values[k]<=limit for k,limit in zip(['body_linear','body_angular','leg_joint','arm_joint','ee_linear'],limits)],-1).all(-1)

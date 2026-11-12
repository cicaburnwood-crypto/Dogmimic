"""Exact deployed 11 Hz notch and delay state, propagated outside learned physics.

XI: physical20, previous applied12, input1/2 (24), DFII z1/z2 (24), ready1.
The physics net retains P1's input dimensions: applied target, physical20 and
previous applied target. Therefore filter memory is not an unobserved disturbance.
"""
import math
import torch
from supermimic.networks import WorldModel

XI_DIM=107
omega=2*math.pi*11./50.
gain=1/(1+math.tan(omega/4.))
B=(gain,-2*gain*math.cos(omega),gain)
A=(1.,B[1],2*gain-1.)

def advance_actuator(action,xi,task):
    q0,scale=task.q0,task.action_scale
    raw=q0+scale*action.clamp(-1.,1.)-xi[...,1:13]
    u=(raw.clamp(task.qmin+.02,task.qmax-.02)-q0)/scale
    ready=xi[...,80:81]>.5
    z1=torch.where(ready,xi[...,56:68],(1-B[0])*u)
    z2=torch.where(ready,xi[...,68:80],(B[2]-A[2])*u)
    y=B[0]*u+z1
    physical=q0+scale*y
    bounded=(physical.clamp(task.qmin+.02,task.qmax-.02)-q0)/scale
    # Detect actual saturation, not cancellation error in normalized units.
    clipped=((physical<task.qmin+.02)|(physical>task.qmax-.02)).any(-1,keepdim=True)
    nz1=B[1]*u-A[1]*y+z2;nz2=B[2]*u-A[2]*y
    nz1=torch.where(clipped,(1-B[0])*bounded,nz1)
    nz2=torch.where(clipped,(B[2]-A[2])*bounded,nz2)
    nxt=torch.cat([xi[...,:20],bounded,u,xi[...,32:44],nz1,nz2,torch.ones_like(xi[...,80:81]),xi[...,81:]],-1)
    return bounded,nxt

def consumed_targets(action,xi,task):
    """Four exact 5 ms targets in physical joint coordinates, for verification."""
    y,nxt=advance_actuator(action,xi,task)
    j=torch.arange(4,device=action.device)
    old=xi[...,20:32]
    values=torch.where((j<xi[...,19:20]*2)[...,None],old[...,None,:],y[...,None,:])
    return task.q0+task.action_scale*values,nxt

class FilteredWorldModel(WorldModel):
    def __init__(self,task,feat_dim,xi_dim,**kwargs):
        from supermimic.networks import mlp,Normalizer
        assert xi_dim==XI_DIM
        super().__init__(task,feat_dim,32+24,**kwargs)
        self.net=mlp(feat_dim+12+32+24,kwargs.get('hidden',(2048,)*3),48,act=torch.nn.SiLU,out_gain=.01)
        self.out_norm=Normalizer(48,eps=1e-4)

    def inputs(self,s,a,xi,feats=None,arm_target=None):
        y,_=advance_actuator(a,xi,self.task)
        bias=xi[...,1:13]/self.task.action_scale
        net_xi=torch.cat([xi[...,:20],xi[...,20:32]+bias],-1)
        old=super().inputs(s,y+bias,net_xi,feats)
        if arm_target is None:raise ValueError('WM requires the executed or explicitly imagined arm target')
        return torch.cat([old,s['aq'],s['aqd'],arm_target,xi[...,93:99]],-1)

    def step(self,s,a,xi,feats=None,arm_target=None):
        from supermimic.networks import run_mlp
        from state import apply_delta
        x=self.inputs(s,a,xi,feats,arm_target)
        d=run_mlp(self.net,self.in_norm(x))*self.out_norm.std+self.out_norm.mean
        return apply_delta(s,d)

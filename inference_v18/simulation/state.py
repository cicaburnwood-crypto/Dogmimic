"""Leg state contract extended by 6 measured arm angles and velocities."""
import torch
from supermimic.task import flat_to_state as leg_flat,state_to_flat as leg_pack
from supermimic.networks import state_delta as leg_delta,state_error as leg_error,apply_delta as leg_apply
SDIM=49
DELTA_DIM=48
def flat_to_state(x):
 s=leg_flat(x[...,:37]);s.update(aq=x[...,37:43],aqd=x[...,43:49]);return s
def state_to_flat(s):return torch.cat([leg_pack(s),s['aq'],s['aqd']],-1)
def state_delta(s,t):return torch.cat([leg_delta(s,t),t['aq']-s['aq'],t['aqd']-s['aqd']],-1)
def state_error(s,t):return torch.cat([leg_error(s,t),s['aq']-t['aq'],s['aqd']-t['aqd']],-1)
def apply_delta(s,d):
 t=leg_apply(s,d[...,:36]);t.update(aq=s['aq']+d[...,36:42],aqd=s['aqd']+d[...,42:48]);return t
state_delta_c=torch.compile(state_delta,dynamic=False)

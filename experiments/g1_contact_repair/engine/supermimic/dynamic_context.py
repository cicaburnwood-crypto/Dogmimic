"""Chronological 5ms actuator queue and force-pulse phase, privileged WM inputs only."""
import torch
BASE=42
QUEUE=9*29
DIM=BASE+QUEUE+1

def advance(xi,action,q0,action_scale):
    queue=xi[...,BASE:BASE+QUEUE].reshape(*xi.shape[:-1],9,29)
    target=q0+action_scale*action-xi[...,1:30]
    updated=torch.cat([queue[...,4:,:],target.unsqueeze(-2).expand(*target.shape[:-1],4,29)],-2)
    return torch.cat([xi[...,:BASE],updated.flatten(-2),xi[...,-1:]+1/200.],-1)

def actual_force(xi):
    frame=(xi[...,-1:]*200.).round()
    return xi[...,39:42]*((frame>=48)&(frame<68))

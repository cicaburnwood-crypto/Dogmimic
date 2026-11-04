"""Exact delayed command sequence consumed by one 20ms physical transition."""
import torch

def consumed_targets(xi,action,q0,action_scale):
    queue=xi[...,42:303].reshape(*xi.shape[:-1],9,29)
    target=q0+action_scale*action-xi[...,1:30]
    timeline=torch.cat([queue,target.unsqueeze(-2).expand(*target.shape[:-1],4,29)],-2)
    # extra[:,4] occupies xi[:,37]; delay units are fractions of 40ms.
    lag=(xi[...,37]*8).round().long()
    indices=9+torch.arange(4,device=xi.device)-lag.unsqueeze(-1)
    return timeline.gather(-2,indices.unsqueeze(-1).expand(*indices.shape,29))
